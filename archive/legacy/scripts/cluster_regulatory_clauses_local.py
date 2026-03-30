#!/usr/bin/env python3
"""
Cluster extracted regulatory clauses using a local no-API embedding:

  TF‑IDF → TruncatedSVD → UMAP → HDBSCAN

This is intended for fast iteration on the PI-requested "instrument discovery"
step without waiting on provider embeddings.

Input
-----
- clauses.jsonl produced by:
    scripts/rehydrate_regulatory_clause_extraction_openai_batch_results.py
  Required fields:
    - clause_id (string)
    - requirement (string) OR clause_text (string)
  Optional fields:
    - doc_id, chunk_id, motive, modality, issue_date, classification_label

Output (output_dir)
-------------------
- clusters.jsonl
- cluster_summary.json
- cluster_keywords.jsonl
- umap_2d.npy + scatter.png
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from os.path import expanduser
from pathlib import Path
from typing import Any

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cluster extracted regulatory clauses (local TF‑IDF → UMAP → HDBSCAN).")
    ap.add_argument("--clauses-jsonl", required=True, help="Path to clauses.jsonl")
    ap.add_argument("--output-dir", required=True, help="Directory to write clusters + plots into")

    ap.add_argument(
        "--text-field",
        default="requirement",
        help="Which field to embed/cluster (default: requirement).",
    )
    ap.add_argument(
        "--filter-motive",
        action="append",
        default=[],
        help="Optional: keep only clauses where motive equals this value (repeatable).",
    )

    ap.add_argument("--min-chars", type=int, default=80, help="Drop clauses with fewer than this many chars.")
    ap.add_argument("--max-docs", type=int, default=0, help="Optional cap on number of clauses (0 disables).")
    ap.add_argument("--max-doc-chars", type=int, default=0, help="Optional: truncate each clause text (0 disables).")

    # TF‑IDF
    ap.add_argument("--max-features", type=int, default=80_000, help="TF‑IDF vocabulary cap (0 disables).")
    ap.add_argument("--min-df", default="2", help="TF‑IDF min_df (count if >=1 else fraction).")
    ap.add_argument("--max-df", default="0.95", help="TF‑IDF max_df (count if >=1 else fraction).")
    ap.add_argument("--ngram-max", type=int, default=2, help="Max ngram size (1=unigrams, 2=uni+bi).")

    # SVD
    ap.add_argument("--svd-dim", type=int, default=200, help="TruncatedSVD output dimension.")
    ap.add_argument("--svd-random-state", type=int, default=0)

    # UMAP
    ap.add_argument("--umap-n-neighbors", type=int, default=25)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    ap.add_argument("--umap-n-components", type=int, default=5, help="UMAP components for clustering (not plotting).")
    ap.add_argument("--umap-metric", default="cosine")
    ap.add_argument("--umap-random-state", type=int, default=0)

    # HDBSCAN
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=25)
    ap.add_argument("--hdbscan-min-samples", type=int, default=10)

    ap.add_argument("--max-points-plot", type=int, default=20000, help="Downsample scatter plot if too many points.")
    ap.add_argument("--top-keywords", type=int, default=12, help="How many TF‑IDF terms to save per cluster.")
    return ap.parse_args()


def _parse_df(name: str, raw: str) -> int | float:
    s = str(raw).strip()
    if not s:
        raise SystemExit(f"--{name} must be non-empty")
    if s.isdigit():
        iv = int(s)
        if iv <= 0:
            raise SystemExit(f"--{name} must be > 0; got {iv}")
        return iv
    try:
        fv = float(s)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"--{name} must be an int count or float fraction; got {raw!r}") from exc
    if fv <= 0:
        raise SystemExit(f"--{name} must be > 0; got {fv}")
    if fv >= 1 and abs(fv - round(fv)) < 1e-9:
        return int(round(fv))
    if fv >= 1:
        raise SystemExit(f"--{name} as a fraction must be < 1; got {fv}. Use an integer count.")
    return float(fv)


def _iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _cluster_top_terms(
    *,
    vectorizer: TfidfVectorizer,
    tfidf,
    ids: list[str],
    labels: np.ndarray,
    k: int,
) -> dict[int, list[str]]:
    """
    Compute cluster keywords by mean TF‑IDF weight within cluster.
    """
    names = vectorizer.get_feature_names_out()
    out: dict[int, list[str]] = {}
    if k <= 0:
        return out

    # Work on sparse matrix.
    # For each cluster, sum the TF‑IDF vectors then divide by n.
    uniq = sorted(set(int(x) for x in labels.tolist()))
    for lab in uniq:
        if lab == -1:
            continue
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue
        v = tfidf[idx].mean(axis=0)
        # v is 1xV matrix
        arr = np.asarray(v).ravel()
        if arr.size == 0:
            continue
        top = np.argsort(-arr)[: int(k)]
        out[int(lab)] = [str(names[i]) for i in top if float(arr[i]) > 0]
    return out


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()

    in_path = Path(expanduser(args.clauses_jsonl)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.is_file():
        raise SystemExit(f"--clauses-jsonl not found: {in_path}")

    text_field = str(args.text_field).strip()
    if not text_field:
        raise SystemExit("--text-field must be non-empty")

    min_chars = int(args.min_chars)
    if min_chars < 0:
        raise SystemExit("--min-chars must be >= 0")
    max_docs = int(args.max_docs)
    if max_docs < 0:
        raise SystemExit("--max-docs must be >= 0")
    max_doc_chars = int(args.max_doc_chars)
    if max_doc_chars < 0:
        raise SystemExit("--max-doc-chars must be >= 0")
    max_doc_chars_opt = max_doc_chars if max_doc_chars > 0 else None

    motives_keep = [str(x).strip() for x in (args.filter_motive or []) if str(x).strip()]
    motives_keep_set = set(motives_keep)

    clause_ids: list[str] = []
    texts: list[str] = []
    metas: list[dict[str, Any]] = []

    for obj in _iter_jsonl(in_path):
        cid = obj.get("clause_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        cid = cid.strip()

        if motives_keep_set:
            mot = obj.get("motive")
            if not isinstance(mot, str) or mot.strip() not in motives_keep_set:
                continue

        txt = obj.get(text_field)
        if not isinstance(txt, str) or not txt.strip():
            continue
        t = txt.strip()
        if max_doc_chars_opt is not None and len(t) > max_doc_chars_opt:
            t = t[:max_doc_chars_opt].rstrip()
        if len(t) < min_chars:
            continue

        clause_ids.append(cid)
        texts.append(t)
        metas.append(obj)

        if max_docs > 0 and len(clause_ids) >= max_docs:
            break

    if len(clause_ids) < 3:
        raise SystemExit(f"Need at least 3 clauses after filtering; got {len(clause_ids)}.")

    # doc_weight: downweight docs with lots of clauses so the clustering isn't driven by a few massive ordinances.
    doc_ids = [str(m.get("doc_id") or "") for m in metas]
    doc_counts = Counter(doc_ids)
    doc_weight = [1.0 / float(doc_counts[d]) if doc_counts.get(d, 0) > 0 and d else 1.0 for d in doc_ids]

    # TF‑IDF
    max_features = int(args.max_features)
    if max_features < 0:
        raise SystemExit("--max-features must be >= 0")
    max_features_opt = max_features if max_features > 0 else None

    min_df = _parse_df("min-df", str(args.min_df))
    max_df = _parse_df("max-df", str(args.max_df))
    if isinstance(min_df, float) and isinstance(max_df, float) and min_df >= max_df:
        raise SystemExit(f"Bad df bounds: min_df={min_df} must be < max_df={max_df}")

    ngram_max = int(args.ngram_max)
    if ngram_max <= 0 or ngram_max > 3:
        raise SystemExit("--ngram-max must be 1..3")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features_opt,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, int(ngram_max)),
    )
    tfidf = vectorizer.fit_transform(texts)
    if tfidf.shape[1] <= 1:
        raise SystemExit(f"TF‑IDF produced too few features (n_features={tfidf.shape[1]}).")

    # SVD
    svd_dim = int(args.svd_dim)
    if svd_dim <= 1:
        raise SystemExit("--svd-dim must be > 1")
    if svd_dim >= tfidf.shape[1]:
        svd_dim = max(2, min(svd_dim, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=int(svd_dim), random_state=int(args.svd_random_state))
    X_svd = svd.fit_transform(tfidf)
    X = normalize(X_svd, norm="l2", copy=False)

    # UMAP for clustering.
    reducer = umap.UMAP(
        n_neighbors=int(args.umap_n_neighbors),
        min_dist=float(args.umap_min_dist),
        n_components=int(args.umap_n_components),
        metric=str(args.umap_metric),
        random_state=int(args.umap_random_state),
    )
    X_umap = reducer.fit_transform(X)

    # HDBSCAN clustering.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(args.hdbscan_min_cluster_size),
        min_samples=int(args.hdbscan_min_samples),
    )
    labels = clusterer.fit_predict(X_umap)
    probs = getattr(clusterer, "probabilities_", None)
    if probs is None:
        probs = np.ones((len(labels),), dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)

    # 2D UMAP for plotting.
    reducer_2d = umap.UMAP(
        n_neighbors=int(args.umap_n_neighbors),
        min_dist=float(args.umap_min_dist),
        n_components=2,
        metric=str(args.umap_metric),
        random_state=int(args.umap_random_state),
    )
    X_2d = reducer_2d.fit_transform(X)
    np.save(out_dir / "umap_2d.npy", X_2d.astype(np.float32))

    # Write per-clause assignments.
    clusters_path = out_dir / "clusters.jsonl"
    with clusters_path.open("w", encoding="utf-8") as f:
        for cid, lab, prob, w, meta in zip(clause_ids, labels.tolist(), probs.tolist(), doc_weight, metas):
            row = {
                "clause_id": cid,
                "cluster_id": int(lab),
                "cluster_prob": float(prob),
                "doc_weight": float(w),
                # keep a small, audit-friendly subset of metadata
                "doc_id": meta.get("doc_id"),
                "chunk_id": meta.get("chunk_id"),
                "motive": meta.get("motive"),
                "modality": meta.get("modality"),
                "instrument_hint": meta.get("instrument_hint"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary + keywords.
    counts = Counter(int(x) for x in labels.tolist())
    keywords = _cluster_top_terms(
        vectorizer=vectorizer,
        tfidf=tfidf,
        ids=clause_ids,
        labels=np.asarray(labels, dtype=int),
        k=int(args.top_keywords),
    )

    summary = {
        "n_clauses": len(clause_ids),
        "n_noise": int(counts.get(-1, 0)),
        "n_clusters": int(len([k for k in counts.keys() if k != -1])),
        "cluster_sizes": {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[0])},
        "params": {
            "text_field": text_field,
            "filter_motive": motives_keep,
            "tfidf": {
                "max_features": max_features_opt,
                "min_df": min_df,
                "max_df": max_df,
                "ngram_max": int(ngram_max),
            },
            "svd": {"dim": int(svd_dim), "random_state": int(args.svd_random_state)},
            "umap": {
                "n_neighbors": int(args.umap_n_neighbors),
                "min_dist": float(args.umap_min_dist),
                "n_components": int(args.umap_n_components),
                "metric": str(args.umap_metric),
                "random_state": int(args.umap_random_state),
            },
            "hdbscan": {
                "min_cluster_size": int(args.hdbscan_min_cluster_size),
                "min_samples": int(args.hdbscan_min_samples),
            },
        },
    }
    (out_dir / "cluster_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Cluster keywords output.
    kw_path = out_dir / "cluster_keywords.jsonl"
    with kw_path.open("w", encoding="utf-8") as f:
        for lab in sorted(keywords.keys()):
            f.write(json.dumps({"cluster_id": int(lab), "keywords": keywords[lab]}, ensure_ascii=False) + "\n")

    # Scatter plot.
    n = X_2d.shape[0]
    if int(args.max_points_plot) > 0 and n > int(args.max_points_plot):
        idx = np.random.RandomState(0).choice(n, size=int(args.max_points_plot), replace=False)
        x = X_2d[idx, 0]
        y = X_2d[idx, 1]
        c = labels[idx]
    else:
        x = X_2d[:, 0]
        y = X_2d[:, 1]
        c = labels

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=c, s=5, cmap="tab20", alpha=0.7, linewidths=0)
    plt.title("UMAP(2D) + HDBSCAN clusters over regulatory clauses (noise=-1)")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=200)
    plt.close()

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"Done. wrote={out_dir} clusters={summary['n_clusters']} noise={summary['n_noise']}/{summary['n_clauses']} elapsed_s={elapsed}")


if __name__ == "__main__":
    main()

