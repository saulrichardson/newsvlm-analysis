#!/usr/bin/env python3
"""
Cluster issue-chunk TEXT documents from OpenAI embedding request shards using TF‑IDF → SVD → UMAP → HDBSCAN.

This is a local, no-API stand-in for:
  - provider embeddings (OpenAI /v1/embeddings) + rehydration
  - scripts/cluster_chunk_topic_embeddings.py (UMAP→HDBSCAN over embedding vectors)

It is intended for:
  - fast local smoke tests of the cluster labeling exporter
  - debugging prompt/example selection issues without waiting on Batch APIs

Inputs (request_dir)
  - openai_requests_shard*.jsonl lines like:
      {"custom_id":"<issue_id>::chunkNNN","url":"/v1/embeddings","body":{"input":"..."}}

Outputs (output_dir)
  - clusters.jsonl: one row per chunk_id with cluster assignment + probability + metadata
  - cluster_summary.json
  - umap_2d.npy + scatter.png
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
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


_CUSTOM_ID_RE = re.compile(r"^(?P<issue_id>.+?)::chunk(?P<chunk_idx>\d{3,})$")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cluster issue-chunk texts from embedding request shards.")
    ap.add_argument("--request-dir", required=True, help="Directory containing openai_requests_shard*.jsonl")
    ap.add_argument("--output-dir", required=True, help="Directory to write clusters + plots into")

    ap.add_argument("--min-chars", type=int, default=800, help="Drop docs with fewer than this many chars (0 disables).")
    ap.add_argument("--max-docs", type=int, default=0, help="Optional cap on number of chunks (0 disables).")
    ap.add_argument("--max-doc-chars", type=int, default=0, help="Optional: truncate each chunk to this many chars (0 disables).")

    # TF‑IDF
    ap.add_argument("--max-features", type=int, default=120_000, help="TF‑IDF vocabulary cap (0 disables).")
    ap.add_argument("--min-df", default="2", help="TF‑IDF min_df (count if >=1 else fraction).")
    ap.add_argument("--max-df", default="0.95", help="TF‑IDF max_df (count if >=1 else fraction).")
    ap.add_argument("--ngram-max", type=int, default=1, help="Max ngram size (1=unigrams, 2=uni+bi).")

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
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    ap.add_argument("--hdbscan-min-samples", type=int, default=10)

    ap.add_argument("--max-points-plot", type=int, default=20000, help="Downsample scatter plot if too many points.")
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
        raise SystemExit(f"--{name} as a fraction must be < 1; got {fv}. Use an integer count like {int(round(fv))}.")
    return float(fv)


def _load_chunks_from_request_dir(
    request_dir: Path,
    *,
    max_doc_chars: int | None,
) -> tuple[list[str], list[str], list[str | None], list[str], list[int]]:
    """
    Returns aligned lists:
      chunk_ids, issue_ids, issue_dates, texts, chunk_chars
    """
    shard_paths = sorted(request_dir.glob("openai_requests_shard*.jsonl"))
    if not shard_paths:
        raise SystemExit(f"No openai_requests_shard*.jsonl found under {request_dir}")

    chunk_ids: list[str] = []
    issue_ids: list[str] = []
    issue_dates: list[str | None] = []
    texts: list[str] = []
    chunk_chars: list[int] = []

    for p in shard_paths:
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id")
                if not isinstance(cid, str) or not cid.strip():
                    continue
                body = obj.get("body") or {}
                if not isinstance(body, dict):
                    raise SystemExit(f"Malformed request body in {p} for custom_id={cid}")
                text = body.get("input")
                if not isinstance(text, str) or not text.strip():
                    continue

                mo = _CUSTOM_ID_RE.match(cid.strip())
                if not mo:
                    # Skip non issue-chunk custom ids.
                    continue
                issue_id = mo.group("issue_id").strip()
                issue_date: str | None = None
                if "__" in issue_id:
                    tail = issue_id.split("__", 1)[1]
                    if re.match(r"^\d{4}-\d{2}-\d{2}$", tail):
                        issue_date = tail

                t = text.strip()
                if max_doc_chars is not None and max_doc_chars > 0 and len(t) > max_doc_chars:
                    t = t[:max_doc_chars].rstrip()

                chunk_ids.append(cid.strip())
                issue_ids.append(issue_id)
                issue_dates.append(issue_date)
                texts.append(t)
                chunk_chars.append(len(t))

    if not chunk_ids:
        raise SystemExit(f"No issue-chunk docs found under {request_dir} (custom_id must match '<issue_id>::chunkNNN').")
    return chunk_ids, issue_ids, issue_dates, texts, chunk_chars


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()

    request_dir = Path(expanduser(args.request_dir)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")

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

    chunk_ids, issue_ids, issue_dates, texts, chunk_chars = _load_chunks_from_request_dir(
        request_dir,
        max_doc_chars=max_doc_chars_opt,
    )

    # Filter by min chars.
    if min_chars > 0:
        keep = [i for i, t in enumerate(texts) if len(t) >= min_chars]
        chunk_ids = [chunk_ids[i] for i in keep]
        issue_ids = [issue_ids[i] for i in keep]
        issue_dates = [issue_dates[i] for i in keep]
        texts = [texts[i] for i in keep]
        chunk_chars = [chunk_chars[i] for i in keep]

    if not chunk_ids:
        raise SystemExit(f"All docs filtered out by --min-chars={min_chars}.")

    if max_docs > 0 and len(chunk_ids) > max_docs:
        chunk_ids = chunk_ids[:max_docs]
        issue_ids = issue_ids[:max_docs]
        issue_dates = issue_dates[:max_docs]
        texts = texts[:max_docs]
        chunk_chars = chunk_chars[:max_docs]

    if len(chunk_ids) < 3:
        raise SystemExit(f"Need at least 3 docs for UMAP; got {len(chunk_ids)}.")

    # Compute doc_weight = 1 / (#chunks in issue) so issue-heavy docs don't dominate.
    chunks_per_issue = Counter(issue_ids)
    doc_weight = [1.0 / float(chunks_per_issue[iid]) if chunks_per_issue[iid] > 0 else 1.0 for iid in issue_ids]

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
        raise SystemExit(
            f"TF‑IDF produced too few features (n_features={tfidf.shape[1]}). "
            "Try lowering --min-df, raising --max-features, or providing more docs."
        )

    # SVD
    svd_dim = int(args.svd_dim)
    if svd_dim <= 1:
        raise SystemExit("--svd-dim must be > 1")
    if svd_dim >= tfidf.shape[1]:
        svd_dim = max(2, min(svd_dim, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=int(svd_dim), random_state=int(args.svd_random_state))
    X_svd = svd.fit_transform(tfidf)
    X = normalize(X_svd, norm="l2", copy=False)

    # UMAP for clustering
    umap_n_components = int(args.umap_n_components)
    if umap_n_components < 2:
        raise SystemExit("--umap-n-components must be >= 2")
    umap_n_neighbors = int(args.umap_n_neighbors)
    if umap_n_neighbors < 2:
        raise SystemExit("--umap-n-neighbors must be >= 2")
    umap_n_neighbors_eff = min(umap_n_neighbors, len(chunk_ids) - 1)

    reducer = umap.UMAP(
        n_neighbors=int(umap_n_neighbors_eff),
        min_dist=float(args.umap_min_dist),
        n_components=int(umap_n_components),
        metric=str(args.umap_metric),
        random_state=int(args.umap_random_state),
    )
    X_umap = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(args.hdbscan_min_cluster_size),
        min_samples=int(args.hdbscan_min_samples),
    )
    labels = clusterer.fit_predict(X_umap)
    probs = getattr(clusterer, "probabilities_", None)
    if probs is None:
        probs = np.ones((len(labels),), dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)

    # 2D UMAP for plotting
    reducer_2d = umap.UMAP(
        n_neighbors=int(umap_n_neighbors_eff),
        min_dist=float(args.umap_min_dist),
        n_components=2,
        metric=str(args.umap_metric),
        random_state=int(args.umap_random_state),
    )
    X_2d = reducer_2d.fit_transform(X)
    np.save(out_dir / "umap_2d.npy", X_2d.astype(np.float32))

    # Write clusters.jsonl
    clusters_path = out_dir / "clusters.jsonl"
    with clusters_path.open("w", encoding="utf-8") as f:
        for i, cid in enumerate(chunk_ids):
            f.write(
                json.dumps(
                    {
                        "chunk_id": cid,
                        "cluster_id": int(labels[i]),
                        "cluster_prob": float(probs[i]),
                        "issue_id": issue_ids[i],
                        "issue_date": issue_dates[i],
                        "chunk_chars": int(chunk_chars[i]),
                        "doc_weight": float(doc_weight[i]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    counts = Counter(int(x) for x in labels.tolist())
    summary = {
        "n_chunks": int(len(chunk_ids)),
        "n_noise": int(counts.get(-1, 0)),
        "n_clusters": int(len([k for k in counts.keys() if k != -1])),
        "cluster_sizes": {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[0])},
        "params": {
            "tfidf": {
                "max_features": int(max_features_opt) if max_features_opt else None,
                "min_df": min_df,
                "max_df": max_df,
                "ngram_range": [1, int(ngram_max)],
                "stop_words": "english",
            },
            "svd": {"n_components": int(svd_dim), "random_state": int(args.svd_random_state)},
            "umap": {
                "n_neighbors": int(umap_n_neighbors_eff),
                "n_neighbors_requested": int(args.umap_n_neighbors),
                "min_dist": float(args.umap_min_dist),
                "n_components": int(umap_n_components),
                "metric": str(args.umap_metric),
                "random_state": int(args.umap_random_state),
            },
            "hdbscan": {
                "min_cluster_size": int(args.hdbscan_min_cluster_size),
                "min_samples": int(args.hdbscan_min_samples),
            },
        },
        "svd": {
            "explained_variance_ratio_sum": float(np.sum(getattr(svd, "explained_variance_ratio_", np.array([])))),
        },
    }
    (out_dir / "cluster_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Scatter plot
    n = X_2d.shape[0]
    max_plot = int(args.max_points_plot)
    if max_plot > 0 and n > max_plot:
        rng = np.random.RandomState(0)
        idx = rng.choice(n, size=max_plot, replace=False)
        x = X_2d[idx, 0]
        y = X_2d[idx, 1]
        c = labels[idx]
    else:
        x = X_2d[:, 0]
        y = X_2d[:, 1]
        c = labels

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=c, s=4, cmap="tab20", alpha=0.7, linewidths=0)
    plt.title("UMAP(2D) + HDBSCAN clusters over issue-chunks (noise=-1)")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=200)
    plt.close()

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"Done. wrote={out_dir} clusters={summary['n_clusters']} noise={summary['n_noise']}/{summary['n_chunks']} elapsed_s={elapsed}")


if __name__ == "__main__":
    main()
