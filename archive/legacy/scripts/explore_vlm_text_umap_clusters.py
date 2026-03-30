#!/usr/bin/env python3
"""
Unstructured exploration of newspaper OCR text via TF‑IDF → SVD → UMAP → HDBSCAN.

This is meant as a *local*, no-API, quick-turn workflow for:
  - projecting page/issue transcripts into 2D (UMAP)
  - discovering topical clusters (HDBSCAN)
  - inspecting clusters via keyword summaries + a scatter plot

It complements (but does not replace) the "issue topic embeddings" pipeline that uses
provider embeddings + Batch APIs.

Inputs
  - --pages: glob or "@file" list of per-page *.vlm.json outputs

Outputs (output_dir)
  - docs.jsonl: one line per doc (page or issue), metadata + text stats
  - clusters.jsonl: one line per doc, cluster assignment + probability
  - cluster_summary.json: sizes + run parameters
  - cluster_keywords.jsonl: top TF‑IDF keywords per cluster
  - umap_2d.npy: 2D UMAP projection aligned with docs.jsonl order
  - scatter.png: quick visual sanity check

No external APIs are called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from glob import glob
from os.path import expanduser
from pathlib import Path
from typing import Any, Literal

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from newsvlm.issue_zoning_classifier import (
    load_page_result,
    page_text_from_boxes,
    parse_page_id,
)


DocLevel = Literal["page", "issue"]
SampleMode = Literal["head", "random"]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_page_paths(spec: str) -> list[Path]:
    """Collect per-page .vlm.json paths from a glob or @file list."""
    spec = spec.strip()
    if spec.startswith("@"):
        list_path = Path(expanduser(spec[1:]))
        if not list_path.is_file():
            raise SystemExit(f"Pages file not found: {list_path}")
        paths: list[Path] = []
        for raw in list_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(expanduser(line))
            if not p.is_file():
                raise SystemExit(f"Page JSON listed but not found: {p}")
            paths.append(p)
        return sorted(paths)

    expanded = expanduser(spec)
    return sorted(Path(p) for p in glob(expanded))


def _safe_preview(text: str, *, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip()


@dataclass(frozen=True)
class Doc:
    doc_id: str
    issue_id: str | None
    issue_date: str | None
    slug: str | None
    page_num: int | None
    text: str
    source_paths: list[str]

    @property
    def text_chars(self) -> int:
        return len(self.text)

    @property
    def text_sha256(self) -> str:
        return _sha256(self.text)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Explore page/issue OCR text with TF‑IDF→UMAP→HDBSCAN.")
    ap.add_argument(
        "--pages",
        required=True,
        help='Glob for per-page *.vlm.json files (absolute OK) or "@file" listing those paths',
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs into")

    ap.add_argument("--doc-level", choices=["page", "issue"], default="page", help="Cluster individual pages or whole issues.")
    ap.add_argument(
        "--min-chars",
        type=int,
        default=800,
        help="Drop docs with fewer than this many transcript chars (0 disables).",
    )
    ap.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional cap on number of docs (0 disables). Applied after filtering.",
    )
    ap.add_argument(
        "--sample-mode",
        choices=["head", "random"],
        default="head",
        help="When --max-docs is set: take the first N docs or a random sample.",
    )
    ap.add_argument("--sample-seed", type=int, default=0, help="RNG seed when --sample-mode=random.")
    ap.add_argument(
        "--max-doc-chars",
        type=int,
        default=0,
        help="Optional: truncate each doc to this many chars before vectorizing (0 disables).",
    )
    ap.add_argument(
        "--skip-bad-pages",
        action="store_true",
        help="Skip pages that fail to load/validate instead of exiting.",
    )

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

    # Outputs
    ap.add_argument("--preview-chars", type=int, default=220, help="Chars of preview text to include in docs.jsonl.")
    ap.add_argument("--keywords-per-cluster", type=int, default=12, help="Top TF‑IDF terms to emit per cluster.")
    ap.add_argument("--max-points-plot", type=int, default=20_000, help="Downsample scatter plot if too many points.")
    return ap.parse_args()


def _coerce_positive_int(name: str, v: int, *, allow_zero: bool) -> int:
    try:
        iv = int(v)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"--{name} must be an integer; got {v!r}") from exc
    if allow_zero and iv == 0:
        return 0
    if iv <= 0:
        raise SystemExit(f"--{name} must be > 0; got {iv}")
    return iv


def _parse_df(name: str, raw: str) -> int | float:
    """
    Parse a df threshold that can be specified as either:
      - integer count (e.g. "2")
      - float fraction (e.g. "0.95")

    We also accept float strings that are integer-like (e.g. "2.0") and coerce them to int,
    because sklearn interprets floats as fractions and would error for >1.
    """
    s = str(raw).strip()
    if not s:
        raise SystemExit(f"--{name} must be non-empty")

    # Fast path: integer count
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

    # If it's an integer-like float >= 1, treat it as a count.
    if fv >= 1 and abs(fv - round(fv)) < 1e-9:
        return int(round(fv))

    if fv >= 1:
        raise SystemExit(f"--{name} as a fraction must be < 1; got {fv}. Use an integer count like {math.ceil(fv)}.")
    return float(fv)


def _load_docs_from_pages(
    page_paths: list[Path],
    *,
    doc_level: DocLevel,
    max_doc_chars: int | None,
    skip_bad_pages: bool,
) -> list[Doc]:
    if doc_level == "page":
        out: list[Doc] = []
        for p in page_paths:
            try:
                page = load_page_result(p)
            except Exception as exc:  # noqa: BLE001
                if skip_bad_pages:
                    print(f"Skipping bad page JSON: {p} ({exc})", file=sys.stderr)
                    continue
                raise SystemExit(f"Failed to load page JSON: {p} ({exc})") from exc

            text = page_text_from_boxes(page)
            if max_doc_chars is not None and max_doc_chars > 0 and len(text) > max_doc_chars:
                text = text[:max_doc_chars].rstrip()

            pid = parse_page_id(page.page_id)
            out.append(
                Doc(
                    doc_id=page.page_id,
                    issue_id=pid.issue_id if pid else None,
                    issue_date=pid.issue_date if pid else None,
                    slug=pid.slug if pid else None,
                    page_num=pid.page_num if pid else None,
                    text=text,
                    source_paths=[str(p)],
                )
            )
        return out

    if doc_level == "issue":
        by_issue: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)  # issue_id -> (page_num, page_id, path, text)
        meta_by_issue: dict[str, dict[str, Any]] = {}
        for p in page_paths:
            try:
                page = load_page_result(p)
            except Exception as exc:  # noqa: BLE001
                if skip_bad_pages:
                    print(f"Skipping bad page JSON: {p} ({exc})", file=sys.stderr)
                    continue
                raise SystemExit(f"Failed to load page JSON: {p} ({exc})") from exc

            pid = parse_page_id(page.page_id)
            if pid is None:
                msg = f"Unparseable page_id for issue-level grouping: {page.page_id} (path={p})"
                if skip_bad_pages:
                    print(f"Skipping: {msg}", file=sys.stderr)
                    continue
                raise SystemExit(msg)

            text = page_text_from_boxes(page)
            if max_doc_chars is not None and max_doc_chars > 0 and len(text) > max_doc_chars:
                text = text[:max_doc_chars].rstrip()

            by_issue[pid.issue_id].append((pid.page_num, pid.page_id, str(p), text))
            meta_by_issue.setdefault(
                pid.issue_id,
                {
                    "issue_id": pid.issue_id,
                    "issue_date": pid.issue_date,
                    "slug": pid.slug,
                },
            )

        docs: list[Doc] = []
        for issue_id in sorted(by_issue.keys()):
            pages = sorted(by_issue[issue_id], key=lambda t: t[0])
            parts: list[str] = []
            source_paths: list[str] = []
            for page_num, page_id, page_path, text in pages:
                parts.append(f"=== PAGE {page_num} {page_id} ===")
                parts.append(text.strip())
                source_paths.append(page_path)
            issue_text = "\n\n".join(p for p in parts if p.strip()).strip()
            meta = meta_by_issue.get(issue_id) or {}
            docs.append(
                Doc(
                    doc_id=issue_id,
                    issue_id=meta.get("issue_id"),
                    issue_date=meta.get("issue_date"),
                    slug=meta.get("slug"),
                    page_num=None,
                    text=issue_text,
                    source_paths=source_paths,
                )
            )
        return docs

    raise ValueError(f"Unknown doc_level: {doc_level}")


def _top_terms_for_clusters(
    tfidf,
    feature_names: list[str],
    labels: np.ndarray,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    out: list[dict[str, Any]] = []
    for cid in sorted(int(x) for x in set(labels.tolist())):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        if idx.size <= 0:
            continue
        # Mean TF‑IDF vector for cluster, then take top-k features.
        v = tfidf[idx].mean(axis=0)
        dense = np.asarray(v).ravel()
        if dense.size != len(feature_names):
            raise RuntimeError("Internal error: feature dimension mismatch")
        k = min(top_k, dense.size)
        if k <= 0:
            terms: list[str] = []
        else:
            top_idx = np.argpartition(-dense, k - 1)[:k]
            top_idx = top_idx[np.argsort(-dense[top_idx])]
            terms = [feature_names[i] for i in top_idx.tolist() if dense[i] > 0]
        out.append(
            {
                "cluster_id": int(cid),
                "cluster_size_docs": int(idx.size),
                "top_terms": terms,
            }
        )
    return out


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()

    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    page_paths = collect_page_paths(args.pages)
    if not page_paths:
        raise SystemExit(f"No pages matched: {args.pages}")

    doc_level: DocLevel = str(args.doc_level)  # type: ignore[assignment]
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

    ngram_max = _coerce_positive_int("ngram-max", args.ngram_max, allow_zero=False)
    if ngram_max > 3:
        raise SystemExit("--ngram-max > 3 is probably a mistake (huge feature space); refusing.")

    svd_dim = _coerce_positive_int("svd-dim", args.svd_dim, allow_zero=False)
    umap_n_components = _coerce_positive_int("umap-n-components", args.umap_n_components, allow_zero=False)
    if umap_n_components < 2:
        raise SystemExit("--umap-n-components must be >= 2")

    if max_docs > 0 and max_docs < 10:
        print(f"Warning: --max-docs={max_docs} is very small; UMAP/HDBSCAN may be unstable.", file=sys.stderr)

    # 1) Load docs.
    docs = _load_docs_from_pages(
        page_paths,
        doc_level=doc_level,
        max_doc_chars=max_doc_chars_opt,
        skip_bad_pages=bool(args.skip_bad_pages),
    )
    if not docs:
        raise SystemExit("No docs loaded (all pages failed or were skipped).")

    # 2) Filter by min chars.
    n_before = len(docs)
    if min_chars > 0:
        docs = [d for d in docs if d.text_chars >= min_chars]
    n_after = len(docs)
    if not docs:
        raise SystemExit(f"All docs filtered out by --min-chars={min_chars} (started with {n_before}).")

    # 3) Optional sampling.
    sample_mode: SampleMode = str(args.sample_mode)  # type: ignore[assignment]
    if max_docs > 0 and len(docs) > max_docs:
        if sample_mode == "head":
            docs = docs[:max_docs]
        elif sample_mode == "random":
            rng = np.random.RandomState(int(args.sample_seed))
            idx = rng.choice(len(docs), size=max_docs, replace=False)
            docs = [docs[i] for i in sorted(idx.tolist())]
        else:
            raise SystemExit(f"Unknown --sample-mode: {sample_mode}")

    doc_ids = [d.doc_id for d in docs]
    texts = [d.text for d in docs]
    if len(docs) < 3:
        raise SystemExit(f"Need at least 3 docs for UMAP; got {len(docs)}. Try lowering --min-chars or increasing inputs.")

    umap_n_neighbors = int(args.umap_n_neighbors)
    if umap_n_neighbors < 2:
        raise SystemExit("--umap-n-neighbors must be >= 2")
    umap_n_neighbors_eff = min(umap_n_neighbors, len(docs) - 1)
    if umap_n_neighbors_eff != umap_n_neighbors:
        print(
            f"Note: capping UMAP n_neighbors from {umap_n_neighbors} to {umap_n_neighbors_eff} "
            f"(only {len(docs)} docs).",
            file=sys.stderr,
        )

    # 4) Vectorize (TF‑IDF).
    max_features = int(args.max_features)
    if max_features < 0:
        raise SystemExit("--max-features must be >= 0")
    max_features_opt = max_features if max_features > 0 else None

    min_df = _parse_df("min-df", str(args.min_df))
    max_df = _parse_df("max-df", str(args.max_df))
    if isinstance(min_df, float) and isinstance(max_df, float) and min_df >= max_df:
        # Common config mistake; yields empty vocab or runtime error.
        raise SystemExit(f"Bad df bounds: min_df={min_df} must be < max_df={max_df}")

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
            "Try lowering --min-df, raising --max-features, or providing more/larger docs."
        )
    feature_names = vectorizer.get_feature_names_out().tolist()

    # 5) SVD -> dense.
    if svd_dim >= tfidf.shape[1]:
        # SVD can't have more components than feature count.
        svd_dim = max(2, min(svd_dim, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=int(svd_dim), random_state=int(args.svd_random_state))
    X_svd = svd.fit_transform(tfidf)
    X = normalize(X_svd, norm="l2", copy=False)

    # 6) UMAP projection (for clustering).
    reducer = umap.UMAP(
        n_neighbors=int(umap_n_neighbors_eff),
        min_dist=float(args.umap_min_dist),
        n_components=int(umap_n_components),
        metric=str(args.umap_metric),
        random_state=int(args.umap_random_state),
    )
    X_umap = reducer.fit_transform(X)

    # 7) HDBSCAN clustering.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(args.hdbscan_min_cluster_size),
        min_samples=int(args.hdbscan_min_samples),
    )
    labels = clusterer.fit_predict(X_umap)
    probs = getattr(clusterer, "probabilities_", None)
    if probs is None:
        probs = np.ones((len(labels),), dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)

    # 8) 2D UMAP for plotting.
    reducer_2d = umap.UMAP(
        n_neighbors=int(umap_n_neighbors_eff),
        min_dist=float(args.umap_min_dist),
        n_components=2,
        metric=str(args.umap_metric),
        random_state=int(args.umap_random_state),
    )
    X_2d = reducer_2d.fit_transform(X)
    np.save(out_dir / "umap_2d.npy", X_2d.astype(np.float32))

    # 9) Write docs + clusters.
    docs_path = out_dir / "docs.jsonl"
    clusters_path = out_dir / "clusters.jsonl"
    if len(doc_ids) != len(labels) or len(doc_ids) != len(probs):
        raise SystemExit("Internal error: doc_id/label/prob length mismatch")

    with docs_path.open("w", encoding="utf-8") as docs_f, clusters_path.open("w", encoding="utf-8") as clus_f:
        for i, d in enumerate(docs):
            docs_f.write(
                json.dumps(
                    {
                        "doc_id": d.doc_id,
                        "doc_level": doc_level,
                        "issue_id": d.issue_id,
                        "issue_date": d.issue_date,
                        "slug": d.slug,
                        "page_num": d.page_num,
                        "text_chars": d.text_chars,
                        "text_sha256": d.text_sha256,
                        "text_preview": _safe_preview(d.text, max_chars=int(args.preview_chars)),
                        "source_paths": d.source_paths,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            clus_f.write(
                json.dumps(
                    {
                        "doc_id": d.doc_id,
                        "cluster_id": int(labels[i]),
                        "cluster_prob": float(probs[i]),
                        "issue_id": d.issue_id,
                        "issue_date": d.issue_date,
                        "slug": d.slug,
                        "page_num": d.page_num,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # 10) Summary + keywords.
    counts = Counter(int(x) for x in labels.tolist())
    summary = {
        "n_docs": int(len(docs)),
        "n_pages_input": int(len(page_paths)),
        "n_noise": int(counts.get(-1, 0)),
        "n_clusters": int(len([k for k in counts.keys() if k != -1])),
        "cluster_sizes": {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[0])},
        "filters": {
            "doc_level": doc_level,
            "min_chars": int(min_chars),
            "max_docs": int(max_docs),
            "sample_mode": str(sample_mode),
            "max_doc_chars": int(max_doc_chars),
        },
        "params": {
            "tfidf": {
                "max_features": int(max_features_opt) if max_features_opt else None,
                "min_df": float(min_df),
                "max_df": float(max_df),
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

    kw_rows = _top_terms_for_clusters(
        tfidf,
        feature_names,
        labels,
        top_k=int(args.keywords_per_cluster),
    )
    (out_dir / "cluster_keywords.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kw_rows) + ("\n" if kw_rows else ""),
        encoding="utf-8",
    )

    # 11) Scatter plot sanity check.
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
    plt.title("UMAP(2D) + HDBSCAN clusters (noise=-1)")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=200)
    plt.close()

    elapsed = round(time.perf_counter() - t0, 2)
    print(
        f"Done. docs={len(docs)} clusters={summary['n_clusters']} noise={summary['n_noise']}/{summary['n_docs']} "
        f"wrote={out_dir} elapsed_s={elapsed}"
    )


if __name__ == "__main__":
    main()
