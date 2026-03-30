#!/usr/bin/env python3
"""
Cluster CHUNK embeddings with UMAP → HDBSCAN ("tweet topics" style).

This is the chunk-level analogue of scripts/cluster_issue_topic_embeddings.py.

Inputs (embedding_dir)
  - chunk_ids.txt
  - chunk_embeddings.npy
  - chunk_metadata.jsonl

Outputs (output_dir)
  - clusters.jsonl: per-chunk cluster assignment + probability + issue/date metadata
  - cluster_summary.json: cluster sizes + basic stats
  - umap_2d.npy: 2D projection aligned with chunk_ids.txt (for plotting)
  - scatter.png: quick visual sanity check

This script does not call any APIs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap


def _load_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Missing ids file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iter_metadata_rows(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"Missing metadata JSONL: {path}")
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cluster chunk embeddings with UMAP→HDBSCAN.")
    ap.add_argument("--embedding-dir", required=True, help="Directory containing chunk_embeddings.npy + chunk_ids.txt")
    ap.add_argument("--output-dir", required=True, help="Directory to write clusters + plots into")

    ap.add_argument("--umap-n-neighbors", type=int, default=25)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    ap.add_argument("--umap-n-components", type=int, default=5, help="UMAP components for clustering (not plotting).")
    ap.add_argument("--umap-metric", default="cosine")
    ap.add_argument("--umap-random-state", type=int, default=0)

    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    ap.add_argument("--hdbscan-min-samples", type=int, default=10)

    ap.add_argument("--max-points-plot", type=int, default=20000, help="Downsample scatter plot if too many points.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    emb_dir = Path(args.embedding_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = _load_ids(emb_dir / "chunk_ids.txt")
    X = np.load(emb_dir / "chunk_embeddings.npy")
    if X.ndim != 2:
        raise SystemExit(f"Expected 2D embedding matrix; got shape {X.shape} from {emb_dir/'chunk_embeddings.npy'}")
    if X.shape[0] != len(chunk_ids):
        raise SystemExit(f"Row mismatch: embeddings rows={X.shape[0]} chunk_ids={len(chunk_ids)}")

    # Read metadata into arrays aligned with chunk_ids order (streaming).
    meta_path = emb_dir / "chunk_metadata.jsonl"
    issue_dates: list[str | None] = []
    issue_ids: list[str | None] = []
    cls_labels: list[str | None] = []
    chunk_counts: list[int | None] = []
    chunk_chars: list[int | None] = []

    meta_iter = _iter_metadata_rows(meta_path)
    for cid in chunk_ids:
        try:
            m = next(meta_iter)
        except StopIteration:
            raise SystemExit(f"Metadata file ended early: {meta_path}")
        got = m.get("chunk_id")
        if got != cid:
            raise SystemExit(f"Metadata misalignment: expected chunk_id={cid} got={got} (refusing to write wrong joins)")
        issue_dates.append(m.get("issue_date") if isinstance(m.get("issue_date"), str) else None)
        issue_ids.append(m.get("issue_id") if isinstance(m.get("issue_id"), str) else None)
        cls_labels.append(m.get("classification_label") if isinstance(m.get("classification_label"), str) else None)
        chunk_counts.append(int(m.get("chunk_count")) if m.get("chunk_count") is not None else None)
        chunk_chars.append(int(m.get("chunk_chars")) if m.get("chunk_chars") is not None else None)

    # Ensure there's no extra metadata (would indicate a mismatch upstream).
    try:
        extra = next(meta_iter)
        raise SystemExit(f"Metadata has extra rows beyond chunk_ids.txt (example: {extra})")
    except StopIteration:
        pass

    # UMAP projection for clustering.
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

    # 2D UMAP for plotting (not used for clustering; helps inspection).
    reducer_2d = umap.UMAP(
        n_neighbors=int(args.umap_n_neighbors),
        min_dist=float(args.umap_min_dist),
        n_components=2,
        metric=str(args.umap_metric),
        random_state=int(args.umap_random_state),
    )
    X_2d = reducer_2d.fit_transform(X)
    np.save(out_dir / "umap_2d.npy", X_2d.astype(np.float32))

    # Per-chunk cluster assignments.
    if len(chunk_ids) != len(labels) or len(chunk_ids) != len(probs):
        raise SystemExit(
            f"Length mismatch: chunk_ids={len(chunk_ids)} labels={len(labels)} probs={len(probs)} "
            "(unexpected; refusing to write misaligned cluster assignments)"
        )
    clusters_path = out_dir / "clusters.jsonl"
    with clusters_path.open("w", encoding="utf-8") as f:
        for i, (cid, lab, prob) in enumerate(zip(chunk_ids, labels.tolist(), probs.tolist())):
            cc = chunk_counts[i]
            doc_weight = 1.0 / float(cc) if isinstance(cc, int) and cc > 0 else 1.0
            row = {
                "chunk_id": cid,
                "cluster_id": int(lab),
                "cluster_prob": float(prob),
                "issue_id": issue_ids[i],
                "issue_date": issue_dates[i],
                "classification_label": cls_labels[i],
                "chunk_count": cc,
                "chunk_chars": chunk_chars[i],
                "doc_weight": float(doc_weight),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary.
    counts = Counter(int(x) for x in labels.tolist())
    summary = {
        "n_chunks": len(chunk_ids),
        "n_noise": int(counts.get(-1, 0)),
        "n_clusters": int(len([k for k in counts.keys() if k != -1])),
        "cluster_sizes": {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[0])},
        "params": {
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

    # Quick scatter plot sanity check.
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
    plt.scatter(x, y, c=c, s=4, cmap="tab20", alpha=0.7, linewidths=0)
    plt.title("UMAP(2D) + HDBSCAN clusters over chunks (noise=-1)")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=200)
    plt.close()

    # Also write a basic frequency-by-year table (raw counts) for quick debugging.
    by_year = defaultdict(Counter)
    for d, lab in zip(issue_dates, labels.tolist()):
        if not isinstance(d, str) or len(d) < 4:
            continue
        year = int(d[:4])
        by_year[year][int(lab)] += 1
    if by_year:
        years = sorted(by_year)
        out_lines = []
        for y in years:
            row = {"year": y, **{str(k): int(v) for k, v in sorted(by_year[y].items(), key=lambda kv: kv[0])}}
            out_lines.append(json.dumps(row, sort_keys=True))
        (out_dir / "cluster_counts_by_year.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"Done. wrote={out_dir} clusters={summary['n_clusters']} noise={summary['n_noise']}/{summary['n_chunks']}")


if __name__ == "__main__":
    main()
