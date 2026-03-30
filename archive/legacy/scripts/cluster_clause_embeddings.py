#!/usr/bin/env python3
"""
Cluster clause embeddings with UMAP → HDBSCAN ("tweet topics" style).

Inputs (embedding_dir)
  - clause_ids.txt
  - clause_embeddings.npy
  - clause_metadata.jsonl (optional but recommended; should be aligned to clause_ids.txt)

Outputs (output_dir)
  - clusters.jsonl: per-clause cluster assignment + probability + selected metadata
  - cluster_summary.json
  - umap_2d.npy + scatter.png

This script does not call any APIs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cluster clause embeddings with UMAP→HDBSCAN.")
    ap.add_argument("--embedding-dir", required=True, help="Directory containing clause_embeddings.npy + clause_ids.txt")
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

    ids = _load_ids(emb_dir / "clause_ids.txt")
    X = np.load(emb_dir / "clause_embeddings.npy")
    if X.ndim != 2:
        raise SystemExit(f"Expected 2D embedding matrix; got shape {X.shape} from {emb_dir/'clause_embeddings.npy'}")
    if X.shape[0] != len(ids):
        raise SystemExit(f"Row mismatch: embeddings rows={X.shape[0]} clause_ids={len(ids)}")

    # Optional aligned metadata.
    meta_path = emb_dir / "clause_metadata.jsonl"
    meta_rows: list[dict[str, Any]] | None = None
    if meta_path.is_file():
        meta_rows = []
        it = _iter_jsonl(meta_path)
        for cid in ids:
            try:
                m = next(it)
            except StopIteration:
                raise SystemExit(f"Metadata file ended early: {meta_path}")
            got = m.get("custom_id") or m.get("clause_id")
            if got != cid:
                raise SystemExit(f"Metadata misalignment: expected clause_id={cid} got={got} (refusing to mis-join)")
            meta_rows.append(m)
        try:
            extra = next(it)
            raise SystemExit(f"Metadata has extra rows beyond clause_ids.txt (example: {extra})")
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

    # Per-clause cluster assignments.
    clusters_path = out_dir / "clusters.jsonl"
    with clusters_path.open("w", encoding="utf-8") as f:
        for i, (cid, lab, prob) in enumerate(zip(ids, labels.tolist(), probs.tolist())):
            row = {"clause_id": cid, "cluster_id": int(lab), "cluster_prob": float(prob)}
            if meta_rows is not None:
                m = meta_rows[i]
                for k in ["doc_id", "chunk_id", "motive", "modality", "instrument_hint", "issue_date", "classification_label", "slug"]:
                    if k in m:
                        row[k] = m.get(k)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(int(x) for x in labels.tolist())
    summary = {
        "n_clauses": len(ids),
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

    # Scatter plot sanity check.
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
    plt.title("UMAP(2D) + HDBSCAN clusters over clauses (noise=-1)")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=200)
    plt.close()

    print(f"Done. wrote={out_dir} clusters={summary['n_clusters']} noise={summary['n_noise']}/{summary['n_clauses']}")


if __name__ == "__main__":
    main()

