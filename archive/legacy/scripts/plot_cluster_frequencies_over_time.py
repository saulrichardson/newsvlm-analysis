#!/usr/bin/env python3
"""
Plot cluster frequencies over time (counts per month/year).

Input:
  - clusters.jsonl (from scripts/cluster_issue_topic_embeddings.py)

Output:
  - counts.csv (bucket x cluster_id counts)
  - plot.png (stacked area for top clusters)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt


Bucket = Literal["month", "year"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot topic/cluster frequencies over time.")
    ap.add_argument("--clusters-jsonl", required=True, help="clusters.jsonl path")
    ap.add_argument("--output-dir", required=True, help="Directory to write plot + csv")
    ap.add_argument(
        "--cluster-labels-jsonl",
        default=None,
        help=(
            "Optional JSONL of cluster labels (one row per cluster_id) with fields like "
            "cluster_id and topic_name. Used to name the legend."
        ),
    )
    ap.add_argument("--bucket", choices=["month", "year"], default="month")
    ap.add_argument(
        "--count-mode",
        choices=["rows", "unique_issue", "weight"],
        default="rows",
        help=(
            "How to count rows into bucket totals. "
            "rows=each JSONL row counts as 1. "
            "unique_issue=each (bucket,cluster_id,issue_id) counts once. "
            "weight=use --weight-field numeric values."
        ),
    )
    ap.add_argument(
        "--weight-field",
        default="",
        help="When --count-mode=weight, take weights from this JSON field (e.g. doc_weight).",
    )
    ap.add_argument("--top-k", type=int, default=12, help="Number of clusters to plot (excluding noise)")
    ap.add_argument("--include-noise", action="store_true", help="Include cluster_id=-1 in outputs")
    ap.add_argument(
        "--omit-other",
        action="store_true",
        help=(
            "Do not aggregate non-top-k clusters into an 'other' series/column. "
            "When set, the plot/CSV will contain ONLY the top-k clusters; "
            "all remaining clusters are dropped (not shown)."
        ),
    )
    ap.add_argument(
        "--legend-mode",
        choices=["inside", "outside_right", "outside_bottom", "none"],
        default="outside_bottom",
        help=(
            "Where to place the legend. "
            "outside_* avoids covering the stacked-area 'mountain' plot. "
            "Default: outside_bottom."
        ),
    )
    return ap.parse_args()


def _bucket_from_date(date_s: str, bucket: Bucket) -> str | None:
    if not isinstance(date_s, str) or len(date_s) < 4:
        return None
    if bucket == "year":
        return date_s[:4]
    if bucket == "month":
        # YYYY-MM
        if len(date_s) >= 7:
            return date_s[:7]
        return None
    raise ValueError(f"Unknown bucket: {bucket}")


def main() -> None:
    args = _parse_args()
    clusters_path = Path(args.clusters_jsonl).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not clusters_path.is_file():
        raise SystemExit(f"--clusters-jsonl not found: {clusters_path}")

    bucket: Bucket = str(args.bucket)  # type: ignore[assignment]
    include_noise = bool(args.include_noise)
    count_mode = str(args.count_mode)
    weight_field = str(args.weight_field or "").strip()
    if count_mode == "weight" and not weight_field:
        raise SystemExit("--weight-field is required when --count-mode=weight")

    label_by_cluster: dict[int, str] = {}
    if args.cluster_labels_jsonl:
        labels_path = Path(args.cluster_labels_jsonl).expanduser().resolve()
        if not labels_path.is_file():
            raise SystemExit(f"--cluster-labels-jsonl not found: {labels_path}")
        with labels_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                obj: dict[str, Any] = json.loads(line)
                cid = obj.get("cluster_id")
                name = obj.get("topic_name")
                if isinstance(cid, int) and isinstance(name, str) and name.strip():
                    label_by_cluster[int(cid)] = name.strip()

    counts = defaultdict(Counter)  # bucket -> cluster_id -> count
    totals = Counter()
    seen_issue = set()

    with clusters_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj: dict[str, Any] = json.loads(line)
            cluster_id = obj.get("cluster_id")
            if not isinstance(cluster_id, int):
                continue
            if cluster_id == -1 and not include_noise:
                continue
            d = obj.get("issue_date")
            if not isinstance(d, str):
                continue
            b = _bucket_from_date(d, bucket=bucket)
            if b is None:
                continue

            if count_mode == "rows":
                v = 1.0
            elif count_mode == "unique_issue":
                issue_id = obj.get("issue_id")
                if not isinstance(issue_id, str) or not issue_id.strip():
                    raise SystemExit("clusters.jsonl missing issue_id rows but --count-mode=unique_issue was requested")
                key = (b, cluster_id, issue_id.strip())
                if key in seen_issue:
                    continue
                seen_issue.add(key)
                v = 1.0
            elif count_mode == "weight":
                w = obj.get(weight_field)
                if not isinstance(w, (int, float)):
                    raise SystemExit(
                        f"clusters.jsonl row missing numeric weight field '{weight_field}' "
                        f"for cluster_id={cluster_id} bucket={b}: got {w!r}"
                    )
                v = float(w)
            else:
                raise SystemExit(f"Unknown count_mode: {count_mode}")

            counts[b][cluster_id] += v
            totals[cluster_id] += v

    if not counts:
        raise SystemExit("No bucketed counts found (clusters.jsonl missing issue_date?)")

    # Choose top clusters by total count (excluding noise unless requested).
    top_k = int(args.top_k)
    if top_k <= 0:
        raise SystemExit("--top-k must be > 0")

    cluster_ids_sorted = [cid for cid, _ in totals.most_common() if include_noise or cid != -1]
    top_clusters = cluster_ids_sorted[:top_k]

    buckets_sorted = sorted(counts.keys())

    # Write CSV
    csv_path = out_dir / f"cluster_counts_by_{bucket}.csv"
    fieldnames = ["bucket"] + [str(cid) for cid in top_clusters]
    if not bool(args.omit_other):
        fieldnames.append("other")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for b in buckets_sorted:
            row: dict[str, Any] = {"bucket": b}
            other = 0.0
            for cid, c in counts[b].items():
                if cid in top_clusters:
                    row[str(cid)] = float(c)
                else:
                    other += float(c)
            for cid in top_clusters:
                row.setdefault(str(cid), 0)
            if "other" in fieldnames:
                row["other"] = other
            w.writerow(row)

    # Plot stacked area (top clusters + other)
    series = {cid: [float(counts[b].get(cid, 0)) for b in buckets_sorted] for cid in top_clusters}

    labels = []
    for cid in top_clusters:
        name = label_by_cluster.get(cid)
        if name:
            labels.append(f"{name} (#{cid})")
        else:
            labels.append(f"cluster {cid}")
    data = [series[cid] for cid in top_clusters]
    if not bool(args.omit_other):
        other_series = []
        for b in buckets_sorted:
            other_series.append(float(sum(v for cid, v in counts[b].items() if cid not in top_clusters)))
        labels.append("other")
        data.append(other_series)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(range(len(buckets_sorted)), data, labels=labels, alpha=0.85)

    tick_step = max(1, len(buckets_sorted) // 12)
    tick_idx = list(range(0, len(buckets_sorted), tick_step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([buckets_sorted[i] for i in tick_idx], rotation=45, ha="right")
    ax.set_title(f"Cluster frequency over time (bucket={bucket}, top_k={top_k})")

    legend_mode = str(args.legend_mode)
    if legend_mode != "none":
        # With many clusters (e.g., top_k=50), putting the legend inside the axes makes
        # the plot unreadable. Default to outside placement.
        n_labels = len(labels)
        if legend_mode == "inside":
            ax.legend(loc="upper left", fontsize=8, ncol=2)
        elif legend_mode == "outside_right":
            # Use more columns as labels grow, to avoid a very tall legend.
            ncol = 1 if n_labels <= 20 else 2 if n_labels <= 40 else 3
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=7,
                ncol=ncol,
                frameon=False,
                labelspacing=0.25,
                handlelength=1.2,
                handletextpad=0.5,
                columnspacing=0.9,
            )
        elif legend_mode == "outside_bottom":
            # Put the legend below the plot in multiple columns.
            # Heuristic: cap the number of legend rows (helps for top_k=50).
            max_rows = 6
            ncol = max(2, min(10, int(math.ceil(n_labels / float(max_rows)))))
            n_rows = int(math.ceil(n_labels / float(ncol)))
            # If the legend is multi-row, push it slightly further down to avoid
            # colliding with rotated x tick labels.
            anchor_y = -0.18 - 0.02 * max(0, min(8, n_rows - 1))
            fontsize = 7 if n_labels <= 30 else 6
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, anchor_y),
                borderaxespad=0.0,
                fontsize=fontsize,
                ncol=ncol,
                frameon=False,
                labelspacing=0.25,
                handlelength=1.2,
                handletextpad=0.5,
                columnspacing=0.9,
            )
        else:
            raise SystemExit(f"Unknown --legend-mode: {legend_mode}")

    # Don't rely on tight_layout with many labels; bbox_inches='tight' is more robust
    # for legends placed outside the axes.
    fig.savefig(out_dir / f"cluster_frequency_{bucket}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Done. wrote={out_dir} buckets={len(buckets_sorted)} clusters_total={len(totals)} top_k={top_k}")


if __name__ == "__main__":
    main()
