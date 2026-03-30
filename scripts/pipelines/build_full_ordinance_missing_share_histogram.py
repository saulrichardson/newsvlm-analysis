#!/usr/bin/env python3
"""
Build a histogram-style summary of estimated missing share from the completed
full-ordinance completeness run.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


LABEL_ORDER = [
    "complete_or_nearly_complete",
    "uncertain",
    "likely_incomplete",
    "definitely_incomplete",
]
LABEL_COLORS = {
    "complete_or_nearly_complete": "#4C956C",
    "uncertain": "#F2C14E",
    "likely_incomplete": "#EE964B",
    "definitely_incomplete": "#C84630",
}
BIN_EDGES = [0, 5, 15, 35, 70, 100]
BIN_LABELS = ["0-5%", "5-15%", "15-35%", "35-70%", "70-100%"]


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Build histogram of estimated missing share for full ordinances.")
    ap.add_argument(
        "--predictions-csv",
        default=str(repo_root / "artifacts/scratch" / "full_ordinance_completeness_eval_20260318_full_gpt54_v10" / "all_predictions.csv"),
    )
    ap.add_argument(
        "--output-png",
        default=str(repo_root / "reports" / "full_ordinance_missing_share_histogram_20260318.png"),
    )
    ap.add_argument(
        "--output-csv",
        default=str(repo_root / "reports" / "full_ordinance_missing_share_histogram_20260318.csv"),
    )
    return ap.parse_args()


def _bin_index(percent: float) -> int:
    for idx in range(len(BIN_EDGES) - 1):
        lo = BIN_EDGES[idx]
        hi = BIN_EDGES[idx + 1]
        if idx == len(BIN_EDGES) - 2:
            if lo <= percent <= hi:
                return idx
        if lo <= percent < hi:
            return idx
    raise ValueError(f"Percent out of range: {percent}")


def main() -> None:
    args = _parse_args()
    predictions_csv = Path(args.predictions_csv).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with predictions_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in {predictions_csv}")

    counts: dict[str, list[int]] = {label: [0] * len(BIN_LABELS) for label in LABEL_ORDER}
    null_by_label = Counter()
    total_numeric = 0
    for row in rows:
        label = row["completeness_label"]
        raw = (row.get("estimated_missing_share_0_to_1") or "").strip()
        if raw in {"", "null", "None"}:
            null_by_label[label] += 1
            continue
        pct = max(0.0, min(100.0, float(raw) * 100.0))
        counts[label][_bin_index(pct)] += 1
        total_numeric += 1

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["bin_label", *LABEL_ORDER, "total"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, bin_label in enumerate(BIN_LABELS):
            row = {"bin_label": bin_label}
            total = 0
            for label in LABEL_ORDER:
                row[label] = counts[label][idx]
                total += counts[label][idx]
            row["total"] = total
            writer.writerow(row)
        writer.writerow(
            {
                "bin_label": "null_estimate",
                **{label: null_by_label.get(label, 0) for label in LABEL_ORDER},
                "total": sum(null_by_label.values()),
            }
        )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bottoms = [0] * len(BIN_LABELS)
    x = range(len(BIN_LABELS))
    for label in LABEL_ORDER:
        vals = counts[label]
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=label,
            color=LABEL_COLORS[label],
            width=0.8,
            edgecolor="white",
            linewidth=0.6,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(list(x), BIN_LABELS)
    ax.set_ylabel("Number of ordinances")
    ax.set_xlabel("Estimated missing share of ordinance body")
    ax.set_title("Estimated Missing Share for Full-Ordinance Artifacts")
    ax.legend(title="Completeness label", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    null_total = sum(null_by_label.values())
    subtitle = (
        f"Source: {predictions_csv.name} | n={len(rows)} total | "
        f"n={total_numeric} with numeric estimate | n={null_total} null estimate"
    )
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    print(
        {
            "predictions_csv": str(predictions_csv),
            "output_png": str(output_png),
            "output_csv": str(output_csv),
            "total_rows": len(rows),
            "numeric_estimates": total_numeric,
            "null_estimates": null_total,
        }
    )


if __name__ == "__main__":
    main()
