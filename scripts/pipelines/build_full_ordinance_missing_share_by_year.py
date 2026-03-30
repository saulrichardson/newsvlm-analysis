#!/usr/bin/env python3
"""
Plot estimated full-ordinance missing share against issue year.

The year used here is the issue/publication year from the live Dropbox full
corpus metadata. That is the only grounded year field available in the public
surface for all rows.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Plot estimated missing share by issue year for the full ordinance corpus.")
    ap.add_argument(
        "--predictions-csv",
        default=str(
            repo_root
            / "artifacts/scratch"
            / "full_ordinance_completeness_eval_20260318_full_gpt54_v10"
            / "all_predictions.csv"
        ),
    )
    ap.add_argument(
        "--metadata-csv",
        default="/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/newspaper_ordinances/metadata.csv",
    )
    ap.add_argument(
        "--output-png",
        default=str(repo_root / "reports" / "full_ordinance_missing_share_by_year_20260319.png"),
    )
    ap.add_argument(
        "--output-csv",
        default=str(repo_root / "reports" / "full_ordinance_missing_share_by_year_20260319.csv"),
    )
    ap.add_argument(
        "--window-years",
        type=int,
        default=5,
        help="Centered rolling window size, in years, for smoothed trend lines.",
    )
    return ap.parse_args()


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("No values for quantile")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def main() -> None:
    args = _parse_args()
    if args.window_years < 1 or args.window_years % 2 == 0:
        raise SystemExit("--window-years must be a positive odd integer")
    predictions_csv = Path(args.predictions_csv).expanduser().resolve()
    metadata_csv = Path(args.metadata_csv).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with predictions_csv.open("r", encoding="utf-8", newline="") as f:
        predictions = {row["issue_id"]: row for row in csv.DictReader(f)}
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        metadata_rows = list(csv.DictReader(f))

    joined_rows: list[dict[str, object]] = []
    yearly_values: dict[int, list[float]] = defaultdict(list)

    for meta in metadata_rows:
        issue_id = meta["issue_id"]
        if issue_id not in predictions:
            continue
        pred = predictions[issue_id]
        year = int(meta["issue_date"][:4])
        raw_share = (pred.get("estimated_missing_share_0_to_1") or "").strip()
        share = None if raw_share in {"", "null", "None"} else float(raw_share)
        if share is None:
            continue
        yearly_values[year].append(share)
        joined_rows.append(
            {
                "issue_id": issue_id,
                "issue_date": meta["issue_date"],
                "issue_year": year,
                "estimated_missing_share_0_to_1": share,
                "completeness_label": pred["completeness_label"],
            }
        )

    if not joined_rows:
        raise SystemExit("No joined rows found")

    years = sorted(yearly_values)
    half_window = args.window_years // 2

    smoothed_stats: dict[int, dict[str, float | int]] = {}
    for year in years:
        pooled_vals: list[float] = []
        for neighbor_year in range(year - half_window, year + half_window + 1):
            pooled_vals.extend(yearly_values.get(neighbor_year, []))
        if pooled_vals:
            pooled_vals = sorted(pooled_vals)
            smoothed_stats[year] = {
                "issue_count_in_window": len(pooled_vals),
                "mean": statistics.fmean(pooled_vals),
                "median": statistics.median(pooled_vals),
                "p25": _quantile(pooled_vals, 0.25),
                "p75": _quantile(pooled_vals, 0.75),
            }
        else:
            smoothed_stats[year] = {
                "issue_count_in_window": 0,
                "mean": math.nan,
                "median": math.nan,
                "p25": math.nan,
                "p75": math.nan,
            }

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "issue_year",
            "issue_count",
            "estimated_missing_share_mean",
            "estimated_missing_share_median",
            "estimated_missing_share_p25",
            "estimated_missing_share_p75",
            "estimated_missing_share_min",
            "estimated_missing_share_max",
            "smoothed_window_years",
            "smoothed_issue_count_in_window",
            "smoothed_estimated_missing_share_mean",
            "smoothed_estimated_missing_share_median",
            "smoothed_estimated_missing_share_p25",
            "smoothed_estimated_missing_share_p75",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for year in years:
            vals = sorted(yearly_values.get(year, []))
            smooth = smoothed_stats[year]
            row = {
                "issue_year": year,
                "issue_count": len(vals),
                "estimated_missing_share_mean": "",
                "estimated_missing_share_median": "",
                "estimated_missing_share_p25": "",
                "estimated_missing_share_p75": "",
                "estimated_missing_share_min": "",
                "estimated_missing_share_max": "",
                "smoothed_window_years": args.window_years,
                "smoothed_issue_count_in_window": smooth["issue_count_in_window"],
                "smoothed_estimated_missing_share_mean": "",
                "smoothed_estimated_missing_share_median": "",
                "smoothed_estimated_missing_share_p25": "",
                "smoothed_estimated_missing_share_p75": "",
            }
            if vals:
                row.update(
                    {
                        "estimated_missing_share_mean": f"{statistics.fmean(vals):.6f}",
                        "estimated_missing_share_median": f"{statistics.median(vals):.6f}",
                        "estimated_missing_share_p25": f"{_quantile(vals, 0.25):.6f}",
                        "estimated_missing_share_p75": f"{_quantile(vals, 0.75):.6f}",
                        "estimated_missing_share_min": f"{vals[0]:.6f}",
                        "estimated_missing_share_max": f"{vals[-1]:.6f}",
                    }
                )
            if smooth["issue_count_in_window"]:
                row.update(
                    {
                        "smoothed_estimated_missing_share_mean": f"{smooth['mean']:.6f}",
                        "smoothed_estimated_missing_share_median": f"{smooth['median']:.6f}",
                        "smoothed_estimated_missing_share_p25": f"{smooth['p25']:.6f}",
                        "smoothed_estimated_missing_share_p75": f"{smooth['p75']:.6f}",
                    }
                )
            writer.writerow(row)

    rng = random.Random(20260319)
    scatter_x: list[float] = []
    scatter_y: list[float] = []
    for row in joined_rows:
        share = row["estimated_missing_share_0_to_1"]
        if share is None:
            continue
        year = int(row["issue_year"])
        scatter_x.append(year + rng.uniform(-0.28, 0.28))
        scatter_y.append(float(share))

    counts = []
    for year in years:
        vals = sorted(yearly_values.get(year, []))
        counts.append(len(vals))
    smoothed_medians = [smoothed_stats[year]["median"] for year in years]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [4.5, 1.3]},
    )

    ax_top.scatter(
        scatter_x,
        scatter_y,
        s=18,
        alpha=0.28,
        color="#4C78A8",
        edgecolors="none",
        label="Issue-level estimate",
    )
    ax_top.plot(
        years,
        smoothed_medians,
        color="#8B1E3F",
        linewidth=3.0,
        label=f"{args.window_years}-year centered median",
    )
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.set_ylabel("Estimated Missing Share")
    ax_top.set_title("Estimated Full-Ordinance Missing Share by Issue Year (Numeric Estimates Only)")
    ax_top.grid(axis="y", linestyle="--", alpha=0.25)
    ax_top.legend(loc="upper left", frameon=True)

    ax_bottom.bar(years, counts, width=0.85, color="#7A7A7A", alpha=0.8, label="Issue count")
    ax_bottom.set_ylabel("Count")
    ax_bottom.set_xlabel("Issue Year")
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.2)
    ax_bottom.legend(loc="upper left", frameon=True)

    xticks = [year for year in years if year % 5 == 0]
    if years[-1] not in xticks:
        xticks.append(years[-1])
    ax_bottom.set_xticks(xticks)
    ax_bottom.set_xlim(min(years) - 1, max(years) + 1)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_png}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
