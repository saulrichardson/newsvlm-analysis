#!/usr/bin/env python3
"""
Compare two longitudinal pilot runs city-by-city.

Typical use:
python scripts/compare_longitudinal_runs.py \
  --base-run-dir reports/runs/pi_v1_run60 \
  --fork-run-dir reports/runs/pi_v1_run60_fullnewspaper
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGES = ("early", "mid", "late")
TREND_COLS = (
    "regulatory_complexity_trend",
    "procedural_intensity_trend",
    "dimensional_specificity_trend",
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare two longitudinal runs.")
    ap.add_argument(
        "--base-run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60",
        help="Baseline run directory.",
    )
    ap.add_argument(
        "--fork-run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Fork run directory.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <fork-run-dir>/longitudinal/comparison_vs_base).",
    )
    return ap.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def _norm_str(x: Any) -> str:
    return str(x or "").strip()


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.ndim != 1 or q.ndim != 1 or len(p) != len(q) or len(p) == 0:
        return math.nan
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    ps = p.sum()
    qs = q.sum()
    if ps <= 0 or qs <= 0:
        return math.nan
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_issue_expansion(city: pd.DataFrame, fig_path: Path) -> None:
    d = city.sort_values("issue_count_delta", ascending=False).head(15).copy()
    d["label"] = d["city_name_base"].fillna(d["city_key"]).astype(str) + ", " + d["state_abbr_base"].fillna("").astype(str).str.upper()
    plt.figure(figsize=(10, 6))
    plt.barh(d["label"], d["issue_count_delta"], color="#2f7aa3")
    plt.gca().invert_yaxis()
    plt.xlabel("Additional issues included in full-newspaper run")
    plt.title("Panel expansion by city (full newspaper vs sampled panel)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def _plot_change_counts(change_counts: pd.DataFrame, fig_path: Path) -> None:
    d = change_counts.sort_values("changed_city_count", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(d["metric"], d["changed_city_count"], color="#d16f32")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Cities with changed value")
    plt.title("How often core outputs changed after full-newspaper rerun")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def _plot_distribution_shift(shift: pd.DataFrame, fig_path: Path) -> None:
    d = shift.sort_values("late_js_divergence", ascending=False).head(15).copy()
    d["label"] = d["city_name"].fillna(d["city_key"]).astype(str) + ", " + d["state_abbr"].fillna("").astype(str).str.upper()
    plt.figure(figsize=(10, 6))
    plt.barh(d["label"], d["late_js_divergence"], color="#7f8f31")
    plt.gca().invert_yaxis()
    plt.xlabel("JS divergence (late-stage category distribution, base vs full)")
    plt.title("Cities with largest late-stage composition changes")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def main() -> None:
    args = _parse_args()
    base_run = Path(args.base_run_dir).expanduser().resolve()
    fork_run = Path(args.fork_run_dir).expanduser().resolve()
    base_long = base_run / "longitudinal"
    fork_long = fork_run / "longitudinal"
    out_dir = Path(args.output_dir).expanduser().resolve() if _norm_str(args.output_dir) else (fork_long / "comparison_vs_base")
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    base_sum = _read_csv(base_long / "city_longitudinal_summary.csv")
    fork_sum = _read_csv(fork_long / "city_longitudinal_summary.csv")
    base_evt = _read_csv(base_long / "city_first_zoning_events.csv")
    fork_evt = _read_csv(fork_long / "city_first_zoning_events.csv")
    base_cat = _read_csv(base_long / "city_category_trajectory.csv")
    fork_cat = _read_csv(fork_long / "city_category_trajectory.csv")

    city = base_sum.merge(fork_sum, on="city_key", how="inner", suffixes=("_base", "_fork"))
    if city.empty:
        raise SystemExit("No overlapping cities between baseline and fork runs.")

    evt_cols = [
        "city_key",
        "first_zoning_issue_id",
        "first_zoning_issue_date",
        "first_zoning_signal_type",
        "first_full_status",
        "first_full_issue_id",
        "first_full_issue_date",
    ]
    city = city.merge(base_evt[evt_cols], on="city_key", how="left")
    city = city.rename(
        columns={
            "first_zoning_issue_id": "first_zoning_issue_id_base",
            "first_zoning_issue_date": "first_zoning_issue_date_base",
            "first_zoning_signal_type": "first_zoning_signal_type_event_base",
            "first_full_status": "first_full_status_event_base",
            "first_full_issue_id": "first_full_issue_id_base",
            "first_full_issue_date": "first_full_issue_date_base",
        }
    )
    city = city.merge(fork_evt[evt_cols], on="city_key", how="left")
    city = city.rename(
        columns={
            "first_zoning_issue_id": "first_zoning_issue_id_fork",
            "first_zoning_issue_date": "first_zoning_issue_date_fork",
            "first_zoning_signal_type": "first_zoning_signal_type_event_fork",
            "first_full_status": "first_full_status_event_fork",
            "first_full_issue_id": "first_full_issue_id_fork",
            "first_full_issue_date": "first_full_issue_date_fork",
        }
    )

    city["issue_count_delta"] = city["selected_issue_count_fork"] - city["selected_issue_count_base"]
    city["issue_count_ratio"] = city["selected_issue_count_fork"] / city["selected_issue_count_base"].replace(0, np.nan)
    city["panel_chars_delta"] = city["panel_chars_original_fork"] - city["panel_chars_original_base"]
    city["panel_chars_ratio"] = city["panel_chars_original_fork"] / city["panel_chars_original_base"].replace(0, np.nan)

    # Change flags for key outcomes.
    city["first_zoning_signal_type_changed"] = city["first_zoning_signal_type_fork"].astype(str) != city["first_zoning_signal_type_base"].astype(str)
    city["first_full_status_changed"] = city["first_full_status_fork"].astype(str) != city["first_full_status_base"].astype(str)
    for col in TREND_COLS:
        city[f"{col}_changed"] = city[f"{col}_fork"].astype(str) != city[f"{col}_base"].astype(str)

    city["n_evolution_phases_delta"] = city["n_evolution_phases_fork"] - city["n_evolution_phases_base"]
    city["n_amendment_waves_delta"] = city["n_amendment_waves_fork"] - city["n_amendment_waves_base"]

    d_base = pd.to_datetime(city["first_zoning_issue_date_base"], errors="coerce")
    d_fork = pd.to_datetime(city["first_zoning_issue_date_fork"], errors="coerce")
    city["first_zoning_date_delta_days"] = (d_fork - d_base).dt.days
    city["first_zoning_issue_id_changed"] = city["first_zoning_issue_id_fork"].astype(str) != city["first_zoning_issue_id_base"].astype(str)

    f_base = pd.to_datetime(city["first_full_issue_date_base"], errors="coerce")
    f_fork = pd.to_datetime(city["first_full_issue_date_fork"], errors="coerce")
    city["first_full_date_delta_days"] = (f_fork - f_base).dt.days
    city["first_full_issue_id_changed"] = city["first_full_issue_id_fork"].astype(str) != city["first_full_issue_id_base"].astype(str)

    # Category deltas.
    cat = base_cat.merge(
        fork_cat,
        on=["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "category"],
        how="inner",
        suffixes=("_base", "_fork"),
    )
    for s in STAGES:
        cat[f"{s}_delta"] = cat[f"{s}_share_fork"] - cat[f"{s}_share_base"]
        cat[f"{s}_abs_delta"] = (cat[f"{s}_delta"]).abs()
    cat["max_abs_stage_delta"] = cat[[f"{s}_abs_delta" for s in STAGES]].max(axis=1)
    cat.to_csv(out_dir / "comparison_category_deltas.csv", index=False)

    # City-level distribution shifts in category composition (base vs fork).
    shift_rows: list[dict[str, Any]] = []
    for city_key, g in cat.groupby("city_key", dropna=False):
        row = g.iloc[0]
        out: dict[str, Any] = {
            "city_key": city_key,
            "city_name": row["city_name"],
            "state_abbr": row["state_abbr"],
            "region": row["region"],
            "urbanicity_proxy": row["urbanicity_proxy"],
        }
        for s in STAGES:
            b = g[f"{s}_share_base"].to_numpy(dtype=float)
            f = g[f"{s}_share_fork"].to_numpy(dtype=float)
            out[f"{s}_l1_half"] = float(np.nansum(np.abs(np.nan_to_num(f) - np.nan_to_num(b))) / 2.0)
            out[f"{s}_js_divergence"] = _js_divergence(b, f)
        shift_rows.append(out)
    shift = pd.DataFrame.from_records(shift_rows)
    shift.to_csv(out_dir / "comparison_city_distribution_shift.csv", index=False)

    # Category-level summary.
    cat_summary = []
    for s in STAGES:
        tmp = (
            cat.groupby("category", dropna=False)[[f"{s}_delta", f"{s}_abs_delta"]]
            .mean()
            .reset_index()
            .rename(
                columns={
                    f"{s}_delta": "mean_delta",
                    f"{s}_abs_delta": "mean_abs_delta",
                }
            )
        )
        tmp.insert(0, "stage", s)
        cat_summary.append(tmp)
    cat_summary_df = pd.concat(cat_summary, ignore_index=True)
    cat_summary_df.to_csv(out_dir / "comparison_category_delta_summary.csv", index=False)

    # Field-level change counts.
    tracked_flags = [
        "first_zoning_signal_type_changed",
        "first_full_status_changed",
        "first_zoning_issue_id_changed",
        "first_full_issue_id_changed",
        "regulatory_complexity_trend_changed",
        "procedural_intensity_trend_changed",
        "dimensional_specificity_trend_changed",
    ]
    counts = pd.DataFrame(
        {
            "metric": tracked_flags,
            "changed_city_count": [int(city[c].fillna(False).astype(bool).sum()) for c in tracked_flags],
            "changed_city_share": [float(city[c].fillna(False).astype(bool).mean()) for c in tracked_flags],
        }
    )
    counts.to_csv(out_dir / "comparison_change_counts.csv", index=False)

    # Transition tables.
    status_transition = pd.crosstab(
        city["first_full_status_base"].astype(str),
        city["first_full_status_fork"].astype(str),
        dropna=False,
    )
    status_transition.to_csv(out_dir / "comparison_first_full_status_transition.csv")

    signal_transition = pd.crosstab(
        city["first_zoning_signal_type_base"].astype(str),
        city["first_zoning_signal_type_fork"].astype(str),
        dropna=False,
    )
    signal_transition.to_csv(out_dir / "comparison_first_signal_transition.csv")

    # Save city summary with key columns first.
    front_cols = [
        "city_key",
        "city_name_base",
        "state_abbr_base",
        "region_base",
        "urbanicity_proxy_base",
        "selected_issue_count_base",
        "selected_issue_count_fork",
        "issue_count_delta",
        "issue_count_ratio",
        "panel_chars_original_base",
        "panel_chars_original_fork",
        "panel_chars_delta",
        "panel_chars_ratio",
        "first_zoning_signal_type_base",
        "first_zoning_signal_type_fork",
        "first_zoning_signal_type_changed",
        "first_zoning_issue_id_base",
        "first_zoning_issue_id_fork",
        "first_zoning_date_delta_days",
        "first_full_status_base",
        "first_full_status_fork",
        "first_full_status_changed",
        "first_full_issue_id_base",
        "first_full_issue_id_fork",
        "first_full_date_delta_days",
        "regulatory_complexity_trend_base",
        "regulatory_complexity_trend_fork",
        "regulatory_complexity_trend_changed",
        "procedural_intensity_trend_base",
        "procedural_intensity_trend_fork",
        "procedural_intensity_trend_changed",
        "dimensional_specificity_trend_base",
        "dimensional_specificity_trend_fork",
        "dimensional_specificity_trend_changed",
        "n_evolution_phases_base",
        "n_evolution_phases_fork",
        "n_evolution_phases_delta",
        "n_amendment_waves_base",
        "n_amendment_waves_fork",
        "n_amendment_waves_delta",
        "overflow_applied_base",
        "overflow_applied_fork",
        "clipped_issue_count_base",
        "clipped_issue_count_fork",
    ]
    keep_cols = [c for c in front_cols if c in city.columns] + [c for c in city.columns if c not in front_cols]
    city[keep_cols].to_csv(out_dir / "comparison_city_summary.csv", index=False)

    # Compact overall metrics JSON.
    overall = {
        "n_cities_compared": int(len(city)),
        "issue_count_mean_base": float(city["selected_issue_count_base"].mean()),
        "issue_count_mean_fork": float(city["selected_issue_count_fork"].mean()),
        "issue_count_mean_delta": float(city["issue_count_delta"].mean()),
        "panel_chars_mean_base": float(city["panel_chars_original_base"].mean()),
        "panel_chars_mean_fork": float(city["panel_chars_original_fork"].mean()),
        "overflow_city_count_base": int(city["overflow_applied_base"].fillna(0).astype(int).sum()),
        "overflow_city_count_fork": int(city["overflow_applied_fork"].fillna(0).astype(int).sum()),
        "changed_first_signal_type_city_count": int(city["first_zoning_signal_type_changed"].sum()),
        "changed_first_full_status_city_count": int(city["first_full_status_changed"].sum()),
        "changed_regulatory_complexity_trend_city_count": int(city["regulatory_complexity_trend_changed"].sum()),
        "changed_procedural_intensity_trend_city_count": int(city["procedural_intensity_trend_changed"].sum()),
        "changed_dimensional_specificity_trend_city_count": int(city["dimensional_specificity_trend_changed"].sum()),
        "mean_n_evolution_phases_base": float(city["n_evolution_phases_base"].mean()),
        "mean_n_evolution_phases_fork": float(city["n_evolution_phases_fork"].mean()),
        "mean_n_amendment_waves_base": float(city["n_amendment_waves_base"].mean()),
        "mean_n_amendment_waves_fork": float(city["n_amendment_waves_fork"].mean()),
        "mean_early_js_divergence_base_vs_fork": float(shift["early_js_divergence"].mean()),
        "mean_mid_js_divergence_base_vs_fork": float(shift["mid_js_divergence"].mean()),
        "mean_late_js_divergence_base_vs_fork": float(shift["late_js_divergence"].mean()),
    }
    with (out_dir / "comparison_overall_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # Figures.
    _plot_issue_expansion(city, fig_dir / "issue_expansion_top15.png")
    _plot_change_counts(counts, fig_dir / "core_output_change_counts.png")
    _plot_distribution_shift(shift, fig_dir / "late_stage_distribution_shift_top15.png")

    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
