#!/usr/bin/env python3
"""
Build v2 LaTeX report for the 20-panel longitudinal pilot.

v2 emphasizes interpretable deterministic metrics, phase-sequence summaries,
and manual transcript checks tied to model-derived claims.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_FIXED_CATEGORIES = [
    "land_use_restrictions",
    "bulk_dimensional_standards",
    "procedural_governance",
    "nonconforming_use",
    "definitions",
    "parking_transport",
    "environmental_protection",
    "historic_preservation",
    "flexible_zoning",
    "inclusionary_affordable_housing",
    "enforcement_penalties",
    "miscellaneous_site_rules",
]

_PHASE_LABEL_DISPLAY = {
    "initial_establishment": "Initial establishment",
    "formalization_enforcement": "Formalization & enforcement",
    "expansion_reclassification": "Expansion & reclassification",
    "flexible_special_use": "Flexible/special use",
    "infrastructure_environment": "Infrastructure/environment",
    "maturation_implementation": "Maturation & implementation",
    "other_unclear": "Other/unclear",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _tex_escape(s: Any) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _short_text(s: Any, max_len: int = 92) -> str:
    t = str(s or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _write_table_tex(path: Path, df: pd.DataFrame, cols: list[str], float_cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("\\begin{tabular}{l}\n\\toprule\nNo data\\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return

    keep = [c for c in cols if c in df.columns]
    if not keep:
        keep = list(df.columns[: min(7, len(df.columns))])

    sub = df[keep].copy()
    float_cols = float_cols or []
    for c in sub.columns:
        if c in float_cols or pd.api.types.is_float_dtype(sub[c]):
            sub[c] = pd.to_numeric(sub[c], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
        else:
            sub[c] = sub[c].map(lambda v: _short_text(v, 96))

    lines: list[str] = []
    lines.append("\\begin{tabular}{" + "l" * len(sub.columns) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(_tex_escape(c) for c in sub.columns) + " \\\\")
    lines.append("\\midrule")
    for row in sub.itertuples(index=False):
        lines.append(" & ".join(_tex_escape(v) for v in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _plot_lag_distribution(events: pd.DataFrame, out: Path) -> None:
    if events.empty:
        return
    d = events.copy()
    d["first_signal_to_full_lag_years"] = pd.to_numeric(d["first_signal_to_full_lag_years"], errors="coerce")
    d = d[(d["lag_observed"] == 1) & d["first_signal_to_full_lag_years"].notna()].copy()
    if d.empty:
        return

    x = d["first_signal_to_full_lag_years"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(0, max(1.0, float(np.nanmax(x))), 10)
    ax.hist(x, bins=bins, color="#4e79a7", edgecolor="white")
    ax.axvline(float(np.nanmedian(x)), color="#e15759", linestyle="--", linewidth=1.7, label=f"Median={np.nanmedian(x):.2f}y")
    ax.set_xlabel("Years between first zoning signal and first full ordinance")
    ax.set_ylabel("Cities")
    ax.set_title("First-signal to first-full ordinance lag")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_phase_sequence_counts(seq_df: pd.DataFrame, out: Path) -> None:
    if seq_df.empty:
        return
    counts = seq_df["phase_sequence"].astype(str).value_counts().head(8)
    if counts.empty:
        return
    labels = [s.replace("_", " ") for s in counts.index.tolist()]
    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    ax.barh(labels[::-1], counts.values[::-1], color="#76b7b2")
    ax.set_xlabel("City count")
    ax.set_ylabel("Phase sequence")
    ax.set_title("Most common phase sequences across pilot cities")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_category_shift_heatmap(cat_shift: pd.DataFrame, out: Path) -> None:
    if cat_shift.empty:
        return
    d = cat_shift.copy()
    d["late_minus_early"] = pd.to_numeric(d["late_minus_early"], errors="coerce")
    d = d.dropna(subset=["late_minus_early"]) 
    if d.empty:
        return

    pivot = (
        d.groupby(["region", "category"], dropna=False)["late_minus_early"]
        .mean()
        .reset_index()
        .pivot(index="category", columns="region", values="late_minus_early")
    )
    if pivot.empty:
        return

    # Stable category order.
    cats = [c for c in _FIXED_CATEGORIES if c in pivot.index]
    pivot = pivot.reindex(cats)

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    vals = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 0.1
    vmax = max(vmax, 0.05)
    im = ax.imshow(vals, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_title("Category shift heatmap: mean (late - early) share by region")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Late - Early share")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_amendment_balance_region(region_wave: pd.DataFrame, out: Path) -> None:
    if region_wave.empty:
        return
    d = region_wave.copy()
    d["count"] = pd.to_numeric(d["count"], errors="coerce")
    d = d.dropna(subset=["count"])
    if d.empty:
        return

    pv = d.pivot_table(index="region", columns="change_type", values="count", aggfunc="sum", fill_value=0)
    if pv.empty:
        return

    cols = [c for c in ["expansion", "tightening", "clarification", "reclassification", "mixed"] if c in pv.columns] + [
        c for c in pv.columns if c not in {"expansion", "tightening", "clarification", "reclassification", "mixed"}
    ]
    pv = pv[cols]

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    pv.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Region")
    ax.set_ylabel("Wave count")
    ax.set_title("Amendment wave type mix by region")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_substantive_shift_top(insight: pd.DataFrame, out: Path) -> None:
    if insight.empty:
        return
    d = insight.copy()
    d["substantive_shift_js"] = pd.to_numeric(d["substantive_shift_js"], errors="coerce")
    d = d.dropna(subset=["substantive_shift_js"]).sort_values("substantive_shift_js", ascending=False).head(10)
    if d.empty:
        return

    labels = [f"{r.city_name}, {str(r.state_abbr).upper()}" for r in d.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    ax.barh(labels[::-1], d["substantive_shift_js"].to_numpy()[::-1], color="#f28e2b")
    ax.set_xlabel("Substantive shift index (JS divergence, early vs late)")
    ax.set_ylabel("City")
    ax.set_title("Top movers in zoning content composition")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_tercile_profile(cat_terc: pd.DataFrame, out: Path) -> None:
    if cat_terc.empty:
        return
    d = cat_terc.copy()
    d["share"] = pd.to_numeric(d["share"], errors="coerce")
    d = d.dropna(subset=["share"])
    if d.empty:
        return

    g = (
        d.groupby(["category", "tercile"], dropna=False)["share"]
        .mean()
        .reset_index()
        .pivot(index="category", columns="tercile", values="share")
        .reset_index()
    )
    for c in ["early", "mid", "late"]:
        if c not in g.columns:
            g[c] = math.nan
    g["avg"] = g[["early", "mid", "late"]].mean(axis=1)
    g = g.sort_values("avg", ascending=False).reset_index(drop=True)

    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.plot(x, g["early"], marker="o", color="#4e79a7", label="Early (first tercile of issues)")
    ax.plot(x, g["mid"], marker="o", color="#59a14f", label="Mid (second tercile)")
    ax.plot(x, g["late"], marker="o", color="#e15759", label="Late (final tercile)")
    ax.set_xticks(x)
    ax.set_xticklabels(g["category"].astype(str).tolist(), rotation=38, ha="right")
    ax.set_ylabel("Mean category share")
    ax.set_title("Deterministic tercile category profiles")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_manual_verdicts(reviewed: pd.DataFrame, out: Path) -> None:
    if reviewed.empty or "manual_verdict" not in reviewed.columns:
        return
    counts = reviewed["manual_verdict"].astype(str).str.strip().str.lower().value_counts()
    if counts.empty:
        return
    order = [c for c in ["pass", "partial", "fail", ""] if c in counts.index]
    if not order:
        order = counts.index.tolist()
    vals = [counts.get(o, 0) for o in order]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = ["#59a14f" if o == "pass" else "#f28e2b" if o == "partial" else "#e15759" for o in order]
    ax.bar(order, vals, color=colors)
    ax.set_xlabel("Manual verdict")
    ax.set_ylabel("Validation cases")
    ax.set_title("Manual transcript validation outcomes")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build v2 longitudinal pilot report.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60",
        help="Run directory containing longitudinal and longitudinal_v2 outputs.",
    )
    ap.add_argument("--longitudinal-dir", default="", help="Longitudinal input dir (default: <run-dir>/longitudinal).")
    ap.add_argument("--v2-dir", default="", help="V2 metrics dir (default: <run-dir>/longitudinal_v2).")
    ap.add_argument("--output-dir", default="", help="Output report dir (default: <run-dir>/reports/longitudinal_pilot20_v2).")
    ap.add_argument("--compile-pdf", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--title",
        default="Longitudinal Pilot v2 (20 Panels): Interpretable Zoning Evolution Metrics",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    lon_dir = Path(args.longitudinal_dir).expanduser().resolve() if str(args.longitudinal_dir).strip() else (run_dir / "longitudinal")
    v2_dir = Path(args.v2_dir).expanduser().resolve() if str(args.v2_dir).strip() else (run_dir / "longitudinal_v2")
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "reports" / "longitudinal_pilot20_v2")

    figs = out_dir / "figures"
    tbls = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    tbls.mkdir(parents=True, exist_ok=True)

    # Inputs.
    event_lags = _read_csv(v2_dir / "city_event_lags.csv")
    seq = _read_csv(v2_dir / "city_phase_sequences.csv")
    cat_shift = _read_csv(v2_dir / "city_category_shift_metrics.csv")
    amend_balance = _read_csv(v2_dir / "city_amendment_balance_metrics.csv")
    insight = _read_csv(v2_dir / "city_insight_indices.csv")
    cat_terc = _read_csv(v2_dir / "city_tercile_category_shares.csv")
    region_waves = _read_csv(v2_dir / "region_wave_type_counts.csv")
    reviewed = _read_csv(v2_dir / "manual_validation_reviewed.csv")
    deep_review = _read_csv(v2_dir / "deep_dive_manual_review.csv")
    summary = _read_csv(lon_dir / "city_longitudinal_summary.csv")
    cmp_dir = lon_dir / "comparison_vs_base"
    cmp_overall: dict[str, Any] = {}
    cmp_change_counts = _read_csv(cmp_dir / "comparison_change_counts.csv")
    cmp_city_summary = _read_csv(cmp_dir / "comparison_city_summary.csv")
    cmp_overall_path = cmp_dir / "comparison_overall_metrics.json"
    if cmp_overall_path.is_file():
        try:
            cmp_overall = json.loads(cmp_overall_path.read_text(encoding="utf-8"))
        except Exception:
            cmp_overall = {}

    # Figures.
    fig_lag = figs / "event_lag_distribution.png"
    fig_seq = figs / "phase_sequence_counts.png"
    fig_heat = figs / "category_shift_heatmap_region.png"
    fig_wave = figs / "amendment_balance_region.png"
    fig_shift = figs / "substantive_shift_top_movers.png"
    fig_terc = figs / "deterministic_tercile_profile.png"
    fig_manual = figs / "manual_validation_verdicts.png"

    _plot_lag_distribution(event_lags, fig_lag)
    _plot_phase_sequence_counts(seq, fig_seq)
    _plot_category_shift_heatmap(cat_shift, fig_heat)
    _plot_amendment_balance_region(region_waves, fig_wave)
    _plot_substantive_shift_top(insight, fig_shift)
    _plot_tercile_profile(cat_terc, fig_terc)
    _plot_manual_verdicts(reviewed, fig_manual)

    cmp_fig_issue = figs / "comparison_issue_expansion_top15.png"
    cmp_fig_changes = figs / "comparison_core_output_change_counts.png"
    cmp_fig_shift = figs / "comparison_late_stage_distribution_shift_top15.png"
    has_cmp_fig_issue = _copy_if_exists(cmp_dir / "figures" / "issue_expansion_top15.png", cmp_fig_issue)
    has_cmp_fig_changes = _copy_if_exists(cmp_dir / "figures" / "core_output_change_counts.png", cmp_fig_changes)
    has_cmp_fig_shift = _copy_if_exists(cmp_dir / "figures" / "late_stage_distribution_shift_top15.png", cmp_fig_shift)

    # Tables.
    metric_def = pd.DataFrame(
        [
            {
                "metric": "Early / Mid / Late",
                "definition": "Issue-tercile bins within city panel (chronological first/middle/final third).",
            },
            {
                "metric": "Substantive shift index",
                "definition": "Jensen-Shannon divergence between early and late 12-category share vectors.",
            },
            {
                "metric": "Proceduralization delta",
                "definition": "(procedural+enforcement+definitions - land_use-bulk) late minus early.",
            },
            {
                "metric": "Flexibility uptake index",
                "definition": "Late flexible_zoning share minus early flexible_zoning share.",
            },
            {
                "metric": "First-signal to first-full lag",
                "definition": "Years between first zoning signal date and first full-ordinance date (when observed).",
            },
            {
                "metric": "Amendment intensity",
                "definition": "Amendment wave count per decade of panel time span.",
            },
        ]
    )

    top_shift = insight.sort_values("substantive_shift_js", ascending=False).head(10)
    seq_counts = seq["phase_sequence"].astype(str).value_counts().reset_index()
    seq_counts.columns = ["phase_sequence", "count"]
    seq_counts = seq_counts.head(10)

    wave_city = amend_balance[[
        "city_key",
        "wave_count",
        "waves_per_decade",
        "expansion",
        "tightening",
        "clarification",
        "reclassification",
        "mixed",
        "expansion_minus_tightening",
    ]].copy() if not amend_balance.empty else pd.DataFrame()

    manual_tbl = reviewed[[
        "claim_group",
        "city_key",
        "issue_id",
        "manual_verdict",
        "manual_note",
    ]].copy() if not reviewed.empty else pd.DataFrame()

    # Add exemplar excerpts for appendix.
    manual_excerpt_tbl = reviewed[[
        "claim_group",
        "city_key",
        "issue_id",
        "manual_verdict",
        "excerpt",
    ]].copy() if not reviewed.empty else pd.DataFrame()

    deep_tbl = deep_review[[
        "city_display",
        "focus_code",
        "substantive_shift_js",
        "proceduralization_delta",
        "flexibility_uptake_index",
        "early_issue_id",
        "late_issue_id",
        "manual_verdict",
        "manual_note",
    ]].copy() if not deep_review.empty else pd.DataFrame()

    _write_table_tex(tbls / "metric_definitions.tex", metric_def, ["metric", "definition"])
    _write_table_tex(
        tbls / "top_substantive_movers.tex",
        top_shift,
        [
            "city_key",
            "region",
            "urbanicity_proxy",
            "substantive_shift_js",
            "proceduralization_delta",
            "flexibility_uptake_index",
            "first_signal_to_full_lag_years",
            "waves_per_decade",
        ],
        float_cols=[
            "substantive_shift_js",
            "proceduralization_delta",
            "flexibility_uptake_index",
            "first_signal_to_full_lag_years",
            "waves_per_decade",
        ],
    )
    _write_table_tex(tbls / "phase_sequence_counts.tex", seq_counts, ["phase_sequence", "count"])
    _write_table_tex(
        tbls / "amendment_balance_city.tex",
        wave_city.sort_values("waves_per_decade", ascending=False).head(12) if not wave_city.empty else wave_city,
        [
            "city_key",
            "wave_count",
            "waves_per_decade",
            "expansion",
            "tightening",
            "clarification",
            "reclassification",
            "mixed",
            "expansion_minus_tightening",
        ],
        float_cols=["waves_per_decade"],
    )
    _write_table_tex(tbls / "manual_validation_summary.tex", manual_tbl, ["claim_group", "city_key", "issue_id", "manual_verdict", "manual_note"])
    _write_table_tex(tbls / "manual_validation_excerpts.tex", manual_excerpt_tbl.head(16), ["claim_group", "city_key", "manual_verdict", "excerpt"])
    _write_table_tex(
        tbls / "deep_dive_town_summary.tex",
        deep_tbl,
        [
            "city_display",
            "focus_code",
            "substantive_shift_js",
            "proceduralization_delta",
            "flexibility_uptake_index",
            "early_issue_id",
            "late_issue_id",
            "manual_verdict",
            "manual_note",
        ],
        float_cols=["substantive_shift_js", "proceduralization_delta", "flexibility_uptake_index"],
    )

    cmp_table = pd.DataFrame()
    if not cmp_change_counts.empty:
        cmp_table = cmp_change_counts.copy()
    _write_table_tex(
        tbls / "comparison_change_counts.tex",
        cmp_table,
        ["metric", "changed_city_count", "changed_city_share"],
        float_cols=["changed_city_share"],
    )

    cmp_status_changes = pd.DataFrame()
    if not cmp_city_summary.empty:
        keep = [
            "city_key",
            "first_full_status_base",
            "first_full_status_fork",
            "first_full_issue_id_base",
            "first_full_issue_id_fork",
            "first_full_date_delta_days",
        ]
        for c in keep:
            if c not in cmp_city_summary.columns:
                cmp_city_summary[c] = ""
        cmp_status_changes = cmp_city_summary[
            cmp_city_summary["first_full_status_base"].astype(str) != cmp_city_summary["first_full_status_fork"].astype(str)
        ][keep].copy()
    _write_table_tex(
        tbls / "comparison_first_full_status_changes.tex",
        cmp_status_changes,
        [
            "city_key",
            "first_full_status_base",
            "first_full_status_fork",
            "first_full_issue_id_base",
            "first_full_issue_id_fork",
            "first_full_date_delta_days",
        ],
        float_cols=["first_full_date_delta_days"],
    )

    # Headline metrics.
    n_city = int(summary["city_key"].nunique()) if not summary.empty else 0
    parse_share = float(pd.to_numeric(summary.get("parse_valid"), errors="coerce").mean()) if not summary.empty else math.nan
    schema_share = float(pd.to_numeric(summary.get("schema_valid"), errors="coerce").mean()) if not summary.empty else math.nan
    phase_nonempty = float((pd.to_numeric(summary.get("n_evolution_phases"), errors="coerce") > 0).mean()) if not summary.empty else math.nan
    wave_nonempty = float((pd.to_numeric(summary.get("n_amendment_waves"), errors="coerce") > 0).mean()) if not summary.empty else math.nan

    lag_obs = pd.to_numeric(event_lags.get("lag_observed"), errors="coerce") if not event_lags.empty else pd.Series(dtype=float)
    lag_years = pd.to_numeric(event_lags.get("first_signal_to_full_lag_years"), errors="coerce") if not event_lags.empty else pd.Series(dtype=float)
    lag_share = float(lag_obs.mean()) if not lag_obs.empty else math.nan
    lag_median = float(lag_years[lag_obs == 1].median()) if not lag_years.empty and (lag_obs == 1).any() else math.nan

    mean_js = float(pd.to_numeric(insight.get("substantive_shift_js"), errors="coerce").mean()) if not insight.empty else math.nan
    mean_proc = float(pd.to_numeric(insight.get("proceduralization_delta"), errors="coerce").mean()) if not insight.empty else math.nan
    mean_flex = float(pd.to_numeric(insight.get("flexibility_uptake_index"), errors="coerce").mean()) if not insight.empty else math.nan

    verdict_counts = reviewed["manual_verdict"].astype(str).str.strip().str.lower().value_counts() if not reviewed.empty else pd.Series(dtype=int)
    pass_n = int(verdict_counts.get("pass", 0))
    partial_n = int(verdict_counts.get("partial", 0))
    fail_n = int(verdict_counts.get("fail", 0))

    deep_verdict_counts = (
        deep_review["manual_verdict"].astype(str).str.strip().str.lower().value_counts() if not deep_review.empty else pd.Series(dtype=int)
    )
    deep_pass_n = int(deep_verdict_counts.get("pass", 0))
    deep_partial_n = int(deep_verdict_counts.get("partial", 0))
    deep_fail_n = int(deep_verdict_counts.get("fail", 0))

    overflow_count = int(pd.to_numeric(summary.get("overflow_applied"), errors="coerce").fillna(0).astype(int).sum()) if not summary.empty else 0
    clipped_issue_total = int(pd.to_numeric(summary.get("clipped_issue_count"), errors="coerce").fillna(0).sum()) if not summary.empty else 0

    report_date = dt.datetime.now().strftime("%Y-%m-%d")
    doc: list[str] = []
    doc.append("\\documentclass[11pt]{article}")
    doc.append("\\usepackage[margin=1in]{geometry}")
    doc.append("\\usepackage{graphicx}")
    doc.append("\\usepackage{booktabs}")
    doc.append("\\usepackage{float}")
    doc.append("\\usepackage{hyperref}")
    doc.append("\\usepackage{longtable}")
    doc.append(f"\\title{{{_tex_escape(args.title)}}}")
    doc.append("\\author{Automated v2 longitudinal pipeline}")
    doc.append(f"\\date{{{_tex_escape(report_date)}}}")
    doc.append("\\begin{document}")
    doc.append("\\maketitle")
    doc.append("\\tableofcontents")
    doc.append("\\clearpage")

    doc.append("\\section{Read this first: how the measurements are built}")
    doc.append(
        "This report combines two layers: (i) deterministic calculations from issue-level category outputs, and "
        "(ii) panel-level longitudinal labels from the LLM (first signal, first full ordinance, phases, and waves). "
        "The deterministic layer is the backbone for the quantitative figures."
    )
    doc.append("\\subsection{Strategy in six steps}")
    doc.append("\\begin{enumerate}")
    doc.append(
        "\\item \\textbf{Issue parsing and normalization.} For each city issue, we read fixed 12-category shares and "
        "normalize shares to sum to 1.0 when parsed output exists."
    )
    doc.append(
        "\\item \\textbf{Deterministic time bins (early/mid/late).} Within each city, issues are ordered chronologically "
        "(\\texttt{issue\\_date}, then \\texttt{issue\\_id}) and split into first/middle/final terciles."
    )
    doc.append(
        "\\item \\textbf{City-tercile aggregation.} We compute city-level mean category shares for each tercile; these "
        "feed profile charts and late-minus-early shift tables."
    )
    doc.append(
        "\\item \\textbf{Interpretable indices.} We compute: (a) substantive shift index as Jensen-Shannon divergence "
        "between early and late 12-category vectors; (b) proceduralization delta; (c) flexibility uptake index."
    )
    doc.append(
        "\\item \\textbf{Panel-level longitudinal structure.} We retain LLM panel outputs for first-zoning signal, first "
        "full ordinance, evolution phases, and amendment waves, then summarize these in counts/distributions. "
        "Operationally, this is one LLM request per city panel with all selected issues included in chronological order "
        "(subject to overflow clipping if a panel exceeds the configured character budget)."
    )
    doc.append(
        "\\item \\textbf{Focused manual validation.} A smaller set of high-change towns is reviewed with long transcript "
        "excerpts (early vs late anchors) to verify that claimed shifts are visible in source text."
    )
    doc.append("\\end{enumerate}")

    doc.append("\\subsection{How to read the main outputs}")
    doc.append("\\begin{itemize}")
    doc.append(
        "\\item \\textbf{Deterministic tercile profile figure:} average category composition in early/mid/late bins."
    )
    doc.append(
        "\\item \\textbf{Top movers figure:} cities ranked by substantive shift index (largest early-to-late composition change)."
    )
    doc.append(
        "\\item \\textbf{Amendment style tables:} counts of expansion/tightening/clarification/reclassification/mixed wave labels."
    )
    doc.append(
        "\\item \\textbf{Deep-dive section:} transcript-grounded checks that tie index claims back to concrete ordinance text."
    )
    doc.append("\\end{itemize}")

    doc.append("\\subsection{Scope guardrails in this report}")
    doc.append("\\begin{itemize}")
    doc.append("\\item No external covariate/adoption-regression narrative is used in this v2 pilot report.")
    doc.append("\\item Early/mid/late are issue-count terciles within each city panel, not calendar-year terciles.")
    if overflow_count == 0:
        doc.append("\\item Overflow clipping did not trigger in this run; panel prompts used full selected issue text.")
    else:
        doc.append(
            f"\\item Overflow clipping triggered for {overflow_count} cities (total clipped issues: {clipped_issue_total}); "
            "all other panels used full selected issue text."
        )
    doc.append("\\item Quantitative claims are tied to deterministic transformations of parsed issue outputs.")
    doc.append("\\end{itemize}")

    if cmp_overall:
        doc.append("\\section{Sampled vs full-newspaper rerun (same 20 cities)}")
        doc.append(
            "This fork keeps the same city set but expands each city panel from sampled issues to all available local-text issues, "
            "then reruns the same panel-level longitudinal extraction."
        )
        doc.append("\\begin{itemize}")
        if "issue_count_mean_base" in cmp_overall and "issue_count_mean_fork" in cmp_overall:
            doc.append(
                f"\\item Mean issues per city: {cmp_overall.get('issue_count_mean_base', float('nan')):.2f} "
                f"$\\rightarrow$ {cmp_overall.get('issue_count_mean_fork', float('nan')):.2f}"
            )
        if "panel_chars_mean_base" in cmp_overall and "panel_chars_mean_fork" in cmp_overall:
            doc.append(
                f"\\item Mean panel characters: {cmp_overall.get('panel_chars_mean_base', float('nan')):,.0f} "
                f"$\\rightarrow$ {cmp_overall.get('panel_chars_mean_fork', float('nan')):,.0f}"
            )
        if "changed_first_signal_type_city_count" in cmp_overall:
            doc.append(f"\\item Cities with changed first-signal type: {int(cmp_overall.get('changed_first_signal_type_city_count', 0))}")
        if "changed_first_full_status_city_count" in cmp_overall:
            doc.append(f"\\item Cities with changed first-full status: {int(cmp_overall.get('changed_first_full_status_city_count', 0))}")
        if "changed_dimensional_specificity_trend_city_count" in cmp_overall:
            doc.append(
                f"\\item Cities with changed dimensional-specificity trend: {int(cmp_overall.get('changed_dimensional_specificity_trend_city_count', 0))}"
            )
        doc.append("\\end{itemize}")
        doc.append("\\begin{table}[H]\\centering\\caption{How often core outputs changed in sampled vs full rerun}\\input{tables/comparison_change_counts.tex}\\end{table}")
        if has_cmp_fig_issue:
            doc.append(
                "\\begin{figure}[H]\\centering\\includegraphics[width=0.88\\textwidth]{figures/comparison_issue_expansion_top15.png}"
                "\\caption{Issue-count expansion for the largest cities under full-newspaper coverage.}\\end{figure}"
            )
        if has_cmp_fig_changes:
            doc.append(
                "\\begin{figure}[H]\\centering\\includegraphics[width=0.82\\textwidth]{figures/comparison_core_output_change_counts.png}"
                "\\caption{Count of cities where core longitudinal outputs changed after full-newspaper rerun.}\\end{figure}"
            )
        if has_cmp_fig_shift:
            doc.append(
                "\\begin{figure}[H]\\centering\\includegraphics[width=0.88\\textwidth]{figures/comparison_late_stage_distribution_shift_top15.png}"
                "\\caption{Largest late-stage category-composition shifts (sampled vs full rerun).}\\end{figure}"
            )
        if not cmp_status_changes.empty:
            doc.append(
                "\\begin{table}[H]\\centering\\caption{Cities where first-full status changed under full-newspaper rerun}"
                "\\input{tables/comparison_first_full_status_changes.tex}\\end{table}"
            )

    doc.append("\\section{Run integrity}")
    doc.append("\\begin{itemize}")
    doc.append(f"\\item Pilot cities: {n_city}")
    doc.append(f"\\item Parse-valid share: {parse_share:.3f}" if pd.notna(parse_share) else "\\item Parse-valid share: NA")
    doc.append(f"\\item Schema-valid share: {schema_share:.3f}" if pd.notna(schema_share) else "\\item Schema-valid share: NA")
    doc.append(f"\\item Non-empty phase share: {phase_nonempty:.3f}" if pd.notna(phase_nonempty) else "\\item Non-empty phase share: NA")
    doc.append(f"\\item Non-empty amendment-wave share: {wave_nonempty:.3f}" if pd.notna(wave_nonempty) else "\\item Non-empty amendment-wave share: NA")
    doc.append("\\end{itemize}")

    doc.append("\\section{Timing of zoning emergence}")
    doc.append(
        "The lag metric shows how quickly city panels move from first zoning signal to observed full ordinance text."
    )
    if fig_lag.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.80\\textwidth]{figures/event_lag_distribution.png}\\caption{Distribution of first-signal to first-full lag (observed cases).}\\end{figure}")
    doc.append("\\begin{itemize}")
    doc.append(f"\\item Lag observed share: {lag_share:.3f}" if pd.notna(lag_share) else "\\item Lag observed share: NA")
    doc.append(f"\\item Median lag among observed: {lag_median:.2f} years" if pd.notna(lag_median) else "\\item Median lag among observed: NA")
    doc.append("\\end{itemize}")

    doc.append("\\section{Content evolution and phase sequences}")
    doc.append(
        "This section combines deterministic category-shift metrics with panel-level phase sequences from the LLM to summarize "
        "how zoning content evolves over time."
    )
    if fig_terc.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.96\\textwidth]{figures/deterministic_tercile_profile.png}\\caption{Deterministic early/mid/late category profiles (issue terciles).}\\end{figure}")
    if fig_heat.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.86\\textwidth]{figures/category_shift_heatmap_region.png}\\caption{Mean category share change (late minus early) by region.}\\end{figure}")
    if fig_seq.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.96\\textwidth]{figures/phase_sequence_counts.png}\\caption{Top phase sequences across pilot cities.}\\end{figure}")
    doc.append("\\begin{table}[H]\\centering\\caption{Top phase sequences}\\input{tables/phase_sequence_counts.tex}\\end{table}")

    doc.append("\\section{Amendment style and intensity}")
    doc.append(
        "Amendment metrics report style mix (expansion/tightening/clarification/reclassification/mixed) and intensity normalized by panel time span."
    )
    if fig_wave.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.94\\textwidth]{figures/amendment_balance_region.png}\\caption{Amendment wave type mix by region.}\\end{figure}")
    doc.append("\\begin{table}[H]\\centering\\caption{City amendment balance (top by waves/decade)}\\input{tables/amendment_balance_city.tex}\\end{table}")

    doc.append("\\section{Top movers and interpretable indices}")
    doc.append("\\begin{itemize}")
    doc.append(f"\\item Mean substantive shift index (JS): {mean_js:.3f}" if pd.notna(mean_js) else "\\item Mean substantive shift index (JS): NA")
    doc.append(f"\\item Mean proceduralization delta: {mean_proc:.3f}" if pd.notna(mean_proc) else "\\item Mean proceduralization delta: NA")
    doc.append(f"\\item Mean flexibility uptake index: {mean_flex:.3f}" if pd.notna(mean_flex) else "\\item Mean flexibility uptake index: NA")
    doc.append("\\end{itemize}")
    if fig_shift.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.92\\textwidth]{figures/substantive_shift_top_movers.png}\\caption{Top cities by substantive shift index.}\\end{figure}")
    doc.append("\\begin{table}[H]\\centering\\caption{Top substantive movers and companion indices}\\input{tables/top_substantive_movers.tex}\\end{table}")

    doc.append("\\section{Manual transcript validation}")
    doc.append(
        "Manual checks were run against original panel transcripts to verify whether high-level LLM claims are visibly supported in the source text."
    )
    doc.append("\\begin{itemize}")
    doc.append(f"\\item Manual checks marked pass: {pass_n}")
    doc.append(f"\\item Manual checks marked partial: {partial_n}")
    doc.append(f"\\item Manual checks marked fail: {fail_n}")
    doc.append("\\end{itemize}")
    if fig_manual.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.58\\textwidth]{figures/manual_validation_verdicts.png}\\caption{Manual validation outcomes for transcript-linked claims.}\\end{figure}")
    doc.append("\\begin{table}[H]\\centering\\caption{Manual validation summary}\\input{tables/manual_validation_summary.tex}\\end{table}")

    if not deep_review.empty:
        doc.append("\\subsection{Focused deep-dive checks (few towns, long excerpts)}")
        doc.append(
            "To improve interpretability, we run a deeper manual pass on a smaller set of towns with large measured shifts. "
            "Each town uses one early anchor issue and one late anchor issue, with long transcript excerpts embedded below."
        )
        doc.append("\\begin{itemize}")
        doc.append(f"\\item Deep-dive towns: {int(deep_review['city_key'].nunique())}")
        doc.append(f"\\item Deep-dive verdict counts: pass={deep_pass_n}, partial={deep_partial_n}, fail={deep_fail_n}")
        doc.append("\\end{itemize}")
        doc.append("\\begin{table}[H]\\centering\\caption{Deep-dive town summary}\\input{tables/deep_dive_town_summary.tex}\\end{table}")

        dd = deep_review.copy()
        dd["selection_score"] = pd.to_numeric(dd.get("selection_score"), errors="coerce")
        dd = dd.sort_values(["selection_score", "city_display"], ascending=[False, True]).reset_index(drop=True)
        for _, r in dd.iterrows():
            city_display = _tex_escape(r.get("city_display", r.get("city_key", "")))
            focus_claim = _tex_escape(r.get("focus_claim", ""))
            verdict = _tex_escape(r.get("manual_verdict", ""))
            note = _tex_escape(r.get("manual_note", ""))

            e_id = _tex_escape(r.get("early_issue_id", ""))
            e_date = _tex_escape(r.get("early_issue_date", ""))
            e_lbl = _tex_escape(r.get("early_classification_label", ""))
            l_id = _tex_escape(r.get("late_issue_id", ""))
            l_date = _tex_escape(r.get("late_issue_date", ""))
            l_lbl = _tex_escape(r.get("late_classification_label", ""))

            e_text = _tex_escape(_short_text(r.get("early_excerpt", ""), 2400))
            l_text = _tex_escape(_short_text(r.get("late_excerpt", ""), 2400))

            doc.append(f"\\subsubsection{{{city_display}}}")
            doc.append(f"\\textbf{{Claim focus:}} {focus_claim}")
            doc.append(f"\\textbf{{Early anchor:}} \\texttt{{{e_id}}} ({e_date}; {e_lbl})")
            doc.append("\\begin{quote}\\small")
            doc.append(e_text)
            doc.append("\\end{quote}")
            doc.append(f"\\textbf{{Late anchor:}} \\texttt{{{l_id}}} ({l_date}; {l_lbl})")
            doc.append("\\begin{quote}\\small")
            doc.append(l_text)
            doc.append("\\end{quote}")
            doc.append(f"\\textbf{{Manual verdict:}} {verdict}")
            if note:
                doc.append(f"\\textbf{{Manual note:}} {note}")

    doc.append("\\appendix")
    doc.append("\\section{Transcript excerpts used in manual checks (illustrative)}")
    doc.append("\\begin{table}[H]\\centering\\caption{Manual-check excerpts (head)}\\input{tables/manual_validation_excerpts.tex}\\end{table}")

    doc.append("\\section{Reproducibility}")
    doc.append(f"Run directory: \\path{{{_tex_escape(str(run_dir))}}}.")
    doc.append(f"Longitudinal directory: \\path{{{_tex_escape(str(lon_dir))}}}.")
    doc.append(f"V2 metrics directory: \\path{{{_tex_escape(str(v2_dir))}}}.")
    doc.append("\\end{document}")

    tex_path = out_dir / "report.tex"
    tex_path.write_text("\n".join(doc) + "\n", encoding="utf-8")

    makefile = [
        "LATEXMK ?= latexmk",
        "",
        "all: report.pdf",
        "",
        "report.pdf: report.tex",
        "\t$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error report.tex",
        "",
        "clean:",
        "\t$(LATEXMK) -C",
        "",
        ".PHONY: all clean",
    ]
    (out_dir / "Makefile").write_text("\n".join(makefile) + "\n", encoding="utf-8")

    compiled_pdf = False
    if bool(args.compile_pdf):
        latexmk = shutil.which("latexmk")
        if latexmk:
            try:
                subprocess.run(
                    [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                    cwd=str(out_dir),
                    check=True,
                )
                compiled_pdf = True
            except Exception:
                compiled_pdf = False

    provenance = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "longitudinal_dir": str(lon_dir),
        "v2_dir": str(v2_dir),
        "output_dir": str(out_dir),
        "compiled_pdf": bool(compiled_pdf),
        "input_presence": {
            "event_lags": not event_lags.empty,
            "phase_sequences": not seq.empty,
            "category_shift": not cat_shift.empty,
            "amendment_balance": not amend_balance.empty,
            "insight_indices": not insight.empty,
            "manual_reviewed": not reviewed.empty,
            "deep_dive_reviewed": not deep_review.empty,
            "comparison_present": bool(cmp_overall),
        },
        "manual_validation_counts": {
            "pass": pass_n,
            "partial": partial_n,
            "fail": fail_n,
        },
        "deep_dive_validation_counts": {
            "pass": deep_pass_n,
            "partial": deep_partial_n,
            "fail": deep_fail_n,
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"report_tex={tex_path} "
        f"report_pdf={(out_dir / 'report.pdf')} "
        f"compiled_pdf={compiled_pdf}"
    )


if __name__ == "__main__":
    main()
