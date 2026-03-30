#!/usr/bin/env python3
"""
Build a focused LaTeX writeup for the panel-level longitudinal pilot.

Primary narrative focuses on:
  - when zoning first appears,
  - when full ordinance appears,
  - how category and complexity content evolves,
  - how amendment waves change topics.

Main outputs:
  - <output-dir>/report.tex
  - <output-dir>/report.pdf (optional)
  - <output-dir>/figures/*.png
  - <output-dir>/tables/*.tex
  - <output-dir>/provenance.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_FIXED_CATEGORY_KEYS = [
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

_PHASE_LABEL_MAP = {
    "initial_establishment": "initial establishment",
    "formalization_enforcement": "formalization & enforcement",
    "expansion_reclassification": "expansion & reclassification",
    "flexible_special_use": "flexible/special use",
    "infrastructure_environment": "infrastructure/environment",
    "maturation_implementation": "maturation & implementation",
    "other_unclear": "other/unclear",
}


def _phase_group_from_label(lbl: str) -> str:
    s = str(lbl or "").strip().lower()
    if not s:
        return "other/unclear"
    if s in _PHASE_LABEL_MAP:
        return _PHASE_LABEL_MAP[s]
    if s == "(blank)":
        return "other/unclear"
    if re.search(r"\b(introduction|initial|emergence|foundational|incipient|establish|adopt)\b", s):
        return "initial establishment"
    if re.search(r"\b(formal|enforc|governance|code development|codif)\b", s):
        return "formalization & enforcement"
    if re.search(r"\b(expansion|broaden|rezon|reclass|subdivision)\b", s):
        return "expansion & reclassification"
    if re.search(r"\b(flexible|conditional|specific use|specialized|special use)\b", s):
        return "flexible/special use"
    if re.search(r"\b(environment|infrastructure|drain|stormwater|habitat)\b", s):
        return "infrastructure/environment"
    if re.search(r"\b(advanced|mature|application|sustained)\b", s):
        return "maturation & implementation"
    return "other/unclear"


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


def _write_table_tex(path: Path, df: pd.DataFrame, cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("\\begin{tabular}{l}\n\\toprule\nNo data\\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return

    keep = [c for c in cols if c in df.columns]
    if not keep:
        keep = list(df.columns)
    sub = df[keep].copy()

    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")

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


def _to_year(s: Any) -> float:
    txt = str(s or "").strip()
    if not txt:
        return math.nan
    try:
        return float(pd.to_datetime(txt, errors="coerce").year)
    except Exception:
        return math.nan


def _plot_first_event_years(first_df: pd.DataFrame, out_path: Path) -> None:
    if first_df.empty:
        return

    d = first_df.copy()
    d["first_zoning_year"] = d["first_zoning_issue_date"].map(_to_year)
    d["first_full_year"] = d["first_full_issue_date"].map(_to_year)
    d = d[(d["first_zoning_year"].notna()) | (d["first_full_year"].notna())].copy()
    if d.empty:
        return

    z = d["first_zoning_year"].dropna().astype(int)
    f = d["first_full_year"].dropna().astype(int)
    years = sorted(set(z.tolist() + f.tolist()))
    if not years:
        return

    zc = pd.Series(z).value_counts().reindex(years, fill_value=0)
    fc = pd.Series(f).value_counts().reindex(years, fill_value=0)

    x = np.arange(len(years))
    w = 0.42
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(x - w / 2, zc.values, width=w, color="#4e79a7", label="First zoning signal")
    ax.bar(x + w / 2, fc.values, width=w, color="#f28e2b", label="First full ordinance")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_ylabel("Cities")
    ax.set_xlabel("Year")
    ax.set_title("Distribution of first-zoning and first-full-ordinance years")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_category_trajectory(cat_df: pd.DataFrame, out_path: Path) -> None:
    if cat_df.empty:
        return

    d = cat_df.copy()
    for c in ("early_share", "mid_share", "late_share"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["category"].isin(_FIXED_CATEGORY_KEYS)]
    d = d.dropna(subset=["early_share", "mid_share", "late_share"])
    if d.empty:
        return

    g = (
        d.groupby("category", dropna=False)
        .agg(
            early_share=("early_share", "mean"),
            mid_share=("mid_share", "mean"),
            late_share=("late_share", "mean"),
        )
        .reset_index()
    )
    g["avg"] = g[["early_share", "mid_share", "late_share"]].mean(axis=1)
    g = g.sort_values("avg", ascending=False).reset_index(drop=True)

    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.plot(x, g["early_share"], marker="o", color="#4e79a7", label="Early")
    ax.plot(x, g["mid_share"], marker="o", color="#59a14f", label="Mid")
    ax.plot(x, g["late_share"], marker="o", color="#e15759", label="Late")
    ax.set_xticks(x)
    ax.set_xticklabels(g["category"].astype(str).tolist(), rotation=40, ha="right")
    ax.set_ylabel("Mean category share")
    ax.set_title("Category trajectory profiles (early vs mid vs late)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_phase_counts(phase_df: pd.DataFrame, out_path: Path) -> None:
    if phase_df.empty:
        return

    d = phase_df.copy()
    d["phase_label"] = d["phase_label"].astype(str).str.strip().str.lower()
    d.loc[d["phase_label"].eq(""), "phase_label"] = "(blank)"
    d["phase_group"] = d["phase_label"].map(_phase_group_from_label)
    counts = d["phase_group"].value_counts().head(12)
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(9.8, 4.7))
    ax.barh(counts.index.tolist()[::-1], counts.values[::-1], color="#76b7b2")
    ax.set_xlabel("Count")
    ax.set_ylabel("Phase archetype group")
    ax.set_title("Evolution phase archetype counts (canonicalized)")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_wave_counts_by_group(wave_df: pd.DataFrame, out_path: Path) -> None:
    if wave_df.empty:
        return

    d = wave_df.copy()
    d["change_type"] = d["change_type"].astype(str).str.strip().str.lower()
    d["region"] = d["region"].astype(str)
    d["urbanicity_proxy"] = d["urbanicity_proxy"].astype(str)

    by_region = (
        d.groupby(["region", "change_type"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    by_urb = (
        d.groupby(["urbanicity_proxy", "change_type"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    if by_region.empty and by_urb.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))

    if not by_region.empty:
        by_region.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab20")
        axes[0].set_title("Amendment wave types by region")
        axes[0].set_xlabel("Region")
        axes[0].set_ylabel("Wave count")
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].grid(axis="y", alpha=0.25, linestyle="--")
        axes[0].legend(frameon=False, fontsize=8)
    else:
        axes[0].axis("off")

    if not by_urb.empty:
        by_urb.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20")
        axes[1].set_title("Amendment wave types by urbanicity")
        axes[1].set_xlabel("Urbanicity")
        axes[1].set_ylabel("Wave count")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].grid(axis="y", alpha=0.25, linestyle="--")
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_extraction_diagnostics(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return

    d = summary_df.copy()
    cols = [
        "parse_valid",
        "schema_valid",
        "first_signal_issue_valid",
        "first_full_issue_valid",
        "phase_evidence_valid",
        "wave_evidence_valid",
    ]
    vals: list[float] = []
    labels: list[str] = []
    for c in cols:
        if c in d.columns:
            v = pd.to_numeric(d[c], errors="coerce")
            if v.notna().any():
                labels.append(c)
                vals.append(float(v.mean()))

    overflow_share = float(pd.to_numeric(d.get("overflow_applied"), errors="coerce").mean()) if "overflow_applied" in d.columns else math.nan
    clipped_rate = math.nan
    if "clipped_issue_count" in d.columns and "selected_issue_count" in d.columns:
        den = pd.to_numeric(d["selected_issue_count"], errors="coerce")
        num = pd.to_numeric(d["clipped_issue_count"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = num / den
        clipped_rate = float(frac.replace([np.inf, -np.inf], np.nan).mean())

    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    x = np.arange(len(labels))
    axes[0].bar(x, vals, color="#4e79a7")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of cities")
    axes[0].set_title("Parse and evidence validity rates")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    diag_labels = ["overflow_applied_share", "clipped_issue_share"]
    diag_vals = [overflow_share, clipped_rate]
    axes[1].bar(np.arange(2), diag_vals, color="#f28e2b")
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(diag_labels, rotation=20, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Share")
    axes[1].set_title("Overflow fallback diagnostics")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build longitudinal pilot LaTeX report.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60",
        help="Base run directory (contains longitudinal output dir and reports/).",
    )
    ap.add_argument(
        "--input-dir",
        default="",
        help="Input longitudinal directory (default: <run-dir>/longitudinal).",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output report directory (default: <run-dir>/reports/longitudinal_pilot20).",
    )
    ap.add_argument(
        "--title",
        default="Panel-Level Longitudinal Pilot: Zoning Emergence and Evolution",
        help="LaTeX report title.",
    )
    ap.add_argument(
        "--compile-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile report.tex to report.pdf using latexmk when available.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    in_dir = Path(args.input_dir).expanduser().resolve() if str(args.input_dir).strip() else (run_dir / "longitudinal")
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "reports" / "longitudinal_pilot20")

    figs_dir = out_dir / "figures"
    tbls_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tbls_dir.mkdir(parents=True, exist_ok=True)

    selection = _read_csv(in_dir / "pilot20_selection_quality.csv")
    summary = _read_csv(in_dir / "city_longitudinal_summary.csv")
    first = _read_csv(in_dir / "city_first_zoning_events.csv")
    phases = _read_csv(in_dir / "city_evolution_phases.csv")
    waves = _read_csv(in_dir / "city_amendment_waves.csv")
    cats = _read_csv(in_dir / "city_category_trajectory.csv")
    cpx = _read_csv(in_dir / "city_complexity_trajectory.csv")
    nums = _read_csv(in_dir / "city_numeric_trajectory.csv")

    # Figures requested in main narrative.
    fig_first = figs_dir / "first_event_year_distributions.png"
    fig_cat = figs_dir / "category_trajectory_profiles.png"
    fig_phase = figs_dir / "evolution_phase_archetypes.png"
    fig_wave = figs_dir / "amendment_wave_types_by_region_urbanicity.png"
    fig_diag = figs_dir / "extraction_diagnostics.png"

    _plot_first_event_years(first, fig_first)
    _plot_category_trajectory(cats, fig_cat)
    _plot_phase_counts(phases, fig_phase)
    _plot_wave_counts_by_group(waves, fig_wave)
    _plot_extraction_diagnostics(summary, fig_diag)

    # Tables for report.
    selected_only = selection[selection.get("selected_for_pilot", 0) == 1].copy() if not selection.empty else pd.DataFrame()
    selected_only = selected_only.sort_values("rank_quality") if "rank_quality" in selected_only.columns else selected_only

    first_table = first.copy()
    if not first_table.empty:
        first_table = first_table[
            [
                "city_key",
                "region",
                "urbanicity_proxy",
                "first_zoning_issue_date",
                "first_zoning_signal_type",
                "first_full_issue_date",
                "first_full_status",
            ]
        ].sort_values(["first_zoning_issue_date", "city_key"], na_position="last")

    phase_counts = pd.DataFrame()
    if not phases.empty:
        d = phases.copy()
        d["phase_label"] = d["phase_label"].astype(str).str.strip().str.lower()
        d.loc[d["phase_label"].eq(""), "phase_label"] = "(blank)"
        d["phase_group"] = d["phase_label"].map(_phase_group_from_label)
        phase_counts = (
            d.groupby(["phase_group"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(20)
        )

    wave_counts = pd.DataFrame()
    if not waves.empty:
        wave_counts = (
            waves.groupby(["region", "urbanicity_proxy", "change_type"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "region", "urbanicity_proxy"], ascending=[False, True, True])
        )

    ambiguous_flags = pd.DataFrame()
    if not summary.empty:
        d = summary.copy()
        for c in (
            "parse_valid",
            "schema_valid",
            "first_signal_issue_valid",
            "first_full_issue_valid",
            "phase_evidence_valid",
            "wave_evidence_valid",
        ):
            d[c] = pd.to_numeric(d.get(c), errors="coerce")
        amb_mask = (
            (d["parse_valid"] != 1)
            | (d["schema_valid"] != 1)
            | (d["first_signal_issue_valid"] != 1)
            | (d["first_full_issue_valid"] != 1)
            | (d["phase_evidence_valid"] != 1)
            | (d["wave_evidence_valid"] != 1)
            | (d.get("first_full_status", "").astype(str) == "uncertain")
        )
        ambiguous_flags = d.loc[
            amb_mask,
            [
                "city_key",
                "parse_valid",
                "schema_valid",
                "first_signal_issue_valid",
                "first_full_issue_valid",
                "phase_evidence_valid",
                "wave_evidence_valid",
                "first_zoning_signal_type",
                "first_full_status",
                "missing_top_level_keys",
                "missing_category_keys",
            ],
        ].copy()

    cpx_counts = pd.DataFrame()
    if not cpx.empty:
        cpx_counts = pd.DataFrame(
            {
                "metric": [
                    "regulatory_complexity_trend",
                    "procedural_intensity_trend",
                    "dimensional_specificity_trend",
                ],
                "up_share": [
                    float((cpx["regulatory_complexity_trend"].astype(str) == "up").mean()),
                    float((cpx["procedural_intensity_trend"].astype(str) == "up").mean()),
                    float((cpx["dimensional_specificity_trend"].astype(str) == "up").mean()),
                ],
                "down_share": [
                    float((cpx["regulatory_complexity_trend"].astype(str) == "down").mean()),
                    float((cpx["procedural_intensity_trend"].astype(str) == "down").mean()),
                    float((cpx["dimensional_specificity_trend"].astype(str) == "down").mean()),
                ],
                "flat_share": [
                    float((cpx["regulatory_complexity_trend"].astype(str) == "flat").mean()),
                    float((cpx["procedural_intensity_trend"].astype(str) == "flat").mean()),
                    float((cpx["dimensional_specificity_trend"].astype(str) == "flat").mean()),
                ],
            }
        )

    numeric_counts = pd.DataFrame()
    if not nums.empty:
        numeric_counts = (
            nums.groupby(["metric", "direction"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["metric", "count"], ascending=[True, False])
        )

    _write_table_tex(
        tbls_dir / "selection_top20_quality.tex",
        selected_only,
        [
            "rank_quality",
            "city_key",
            "region",
            "urbanicity_proxy",
            "selected_issue_count",
            "parsed_issue_count",
            "parse_success_rate",
            "post_filter_noise_share",
            "lexical_noise_share",
        ],
    )
    _write_table_tex(
        tbls_dir / "first_zoning_events.tex",
        first_table.head(25),
        [
            "city_key",
            "region",
            "urbanicity_proxy",
            "first_zoning_issue_date",
            "first_zoning_signal_type",
            "first_full_issue_date",
            "first_full_status",
        ],
    )
    _write_table_tex(
        tbls_dir / "phase_archetype_counts.tex",
        phase_counts,
        ["phase_group", "count"],
    )
    _write_table_tex(
        tbls_dir / "wave_counts_by_region_urbanicity.tex",
        wave_counts.head(40),
        ["region", "urbanicity_proxy", "change_type", "count"],
    )
    _write_table_tex(
        tbls_dir / "complexity_trend_distribution.tex",
        cpx_counts,
        ["metric", "up_share", "down_share", "flat_share"],
    )
    _write_table_tex(
        tbls_dir / "numeric_direction_counts.tex",
        numeric_counts,
        ["metric", "direction", "count"],
    )
    _write_table_tex(
        tbls_dir / "ambiguous_city_flags.tex",
        ambiguous_flags,
        [
            "city_key",
            "parse_valid",
            "schema_valid",
            "first_signal_issue_valid",
            "first_full_issue_valid",
            "phase_evidence_valid",
            "wave_evidence_valid",
            "first_zoning_signal_type",
            "first_full_status",
        ],
    )

    # Narrative metrics.
    n_city = int(summary["city_key"].nunique()) if not summary.empty else 0
    parse_share = float(pd.to_numeric(summary.get("parse_valid"), errors="coerce").mean()) if not summary.empty else math.nan
    schema_share = float(pd.to_numeric(summary.get("schema_valid"), errors="coerce").mean()) if not summary.empty else math.nan
    phase_nonempty_share = float((pd.to_numeric(summary.get("n_evolution_phases"), errors="coerce") > 0).mean()) if not summary.empty else math.nan
    wave_nonempty_share = float((pd.to_numeric(summary.get("n_amendment_waves"), errors="coerce") > 0).mean()) if not summary.empty else math.nan
    overflow_share = float(pd.to_numeric(summary.get("overflow_applied"), errors="coerce").mean()) if not summary.empty else math.nan

    first_zoning_year_med = math.nan
    first_full_year_med = math.nan
    if not first.empty:
        zy = first["first_zoning_issue_date"].map(_to_year)
        fy = first["first_full_issue_date"].map(_to_year)
        first_zoning_year_med = float(zy.dropna().median()) if zy.notna().any() else math.nan
        first_full_year_med = float(fy.dropna().median()) if fy.notna().any() else math.nan

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
    doc.append("\\author{Automated longitudinal pilot pipeline}")
    doc.append(f"\\date{{{_tex_escape(report_date)}}}")
    doc.append("\\begin{document}")
    doc.append("\\maketitle")
    doc.append("\\tableofcontents")
    doc.append("\\clearpage")

    doc.append("\\section{Pilot objective and scope}")
    doc.append(
        "This pilot estimates zoning emergence and evolution from the newspaper panel data as currently available. "
        "It uses whole-panel longitudinal prompting with deterministic overflow fallback and does not use covariates "
        "or causal adoption models in the main analysis narrative."
    )

    doc.append("\\section{Panel selection and data quality}")
    doc.append(
        f"The pilot includes {n_city} city panels selected by parse-completeness and contamination quality ranking "
        "from the 60-panel run."
    )
    doc.append("\\begin{table}[H]\\centering\\caption{Top-20 quality-selected panels}\\input{tables/selection_top20_quality.tex}\\end{table}")

    doc.append("\\section{When zoning first appears}")
    doc.append(
        "The first event layer separates first zoning signal from first full ordinance and stores confidence and evidence. "
        "This avoids conflating early mentions/amendments with full code adoption text."
    )
    if fig_first.is_file():
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.92\\textwidth]{figures/first_event_year_distributions.png}"
            "\\caption{First-zoning and first-full-ordinance year distributions.}\\end{figure}"
        )
    doc.append("\\begin{table}[H]\\centering\\caption{Illustrative first-event rows}\\input{tables/first_zoning_events.tex}\\end{table}")

    doc.append("\\section{How content evolves over time}")
    doc.append(
        "Category trajectories summarize early/mid/late shares for a fixed 12-category zoning taxonomy; "
        "complexity trajectories summarize directional shifts in regulatory complexity, procedural intensity, and dimensional specificity."
    )
    if fig_cat.is_file():
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\textwidth]{figures/category_trajectory_profiles.png}"
            "\\caption{Mean early/mid/late category shares across pilot cities.}\\end{figure}"
        )
    doc.append("\\begin{table}[H]\\centering\\caption{Complexity trend direction shares}\\input{tables/complexity_trend_distribution.tex}\\end{table}")
    doc.append("\\begin{table}[H]\\centering\\caption{Numeric trajectory direction counts}\\input{tables/numeric_direction_counts.tex}\\end{table}")

    doc.append("\\section{Amendment waves and phase archetypes}")
    doc.append(
        "The longitudinal extraction identifies evolution phases and amendment waves directly from each full city panel, "
        "preserving cross-issue context in a single pass."
    )
    if fig_phase.is_file():
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.86\\textwidth]{figures/evolution_phase_archetypes.png}"
            "\\caption{Most common evolution phase archetypes.}\\end{figure}"
        )
    if fig_wave.is_file():
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.98\\textwidth]{figures/amendment_wave_types_by_region_urbanicity.png}"
            "\\caption{Amendment wave type counts by region and urbanicity.}\\end{figure}"
        )
    doc.append("\\begin{table}[H]\\centering\\caption{Phase archetype counts}\\input{tables/phase_archetype_counts.tex}\\end{table}")
    doc.append("\\begin{table}[H]\\centering\\caption{Wave counts by region and urbanicity (head)}\\input{tables/wave_counts_by_region_urbanicity.tex}\\end{table}")

    doc.append("\\section{Key pilot diagnostics}")
    doc.append("\\begin{itemize}")
    doc.append(f"\\item Parsed JSON share: {parse_share:.3f}" if pd.notna(parse_share) else "\\item Parsed JSON share: NA")
    doc.append(f"\\item Schema-valid share: {schema_share:.3f}" if pd.notna(schema_share) else "\\item Schema-valid share: NA")
    doc.append(
        f"\\item Cities with non-empty evolution phases: {phase_nonempty_share:.3f}"
        if pd.notna(phase_nonempty_share)
        else "\\item Cities with non-empty evolution phases: NA"
    )
    doc.append(
        f"\\item Cities with non-empty amendment waves: {wave_nonempty_share:.3f}"
        if pd.notna(wave_nonempty_share)
        else "\\item Cities with non-empty amendment waves: NA"
    )
    doc.append(f"\\item Overflow fallback applied share: {overflow_share:.3f}" if pd.notna(overflow_share) else "\\item Overflow fallback applied share: NA")
    doc.append(
        f"\\item Median first-zoning year: {first_zoning_year_med:.0f}"
        if pd.notna(first_zoning_year_med)
        else "\\item Median first-zoning year: NA"
    )
    doc.append(
        f"\\item Median first-full-ordinance year: {first_full_year_med:.0f}"
        if pd.notna(first_full_year_med)
        else "\\item Median first-full-ordinance year: NA"
    )
    doc.append("\\end{itemize}")

    doc.append("\\appendix")
    doc.append("\\section{Extraction Diagnostics}")
    doc.append(
        "This appendix reports parse/evidence validity and overflow fallback diagnostics, plus cities flagged as ambiguous "
        "for additional review."
    )
    if fig_diag.is_file():
        doc.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\textwidth]{figures/extraction_diagnostics.png}"
            "\\caption{Parse/evidence validity and overflow diagnostics.}\\end{figure}"
        )
    doc.append("\\begin{table}[H]\\centering\\caption{Ambiguous city flags}\\input{tables/ambiguous_city_flags.tex}\\end{table}")

    doc.append("\\section{Reproducibility}")
    doc.append(f"Run directory: \\path{{{_tex_escape(str(run_dir))}}}.")
    doc.append(f"Longitudinal input directory: \\path{{{_tex_escape(str(in_dir))}}}.")
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

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "compiled_pdf": bool(compiled_pdf),
        "inputs_present": {
            "selection_quality": not selection.empty,
            "city_longitudinal_summary": not summary.empty,
            "city_first_zoning_events": not first.empty,
            "city_evolution_phases": not phases.empty,
            "city_amendment_waves": not waves.empty,
            "city_category_trajectory": not cats.empty,
            "city_complexity_trajectory": not cpx.empty,
            "city_numeric_trajectory": not nums.empty,
        },
        "key_metrics": {
            "n_city": int(n_city),
            "parse_valid_share": float(parse_share) if pd.notna(parse_share) else None,
            "schema_valid_share": float(schema_share) if pd.notna(schema_share) else None,
            "phase_nonempty_share": float(phase_nonempty_share) if pd.notna(phase_nonempty_share) else None,
            "wave_nonempty_share": float(wave_nonempty_share) if pd.notna(wave_nonempty_share) else None,
            "overflow_share": float(overflow_share) if pd.notna(overflow_share) else None,
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"report_tex={tex_path} "
        f"report_pdf={(out_dir / 'report.pdf')} "
        f"compiled_pdf={compiled_pdf}"
    )


if __name__ == "__main__":
    main()
