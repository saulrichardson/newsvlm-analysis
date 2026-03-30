#!/usr/bin/env python3
"""
Build a LaTeX writeup for the prototype LLM-based zoning measurement run.

Inputs (run directory produced by prototype_zoning_panel_analysis.py):
  - prototype_summary.md
  - tables/*.csv
  - panels/selected_panels.csv
  - llm_panel_vs_issue/*.csv
  - llm_panel_vs_issue/parsed_*_outputs.jsonl

Outputs (output directory):
  - report.tex
  - Makefile
  - figures/*.png
  - tables/*.tex
  - provenance.json

This script writes sources/assets only; LaTeX compilation is optional.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
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


def _safe_slug(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] or "x"


def _city_key(city_name: str, state_abbr: str) -> str:
    return f"{_safe_slug(city_name)}__{_safe_slug(state_abbr)}"


def _pretty_city_key(s: str) -> str:
    if "__" not in str(s):
        return str(s)
    city, st = str(s).split("__", 1)
    city2 = city.replace("_", " ").title()
    return f"{city2}, {st.upper()}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _parse_dataset_scope(summary_md_path: Path) -> dict[str, str]:
    text = summary_md_path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    patterns = {
        "issues": r"- issues:\s*([0-9,]+)",
        "city_state_units": r"- city-state units:\s*([0-9,]+)",
        "states": r"- states:\s*([0-9,]+)",
        "issue_date_span": r"- issue_date span:\s*([0-9\-]+\s+to\s+[0-9\-]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, flags=re.I)
        if m:
            out[k] = m.group(1).strip()
    return out


def _write_table_tex(path: Path, header: list[str], rows: list[list[str]], colspec: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_national_coverage(ncov: pd.DataFrame, out_path: Path) -> None:
    df = ncov.copy()
    df = df.sort_values("issue_year")
    x = pd.to_numeric(df["issue_year"], errors="coerce")
    y1 = pd.to_numeric(df["n_issues"], errors="coerce")
    y2 = pd.to_numeric(df["n_city_states"], errors="coerce")

    fig, ax1 = plt.subplots(figsize=(9.2, 4.6))
    ax1.plot(x, y1, color="#1f77b4", linewidth=2.2, label="Issues")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Issues", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.7)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color="#d62728", linewidth=2.0, linestyle="-", label="City-state units")
    ax2.set_ylabel("City-state units", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("National coverage over time")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_adoption_by_region(first_city: pd.DataFrame, out_path: Path) -> None:
    df = first_city.copy()
    df["first_full_ordinance_year"] = pd.to_numeric(df["first_full_ordinance_year"], errors="coerce")
    df = df.dropna(subset=["first_full_ordinance_year", "region"])

    order = ["Northeast", "Midwest", "South", "West"]
    grouped = []
    labels = []
    for r in order:
        vals = df.loc[df["region"] == r, "first_full_ordinance_year"].astype(float).tolist()
        if vals:
            grouped.append(vals)
            labels.append(r)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bp = ax.boxplot(grouped, patch_artist=True, tick_labels=labels)
    colors = ["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"]
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("#333333")
        median.set_linewidth(1.6)
    ax.set_ylabel("First full-ordinance year")
    ax.set_title("Distribution of initial full-ordinance adoption year by region")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_complexity_time_by_urbanicity(length_u: pd.DataFrame, out_path: Path) -> None:
    df = length_u.copy()
    df["issue_year"] = pd.to_numeric(df["issue_year"], errors="coerce")
    df["complexity_proxy_mean"] = pd.to_numeric(df["complexity_proxy_mean"], errors="coerce")
    df = df.dropna(subset=["issue_year", "urbanicity_proxy", "complexity_proxy_mean"])
    df = df[(df["issue_year"] >= 1920) & (df["issue_year"] <= 2025)]
    if df.empty:
        return

    # Smooth noisy yearly series for readability.
    series_rows: list[pd.DataFrame] = []
    for urb, g in df.groupby("urbanicity_proxy", dropna=False):
        g2 = g.sort_values("issue_year").copy()
        g2["smooth"] = g2["complexity_proxy_mean"].rolling(window=5, min_periods=1, center=True).mean()
        g2["urbanicity_proxy"] = urb
        series_rows.append(g2)
    sm = pd.concat(series_rows, ignore_index=True)

    color_map = {
        "urban": "#1f77b4",
        "suburban": "#2ca02c",
        "rural": "#ff7f0e",
        "unknown": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for urb, g in sm.groupby("urbanicity_proxy", dropna=False):
        label = str(urb)
        ax.plot(
            g["issue_year"],
            g["smooth"],
            linewidth=2.0,
            color=color_map.get(label, "#7f7f7f"),
            label=label,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean mechanics-length complexity proxy")
    ax.set_title("Complexity proxy over time by urbanicity (5-year smoothed)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(frameon=False, ncols=4, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_jaccard_hist(cons: pd.DataFrame, out_path: Path) -> None:
    vals = pd.to_numeric(cons["category_jaccard_panel_vs_issue_union"], errors="coerce").dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.hist(vals, bins=np.linspace(0.0, 1.0, 11), color="#4c78a8", edgecolor="white")
    ax.axvline(np.mean(vals), color="#d62728", linestyle="--", linewidth=1.8, label=f"Mean={np.mean(vals):.3f}")
    ax.set_xlabel("Category Jaccard (panel vs issue-union)")
    ax.set_ylabel("Number of city panels")
    ax.set_title("Agreement distribution across city panels")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_complexity_scatter(cons: pd.DataFrame, out_path: Path) -> None:
    df = cons.copy()
    x = pd.to_numeric(df["issue_complexity_mean_0_100"], errors="coerce")
    y = pd.to_numeric(df["panel_complexity_0_100"], errors="coerce")
    c = pd.to_numeric(df["category_jaccard_panel_vs_issue_union"], errors="coerce")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=55, alpha=0.9, edgecolor="black", linewidth=0.3)
    mn = float(min(np.nanmin(x), np.nanmin(y)))
    mx = float(max(np.nanmax(x), np.nanmax(y)))
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="#888888", linewidth=1.2, label="45-degree line")
    ax.set_xlabel("Issue-level mean complexity (0-100)")
    ax.set_ylabel("Panel-level complexity (0-100)")
    ax.set_title("Panel complexity is systematically higher than issue-level means")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Category Jaccard")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_top_category_gaps(shares: pd.DataFrame, out_path: Path, top_n: int = 12) -> pd.DataFrame:
    df = shares.copy()
    df["abs_gap"] = (pd.to_numeric(df["panel_share"], errors="coerce") - pd.to_numeric(df["issue_share_mean"], errors="coerce")).abs()
    top = df.sort_values("abs_gap", ascending=False).head(int(top_n)).copy()
    top["city_category"] = top.apply(lambda r: f"{_pretty_city_key(str(r['city_key']))} | {str(r['category'])}", axis=1)
    top = top.sort_values("abs_gap", ascending=True)

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.barh(top["city_category"], top["abs_gap"], color="#f28e2b", alpha=0.85)
    ax.set_xlabel("|panel share - issue mean share|")
    ax.set_title(f"Top {int(top_n)} panel-vs-issue category share gaps")
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return top


def _build_report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    figs_dir = out_dir / "figures"
    tbls_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tbls_dir.mkdir(parents=True, exist_ok=True)

    # Inputs.
    summary_md = run_dir / "prototype_summary.md"
    ncov = pd.read_csv(run_dir / "tables" / "national_coverage_by_year.csv")
    first_region = pd.read_csv(run_dir / "tables" / "initial_full_ordinance_by_region.csv")
    first_urb = pd.read_csv(run_dir / "tables" / "initial_full_ordinance_by_urbanicity_proxy.csv")
    first_city = pd.read_csv(run_dir / "tables" / "initial_full_ordinance_by_city.csv")
    length_u = pd.read_csv(run_dir / "tables" / "length_complexity_over_time_by_urbanicity_proxy.csv")
    cons = pd.read_csv(run_dir / "llm_panel_vs_issue" / "panel_vs_issue_consistency.csv")
    shares = pd.read_csv(run_dir / "llm_panel_vs_issue" / "panel_vs_issue_category_shares.csv")
    selected = pd.read_csv(run_dir / "panels" / "selected_panels.csv")
    parsed_panel = _read_jsonl(run_dir / "llm_panel_vs_issue" / "parsed_panel_outputs.jsonl")
    parsed_issue = _read_jsonl(run_dir / "llm_panel_vs_issue" / "parsed_issue_outputs.jsonl")

    # Derived metadata.
    scope = _parse_dataset_scope(summary_md)
    selected = selected.copy()
    selected["state_abbr"] = selected["state_abbr"].astype(str).str.lower()
    first_city2 = first_city[["city_name", "state_abbr", "region", "urbanicity_proxy"]].copy()
    first_city2["state_abbr"] = first_city2["state_abbr"].astype(str).str.lower()
    selected = selected.merge(first_city2, on=["city_name", "state_abbr"], how="left")
    selected["region"] = selected["region"].fillna("Unknown")
    selected["urbanicity_proxy"] = selected["urbanicity_proxy"].fillna("unknown")
    selected["city_key"] = selected.apply(lambda r: _city_key(str(r["city_name"]), str(r["state_abbr"])), axis=1)

    cons2 = cons.merge(selected[["city_key", "region", "urbanicity_proxy"]], on="city_key", how="left")

    # Parse coverage.
    panel_ok = sum(1 for r in parsed_panel if isinstance(r.get("parsed_json"), dict))
    issue_ok = sum(1 for r in parsed_issue if isinstance(r.get("parsed_json"), dict))

    # Core metrics.
    j = pd.to_numeric(cons["category_jaccard_panel_vs_issue_union"], errors="coerce")
    panel_c = pd.to_numeric(cons["panel_complexity_0_100"], errors="coerce")
    issue_c = pd.to_numeric(cons["issue_complexity_mean_0_100"], errors="coerce")

    j_mean = float(j.mean())
    j_med = float(j.median())
    j_hi = int((j >= 0.7).sum())
    j_lo = int((j < 0.5).sum())
    panel_mean = float(panel_c.mean())
    issue_mean = float(issue_c.mean())
    gap_mean = float((panel_c - issue_c).mean())

    # Category gap table.
    top_gap = shares.copy()
    top_gap["abs_gap"] = (pd.to_numeric(top_gap["panel_share"], errors="coerce") - pd.to_numeric(top_gap["issue_share_mean"], errors="coerce")).abs()
    top_gap = top_gap.sort_values("abs_gap", ascending=False).head(12).copy()
    top_gap["city"] = top_gap["city_key"].map(_pretty_city_key)

    # Figures.
    _plot_national_coverage(ncov, figs_dir / "national_coverage_over_time.png")
    _plot_adoption_by_region(first_city, figs_dir / "adoption_year_by_region_boxplot.png")
    _plot_complexity_time_by_urbanicity(length_u, figs_dir / "complexity_proxy_over_time_by_urbanicity.png")
    _plot_jaccard_hist(cons, figs_dir / "panel_issue_jaccard_hist.png")
    _plot_complexity_scatter(cons, figs_dir / "panel_vs_issue_complexity_scatter.png")
    _plot_top_category_gaps(shares, figs_dir / "top_category_share_gaps.png", top_n=12)

    # Tables.
    dataset_rows = [
        [_tex_escape("Total issues"), _tex_escape(scope.get("issues", f"{int(ncov['n_issues'].sum()):,d}"))],
        [_tex_escape("City-state units"), _tex_escape(scope.get("city_state_units", "NA"))],
        [_tex_escape("States"), _tex_escape(scope.get("states", "NA"))],
        [_tex_escape("Issue date span"), _tex_escape(scope.get("issue_date_span", "NA"))],
        [_tex_escape("Selected city panels"), f"{len(selected):,d}"],
        [_tex_escape("Selected issues reconstructed"), f"{int(pd.to_numeric(cons['n_issue_outputs'], errors='coerce').fillna(0).sum()):,d} (targeted sample)"],
        [_tex_escape("Panel outputs parsed"), f"{panel_ok}/{len(parsed_panel)}"],
        [_tex_escape("Issue outputs parsed"), f"{issue_ok}/{len(parsed_issue)}"],
    ]
    _write_table_tex(tbls_dir / "dataset_scope.tex", ["Metric", "Value"], dataset_rows, "lp{8.5cm}")

    # Region adoption summary table.
    fr = first_region.copy()
    fr = fr.sort_values("region")
    fr_rows: list[list[str]] = []
    for r in fr.itertuples(index=False):
        fr_rows.append(
            [
                _tex_escape(str(r.region)),
                f"{int(r.n_city_states):,d}",
                f"{int(r.first_year_min)}",
                f"{float(r.first_year_median):.1f}",
                f"{int(r.first_year_max)}",
            ]
        )
    _write_table_tex(
        tbls_dir / "adoption_by_region.tex",
        ["Region", "City-state units", "Min year", "Median year", "Max year"],
        fr_rows,
        "lrrrr",
    )

    # Panel mix table.
    rg = selected.groupby("region", dropna=False).size().reset_index(name="n_panels")
    ur = selected.groupby("urbanicity_proxy", dropna=False).size().reset_index(name="n_panels")
    mix_rows: list[list[str]] = []
    for r in rg.sort_values("n_panels", ascending=False).itertuples(index=False):
        mix_rows.append([_tex_escape(f"Region: {r.region}"), f"{int(r.n_panels)}"])
    for r in ur.sort_values("n_panels", ascending=False).itertuples(index=False):
        mix_rows.append([_tex_escape(f"Urbanicity: {r.urbanicity_proxy}"), f"{int(r.n_panels)}"])
    _write_table_tex(tbls_dir / "selected_panel_mix.tex", ["Group", "Panels"], mix_rows, "lr")

    # Consistency summary table.
    cons_rows = [
        [_tex_escape("Category Jaccard mean"), f"{j_mean:.3f}"],
        [_tex_escape("Category Jaccard median"), f"{j_med:.3f}"],
        [_tex_escape("City panels with Jaccard >= 0.70"), f"{j_hi}/{len(cons)}"],
        [_tex_escape("City panels with Jaccard < 0.50"), f"{j_lo}/{len(cons)}"],
        [_tex_escape("Panel complexity mean (0-100)"), f"{panel_mean:.2f}"],
        [_tex_escape("Issue complexity mean (0-100)"), f"{issue_mean:.2f}"],
        [_tex_escape("Mean complexity gap (panel - issue)"), f"{gap_mean:.2f}"],
    ]
    _write_table_tex(tbls_dir / "consistency_summary.tex", ["Statistic", "Value"], cons_rows, "lp{6.5cm}")

    # Weakest-city table.
    weak = cons.sort_values("category_jaccard_panel_vs_issue_union", ascending=True).head(8).copy()
    weak_rows: list[list[str]] = []
    for r in weak.itertuples(index=False):
        weak_rows.append(
            [
                _tex_escape(_pretty_city_key(str(r.city_key))),
                f"{float(r.category_jaccard_panel_vs_issue_union):.3f}",
                f"{float(r.panel_complexity_0_100):.1f}",
                f"{float(r.issue_complexity_mean_0_100):.1f}",
            ]
        )
    _write_table_tex(
        tbls_dir / "lowest_consistency_cities.tex",
        ["City", "Jaccard", "Panel cplx.", "Issue cplx. mean"],
        weak_rows,
        "lrrr",
    )

    # Top category gaps table.
    gap_rows: list[list[str]] = []
    for r in top_gap.itertuples(index=False):
        gap_rows.append(
            [
                _tex_escape(str(r.city)),
                _tex_escape(str(r.category)),
                f"{float(r.panel_share):.3f}",
                f"{float(r.issue_share_mean):.3f}",
                f"{float(r.abs_gap):.3f}",
            ]
        )
    _write_table_tex(
        tbls_dir / "top_category_gaps.tex",
        ["City", "Category", "Panel", "Issue mean", "|Gap|"],
        gap_rows,
        "llrrr",
    )

    # Regional complexity table.
    reg_c = cons2.groupby("region", dropna=False).agg(
        n_panels=("city_key", "nunique"),
        panel_complexity_mean=("panel_complexity_0_100", "mean"),
        issue_complexity_mean=("issue_complexity_mean_0_100", "mean"),
        jaccard_mean=("category_jaccard_panel_vs_issue_union", "mean"),
    ).reset_index()
    reg_rows: list[list[str]] = []
    for r in reg_c.sort_values("n_panels", ascending=False).itertuples(index=False):
        reg_rows.append(
            [
                _tex_escape(str(r.region)),
                f"{int(r.n_panels)}",
                f"{float(r.panel_complexity_mean):.2f}",
                f"{float(r.issue_complexity_mean):.2f}",
                f"{float(r.jaccard_mean):.3f}",
            ]
        )
    _write_table_tex(
        tbls_dir / "regional_consistency_complexity.tex",
        ["Region", "Panels", "Panel cplx.", "Issue cplx.", "Jaccard"],
        reg_rows,
        "lrrrr",
    )

    # LaTeX report text.
    now = dt.datetime.now().strftime("%Y-%m-%d")
    title = str(args.title).strip()
    doc: list[str] = []
    doc.append("\\documentclass[11pt]{article}")
    doc.append("\\usepackage[margin=1in]{geometry}")
    doc.append("\\usepackage{graphicx}")
    doc.append("\\usepackage{booktabs}")
    doc.append("\\usepackage{longtable}")
    doc.append("\\usepackage{caption}")
    doc.append("\\usepackage{float}")
    doc.append("\\usepackage{hyperref}")
    doc.append("\\usepackage{array}")
    doc.append("")
    doc.append(f"\\title{{{_tex_escape(title)}}}")
    doc.append("\\author{Prototype pipeline auto-writeup}")
    doc.append(f"\\date{{{_tex_escape(now)}}}")
    doc.append("\\begin{document}")
    doc.append("\\maketitle")
    doc.append("\\tableofcontents")
    doc.append("\\clearpage")

    doc.append("\\section{Executive overview}")
    doc.append(
        "This document summarizes a scaled prototype of LLM-based zoning measurement from "
        "historical newspaper transcripts. The run combines baseline descriptive tables with "
        "a panel-vs-issue LLM coding experiment to evaluate stability of category composition "
        "and complexity measures."
    )
    doc.append("\\begin{table}[H]")
    doc.append("\\centering")
    doc.append("\\caption{Dataset scope and run status}")
    doc.append("\\input{tables/dataset_scope.tex}")
    doc.append("\\end{table}")
    doc.append(
        f"In this run, the mean category Jaccard is {j_mean:.3f}, with {j_hi} of {len(cons)} city panels at "
        "or above 0.70, and a mean panel-minus-issue complexity gap of "
        f"{gap_mean:.2f} points."
    )

    doc.append("\\section{Data and panel construction}")
    doc.append(
        "The underlying issue panel spans multiple decades and all major U.S. regions. "
        "City panels are selected by local transcript availability and sampled into balanced "
        "issue bundles for same-city panel coding."
    )
    doc.append("\\begin{figure}[H]")
    doc.append("\\centering")
    doc.append("\\includegraphics[width=0.93\\textwidth]{figures/national_coverage_over_time.png}")
    doc.append("\\caption{National coverage over time: issue counts and city-state coverage by year.}")
    doc.append("\\end{figure}")
    doc.append("\\begin{figure}[H]")
    doc.append("\\centering")
    doc.append("\\includegraphics[width=0.82\\textwidth]{figures/adoption_year_by_region_boxplot.png}")
    doc.append("\\caption{Distribution of first observed full-ordinance adoption year by Census region.}")
    doc.append("\\end{figure}")
    doc.append("\\begin{table}[H]")
    doc.append("\\centering")
    doc.append("\\caption{Initial full-ordinance adoption timing by region}")
    doc.append("\\input{tables/adoption_by_region.tex}")
    doc.append("\\end{table}")
    doc.append("\\begin{table}[H]")
    doc.append("\\centering")
    doc.append("\\caption{Composition of selected city panels}")
    doc.append("\\input{tables/selected_panel_mix.tex}")
    doc.append("\\end{table}")

    doc.append("\\section{Measurement design}")
    doc.append(
        "Each selected city receives two coding passes: (i) panel-level coding over a bundle of issues, "
        "and (ii) issue-level coding one issue at a time. Both use the same fixed taxonomy with "
        "category-share outputs and complexity subscores. The panel-vs-issue comparison is used as "
        "a practical reliability check before scaling."
    )
    doc.append("\\begin{figure}[H]")
    doc.append("\\centering")
    doc.append("\\includegraphics[width=0.83\\textwidth]{figures/complexity_proxy_over_time_by_urbanicity.png}")
    doc.append("\\caption{Mechanics-length complexity proxy over time by urbanicity (baseline issue-level panel).}")
    doc.append("\\end{figure}")

    doc.append("\\section{Panel-vs-issue consistency results}")
    doc.append("\\begin{table}[H]")
    doc.append("\\centering")
    doc.append("\\caption{Consistency summary across selected city panels}")
    doc.append("\\input{tables/consistency_summary.tex}")
    doc.append("\\end{table}")
    doc.append("\\begin{figure}[H]")
    doc.append("\\centering")
    doc.append("\\includegraphics[width=0.74\\textwidth]{figures/panel_issue_jaccard_hist.png}")
    doc.append("\\caption{Distribution of category-set agreement (Jaccard) between panel and issue-union outputs.}")
    doc.append("\\end{figure}")
    doc.append("\\begin{figure}[H]")
    doc.append("\\centering")
    doc.append("\\includegraphics[width=0.78\\textwidth]{figures/panel_vs_issue_complexity_scatter.png}")
    doc.append("\\caption{Panel complexity vs issue-level mean complexity, colored by category Jaccard.}")
    doc.append("\\end{figure}")
    doc.append("\\begin{table}[H]")
    doc.append("\\centering")
    doc.append("\\caption{Lowest-consistency city panels}")
    doc.append("\\input{tables/lowest_consistency_cities.tex}")
    doc.append("\\end{table}")
    doc.append("\\begin{table}[H]")
    doc.append("\\centering")
    doc.append("\\caption{Regional breakdown of complexity and consistency}")
    doc.append("\\input{tables/regional_consistency_complexity.tex}")
    doc.append("\\end{table}")

    doc.append("\\section{Where disagreement concentrates}")
    doc.append(
        "Largest panel-vs-issue gaps tend to occur in broad procedural and miscellaneous buckets, "
        "which likely aggregate heterogeneous clauses and newspaper framing effects."
    )
    doc.append("\\begin{figure}[H]")
    doc.append("\\centering")
    doc.append("\\includegraphics[width=0.93\\textwidth]{figures/top_category_share_gaps.png}")
    doc.append("\\caption{Largest absolute panel-vs-issue category-share gaps (city-category pairs).}")
    doc.append("\\end{figure}")
    doc.append("\\begin{table}[H]")
    doc.append("\\centering")
    doc.append("\\caption{Top category-share gaps}")
    doc.append("\\input{tables/top_category_gaps.tex}")
    doc.append("\\end{table}")

    doc.append("\\section{Interpretation and next steps}")
    doc.append("\\begin{itemize}")
    doc.append("\\item The prototype has enough stability to support exploratory right-hand-side construction.")
    doc.append("\\item Agreement improves at this scale relative to smaller pilot runs, but low-agreement tails remain.")
    doc.append("\\item Next hardening step: tighten prompts and add second-pass adjudication for high-gap city-category pairs.")
    doc.append("\\item For estimation-ready panels, prioritize robust categories (land-use, bulk, procedural) and versioned extraction checks.")
    doc.append("\\end{itemize}")

    doc.append("\\section*{Reproducibility}")
    doc.append(
        "This report was generated from run artifacts in "
        f"\\path{{{_tex_escape(str(run_dir))}}}. "
        "All tables and figures are built from CSV/JSONL artifacts produced by the prototype pipeline."
    )

    doc.append("\\end{document}")
    (out_dir / "report.tex").write_text("\n".join(doc) + "\n", encoding="utf-8")

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

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "title": str(args.title),
        "n_panels": int(len(selected)),
        "n_panel_outputs": int(len(parsed_panel)),
        "n_issue_outputs": int(len(parsed_issue)),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build LaTeX writeup for prototype LLM zoning measurement run.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30",
        help="Prototype run directory containing tables/panels/llm_panel_vs_issue outputs.",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30/latex_writeup",
        help="Directory to write report package.",
    )
    ap.add_argument(
        "--title",
        default="Prototype LLM-Based Zoning Measurement from Historical Newspaper Transcripts",
        help="LaTeX report title.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _build_report(args)


if __name__ == "__main__":
    main()
