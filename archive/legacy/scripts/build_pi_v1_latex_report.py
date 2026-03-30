#!/usr/bin/env python3
"""
Build PI-facing LaTeX writeup for defensible v1 zoning measurement package.

Outputs:
  - report.tex
  - report.pdf (optional compile)
  - tables/*.tex
  - figures/*.png
  - provenance.json
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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _tex_escape(s: str) -> str:
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


def _write_table_tex(path: Path, df: pd.DataFrame, *, caption_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("\\begin{tabular}{l}\n\\toprule\nNo data\\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return
    cols = [c for c in caption_cols if c in df.columns]
    if not cols:
        cols = list(df.columns[: min(5, len(df.columns))])
    sub = df[cols].copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    lines: list[str] = []
    lines.append("\\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(_tex_escape(c) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for r in sub.itertuples(index=False):
        lines.append(" & ".join(_tex_escape(v) for v in r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_national_coverage(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    x = pd.to_numeric(df["issue_year"], errors="coerce")
    y1 = pd.to_numeric(df.get("n_issues"), errors="coerce")
    y2 = pd.to_numeric(df.get("n_city_states"), errors="coerce")
    mask = x.notna() & y1.notna() & y2.notna()
    if not mask.any():
        return
    x = x[mask]
    y1 = y1[mask]
    y2 = y2[mask]
    fig, ax1 = plt.subplots(figsize=(9, 4.7))
    ax1.plot(x, y1, color="#1f77b4", lw=2.1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Issues", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.25, linestyle="--")
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color="#d62728", lw=2.1)
    ax2.set_ylabel("City-state units", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("National coverage over time")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_panel_mix(selected_panels: pd.DataFrame, out_path: Path) -> None:
    if selected_panels.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    reg = selected_panels["region"].astype(str).value_counts().sort_values(ascending=False)
    urb = selected_panels["urbanicity_proxy"].astype(str).value_counts().sort_values(ascending=False)
    axes[0].bar(reg.index.tolist(), reg.values, color="#4e79a7")
    axes[0].set_title("Panel count by region")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(urb.index.tolist(), urb.values, color="#f28e2b")
    axes[1].set_title("Panel count by urbanicity proxy")
    axes[1].tick_params(axis="x", rotation=35)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_jaccard(consistency: pd.DataFrame, out_path: Path) -> None:
    if consistency.empty:
        return
    x = pd.to_numeric(consistency["category_jaccard_panel_vs_issue_union"], errors="coerce").dropna()
    if x.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.hist(x.to_numpy(), bins=np.linspace(0.0, 1.0, 11), color="#59a14f", edgecolor="white")
    mu = float(x.mean())
    ax.axvline(mu, color="#e15759", linestyle="--", linewidth=1.6, label=f"Mean={mu:.3f}")
    ax.set_xlabel("Panel vs issue category Jaccard")
    ax.set_ylabel("Panels")
    ax.set_title("Consistency distribution")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_contamination(contam: pd.DataFrame, out_path: Path) -> None:
    if contam.empty:
        return
    g = (
        contam.groupby("classification_label", dropna=False)
        .agg(
            pre_filter_noise_share=("pre_filter_noise_share", "mean"),
            post_filter_noise_share=("post_filter_noise_share", "mean"),
        )
        .reset_index()
        .sort_values("classification_label")
    )
    if g.empty:
        return
    x = np.arange(len(g))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.bar(x - w / 2, g["pre_filter_noise_share"], width=w, label="Pre-filter", color="#e15759")
    ax.bar(x + w / 2, g["post_filter_noise_share"], width=w, label="Post-filter", color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels(g["classification_label"].astype(str).tolist(), rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Noise share")
    ax.set_title("Contamination diagnostics by label")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_purpose_similarity(pairs: pd.DataFrame, out_path: Path) -> None:
    if pairs.empty:
        return
    d = pairs.copy()
    d["section_embedding_cosine"] = pd.to_numeric(d.get("section_embedding_cosine"), errors="coerce")
    d["purpose_js_similarity"] = pd.to_numeric(d.get("purpose_js_similarity"), errors="coerce")
    d = d.dropna(subset=["section_embedding_cosine", "purpose_js_similarity"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(d["section_embedding_cosine"], d["purpose_js_similarity"], s=20, alpha=0.55, color="#9c755f", edgecolor="none")
    ax.set_xlabel("Purpose section embedding cosine")
    ax.set_ylabel("Purpose motive JS similarity")
    ax.set_title("Purpose similarity across municipalities")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_model_coeffs(adoption_table: pd.DataFrame, out_path: Path) -> None:
    if adoption_table.empty:
        return
    d = adoption_table.copy()
    chosen_model = ""
    if "model_name" in d.columns:
        preferred = [
            "adoption_hazard_logit_text_region_restricted",
            "adoption_hazard_logit_text_region_decade_restricted",
            "adoption_hazard_logit_baseline_region_decade",
        ]
        available = [m for m in preferred if m in d["model_name"].astype(str).unique().tolist()]
        if available:
            chosen_model = available[0]
        else:
            vc = d["model_name"].astype(str).value_counts()
            if not vc.empty:
                chosen_model = str(vc.index[0])
        if chosen_model:
            d = d[d["model_name"].astype(str) == chosen_model].copy()

    d["term"] = d["term"].astype(str)
    d["coef"] = pd.to_numeric(d["coef"], errors="coerce")
    d["ci_95_low"] = pd.to_numeric(d["ci_95_low"], errors="coerce")
    d["ci_95_high"] = pd.to_numeric(d["ci_95_high"], errors="coerce")
    d = d.dropna(subset=["coef", "ci_95_low", "ci_95_high"]).copy()
    d = d[d["term"] != "Intercept"].copy()
    # Prefer substantive terms over FE dummies in coefficient figure.
    d_sub = d[~d["term"].str.startswith("C(")].copy()
    if not d_sub.empty:
        d = d_sub
    if d.empty:
        return
    d["abs_coef"] = d["coef"].abs()
    d = d.sort_values("abs_coef", ascending=False).head(12).sort_values("coef")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(d))
    ax.errorbar(
        d["coef"].astype(float).to_numpy(),
        y,
        xerr=[
            (d["coef"] - d["ci_95_low"]).abs().astype(float).to_numpy(),
            (d["ci_95_high"] - d["coef"]).abs().astype(float).to_numpy(),
        ],
        fmt="o",
        color="#4e79a7",
        ecolor="#4e79a7",
        capsize=3,
    )
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(d["term"].astype(str).tolist())
    ax.set_xlabel("Logit coefficient")
    title = "Adoption model coefficients"
    if chosen_model:
        title += f" ({chosen_model})"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build PI v1 LaTeX report package.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30",
        help="Prototype run directory.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-dir>/reports/pi_v1_analysis).",
    )
    ap.add_argument(
        "--title",
        default="Defensible v1 Zoning Measurement: Full Ordinances + Amendments",
        help="Report title.",
    )
    ap.add_argument(
        "--compile-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile report.tex to report.pdf with latexmk when available.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "reports" / "pi_v1_analysis")
    figs = out_dir / "figures"
    tbls = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    tbls.mkdir(parents=True, exist_ok=True)

    # Core data.
    national = _read_csv(run_dir / "tables" / "national_coverage_by_year.csv")
    selected_panels = _read_csv(run_dir / "panels" / "selected_panels.csv")
    selection_bias = _read_csv(run_dir / "tables" / "panel_selection_coverage_bias.csv")
    selection_targets = _read_csv(run_dir / "tables" / "panel_selection_targets.csv")
    consistency = _read_csv(run_dir / "llm_panel_vs_issue" / "panel_vs_issue_consistency.csv")
    cat_shares = _read_csv(run_dir / "llm_panel_vs_issue" / "panel_vs_issue_category_shares.csv")
    open_rel = _read_csv(run_dir / "llm_panel_vs_issue" / "open_fixed_reliability_summary.csv")
    contamination = _read_csv(run_dir / "tables" / "contamination_panel_summary.csv")
    init_region = _read_csv(run_dir / "tables" / "initial_full_ordinance_by_region.csv")
    dist_label = _read_csv(run_dir / "tables" / "distribution_region_urbanicity_label.csv")
    numeric_ext = _read_csv(run_dir / "tables" / "numeric_constraints_extremes_by_region_urbanicity.csv")
    lot_ext = _read_csv(run_dir / "tables" / "single_family_lot_size_extremes.csv")

    # Section analysis.
    purpose_pairs = _read_csv(run_dir / "section_analysis" / "purpose_section_similarity_pairs.csv")
    purpose_rel = _read_csv(run_dir / "section_analysis" / "purpose_similarity_by_region_time_urbanicity.csv")
    purpose_examples = _read_csv(run_dir / "section_analysis" / "purpose_section_examples.csv")

    # Amendments.
    amend_delta = _read_csv(run_dir / "amendments" / "amendment_delta_category_summary.csv")
    amend_match = _read_csv(run_dir / "amendments" / "amendment_delta_matching_diagnostics.csv")
    amend_topics = _read_csv(run_dir / "amendments" / "amendment_topic_shares_by_region_urbanicity_year.csv")

    # Covariates + models.
    cov_missing = _read_csv(run_dir / "covariates" / "covariate_missingness_report.csv")
    place_xw = _read_csv(run_dir / "covariates" / "city_place_crosswalk.csv")
    county_xw = _read_csv(run_dir / "covariates" / "city_county_crosswalk.csv")
    adoption_table = _read_csv(run_dir / "models" / "adoption_model_table.csv")
    content_table = _read_csv(run_dir / "models" / "content_model_table.csv")
    adoption_option_grid = _read_csv(run_dir / "models" / "adoption_option_grid_ml_groupcv_estimable.csv")
    if adoption_option_grid.empty:
        adoption_option_grid = _read_csv(run_dir / "models" / "adoption_model_option_diagnostics.csv")
    model_manifest_path = run_dir / "models" / "model_spec_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8")) if model_manifest_path.is_file() else {}
    section_option_grid = _read_csv(run_dir / "section_analysis" / "section_option_grid.csv")

    if not selected_panels.empty and ("region" not in selected_panels.columns or "urbanicity_proxy" not in selected_panels.columns):
        sel_issue = _read_csv(run_dir / "panels" / "selected_panel_issues.csv")
        if sel_issue.empty:
            # Backward compatibility for older runs without selected_panel_issues.csv.
            recs: list[dict[str, Any]] = []
            for city_dir in sorted((run_dir / "panels").iterdir() if (run_dir / "panels").is_dir() else []):
                if not city_dir.is_dir():
                    continue
                p = city_dir / "issue_texts.jsonl"
                if not p.is_file():
                    continue
                for raw in p.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    recs.append(
                        {
                            "city_name": str(obj.get("city_name") or ""),
                            "state_abbr": str(obj.get("state_abbr") or "").lower(),
                            "region": str(obj.get("region") or ""),
                            "urbanicity_proxy": str(obj.get("urbanicity_proxy") or ""),
                        }
                    )
            sel_issue = pd.DataFrame.from_records(recs)
        if not sel_issue.empty:
            aux = (
                sel_issue.groupby(["city_name", "state_abbr"], dropna=False)
                .agg(region=("region", "first"), urbanicity_proxy=("urbanicity_proxy", "first"))
                .reset_index()
            )
            selected_panels["state_abbr"] = selected_panels["state_abbr"].astype(str).str.lower()
            selected_panels = selected_panels.merge(aux, on=["city_name", "state_abbr"], how="left")

    # Figures.
    fig_national = figs / "national_coverage.png"
    fig_panel_mix = figs / "panel_mix.png"
    fig_jaccard = figs / "consistency_jaccard_hist.png"
    fig_contam = figs / "contamination_pre_post.png"
    fig_purpose = figs / "purpose_similarity_scatter.png"
    fig_adopt_coef = figs / "adoption_model_coeffs.png"
    _plot_national_coverage(national, fig_national)
    _plot_panel_mix(selected_panels, fig_panel_mix)
    _plot_jaccard(consistency, fig_jaccard)
    _plot_contamination(contamination, fig_contam)
    _plot_purpose_similarity(purpose_pairs, fig_purpose)
    _plot_model_coeffs(adoption_table, fig_adopt_coef)

    # Tables.
    _write_table_tex(tbls / "selection_targets.tex", selection_targets, caption_cols=["target_type", "group", "target_n", "selected_n"])
    _write_table_tex(tbls / "selection_bias.tex", selection_bias, caption_cols=["dimension", "group", "eligible_n", "selected_n", "eligible_share", "selected_share"])
    _write_table_tex(tbls / "consistency_summary.tex", consistency.describe(include="all"), caption_cols=["category_jaccard_panel_vs_issue_union", "panel_complexity_0_100", "issue_complexity_mean_0_100"])
    _write_table_tex(tbls / "contamination_summary.tex", contamination, caption_cols=["classification_label", "n_issues", "pre_filter_noise_share", "post_filter_noise_share"])
    _write_table_tex(tbls / "adoption_by_region.tex", init_region, caption_cols=list(init_region.columns[: min(6, len(init_region.columns))]))
    _write_table_tex(tbls / "purpose_relations.tex", purpose_rel, caption_cols=list(purpose_rel.columns[: min(6, len(purpose_rel.columns))]))
    _write_table_tex(tbls / "amendment_matching.tex", amend_match, caption_cols=list(amend_match.columns[: min(6, len(amend_match.columns))]))
    _write_table_tex(tbls / "model_adoption_terms.tex", adoption_table.head(30), caption_cols=["term", "coef", "std_err", "p_value", "odds_ratio"])
    _write_table_tex(tbls / "model_content_terms.tex", content_table.head(40), caption_cols=["subset", "dependent_variable", "term", "coef", "std_err", "p_value"])
    _write_table_tex(tbls / "lot_size_extremes.tex", lot_ext.head(15), caption_cols=["city_name", "state_abbr", "issue_date", "max_min_lot_size_sqft", "issue_id"])
    _write_table_tex(tbls / "covariate_missingness.tex", cov_missing[cov_missing.get("dimension", "") == "variable"], caption_cols=["variable", "coverage_share", "n_rows", "non_missing_n"])
    _write_table_tex(
        tbls / "adoption_option_grid.tex",
        adoption_option_grid.head(10),
        caption_cols=["option", "status", "n_rows", "n_events", "group_cv_auc_mean", "group_cv_logloss_mean"],
    )
    _write_table_tex(
        tbls / "section_option_grid.tex",
        section_option_grid,
        caption_cols=["option", "docs_nonempty", "pairs", "mean_js", "purpose_title_match_docs"],
    )

    # Key metrics for narrative.
    n_panels = int(selected_panels["city_name"].nunique()) if not selected_panels.empty else 0
    j = pd.to_numeric(consistency.get("category_jaccard_panel_vs_issue_union"), errors="coerce")
    mean_j = float(j.mean()) if j.notna().any() else math.nan
    low_j_share = float((j < 0.50).mean()) if j.notna().any() else math.nan
    contam_drop = math.nan
    if not contamination.empty:
        pre = pd.to_numeric(contamination["pre_filter_noise_share"], errors="coerce").mean()
        post = pd.to_numeric(contamination["post_filter_noise_share"], errors="coerce").mean()
        contam_drop = float(pre - post) if pd.notna(pre) and pd.notna(post) else math.nan
    place_match = float((place_xw["match_method"] != "unmatched").mean()) if not place_xw.empty else math.nan
    county_match = float((county_xw["county_fips"].astype(str).str.len() == 5).mean()) if not county_xw.empty else math.nan
    purpose_pairs_n = int(len(purpose_pairs))
    amend_match_rate = float(pd.to_numeric(amend_match.get("matched_share"), errors="coerce").mean()) if not amend_match.empty else math.nan
    best_adoption_option = ""
    if not adoption_option_grid.empty and "option" in adoption_option_grid.columns:
        ag = adoption_option_grid.copy()
        ag["group_cv_auc_mean"] = pd.to_numeric(ag.get("group_cv_auc_mean"), errors="coerce")
        ag["group_cv_logloss_mean"] = pd.to_numeric(ag.get("group_cv_logloss_mean"), errors="coerce")
        ag = ag.sort_values(["group_cv_auc_mean", "group_cv_logloss_mean"], ascending=[False, True], na_position="last")
        if not ag.empty:
            best_adoption_option = str(ag.iloc[0]["option"])
    best_section_option = ""
    if not section_option_grid.empty and "option" in section_option_grid.columns:
        sg = section_option_grid.copy()
        sg["mean_js"] = pd.to_numeric(sg.get("mean_js"), errors="coerce")
        sg["docs_nonempty"] = pd.to_numeric(sg.get("docs_nonempty"), errors="coerce")
        sg = sg.sort_values(["mean_js", "docs_nonempty"], ascending=[False, False], na_position="last")
        if not sg.empty:
            best_section_option = str(sg.iloc[0]["option"])

    now = dt.datetime.now().strftime("%Y-%m-%d")
    doc: list[str] = []
    doc.append("\\documentclass[11pt]{article}")
    doc.append("\\usepackage[margin=1in]{geometry}")
    doc.append("\\usepackage{graphicx}")
    doc.append("\\usepackage{booktabs}")
    doc.append("\\usepackage{float}")
    doc.append("\\usepackage{hyperref}")
    doc.append("\\usepackage{longtable}")
    doc.append(f"\\title{{{_tex_escape(args.title)}}}")
    doc.append("\\author{Automated PI v1 analysis pipeline}")
    doc.append(f"\\date{{{_tex_escape(now)}}}")
    doc.append("\\begin{document}")
    doc.append("\\maketitle")
    doc.append("\\tableofcontents")
    doc.append("\\clearpage")

    doc.append("\\section{Executive Summary}")
    doc.append(
        f"This package scales the zoning measurement workflow to {n_panels} selected city panels, "
        "adds contamination diagnostics, a fixed-taxonomy/open-coding bridge, purpose-section similarity "
        "analysis, amendment delta measurement, and merged city-year covariates for adoption/content models."
    )
    doc.append("\\begin{itemize}")
    doc.append(f"\\item Mean panel-vs-issue category Jaccard: {mean_j:.3f}" if pd.notna(mean_j) else "\\item Mean panel-vs-issue category Jaccard: NA")
    doc.append(f"\\item Share of panels with Jaccard $<$ 0.50: {low_j_share:.3f}" if pd.notna(low_j_share) else "\\item Share of panels with Jaccard $<$ 0.50: NA")
    doc.append(f"\\item Mean contamination reduction (pre-filter minus post-filter): {contam_drop:.3f}" if pd.notna(contam_drop) else "\\item Mean contamination reduction: NA")
    doc.append(f"\\item Place crosswalk match rate: {place_match:.3f}" if pd.notna(place_match) else "\\item Place crosswalk match rate: NA")
    doc.append(f"\\item County crosswalk match rate: {county_match:.3f}" if pd.notna(county_match) else "\\item County crosswalk match rate: NA")
    doc.append(f"\\item Purpose similarity pairs computed: {purpose_pairs_n:,d}")
    doc.append(f"\\item Amendment prior-match rate (mean): {amend_match_rate:.3f}" if pd.notna(amend_match_rate) else "\\item Amendment prior-match rate: NA")
    doc.append("\\end{itemize}")

    doc.append("\\section{Coverage and Selection Diagnostics}")
    if fig_national.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.92\\textwidth]{figures/national_coverage.png}\\caption{National coverage over time.}\\end{figure}")
    if fig_panel_mix.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.92\\textwidth]{figures/panel_mix.png}\\caption{Selected panel composition by region and urbanicity proxy.}\\end{figure}")
    doc.append("\\begin{table}[H]\\centering\\caption{Selection targets}\\input{tables/selection_targets.tex}\\end{table}")
    doc.append("\\begin{table}[H]\\centering\\caption{Eligible vs selected composition}\\input{tables/selection_bias.tex}\\end{table}")
    doc.append("\\begin{table}[H]\\centering\\caption{Initial full ordinance adoption by region}\\input{tables/adoption_by_region.tex}\\end{table}")

    doc.append("\\section{Reliability and Contamination}")
    if fig_jaccard.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.75\\textwidth]{figures/consistency_jaccard_hist.png}\\caption{Panel-vs-issue consistency distribution.}\\end{figure}")
    if fig_contam.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.82\\textwidth]{figures/contamination_pre_post.png}\\caption{Contamination rates before and after deterministic filtering.}\\end{figure}")
    doc.append("\\begin{table}[H]\\centering\\caption{Contamination panel summary}\\input{tables/contamination_summary.tex}\\end{table}")

    doc.append("\\section{Section-Level Purpose Analysis}")
    doc.append(
        "Purpose sections are selected via title matching (purpose/intent/objective/findings/declaration), "
        "with first-section fallback. Similarity is measured by section text embedding cosine and "
        "Jensen-Shannon similarity on motive distributions."
    )
    if fig_purpose.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.78\\textwidth]{figures/purpose_similarity_scatter.png}\\caption{Pairwise purpose-section similarity across municipalities.}\\end{figure}")
    doc.append("\\begin{table}[H]\\centering\\caption{Purpose similarity by region/time/urbanicity relation}\\input{tables/purpose_relations.tex}\\end{table}")

    doc.append("\\section{Amendment Delta Measurement}")
    doc.append(
        "Amendment deltas are computed against deterministic nearest-prior same-city documents "
        "(prefer prior full ordinance, otherwise prior amendment, max 10-year lookback)."
    )
    doc.append("\\begin{table}[H]\\centering\\caption{Amendment matching diagnostics}\\input{tables/amendment_matching.tex}\\end{table}")

    doc.append("\\section{Numeric Constraints and Complexity}")
    doc.append("\\begin{table}[H]\\centering\\caption{Single-family lot-size extremes (illustrative)}\\input{tables/lot_size_extremes.tex}\\end{table}")

    doc.append("\\section{Option Testing and Recommended Defaults}")
    doc.append(
        "The run includes direct option tests for adoption model specifications and section-recovery strategies. "
        "These diagnostics are used to pick defaults for the next scaling iteration."
    )
    if best_adoption_option:
        doc.append(f"Best adoption option by grouped CV: \\texttt{{{_tex_escape(best_adoption_option)}}}.")
    if best_section_option:
        doc.append(f"Best section strategy by purpose-similarity quality/coverage: \\texttt{{{_tex_escape(best_section_option)}}}.")
    doc.append("\\begin{table}[H]\\centering\\caption{Adoption option grid (grouped CV)}\\input{tables/adoption_option_grid.tex}\\end{table}")
    doc.append("\\begin{table}[H]\\centering\\caption{Section strategy option grid}\\input{tables/section_option_grid.tex}\\end{table}")

    doc.append("\\section{Covariates and Models}")
    doc.append("\\begin{table}[H]\\centering\\caption{Covariate missingness (variable-level)}\\input{tables/covariate_missingness.tex}\\end{table}")
    if fig_adopt_coef.is_file():
        doc.append("\\begin{figure}[H]\\centering\\includegraphics[width=0.78\\textwidth]{figures/adoption_model_coeffs.png}\\caption{Adoption hazard model coefficients (selected covariates).}\\end{figure}")
    doc.append("\\begin{table}[H]\\centering\\caption{Adoption model terms}\\input{tables/model_adoption_terms.tex}\\end{table}")
    doc.append("\\begin{table}[H]\\centering\\caption{Content model terms (head)}\\input{tables/model_content_terms.tex}\\end{table}")

    doc.append("\\section{PI Risk Register and Next Iteration}")
    doc.append("\\begin{itemize}")
    doc.append("\\item Coverage balance: verify selected-vs-eligible distortions in underrepresented strata.")
    doc.append("\\item Measurement validity: inspect low-Jaccard tails and high contamination panels for prompt/parse failure modes.")
    doc.append("\\item Purpose comparability: check whether fallback-first-section cases drive similarity attenuation.")
    doc.append("\\item Amendment matching: audit unmatched and long-lag events for true prior-document availability.")
    doc.append("\\item Covariates: prioritize richer pre-ACS historical proxies for early decades.")
    doc.append("\\item Models: report robustness with alternative FE structures and restricted balanced samples.")
    doc.append("\\end{itemize}")

    doc.append("\\section*{Reproducibility}")
    doc.append(f"Run directory: \\path{{{_tex_escape(str(run_dir))}}}.")
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
        "output_dir": str(out_dir),
        "compiled_pdf": compiled_pdf,
        "model_manifest_present": bool(model_manifest),
        "inputs_present": {
            "national_coverage": not national.empty,
            "selected_panels": not selected_panels.empty,
            "consistency": not consistency.empty,
            "contamination": not contamination.empty,
            "purpose_pairs": not purpose_pairs.empty,
            "amendment_matching": not amend_match.empty,
            "covariate_missingness": not cov_missing.empty,
            "adoption_model_table": not adoption_table.empty,
            "content_model_table": not content_table.empty,
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
