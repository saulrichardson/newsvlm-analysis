#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PURPOSE_ORDER = [
    "public_health_safety",
    "growth_management",
    "externality_control",
    "infrastructure_coordination",
    "aesthetic_design_control",
    "environmental_protection",
    "exclusion",
    "value_capture",
    "fiscal_zoning",
]
REGION_ORDER = ["northeast", "midwest", "south", "west"]
LEVEL_ORDER = ["city", "village", "borough", "town", "township", "county", "city_and_county"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _tex_escape(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.000"


def _clip(text: Any, limit: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + "..."


def _table_block(*, caption: str, label: str, colspec: str, header: list[str], rows: list[list[str]]) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows) if rows else r"\multicolumn{99}{c}{No rows available.} \\"
    uses_x = "X" in colspec
    env = "tabularx" if uses_x else "tabular"
    env_open = rf"\begin{{{env}}}{{\textwidth}}{{{colspec}}}" if uses_x else rf"\begin{{{env}}}{{{colspec}}}"
    env_close = rf"\end{{{env}}}"
    return rf"""
\begin{{table}}[H]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{0.98}}
{env_open}
\toprule
{' & '.join(_tex_escape(cell) for cell in header)} \\\\
\midrule
{body}
\bottomrule
{env_close}
\caption{{{caption}}}
\label{{{label}}}
\end{{table}}
""".strip()


def _figure_block(path: str, caption: str, width: str = "0.9\\textwidth") -> str:
    return rf"""
\IfFileExists{{{path}}}{{
\begin{{figure}}[H]
  \centering
  \includegraphics[width={width}]{{{path}}}
  \caption{{{caption}}}
\end{{figure}}
}}{{}}
""".strip()


def _rows_from_df(df: pd.DataFrame, *, columns: list[str], formatters: dict[str, Any] | None = None) -> list[list[str]]:
    formatters = formatters or {}
    rows: list[list[str]] = []
    for row in df.to_dict(orient="records"):
        formatted: list[str] = []
        for col in columns:
            value = row.get(col, "")
            if col in formatters:
                value = formatters[col](value)
            else:
                value = _tex_escape(value)
            formatted.append(str(value))
        rows.append(formatted)
    return rows


def _display_label(value: Any) -> str:
    return str(value or "").strip().replace("_", " ")


def _category_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text == "city_and_county":
        return "City and County"
    return text.replace("_", " ").title()


def _top_purpose_lifts(df: pd.DataFrame, *, group_col: str, top_n: int = 2) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[df["motive"].astype(str).isin(PURPOSE_ORDER)].copy()
    out["abs_lift_deviation"] = (pd.to_numeric(out["purpose_weight_lift_vs_overall_vintage_adjusted"], errors="coerce") - 1.0).abs()
    out = out.sort_values(
        [group_col, "abs_lift_deviation", "purpose_weight_lift_vs_overall_vintage_adjusted", "purpose_weight__vintage_adjusted", "motive"],
        ascending=[True, False, False, False, True],
    )
    return out.groupby(group_col, dropna=False).head(top_n).reset_index(drop=True)


def _purpose_alignment_summary(df: pd.DataFrame, *, min_docs: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[df["doc_primary_motive"].astype(str).isin(PURPOSE_ORDER)].copy()
    out = out[out["ordinance_count"].fillna(0) >= min_docs].copy()
    return out.sort_values(
        ["mean_non_other_alignment_share", "ordinance_count", "doc_primary_motive"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _fmt_pct_diff_from_lift(value: Any) -> str:
    try:
        pct = (float(value) - 1.0) * 100.0
        return _tex_escape(f"{pct:+.1f}%")
    except Exception:
        return _tex_escape("+0.0%")


def _plot_purpose_lift_heatmap(df: pd.DataFrame, *, group_col: str, out_path: Path, title: str) -> None:
    if df.empty:
        return
    plot_df = df[df["motive"].astype(str).isin(PURPOSE_ORDER)].copy()
    if plot_df.empty:
        return
    pivot = plot_df.pivot_table(
        index=group_col,
        columns="motive",
        values="purpose_weight_lift_vs_overall_vintage_adjusted",
        aggfunc="first",
    )
    desired_rows = REGION_ORDER if group_col == "jurisdiction_region" else LEVEL_ORDER
    row_order = [item for item in desired_rows if item in pivot.index]
    col_order = [item for item in PURPOSE_ORDER if item in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)
    if pivot.empty:
        return
    lift_values = pivot.to_numpy(dtype=float)
    values = (lift_values - 1.0) * 100.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    delta = max(abs(float(finite.min())), abs(float(finite.max())), 1.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(col_order) + 2), max(4.5, 0.62 * len(row_order) + 1.5)))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-delta, vmax=delta)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([_category_label(x) for x in col_order], rotation=35, ha="right")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([_category_label(x) for x in row_order])
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if abs(val) > delta * 0.55 else "#1a202c"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Vintage-adjusted % difference vs sample baseline")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_full_writeup_tex(*, run_root: Path, output_path: Path | None = None) -> Path:
    run_root = run_root.resolve()
    reports_dir = run_root / "reports"
    research_dir = run_root / "research"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_path or (reports_dir / "frontier_llm_v2_full_writeup.tex")).resolve()

    research_summary = _load_json(next(iter(sorted(research_dir.glob("research_summary__*.json")))))
    baseline_summary = _load_json(reports_dir / "baseline_local_comparison_summary.json")

    sections = _read_parquet(run_root / "structure" / "sections__pilot.parquet")
    vintage_counts = _read_parquet(research_dir / "vintage" / "vintage_counts__pilot.parquet")
    region_contrasts = _read_parquet(research_dir / "section_clusters" / "top_region_contrasts_vintage_adjusted__pilot.parquet")
    level_contrasts = _read_parquet(research_dir / "section_clusters" / "top_level_contrasts_vintage_adjusted__pilot.parquet")
    weighted_purpose_region = _read_parquet(research_dir / "purposes" / "weighted_purpose_by_region_vintage_adjusted__pilot.parquet")
    weighted_purpose_level = _read_parquet(research_dir / "purposes" / "weighted_purpose_by_level_vintage_adjusted__pilot.parquet")
    purpose_alignment = _read_parquet(research_dir / "purposes" / "purpose_alignment_by_primary_motive__pilot.parquet")
    joint_effects = _read_parquet(research_dir / "section_clusters" / "top_cluster_presence_region_level_effects__pilot.parquet")
    complexity_effects = _read_parquet(research_dir / "complexity" / "top_complexity_region_level_effects__pilot.parquet")
    positive_cases = _read_parquet(research_dir / "coherence" / "coherence_positive_residuals__pilot.parquet")
    negative_cases = _read_parquet(research_dir / "coherence" / "coherence_negative_residuals__pilot.parquet")
    style_summary = _read_parquet(research_dir / "style_families" / "regulatory_style_family_summary__pilot.parquet")
    largest_clause_clusters = _read_parquet(research_dir / "clause_clusters" / "largest_clause_clusters__pilot.parquet")
    coverage_by_region = _read_parquet(research_dir / "coverage" / "clause_enrichment_coverage_by_region_vintage_adjusted__pilot.parquet")
    policy_clusters = _read_csv(reports_dir / "tables" / "policy_signature_clusters.csv")
    top_instrument_lifts = _read_parquet(research_dir / "instruments" / "top_region_instrument_share_lifts__pilot.parquet")

    section_count = len(sections)
    enriched_doc_coverage = float(research_summary.get("enriched_clause_doc_coverage_rate", 0.0))
    reportable_policy_clusters = int(research_summary.get("policy_signature_reportable_cluster_count", 0))
    policy_signature_section_presence_available = bool(research_summary.get("policy_signature_section_presence_available", False))
    policy_signature_section_presence_reportable = int(research_summary.get("policy_signature_section_presence_reportable_cluster_count", 0))
    policy_signature_section_presence_silhouette = float(research_summary.get("policy_signature_section_presence_silhouette_score", 0.0))
    policy_signature_section_presence_labels = [
        _tex_escape(_clip(label, 52))
        for label in (research_summary.get("policy_signature_section_presence_top_labels") or [])
        if str(label).strip()
    ]
    south_coverage = None
    if not coverage_by_region.empty:
        south_rows = coverage_by_region[
            coverage_by_region["jurisdiction_region"].astype(str).str.lower() == "south"
        ]
        if not south_rows.empty:
            south_coverage = float(south_rows.iloc[0]["has_clause_enrichment__vintage_adjusted"])

    region_contrasts = (
        region_contrasts.sort_values(
            ["jurisdiction_region", "presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted"],
            ascending=[True, False, False],
        )
        .groupby("jurisdiction_region", dropna=False)
        .head(2)
        .reset_index(drop=True)
    )
    level_contrasts = level_contrasts[level_contrasts["ordinance_count"].fillna(0) >= 5].copy()
    level_contrasts = (
        level_contrasts.sort_values(
            ["jurisdiction_level", "presence_lift_vs_overall_vintage_adjusted", "presence_rate_vintage_adjusted"],
            ascending=[True, False, False],
        )
        .groupby("jurisdiction_level", dropna=False)
        .head(2)
        .reset_index(drop=True)
    )
    purpose_region_top = _top_purpose_lifts(weighted_purpose_region, group_col="jurisdiction_region", top_n=2)
    purpose_level_top = _top_purpose_lifts(weighted_purpose_level, group_col="jurisdiction_level", top_n=2)
    purpose_alignment = _purpose_alignment_summary(purpose_alignment)
    _plot_purpose_lift_heatmap(
        weighted_purpose_region,
        group_col="jurisdiction_region",
        out_path=figures_dir / "purpose_lift_by_region_heatmap__pilot.png",
        title="Pilot Document-Purpose Deviations by Region (%)",
    )
    _plot_purpose_lift_heatmap(
        weighted_purpose_level,
        group_col="jurisdiction_level",
        out_path=figures_dir / "purpose_lift_by_level_heatmap__pilot.png",
        title="Pilot Document-Purpose Deviations by Jurisdiction Type (%)",
    )
    joint_region_effects = joint_effects[joint_effects["group_name"].astype(str) == "jurisdiction_region"].head(6).copy()
    joint_level_effects = joint_effects[
        (joint_effects["group_name"].astype(str) == "jurisdiction_level")
        & (~joint_effects["group_value"].astype(str).isin(["borough", "city_and_county"]))
    ].head(6).copy()
    complexity_region_effects = complexity_effects[complexity_effects["group_name"].astype(str) == "jurisdiction_region"].head(4).copy()

    summary_table = _table_block(
        caption="Current Frontier LLM v2 pilot checkpoint. This table is the high-level inventory of the semantic pilot: it shows how much of the pilot is covered by the section and clause layers, how large the current clause surface is, and which headline diagnostics currently characterize the pilot.",
        label="tab:v2-writeup-summary",
        colspec=r">{\raggedright\arraybackslash}Xr",
        header=["Metric", "Value"],
        rows=[
            ["Pilot ordinances", _fmt_int(research_summary["doc_count"])],
            ["Sections", _fmt_int(section_count)],
            ["Enriched-clause ordinances", _fmt_int(research_summary["enriched_clause_doc_count"])],
            ["Enriched clauses", _fmt_int(research_summary["enriched_clause_row_count"])],
            ["Clause clusters", _fmt_int(research_summary["clause_cluster_count"])],
            ["Regulatory style families", _fmt_int(research_summary["regulatory_style_family_count"])],
            ["Policy-signature documents", _fmt_int(research_summary["policy_signature_item_count"])],
            ["Policy-signature silhouette", _fmt_float(research_summary["policy_signature_silhouette_score"])],
            ["OpenAI section surface winner", _tex_escape(research_summary["embedding_surface_winner"])],
            ["Complexity/alignment raw correlation", _fmt_float(research_summary["complexity_alignment_correlation"])],
            ["Complexity/alignment adjusted slope", _fmt_float(research_summary["complexity_alignment_adjusted_coef"])],
        ],
    )

    vintage_table = _table_block(
        caption="Pilot vintage composition. These counts matter because the pilot is not a balanced panel over time, so later regional and jurisdiction-type contrasts are reported after vintage adjustment rather than as raw tabulations.",
        label="tab:v2-vintage",
        colspec="lr",
        header=["Vintage bin", "Ordinances"],
        rows=_rows_from_df(vintage_counts, columns=["vintage_bin", "ordinance_count"], formatters={"vintage_bin": _tex_escape, "ordinance_count": _fmt_int}),
    )

    region_table = _table_block(
        caption="Top vintage-adjusted section-family lifts by region (top two per region). `Adj. presence` is the section-family presence rate after reweighting the pilot to a common vintage mix, and `Adj. lift` compares that adjusted presence to the pilot-wide baseline. Values above one indicate section families that are relatively overrepresented in that region.",
        label="tab:v2-region-clusters",
        colspec=r"l>{\raggedright\arraybackslash}Xrr",
        header=["Region", "Section family", "Adj. presence", "Adj. lift"],
        rows=_rows_from_df(
            region_contrasts,
            columns=["jurisdiction_region", "cluster_label", "presence_rate_vintage_adjusted", "presence_lift_vs_overall_vintage_adjusted"],
            formatters={
                "jurisdiction_region": _tex_escape,
                "cluster_label": lambda x: _tex_escape(_clip(x, 44)),
                "presence_rate_vintage_adjusted": lambda x: _fmt_float(x, 3),
                "presence_lift_vs_overall_vintage_adjusted": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    level_table = _table_block(
        caption="Top vintage-adjusted section-family lifts by jurisdiction level (levels with at least five ordinances in the pilot). Read `Adj. presence` and `Adj. lift` the same way as Table~\\ref{tab:v2-region-clusters}, but the comparison is across institutional forms rather than regions.",
        label="tab:v2-level-clusters",
        colspec=r"l>{\raggedright\arraybackslash}Xrr",
        header=["Level", "Section family", "Adj. presence", "Adj. lift"],
        rows=_rows_from_df(
            level_contrasts,
            columns=["jurisdiction_level", "cluster_label", "presence_rate_vintage_adjusted", "presence_lift_vs_overall_vintage_adjusted"],
            formatters={
                "jurisdiction_level": _tex_escape,
                "cluster_label": lambda x: _tex_escape(_clip(x, 44)),
                "presence_rate_vintage_adjusted": lambda x: _fmt_float(x, 3),
                "presence_lift_vs_overall_vintage_adjusted": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    purpose_region_table = _table_block(
        caption="Top two document-purpose deviations by region on the pilot semantic surface after vintage adjustment. Construction: ordinance-level LLM purpose weights are normalized to sum to one, averaged within region-by-vintage cells, and then reweighted to the pilot-wide vintage mix. Adj. share is the adjusted mean share. Adj. percent diff compares that adjusted share with the pilot-wide benchmark for the same purpose. Because the normalized purpose shares already sum to one, meaningful heterogeneity often appears as single-digit percentage deviations.",
        label="tab:v2-purpose-region",
        colspec="llrr",
        header=["Region", "Purpose", "Adj. share", "Adj. % diff"],
        rows=_rows_from_df(
            purpose_region_top,
            columns=[
                "jurisdiction_region",
                "motive",
                "purpose_weight__vintage_adjusted",
                "purpose_weight_lift_vs_overall_vintage_adjusted",
            ],
            formatters={
                "jurisdiction_region": lambda x: _tex_escape(_category_label(x)),
                "motive": lambda x: _tex_escape(_category_label(x)),
                "purpose_weight__vintage_adjusted": lambda x: _fmt_float(x, 3),
                "purpose_weight_lift_vs_overall_vintage_adjusted": _fmt_pct_diff_from_lift,
            },
        ),
    )

    purpose_level_table = _table_block(
        caption="Top two document-purpose deviations by jurisdiction type on the pilot semantic surface after vintage adjustment. Read this exactly like Table~\\ref{tab:v2-purpose-region}, but as an institutional-form comparison rather than a geographic comparison. Adj. share is the adjusted average purpose share within that jurisdiction type, and Adj. percent diff compares it with the pilot-wide benchmark for the same purpose.",
        label="tab:v2-purpose-level",
        colspec="llrr",
        header=["Level", "Purpose", "Adj. share", "Adj. % diff"],
        rows=_rows_from_df(
            purpose_level_top,
            columns=[
                "jurisdiction_level",
                "motive",
                "purpose_weight__vintage_adjusted",
                "purpose_weight_lift_vs_overall_vintage_adjusted",
            ],
            formatters={
                "jurisdiction_level": lambda x: _tex_escape(_category_label(x)),
                "motive": lambda x: _tex_escape(_category_label(x)),
                "purpose_weight__vintage_adjusted": lambda x: _fmt_float(x, 3),
                "purpose_weight_lift_vs_overall_vintage_adjusted": _fmt_pct_diff_from_lift,
            },
        ),
    )

    purpose_alignment_table = _table_block(
        caption="Purpose-to-section alignment on the pilot. Construction: a dedicated `gpt-5-mini` document-purpose pass assigns each ordinance one primary purpose, then section-level motives are aggregated within the same ordinance using section word counts. `Mean non-other alignment` is the average share of non-`other`, non-`unclear` section-word mass that lands on the same motive as the document purpose. `Filtered exact match` is the share of ordinances where the largest filtered section-motive bucket equals the document purpose. Example: if a row shows `0.30`, that means about 30 percent of the filtered section-word mass aligns with the ordinance's primary document purpose.",
        label="tab:v2-purpose-alignment",
        colspec="lrrrr",
        header=["Primary purpose", "Docs", "Mean non-other alignment", "Filtered exact match", "Mean distinct section motives"],
        rows=_rows_from_df(
            purpose_alignment.head(8),
            columns=[
                "doc_primary_motive",
                "ordinance_count",
                "mean_non_other_alignment_share",
                "filtered_match_rate",
                "mean_distinct_section_motives",
            ],
            formatters={
                "doc_primary_motive": lambda x: _tex_escape(_category_label(x)),
                "ordinance_count": _fmt_int,
                "mean_non_other_alignment_share": lambda x: _fmt_float(x, 3),
                "filtered_match_rate": lambda x: _fmt_float(x, 3),
                "mean_distinct_section_motives": lambda x: _fmt_float(x, 2),
            },
        ),
    )

    region_effect_table = _table_block(
        caption="Strongest section-family region effects after controlling jointly for vintage and jurisdiction type. Coefficients are adjusted contrasts relative to omitted baseline categories, so positive values indicate a section family that remains region-associated even after the joint controls.",
        label="tab:v2-region-effects",
        colspec=r"l>{\raggedright\arraybackslash}Xr",
        header=["Region", "Section family", "Coef."],
        rows=_rows_from_df(
            joint_region_effects,
            columns=["group_value", "cluster_label", "coefficient"],
            formatters={
                "group_value": _tex_escape,
                "cluster_label": lambda x: _tex_escape(_clip(x, 48)),
                "coefficient": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    level_effect_table = _table_block(
        caption="Strongest section-family jurisdiction-type effects after controlling jointly for vintage and region. Positive coefficients indicate section families that remain institutionally distinctive after controlling for geography and vintage.",
        label="tab:v2-level-effects",
        colspec=r"l>{\raggedright\arraybackslash}Xr",
        header=["Level", "Section family", "Coef."],
        rows=_rows_from_df(
            joint_level_effects,
            columns=["group_value", "cluster_label", "coefficient"],
            formatters={
                "group_value": _tex_escape,
                "cluster_label": lambda x: _tex_escape(_clip(x, 48)),
                "coefficient": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    coherence_table = _table_block(
        caption="Illustrative coherence residual cases after controlling for complexity, vintage, and jurisdiction level. The residual is observed semantic alignment minus model-predicted alignment, so positive values identify unexpectedly coherent ordinances and negative values identify unexpectedly fragmented ones.",
        label="tab:v2-coherence",
        colspec=r"l>{\raggedright\arraybackslash}Xr",
        header=["Case type", "Ordinance", "Residual"],
        rows=[
            ["Unexpectedly coherent", _tex_escape(_clip(positive_cases.iloc[0]["ordinance_doc_id"], 48)), _fmt_float(positive_cases.iloc[0]["residual"])]
            if not positive_cases.empty
            else ["Unexpectedly coherent", "Unavailable", "0.000"],
            ["Unexpectedly fragmented", _tex_escape(_clip(negative_cases.iloc[0]["ordinance_doc_id"], 48)), _fmt_float(negative_cases.iloc[0]["residual"])]
            if not negative_cases.empty
            else ["Unexpectedly fragmented", "Unavailable", "0.000"],
        ],
    )

    style_table = _table_block(
        caption="Current regulatory style families on the enriched-clause pilot surface. `Direct`, `Discretion`, and `Except./Cond.` are mean shares of operative clauses, so the rows describe regulatory composition rather than raw volume. They need not sum to one because other operative categories are omitted here.",
        label="tab:v2-style",
        colspec=r">{\raggedright\arraybackslash}Xrrrr",
        header=["Family", "Docs", "Direct", "Discretion", "Except./Cond."],
        rows=_rows_from_df(
            style_summary.sort_values("ordinance_count", ascending=False).head(4),
            columns=[
                "cluster_label",
                "ordinance_count",
                "mean_direct_rule_share",
                "mean_discretion_share",
                "mean_exception_or_condition_share",
            ],
            formatters={
                "cluster_label": lambda x: _tex_escape(_clip(x, 58)),
                "ordinance_count": _fmt_int,
                "mean_direct_rule_share": lambda x: _fmt_float(x, 3),
                "mean_discretion_share": lambda x: _fmt_float(x, 3),
                "mean_exception_or_condition_share": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    clause_table = _table_block(
        caption="Largest current clause families on the enriched pilot surface. `Clauses` counts atomic enriched clauses and `Docs` counts the ordinances touched by the family. Large values can reflect long enumerations, so the table should be read as a map of clause-space coverage rather than a direct welfare ranking of legal importance.",
        label="tab:v2-largest-clause-families",
        colspec=r"r>{\raggedright\arraybackslash}Xrr",
        header=["ID", "Clause family", "Clauses", "Docs"],
        rows=_rows_from_df(
            largest_clause_clusters.head(10),
            columns=["cluster_id", "cluster_label", "clause_count", "ordinance_count"],
            formatters={
                "cluster_id": _fmt_int,
                "cluster_label": lambda x: _tex_escape(_clip(x, 46)),
                "clause_count": _fmt_int,
                "ordinance_count": _fmt_int,
            },
        ),
    )

    coverage_table = _table_block(
        caption="Vintage-adjusted clause-enrichment coverage by region. `Doc coverage` is the adjusted share of ordinances with clause enrichment in that region, while `Clause share` and `Section share` summarize how much of the underlying clause and section mass is currently represented. This is primarily a coverage-bias diagnostic.",
        label="tab:v2-coverage",
        colspec="lrrr",
        header=["Region", "Doc coverage", "Clause share", "Section share"],
        rows=_rows_from_df(
            coverage_by_region.sort_values("has_clause_enrichment__vintage_adjusted", ascending=False),
            columns=[
                "jurisdiction_region",
                "has_clause_enrichment__vintage_adjusted",
                "clause_enrichment_share__vintage_adjusted",
                "section_enrichment_share__vintage_adjusted",
            ],
            formatters={
                "jurisdiction_region": _tex_escape,
                "has_clause_enrichment__vintage_adjusted": lambda x: _fmt_float(x, 3),
                "clause_enrichment_share__vintage_adjusted": lambda x: _fmt_float(x, 3),
                "section_enrichment_share__vintage_adjusted": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    instrument_table = _table_block(
        caption="Top vintage-adjusted instrument-share lifts by region on the current enriched surface. `Adj. share` is the vintage-adjusted clause-share for the instrument in that region, and `Adj. lift` divides that share by the pilot-wide baseline. Values above one therefore mark instruments that are relatively overrepresented in that region.",
        label="tab:v2-instruments",
        colspec="llrr",
        header=["Region", "Instrument", "Adj. share", "Adj. lift"],
        rows=_rows_from_df(
            top_instrument_lifts.sort_values(
                ["jurisdiction_region", "instrument_clause_share_lift_vs_overall_vintage_adjusted"],
                ascending=[True, False],
            ).groupby("jurisdiction_region", dropna=False).head(2),
            columns=[
                "jurisdiction_region",
                "instrument_type",
                "instrument_clause_share__vintage_adjusted",
                "instrument_clause_share_lift_vs_overall_vintage_adjusted",
            ],
            formatters={
                "jurisdiction_region": _tex_escape,
                "instrument_type": _tex_escape,
                "instrument_clause_share__vintage_adjusted": lambda x: _fmt_float(x, 3),
                "instrument_clause_share_lift_vs_overall_vintage_adjusted": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    policy_table = _table_block(
        caption="Current policy-signature clusters. These are document-level typology groupings built from clause-derived composition rather than raw text. `Docs` is the number of ordinances in the cluster and `Reportable` indicates whether the cluster is large enough to interpret seriously at the current pilot checkpoint.",
        label="tab:v2-policy",
        colspec=r"r>{\raggedright\arraybackslash}Xrl",
        header=["ID", "Signature label", "Docs", "Reportable"],
        rows=_rows_from_df(
            policy_clusters.head(6),
            columns=["policy_signature_cluster_id", "cluster_label", "cluster_size", "reportable_cluster_flag"],
            formatters={
                "policy_signature_cluster_id": _fmt_int,
                "cluster_label": lambda x: _tex_escape(_clip(x, 52)),
                "cluster_size": _fmt_int,
                "reportable_cluster_flag": lambda x: _tex_escape("yes" if int(x) else "no"),
            },
        ),
    )

    baseline_table = _table_block(
        caption="Pilot-level comparison between the older baseline-local layer and the current v2 semantic layer. Agreement rates summarize how often the older baseline and the new LLM-native semantic layer produce the same dominant interpretation on the shared pilot documents.",
        label="tab:v2-baseline-comparison",
        colspec=r">{\raggedright\arraybackslash}Xr",
        header=["Comparison metric", "Value"],
        rows=[
            ["Shared documents", _fmt_int(baseline_summary["shared_document_count"])],
            ["Primary motive agreement rate", _fmt_float(baseline_summary["primary_motive_agreement_rate"], 3)],
            ["Dominant instrument agreement rate", _fmt_float(baseline_summary["dominant_instrument_agreement_rate"], 3)],
            ["Operative dominant instrument agreement rate", _fmt_float(baseline_summary["operative_dominant_instrument_agreement_rate"], 3)],
        ],
    )

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{float}}
\usepackage{{graphicx}}
\usepackage[bookmarks=false]{{hyperref}}
\usepackage{{enumitem}}
\usepackage{{array}}
\usepackage{{caption}}
\captionsetup[table]{{position=bottom}}
\captionsetup[figure]{{position=bottom}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\title{{Frontier LLM v2 Pilot Writeup}}
\author{{Codex research build}}
\date{{March 10, 2026}}
\begin{{document}}
\maketitle

\section*{{Overview}}
\begin{{itemize}}[leftmargin=1.2em]
\item This document summarizes the current Frontier LLM v2 pilot.
\item The pilot validates the semantic pipeline on 64 ordinances before broader expansion.
\item The section side is stable; the clause side is informative but still coverage-limited.
\end{{itemize}}

{summary_table}

\section{{Data, Sample, and Corpus Construction}}
\begin{{itemize}}[leftmargin=1.2em]
\item The pilot is a vintage-aware cross-section rather than a within-place panel.
\item Vintage adjustment matters because the pilot is concentrated in the 1960s and 1970s.
\end{{itemize}}

{vintage_table}

\begin{{itemize}}[leftmargin=1.2em]
\item The pilot is the canonical v2 semantic checkpoint.
\item Clause coverage is {_fmt_int(research_summary["enriched_clause_doc_count"])} ordinances and {_fmt_int(research_summary["enriched_clause_row_count"])} enriched clauses, or about {_fmt_float(enriched_doc_coverage, 3)} of the pilot document surface.
\item That is enough for governance style and clause families, but not enough to treat enriched-clause typologies as fully mature.
\end{{itemize}}

\section{{Pipeline and Design Choice}}
\begin{{itemize}}[leftmargin=1.2em]
\item The v2 design is intentionally LLM-native for semantics.
\item Deterministic code is used for batching, validation, provenance, aggregation, clustering, diagnostics, and reporting.
\end{{itemize}}
\medskip
Current canonical path:
\begin{{enumerate}}[leftmargin=1.4em]
\item section segmentation with contiguous line ranges,
\item document purpose extraction,
\item section-motive labeling,
\item clause extraction,
\item clause enrichment,
\item OpenAI section embeddings with pooled segment embeddings for oversized sections,
\item section and clause clustering,
\item ordinance-level aggregation into governance-style and policy-signature surfaces.
\end{{enumerate}}

{baseline_table}

\section{{Document Purposes as a Heterogeneity Surface}}
\begin{{itemize}}[leftmargin=1.2em]
\item High level: the purpose surface records what ordinances say they are trying to do. It is weighted and multi-label, so one ordinance can contribute mass to several purpose families.
\item Exact construction: the LLM can emit multiple purpose rows per ordinance. Those raw purpose weights are normalized within ordinance to sum to one, averaged within group-by-vintage cells, and then reweighted to the pilot-wide vintage distribution.
\item Table formula: `Adj. share` is the vintage-adjusted average purpose share for that group. `Adj. \% diff` is `100 * (Adj. share / pilot-wide purpose share - 1)`. Small values are expected because the normalized purpose shares already sum to one within each ordinance.
\item Heatmap formula: every cell reports the same percent-difference object. Zero means the group is at the pilot-wide baseline for that purpose, positive values indicate overrepresentation, and negative values indicate underrepresentation.
\item What to notice: use the tables to see the biggest purpose shifts and the heatmaps to see whether those shifts are isolated or part of a broader profile.
\end{{itemize}}

{purpose_region_table}

{purpose_level_table}

{_figure_block("figures/purpose_lift_by_region_heatmap__pilot.png", "Pilot document-purpose deviations by region. Construction: normalize LLM purpose weights within ordinance, average them within region-by-vintage cells, and reweight those means to the pilot-wide vintage mix. Each cell reports `100 * (group share / pilot-wide share - 1)`. Values near zero mean the region is close to the pilot-wide baseline for that purpose; `+3.0` means three percent above baseline and `-3.0` means three percent below. Read rows across for a region's overall purpose profile and columns down for which regions over-index on a specific purpose family.", "0.86\\textwidth")}

{_figure_block("figures/purpose_lift_by_level_heatmap__pilot.png", "Pilot document-purpose deviations by jurisdiction type. This figure is built exactly like the regional heatmap, but groups ordinances by jurisdiction type rather than geography. Values near zero indicate near-baseline purpose weights, while larger positive or negative values indicate institutional-form heterogeneity after holding vintage composition fixed.", "0.90\\textwidth")}

{purpose_alignment_table}

\section{{Section-Level Legal Architecture}}
\begin{{itemize}}[leftmargin=1.2em]
\item This is the strongest current pilot result.
\item OpenAI section embeddings beat the local baseline.
\item The most important section-family differences survive vintage adjustment.
\end{{itemize}}

{_figure_block("figures/section_semantic_hdbscan_umap.png", "OpenAI section semantic manifold for the current pilot.", "0.88\\textwidth")}

{_figure_block("figures/section_semantic_hdbscan_local_baseline_umap.png", "Local baseline section manifold for comparison.", "0.88\\textwidth")}

{region_table}

{region_effect_table}

\section{{Jurisdiction Type as Legal Form}}
\begin{{itemize}}[leftmargin=1.2em]
\item Jurisdiction type is treated as a substantive legal-form axis rather than a metadata control.
\item The level tables are intended to separate institutional-form heterogeneity from purely geographic heterogeneity.
\end{{itemize}}

{level_table}

{level_effect_table}

\section{{Complexity, Coherence, and Fragmentation}}
\begin{{itemize}}[leftmargin=1.2em]
\item The pilot does not support a simple “more complexity means less coherence” story.
\item The stronger interpretation is residual coherence versus fragmentation after controls.
\end{{itemize}}

{coherence_table}

\section{{Clause Families, Governance Style, and Instrument Mix}}
\begin{{itemize}}[leftmargin=1.2em]
\item The enriched-clause surface is now substantive enough for clause-family and governance-style analysis.
\item The style-family surface is currently stronger than the policy-signature surface.
\end{{itemize}}

{clause_table}

{coverage_table}

{style_table}

{instrument_table}

\begin{{itemize}}[leftmargin=1.2em]
\item The dominant style family is characterized by large operative scale, high instrument diversity, and high instrument variety.
\item Smaller style families are best read as high-information outliers rather than stable mass categories.
\item Clause-side regional comparisons remain emerging rather than final{f", with South coverage at about {_fmt_float(south_coverage, 3)} on the enriched surface" if south_coverage is not None else ""}.
\end{{itemize}}

{_figure_block("figures/clause_instrument_hdbscan_umap.png", "Clause-instrument manifold on the current enriched-clause surface.", "0.88\\textwidth")}

\section{{Document-Level Typology and Policy Signatures}}
\begin{{itemize}}[leftmargin=1.2em]
\item Policy signatures are the first document-level surface built from what ordinances do rather than what they say.
\item They remain provisional because they are limited to the clause-enriched pilot subset.
\item Currently {_fmt_int(reportable_policy_clusters)} cluster{'s are' if reportable_policy_clusters != 1 else ' is'} reportable on the raw clause-weighted surface.
\end{{itemize}}

{policy_table}

{("\\begin{itemize}[leftmargin=1.2em]\n"
 + "\\item The section-normalized comparison averages clause instruments within sections before moving to the document.\n"
 + "\\item It currently yields "
 + _fmt_int(policy_signature_section_presence_reportable)
 + " reportable clusters at silhouette "
 + _fmt_float(policy_signature_section_presence_silhouette)
 + ".\n"
 + "\\item The raw-clause surface remains tighter, but the section-normalized surface is less distorted by long permitted-use and definition lists.\n"
 + "\\end{itemize}")
 if policy_signature_section_presence_available else ""}

{_table_block(
        caption="Section-normalized policy-signature comparison. This surface averages clause instruments within sections before aggregating to the document, which makes it less sensitive to very long use lists or definition sections. The table reports how many clusters remain reportable and how geometrically tight the resulting typology is.",
        label="tab:v2-policy-section-presence",
        colspec=r">{\raggedright\arraybackslash}Xrr",
        header=["Leading section-normalized signature", "Reportable clusters", "Silhouette"],
        rows=[[policy_signature_section_presence_labels[0] if policy_signature_section_presence_labels else "Unavailable", _fmt_int(policy_signature_section_presence_reportable), _fmt_float(policy_signature_section_presence_silhouette)]]
    ) if policy_signature_section_presence_available else ""}

\IfFileExists{{figures/policy_signature_by_region_heatmap.png}}{{
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.82\textwidth]{{figures/policy_signature_by_region_heatmap.png}}
  \caption{{Current policy-signature distribution by region on the covered pilot surface.}}
\end{{figure}}
}}{{}}

\section{{What Is Stable, What Is Emerging}}
\textbf{{Stable now.}} The most defensible pilot claims are:
\begin{{itemize}}[leftmargin=1.2em]
\item OpenAI section embeddings outperform the local baseline.
\item Regional section-cluster differences survive vintage adjustment.
\item Jurisdiction type matters as a legal-form axis, not just as metadata.
\item The interesting coherence frontier is residual coherence versus fragmentation, not raw complexity alone.
\end{{itemize}}

\textbf{{Emerging but still coverage-limited.}} The following are already promising, but should still be presented as partial:
\begin{{itemize}}[leftmargin=1.2em]
\item governance-style regional differences,
\item enriched-clause policy signatures,
\item clause-side regional instrument contrasts.
\end{{itemize}}

\section{{Current Story for the Paper}}
\begin{{enumerate}}[leftmargin=1.4em]
\item make section-level legal architecture the main descriptive contribution,
\item use coherence versus fragmentation as the deeper legal-form frontier,
\item present governance style and policy signatures as the clause-side layer that is becoming mature on top of enriched clauses.
\end{{enumerate}}

\section{{Figures and Reference Surfaces}}
{_figure_block("figures/motive_by_region_heatmap.png", "Regional motive shares on the current pilot surface.", "0.78\\textwidth")}

{_figure_block("figures/instrument_by_region_heatmap.png", "Regional substantive instrument shares on the current pilot surface.", "0.78\\textwidth")}

{_figure_block("figures/jurisdiction_match_status.png", "Publication-versus-jurisdiction match status in the broader corpus shell.", "0.72\\textwidth")}

{_figure_block("figures/duplicate_group_sizes.png", "Duplicate-publication group sizes in the broader corpus shell.", "0.72\\textwidth")}

\section{{Next Work}}
\begin{{itemize}}[leftmargin=1.2em]
\item widen enriched-clause document coverage,
\item rebuild governance-style and policy-signature outputs,
\item keep the paper narrative anchored in the stable section-side surfaces while the clause-side surfaces mature.
\end{{itemize}}

\end{{document}}
"""

    output_path.write_text(tex.strip() + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fuller Frontier LLM v2 pilot TeX writeup.")
    parser.add_argument("--run-root", required=True, help="Frontier LLM v2 run root.")
    parser.add_argument("--output-path", default="", help="Optional explicit output path.")
    args = parser.parse_args()
    output = build_full_writeup_tex(
        run_root=Path(args.run_root),
        output_path=Path(args.output_path) if args.output_path else None,
    )
    print(f"built_frontier_llm_v2_full_writeup_tex output={output}")


if __name__ == "__main__":
    main()
