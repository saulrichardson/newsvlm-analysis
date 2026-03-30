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


def _figure_block(path: str, caption: str, width: str = "0.88\\textwidth") -> str:
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
        rendered: list[str] = []
        for col in columns:
            value = row.get(col, "")
            if col in formatters:
                value = formatters[col](value)
            else:
                value = _tex_escape(value)
            rendered.append(str(value))
        rows.append(rendered)
    return rows


def _display_label(value: Any) -> str:
    text = str(value or "").strip().replace("_", " ")
    return text if text else ""


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


def _purpose_alignment_summary(df: pd.DataFrame, *, min_docs: int = 8) -> pd.DataFrame:
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


def _plot_purpose_lift_heatmap(
    df: pd.DataFrame,
    *,
    group_col: str,
    out_path: Path,
    title: str,
) -> None:
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
    vmin, vmax = -delta, delta
    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(col_order) + 2), max(4.5, 0.62 * len(row_order) + 1.5)))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
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
            text_color = "white" if abs(val) > delta * 0.55 else "#1a202c"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Vintage-adjusted % difference vs full-sample baseline")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_integrated_writeup_tex(*, run_root: Path, output_path: Path | None = None) -> Path:
    run_root = run_root.resolve()
    research_dir = run_root / "research"
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_path or (reports_dir / "frontier_llm_v2_integrated_writeup.tex")).resolve()
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    full_summary = _load_json(research_dir / "research_summary__full.json")
    pilot_summary = _load_json(research_dir / "research_summary__pilot.json")
    baseline_summary = _load_json(reports_dir / "baseline_local_comparison_summary.json")
    coverage_audit = _load_json(research_dir / "coverage" / "research_surface_coverage_audit.json")
    full_section_rows = len(_read_parquet(run_root / "structure" / "sections__full.parquet"))

    vintage_counts = _read_parquet(research_dir / "vintage" / "vintage_counts__full.parquet")
    region_contrasts = _read_parquet(research_dir / "section_clusters" / "top_region_contrasts_vintage_adjusted__full.parquet")
    level_contrasts = _read_parquet(research_dir / "section_clusters" / "top_level_contrasts_vintage_adjusted__full.parquet")
    weighted_purpose_region = _read_parquet(research_dir / "purposes" / "weighted_purpose_by_region_vintage_adjusted__full.parquet")
    weighted_purpose_level = _read_parquet(research_dir / "purposes" / "weighted_purpose_by_level_vintage_adjusted__full.parquet")
    purpose_lifts = _read_parquet(research_dir / "purposes" / "top_weighted_purpose_region_lifts__full.parquet")
    purpose_level_lifts = _read_parquet(research_dir / "purposes" / "top_weighted_purpose_level_lifts__full.parquet")
    purpose_alignment = _read_parquet(research_dir / "purposes" / "purpose_alignment_by_primary_motive__full.parquet")
    complexity_effects = _read_parquet(research_dir / "complexity" / "top_complexity_region_level_effects__full.parquet")
    positive_cases = _read_parquet(research_dir / "coherence" / "coherence_positive_residuals__full.parquet")
    negative_cases = _read_parquet(research_dir / "coherence" / "coherence_negative_residuals__full.parquet")
    architecture_summary = _read_parquet(research_dir / "architecture" / "architecture_family_summary__full.parquet")
    neighbor_region = _read_parquet(research_dir / "retrieval" / "neighbor_similarity_by_region__full.parquet")
    surface_comparison = _read_parquet(research_dir / "embedding_comparison" / "surface_comparison__full.parquet")

    pilot_style_summary = _read_parquet(research_dir / "style_families" / "regulatory_style_family_summary__pilot.parquet")
    pilot_largest_clause = _read_parquet(research_dir / "clause_clusters" / "largest_clause_clusters__pilot.parquet")
    pilot_coverage_region = _read_parquet(research_dir / "coverage" / "clause_enrichment_coverage_by_region_vintage_adjusted__pilot.parquet")
    raw_policy_summary = _load_json(run_root / "clusters" / "policy_signature_kmeans_probe" / "summary.json")
    section_norm_policy_summary = _load_json(run_root / "clusters" / "policy_signature_section_normalized_clause_share_probe" / "summary.json")
    section_presence_policy_summary = _load_json(run_root / "clusters" / "policy_signature_section_presence_probe" / "summary.json")

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
    purpose_lifts = _top_purpose_lifts(purpose_lifts if not purpose_lifts.empty else weighted_purpose_region, group_col="jurisdiction_region", top_n=2)
    purpose_level_lifts = _top_purpose_lifts(
        purpose_level_lifts if not purpose_level_lifts.empty else weighted_purpose_level,
        group_col="jurisdiction_level",
        top_n=2,
    )
    purpose_alignment = _purpose_alignment_summary(purpose_alignment)
    _plot_purpose_lift_heatmap(
        weighted_purpose_region,
        group_col="jurisdiction_region",
        out_path=figures_dir / "purpose_lift_by_region_heatmap__full.png",
        title="Full-Sample Document-Purpose Deviations by Region (%)",
    )
    _plot_purpose_lift_heatmap(
        weighted_purpose_level,
        group_col="jurisdiction_level",
        out_path=figures_dir / "purpose_lift_by_level_heatmap__full.png",
        title="Full-Sample Document-Purpose Deviations by Jurisdiction Type (%)",
    )
    if not pilot_largest_clause.empty and "cluster_label" in pilot_largest_clause.columns:
        labeled_clause_mask = (
            pilot_largest_clause["cluster_label"].fillna("").str.strip().ne("")
            & ~pilot_largest_clause["cluster_label"].fillna("").str.fullmatch(r"cluster \d+")
            & pilot_largest_clause["cluster_label"].fillna("").str.lower().ne("noise")
        )
        labeled_clause_clusters = pilot_largest_clause[labeled_clause_mask].copy()
        pilot_largest_clause_for_table = (
            labeled_clause_clusters
            if not labeled_clause_clusters.empty
            else pilot_largest_clause.copy()
        )
    else:
        pilot_largest_clause_for_table = pilot_largest_clause.copy()
    complexity_effects = complexity_effects.head(6).copy()

    full_summary_table = _table_block(
        caption="Full-sample section-side checkpoint on the 728 non-empty ordinance texts. This table is an inventory of the national semantic surface: counts describe how much of the corpus is covered, alignment shares summarize how closely document purposes line up with weighted section motives, and the embedding winner reports which surface produces cleaner clustering under the automated comparison metrics.",
        label="tab:full-summary",
        colspec=r">{\raggedright\arraybackslash}Xr",
        header=["Metric", "Value"],
        rows=[
            ["Ordinances in full semantic sample", _fmt_int(full_summary["doc_count"])],
            ["Full section rows", _fmt_int(full_section_rows)],
            ["Section semantic clusters", _fmt_int(full_summary["section_cluster_count"])],
            ["Mean purpose/section alignment share", _fmt_float(full_summary["mean_alignment_share"])],
            ["Mean non-`other` alignment share", _fmt_float(full_summary["mean_non_other_alignment_share"])],
            ["Complexity/alignment raw correlation", _fmt_float(full_summary["complexity_alignment_correlation"])],
            ["Complexity/alignment adjusted slope", _fmt_float(full_summary["complexity_alignment_adjusted_coef"])],
            ["OpenAI vs local winner on automated surface metric", _tex_escape(full_summary["embedding_surface_winner"])],
            ["Architecture family count", _fmt_int(full_summary["architecture_family_count"])],
            ["Same-region neighbor lift rows", _fmt_int(full_summary["neighbor_homophily_region_rows"])],
        ],
    )

    vintage_table = _table_block(
        caption="Vintage composition of the full section-side semantic sample. These counts matter because the analysis treats publication date as ordinance vintage, so later tables report vintage-adjusted contrasts rather than naive raw regional differences.",
        label="tab:full-vintage",
        colspec="lr",
        header=["Vintage bin", "Ordinances"],
        rows=_rows_from_df(
            vintage_counts,
            columns=["vintage_bin", "ordinance_count"],
            formatters={"vintage_bin": _tex_escape, "ordinance_count": _fmt_int},
        ),
    )

    purpose_table = _table_block(
        caption="Top two document-purpose deviations by region after vintage adjustment. Construction: ordinance-level LLM purpose weights are normalized to sum to one, averaged within region-by-vintage cells, and then reweighted to the national vintage mix. Adj. share is the resulting adjusted mean share. Adj. percent diff compares that adjusted share with the national benchmark for the same purpose. Because the normalized purpose shares already sum to one, meaningful heterogeneity often appears as single-digit percentage deviations rather than huge swings.",
        label="tab:full-purpose-lifts",
        colspec="llrr",
        header=["Region", "Purpose", "Adj. share", "Adj. % diff"],
        rows=_rows_from_df(
            purpose_lifts,
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
        caption="Top two document-purpose deviations by jurisdiction type after vintage adjustment. This is the institutional analog to Table~\\ref{tab:full-purpose-lifts}. Adj. share is the adjusted mean purpose share within that jurisdiction type, and Adj. percent diff compares it with the full-sample benchmark for the same purpose. Small numbers are expected because this is a normalized multi-label surface rather than a winner-take-all classification.",
        label="tab:full-purpose-level-lifts",
        colspec="llrr",
        header=["Level", "Purpose", "Adj. share", "Adj. % diff"],
        rows=_rows_from_df(
            purpose_level_lifts,
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
        caption="Purpose-to-section alignment by document primary purpose on the full sample. Construction: a dedicated `gpt-5-mini` document-purpose pass assigns each ordinance one primary purpose, then section-level motives are aggregated within the same ordinance using section word counts. `Mean non-other alignment` is the average share of non-`other`, non-`unclear` section-word mass that lands on the same motive as the document purpose. `Filtered exact match` is the share of ordinances where the largest filtered section-motive bucket equals the document purpose. Example: the `0.535` for Externality Control means that ordinances in that row devote about 53.5 percent of filtered section-word mass to sections with the same motive as the document-level purpose.",
        label="tab:full-purpose-alignment",
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

    region_table = _table_block(
        caption="Top vintage-adjusted section-family lifts by region (top two per region). `Adj. presence` is the vintage-adjusted section-family presence rate and `Adj. lift` compares that presence to the national baseline, so values above one indicate section families that are relatively overrepresented in that region.",
        label="tab:full-region-families",
        colspec=r"l>{\raggedright\arraybackslash}Xrr",
        header=["Region", "Section family", "Adj. presence", "Adj. lift"],
        rows=_rows_from_df(
            region_contrasts,
            columns=[
                "jurisdiction_region",
                "cluster_label",
                "presence_rate_vintage_adjusted",
                "presence_lift_vs_overall_vintage_adjusted",
            ],
            formatters={
                "jurisdiction_region": _tex_escape,
                "cluster_label": lambda x: _tex_escape(_clip(x, 52)),
                "presence_rate_vintage_adjusted": lambda x: _fmt_float(x, 3),
                "presence_lift_vs_overall_vintage_adjusted": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    level_table = _table_block(
        caption="Top vintage-adjusted section-family lifts by jurisdiction level (top two per level). Read `Adj. presence` and `Adj. lift` the same way as in Table~\\ref{tab:full-region-families}, but here the comparison is institutional form rather than geography.",
        label="tab:full-level-families",
        colspec=r"l>{\raggedright\arraybackslash}Xrr",
        header=["Level", "Section family", "Adj. presence", "Adj. lift"],
        rows=_rows_from_df(
            level_contrasts,
            columns=[
                "jurisdiction_level",
                "cluster_label",
                "presence_rate_vintage_adjusted",
                "presence_lift_vs_overall_vintage_adjusted",
            ],
            formatters={
                "jurisdiction_level": _tex_escape,
                "cluster_label": lambda x: _tex_escape(_clip(x, 52)),
                "presence_rate_vintage_adjusted": lambda x: _fmt_float(x, 3),
                "presence_lift_vs_overall_vintage_adjusted": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    coherence_table = _table_block(
        caption="Illustrative coherence residual cases after controlling for vintage and jurisdiction level. Residuals are observed alignment minus model-predicted alignment, so positive values identify ordinances that are more semantically coherent than expected and negative values identify ordinances that are more fragmented than expected given their vintage and institutional form.",
        label="tab:full-coherence-residuals",
        colspec=r"l>{\raggedright\arraybackslash}Xlrr",
        header=["Case", "Ordinance", "Region", "Complexity", "Residual"],
        rows=[
            [
                "Unexpectedly coherent",
                _tex_escape(_clip(positive_cases.iloc[0]["ordinance_doc_id"], 52)),
                _tex_escape(positive_cases.iloc[0]["jurisdiction_region"]),
                _fmt_float(positive_cases.iloc[0]["complexity_index_exploratory"]),
                _fmt_float(positive_cases.iloc[0]["residual"]),
            ]
            if not positive_cases.empty
            else ["Unexpectedly coherent", "Unavailable", "", "0.000", "0.000"],
            [
                "Unexpectedly fragmented",
                _tex_escape(_clip(negative_cases.iloc[0]["ordinance_doc_id"], 52)),
                _tex_escape(negative_cases.iloc[0]["jurisdiction_region"]),
                _fmt_float(negative_cases.iloc[0]["complexity_index_exploratory"]),
                _fmt_float(negative_cases.iloc[0]["residual"]),
            ]
            if not negative_cases.empty
            else ["Unexpectedly fragmented", "Unavailable", "", "0.000", "0.000"],
        ],
    )

    complexity_effect_table = _table_block(
        caption="Largest adjusted complexity effects by region or jurisdiction type in the full sample. Coefficients are regression effects relative to omitted baseline categories, so positive values mean higher complexity than the baseline on the target dimension and negative values mean lower complexity. Rows should be compared within this table as adjusted contrasts, not as raw means.",
        label="tab:full-complexity-effects",
        colspec="llrr",
        header=["Group kind", "Group", "Coef.", "p-value"],
        rows=_rows_from_df(
            complexity_effects,
            columns=["group_name", "group_value", "coefficient", "pvalue"],
            formatters={
                "group_name": _tex_escape,
                "group_value": _tex_escape,
                "coefficient": lambda x: _fmt_float(x, 3),
                "pvalue": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    architecture_table = _table_block(
        caption="Largest section-architecture families in the full sample. `Docs` shows how many ordinances fall into each family and `Typical opening` gives the dominant first substantive section kind, which helps interpret whether a family tends to begin with definitions, district rules, procedure, or another opening architecture.",
        label="tab:full-architecture-families",
        colspec=r"r>{\raggedright\arraybackslash}Xr>{\raggedright\arraybackslash}X",
        header=["ID", "Family label", "Docs", "Typical opening"],
        rows=_rows_from_df(
            architecture_summary.sort_values("ordinance_count", ascending=False).head(6),
            columns=[
                "architecture_family_id",
                "cluster_label",
                "ordinance_count",
                "dominant_first_substantive_section_kind",
            ],
            formatters={
                "architecture_family_id": _fmt_int,
                "cluster_label": lambda x: _tex_escape(_clip(x, 56)),
                "ordinance_count": _fmt_int,
                "dominant_first_substantive_section_kind": lambda x: _tex_escape(_clip(x, 42)),
            },
        ),
    )

    neighbor_table = _table_block(
        caption="Same-region neighbor lift on the full section-embedding surface. `Same-region share` is the mean fraction of nearest-neighbor ordinances drawn from the same region, and `Lift vs baseline` compares that share to the region's prevalence in the sample. Values above one therefore indicate region-specific clustering in the embedding space.",
        label="tab:full-neighbor-lift",
        colspec="lrrr",
        header=["Region", "Docs", "Same-region share", "Lift vs baseline"],
        rows=_rows_from_df(
            neighbor_region,
            columns=[
                "jurisdiction_region",
                "ordinance_count",
                "mean_same_region_neighbor_share",
                "same_region_neighbor_lift",
            ],
            formatters={
                "jurisdiction_region": _tex_escape,
                "ordinance_count": _fmt_int,
                "mean_same_region_neighbor_share": lambda x: _fmt_float(x, 3),
                "same_region_neighbor_lift": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    surface_table = _table_block(
        caption="Automated surface comparison on the full sample. More clusters with low noise can be helpful, but the key interpretation columns are `Top-motive share` and `Motive entropy`: higher top-share and lower entropy indicate a cleaner semantic partition. The winning surface is the one that better balances separability and interpretability under these metrics.",
        label="tab:full-surface-comparison",
        colspec="lrrrr",
        header=["Surface", "Clusters", "Noise rate", "Top-motive share", "Motive entropy"],
        rows=_rows_from_df(
            surface_comparison,
            columns=[
                "surface_name",
                "cluster_count",
                "noise_rate",
                "mean_top_motive_share",
                "mean_motive_entropy",
            ],
            formatters={
                "surface_name": _tex_escape,
                "cluster_count": _fmt_int,
                "noise_rate": lambda x: _fmt_float(x, 3),
                "mean_top_motive_share": lambda x: _fmt_float(x, 3),
                "mean_motive_entropy": lambda x: _fmt_float(x, 3),
            },
        ),
    )

    pilot_clause_summary = _table_block(
        caption="Current pilot clause-side frontier checkpoint. This is a coverage and maturity table for the enriched-clause layer: it tells you how much of the pilot clause surface is available for governance-style and document-composition analysis, not the size of the full national corpus.",
        label="tab:pilot-clause-summary",
        colspec=r">{\raggedright\arraybackslash}Xr",
        header=["Metric", "Value"],
        rows=[
            ["Pilot ordinances", _fmt_int(pilot_summary["doc_count"])],
            ["Enriched-clause ordinances", _fmt_int(pilot_summary["enriched_clause_doc_count"])],
            ["Enriched clauses", _fmt_int(pilot_summary["enriched_clause_row_count"])],
            ["Clause clusters", _fmt_int(pilot_summary["clause_cluster_count"])],
            ["Regulatory style families", _fmt_int(pilot_summary["regulatory_style_family_count"])],
            ["Policy-signature clusters", _fmt_int(pilot_summary["policy_signature_cluster_count"])],
            ["Policy-signature silhouette", _fmt_float(pilot_summary["policy_signature_silhouette_score"])],
        ],
    )

    pilot_style_table = _table_block(
        caption="Current pilot governance-style families on the enriched clause surface. `Direct`, `Discretion`, and `Except./Cond.` are mean shares of operative clauses, so these rows describe regulatory composition rather than raw counts. They need not sum to one because other operative categories are omitted from this compact display.",
        label="tab:pilot-style-families",
        colspec=r">{\raggedright\arraybackslash}Xrrrr",
        header=["Family", "Docs", "Direct", "Discretion", "Except./Cond."],
        rows=_rows_from_df(
            pilot_style_summary.sort_values("ordinance_count", ascending=False).head(4),
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

    pilot_clause_table = _table_block(
        caption="Largest current clause families on the pilot enriched surface. `Clauses` counts atomic enriched clauses in the family, while `Docs` counts pilot ordinances touched by that family. Large clause counts may partly reflect long enumerated lists, so this table is best read as a coverage map of the clause space rather than a direct ranking of substantive importance.",
        label="tab:pilot-largest-clauses",
        colspec=r"r>{\raggedright\arraybackslash}Xrr",
        header=["ID", "Clause family", "Clauses", "Docs"],
        rows=_rows_from_df(
            pilot_largest_clause_for_table.head(10),
            columns=["cluster_id", "cluster_label", "clause_count", "ordinance_count"],
            formatters={
                "cluster_id": _fmt_int,
                "cluster_label": lambda x: _tex_escape(_clip(x, 54)),
                "clause_count": _fmt_int,
                "ordinance_count": _fmt_int,
            },
        ),
    )

    pilot_coverage_table = _table_block(
        caption="Pilot clause-enrichment coverage by region after vintage adjustment. `Doc coverage` is the adjusted share of pilot ordinances in that region with enriched-clause coverage, while `Clause share` and `Section share` show how much of the clause and section mass is currently covered. This table is mainly a bias diagnostic.",
        label="tab:pilot-clause-coverage",
        colspec="lrrr",
        header=["Region", "Doc coverage", "Clause share", "Section share"],
        rows=_rows_from_df(
            pilot_coverage_region.sort_values("has_clause_enrichment__vintage_adjusted", ascending=False),
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

    policy_mode_table = _table_block(
        caption="Document-level policy-signature aggregation comparison on the pilot enriched clause surface. `Items` is the number of ordinances entering the typology, `Reportable clusters` counts clusters large enough to interpret, and `Silhouette` is a geometric tightness measure. Higher silhouette means cleaner clustering, but reportable-cluster count matters for whether the surface yields useful document families.",
        label="tab:pilot-policy-modes",
        colspec="lrrr",
        header=["Aggregation mode", "Items", "Reportable clusters", "Silhouette"],
        rows=[
            [
                "Raw atomic clause shares",
                _fmt_int(raw_policy_summary["item_count"]),
                _fmt_int(raw_policy_summary["reportable_cluster_count"]),
                _fmt_float(raw_policy_summary["silhouette_score"]),
            ],
            [
                "Section-normalized clause shares",
                _fmt_int(section_norm_policy_summary["item_count"]),
                _fmt_int(section_norm_policy_summary["reportable_cluster_count"]),
                _fmt_float(section_norm_policy_summary["silhouette_score"]),
            ],
            [
                "Section presence only",
                _fmt_int(section_presence_policy_summary["item_count"]),
                _fmt_int(section_presence_policy_summary["reportable_cluster_count"]),
                _fmt_float(section_presence_policy_summary["silhouette_score"]),
            ],
        ],
    )

    top_pilot_labels = [
        _tex_escape(_clip(label, 70))
        for label in (pilot_summary.get("policy_signature_top_labels") or [])
        if str(label).strip()
    ]
    stable_surface_count = int((coverage_audit.get("summary") or {}).get("stable_surface_count", 0))
    emerging_surface_count = int((coverage_audit.get("summary") or {}).get("emerging_surface_count", 0))
    partial_surface_count = int((coverage_audit.get("summary") or {}).get("partial_surface_count", 0))
    deferred_surface_count = int((coverage_audit.get("summary") or {}).get("deferred_surface_count", 0))

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
\title{{Frontier LLM v2 Integrated Checkpoint}}
\author{{Codex research build}}
\date{{March 11, 2026}}
\begin{{document}}
\maketitle

\section*{{Overview}}
\begin{{itemize}}[leftmargin=1.2em]
\item This checkpoint combines the current full-sample section backbone with the pilot clause frontier.
\item Stable surfaces: {_fmt_int(stable_surface_count)}. Emerging: {_fmt_int(emerging_surface_count)}. Partial: {_fmt_int(partial_surface_count)}. Deferred: {_fmt_int(deferred_surface_count)}.
\item Read the full-sample section and coherence surfaces as the paper backbone. Read the clause surfaces as pilot-scale but already informative.
\end{{itemize}}

{full_summary_table}

\section{{Data and Pipeline}}
\begin{{enumerate}}[leftmargin=1.4em]
\item full-ordinance section segmentation,
\item document purpose extraction,
\item section-level motive labeling,
\item OpenAI section embeddings,
\item section clustering and cluster labeling,
\item clause extraction and enrichment on the pilot frontier,
\item governance-style and policy-signature aggregation on the enriched pilot surface.
\end{{enumerate}}
\begin{{itemize}}[leftmargin=1.2em]
\item The corpus is PID6-linked and explicitly cross-sectional.
\item All region and jurisdiction-type comparisons are interpreted through vintage-adjusted summaries, not as within-place change.
\end{{itemize}}

{vintage_table}

\section{{Full-Sample Architecture Story}}
\begin{{itemize}}[leftmargin=1.2em]
\item This is the strongest national semantic result.
\item The full sample currently has {_fmt_int(full_summary["section_cluster_count"])} labeled non-noise section families.
\item Read the tables as vintage-adjusted contrasts and the UMAPs as semantic maps of section families.
\end{{itemize}}

{_figure_block("figures/section_semantic_hdbscan_umap.png", "OpenAI section manifold on the full-sample architecture surface.")}

{region_table}

{level_table}

\section{{Purpose, Coherence, and Complexity}}
\begin{{itemize}}[leftmargin=1.2em]
\item High level: the purpose surface records what each ordinance says it is trying to do. It is a weighted multi-label object, not a single-label classification.
\item Exact construction: the LLM can emit multiple purpose rows per ordinance. Those rows are normalized within ordinance to sum to one, averaged within group-by-vintage cells, and then reweighted to the national vintage distribution.
\item Table formula: `Adj. share` is the vintage-adjusted average purpose share for that group. `Adj. \% diff` is `100 * (Adj. share / national purpose share - 1)`. Small values are normal because the underlying shares are normalized and already add to one.
\item Heatmap formula: every cell reports the same percent-difference object. Zero means the group is at the national baseline for that purpose, positive values mean overrepresentation, and negative values mean underrepresentation.
\item What to notice: use the tables to identify the largest group-purpose shifts and the heatmaps to assess whether those shifts are isolated or part of a broader purpose profile. The coherence tables then ask whether those document-level purposes line up with section-level semantic structure.
\end{{itemize}}

{purpose_table}

{purpose_level_table}

{_figure_block("figures/purpose_lift_by_region_heatmap__full.png", "Full-sample purpose heterogeneity by region. Construction: normalize LLM purpose weights within ordinance, average them within region-by-vintage cells, and reweight those means to the national vintage mix. Each cell reports `100 * (group share / national share - 1)`. Values near zero mean the region is close to the national baseline for that purpose; `+3.0` means three percent above baseline and `-3.0` means three percent below. Read rows across for a region's overall purpose profile and columns down for where each purpose family is relatively concentrated.", "0.86\\textwidth")}

{_figure_block("figures/purpose_lift_by_level_heatmap__full.png", "Full-sample purpose heterogeneity by jurisdiction type. This figure is built exactly like the regional heatmap, but the grouping variable is jurisdiction type rather than region. Values near zero indicate near-baseline purpose weights, while larger positive or negative values indicate institutional-form heterogeneity after holding vintage composition fixed.", "0.90\\textwidth")}

{purpose_alignment_table}

{coherence_table}

{complexity_effect_table}

\begin{{itemize}}[leftmargin=1.2em]
\item Mean purpose/section alignment share: {_fmt_float(full_summary["mean_alignment_share"])}.
\item Mean non-`other` alignment share: {_fmt_float(full_summary["mean_non_other_alignment_share"])}.
\item Adjusted complexity slope: {_fmt_float(full_summary["complexity_alignment_adjusted_coef"])} with p-value {_fmt_float(full_summary["complexity_alignment_adjusted_pvalue"])}.
\item The right interpretation is residual coherence versus fragmentation, not a mechanical “more complexity means less coherence” claim.
\end{{itemize}}

\section{{Surface Comparison and Retrieval}}
\begin{{itemize}}[leftmargin=1.2em]
\item The current automated comparison favors the OpenAI section surface.
\item The main use of this section is to show that the geometry is strong enough for clustering, retrieval, and regional homophily.
\end{{itemize}}

{surface_table}

{neighbor_table}

\section{{Architecture Families}}
\begin{{itemize}}[leftmargin=1.2em]
\item This is an exploratory coarse-grained layer above individual section clusters.
\item Use it to distinguish common ordinance templates from unusual organizations.
\end{{itemize}}

{architecture_table}

{_figure_block("figures/section_semantic_hdbscan_local_baseline_umap.png", "Local baseline section manifold for comparison with the OpenAI section surface.")}

\section{{Pilot Clause Frontier}}
\begin{{itemize}}[leftmargin=1.2em]
\item The clause side remains the deeper frontier rather than the national backbone.
\item It is now far enough along to support governance-style summaries, clause-family analysis, and a provisional document typology.
\end{{itemize}}

{pilot_clause_summary}

{pilot_style_table}

{pilot_clause_table}

{pilot_coverage_table}

Current leading pilot policy-signature labels:
\begin{{itemize}}[leftmargin=1.2em]
{''.join(f'\\item {label}\n' for label in top_pilot_labels) if top_pilot_labels else '\\item No stable pilot policy-signature labels yet.\n'}
\end{{itemize}}

{policy_mode_table}
\begin{{itemize}}[leftmargin=1.2em]
\item Aggregation choice matters.
\item Raw atomic clause shares cluster tighter.
\item Section-normalized clause shares are less dominated by long use lists and definition sections, so they are a better candidate for the long-run document typology.
\end{{itemize}}

\section{{Baseline Comparison and Interpretation}}
The original local baseline remains useful as a benchmark, but it is no longer the canonical semantic layer. Current shared-pilot comparison:
\begin{{itemize}}[leftmargin=1.4em]
\item shared documents: {_fmt_int(baseline_summary["shared_document_count"])}
\item primary motive agreement: {_fmt_float(baseline_summary["primary_motive_agreement_rate"])}
\item dominant instrument agreement: {_fmt_float(baseline_summary["dominant_instrument_agreement_rate"])}
\item operative dominant instrument agreement: {_fmt_float(baseline_summary["operative_dominant_instrument_agreement_rate"])}
\end{{itemize}}

\section{{Current Paper Direction}}
\begin{{enumerate}}[leftmargin=1.4em]
\item national cross-section of linked zoning ordinances,
\item section-level legal architecture as the main empirical surface,
\item regional and jurisdiction-type differences as the main descriptive contribution,
\item coherence versus fragmentation as the deeper legal-form frontier,
\item pilot clause-side governance style as the next semantic layer beyond architecture.
\end{{enumerate}}

\section{{Next Steps}}
\begin{{itemize}}[leftmargin=1.4em]
\item keep harvesting pilot clause-enrichment coverage and refresh the enriched pilot surface again once the raw checkpoint materially advances,
\item rerun governance-style and policy-signature summaries on that broader clause coverage,
\item regenerate clean joint region-plus-level coefficient surfaces after the latest section reclustering so the architecture section can use both labeled contrasts and clean model tables,
\item promote the strongest stable results into a direct paper draft rather than another checkpoint memo.
\end{{itemize}}

\end{{document}}
"""

    output_path.write_text(tex, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an integrated Frontier LLM v2 TeX writeup from full-sample and pilot research outputs.")
    parser.add_argument("--run-root", required=True, help="Run root under artifacts/runs/frontier_llm_v2/...")
    parser.add_argument("--output-path", help="Optional explicit output .tex path")
    args = parser.parse_args()

    out = build_integrated_writeup_tex(
        run_root=Path(args.run_root),
        output_path=Path(args.output_path) if args.output_path else None,
    )
    print(out)


if __name__ == "__main__":
    main()
