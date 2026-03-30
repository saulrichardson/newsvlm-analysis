#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

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


def _resolve_sample(run_root: Path, sample: Literal["pilot", "full"] | None) -> Literal["pilot", "full"]:
    if sample is not None:
        target = str(sample).strip().lower()
        if target not in {"pilot", "full"}:
            raise SystemExit(f"Unsupported sample: {sample}")
        return target  # type: ignore[return-value]
    available = sorted(path.stem.removeprefix("research_summary__") for path in (run_root / "research").glob("research_summary__*.json"))
    if len(available) == 1 and available[0] in {"pilot", "full"}:
        return available[0]  # type: ignore[return-value]
    if not available:
        raise SystemExit(f"No research_summary__*.json files found under {run_root / 'research'}")
    raise SystemExit(
        "Multiple research summary files found. Pass --sample explicitly so the entrypoint does not mix pilot/full surfaces: "
        + ", ".join(available)
    )


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


def _figure_block(path: str, caption: str, width: str = "0.92\\textwidth") -> str:
    return rf"""
\IfFileExists{{{path}}}{{
\begin{{figure}}[H]
  \centering
  \includegraphics[width={width}]{{{path}}}
  \caption{{{caption}}}
\end{{figure}}
}}{{}}
""".strip()


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


def build_entrypoint_tex(
    *,
    run_root: Path,
    output_path: Path | None = None,
    sample: Literal["pilot", "full"] | None = "pilot",
) -> Path:
    run_root = run_root.resolve()
    reports_dir = run_root / "reports"
    figures_dir = reports_dir / "figures"
    research_dir = run_root / "research"
    output_path = (output_path or (reports_dir / "frontier_llm_v2_entrypoint.tex")).resolve()
    target_sample = _resolve_sample(run_root, sample)
    research_summary = _load_json(research_dir / f"research_summary__{target_sample}.json")
    report_summary = _load_json(run_root / "story" / "story_summary.json")
    review_summary = _load_json(run_root / "review" / "review_summary.json")
    semantics_summary = _load_json(run_root / "semantics" / "semantics_summary.json")
    candidate_storylines = research_dir / f"candidate_storylines__{target_sample}.md"
    if not candidate_storylines.is_file():
        candidate_storylines = None

    style_summary_path = research_dir / "style_families" / f"regulatory_style_family_summary__{target_sample}.parquet"
    style_summary = pd.read_parquet(style_summary_path) if style_summary_path.is_file() else pd.DataFrame()
    clause_lifts_path = research_dir / "clause_clusters" / f"top_region_clause_cluster_lifts__{target_sample}.parquet"
    clause_lifts = pd.read_parquet(clause_lifts_path) if clause_lifts_path.is_file() else pd.DataFrame()
    purpose_region_path = research_dir / "purposes" / f"weighted_purpose_by_region_vintage_adjusted__{target_sample}.parquet"
    purpose_region = pd.read_parquet(purpose_region_path) if purpose_region_path.is_file() else pd.DataFrame()
    purpose_region_top = _top_purpose_lifts(purpose_region, group_col="jurisdiction_region", top_n=2)
    purpose_level_path = research_dir / "purposes" / f"weighted_purpose_by_level_vintage_adjusted__{target_sample}.parquet"
    purpose_level = pd.read_parquet(purpose_level_path) if purpose_level_path.is_file() else pd.DataFrame()
    purpose_level_top = _top_purpose_lifts(purpose_level, group_col="jurisdiction_level", top_n=2)
    purpose_alignment_path = research_dir / "purposes" / f"purpose_alignment_by_primary_motive__{target_sample}.parquet"
    purpose_alignment = pd.read_parquet(purpose_alignment_path) if purpose_alignment_path.is_file() else pd.DataFrame()
    positive_cases_path = research_dir / "coherence" / f"coherence_positive_residuals__{target_sample}.parquet"
    negative_cases_path = research_dir / "coherence" / f"coherence_negative_residuals__{target_sample}.parquet"
    positive_cases = pd.read_parquet(positive_cases_path) if positive_cases_path.is_file() else pd.DataFrame()
    negative_cases = pd.read_parquet(negative_cases_path) if negative_cases_path.is_file() else pd.DataFrame()

    summary_table = _table_block(
        caption="Frontier LLM v2 pilot summary at the current checkpoint. This table is an inventory of the semantic pilot, showing how much of the pilot is covered by the section and clause layers and where the main quantitative frontier currently sits.",
        label="tab:v2-pilot-summary",
        colspec=r">{\raggedright\arraybackslash}Xr",
        header=["Metric", "Value"],
        rows=[
            ["Pilot ordinances", _fmt_int(research_summary["doc_count"])],
            ["Section semantic clusters", _fmt_int(research_summary["section_cluster_count"])],
            ["Enriched-clause ordinances", _fmt_int(research_summary["enriched_clause_doc_count"])],
            ["Enriched clauses", _fmt_int(research_summary["enriched_clause_row_count"])],
            ["Clause clusters", _fmt_int(research_summary["clause_cluster_count"])],
            ["Regulatory style families", _fmt_int(research_summary["regulatory_style_family_count"])],
            ["Policy-signature documents", _fmt_int(research_summary["policy_signature_item_count"])],
            ["OpenAI section embedding winner", _tex_escape(research_summary["embedding_surface_winner"])],
            ["Complexity/alignment raw correlation", _fmt_float(research_summary["complexity_alignment_correlation"])],
            ["Complexity/alignment adjusted slope", _fmt_float(research_summary["complexity_alignment_adjusted_coef"])],
        ],
    )

    style_rows = []
    if not style_summary.empty:
        for row in style_summary.sort_values("ordinance_count", ascending=False).head(4).itertuples(index=False):
            style_rows.append(
                [
                    _tex_escape(_clip(getattr(row, "cluster_label", ""), 52)),
                    _fmt_int(getattr(row, "ordinance_count", 0)),
                    _fmt_float(getattr(row, "mean_direct_rule_share", 0.0)),
                    _fmt_float(getattr(row, "mean_discretion_share", 0.0)),
                    _fmt_float(getattr(row, "mean_exception_or_condition_share", 0.0)),
                ]
            )
    style_table = _table_block(
        caption="Current regulatory style families on the enriched-clause pilot surface. `Direct`, `Discretion`, and `Except./Cond.` are mean shares of operative clauses, so the rows describe regulatory composition rather than raw clause volume.",
        label="tab:v2-style-families",
        colspec=r">{\raggedright\arraybackslash}Xrrrr",
        header=["Family", "Docs", "Direct", "Discretion", "Except./Cond."],
        rows=style_rows,
    )

    clause_rows = []
    if not clause_lifts.empty:
        for row in clause_lifts.head(8).itertuples(index=False):
            clause_rows.append(
                [
                    _tex_escape(getattr(row, "jurisdiction_region", "")),
                    _tex_escape(_clip(getattr(row, "cluster_label", ""), 42)),
                    _fmt_float(getattr(row, "cluster_clause_share_lift_vs_overall_vintage_adjusted", 0.0)),
                ]
            )
    clause_table = _table_block(
        caption="Top vintage-adjusted regional clause-cluster lifts on the current partial enriched surface. `Lift` compares the region-specific clause-family share to the pilot-wide baseline after vintage adjustment, so values above one identify clause families that are relatively overrepresented in that region.",
        label="tab:v2-clause-lifts",
        colspec=r">{\raggedright\arraybackslash}l>{\raggedright\arraybackslash}Xr",
        header=["Region", "Clause family", "Lift"],
        rows=clause_rows,
    )

    coherence_rows = []
    for name, frame in [("Positive residual", positive_cases), ("Negative residual", negative_cases)]:
        if frame.empty:
            continue
        row = frame.iloc[0]
        coherence_rows.append(
            [
                _tex_escape(name),
                _tex_escape(_clip(row.get("ordinance_doc_id", ""), 44)),
                _tex_escape(_clip(row.get("doc_primary_motive", ""), 24)),
                _fmt_float(row.get("purpose_alignment_residual_vintage_level", 0.0)),
            ]
        )
    coherence_table = _table_block(
        caption="Illustrative coherence residual cases from the pilot. Residuals are observed semantic alignment minus model-predicted alignment, so positive values indicate unexpectedly coherent ordinances and negative values indicate unexpectedly fragmented ones.",
        label="tab:v2-coherence-cases",
        colspec=r">{\raggedright\arraybackslash}l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}lr",
        header=["Case", "Ordinance", "Doc motive", "Residual"],
        rows=coherence_rows,
    )

    purpose_table = _table_block(
        caption="Top two document-purpose deviations by region on the current sample. Construction: ordinance-level LLM purpose weights are normalized to sum to one, averaged within region-by-vintage cells, and then reweighted to the sample-wide vintage mix. Adj. share is the adjusted mean share. Adj. percent diff compares that adjusted share with the sample-wide benchmark for the same purpose. Because the normalized purpose shares already sum to one, meaningful heterogeneity often appears as single-digit percentage deviations.",
        label="tab:v2-purpose-lifts",
        colspec="llrr",
        header=["Region", "Purpose", "Adj. share", "Adj. % diff"],
        rows=[
            [
                _tex_escape(_category_label(row.get("jurisdiction_region", ""))),
                _tex_escape(_category_label(row.get("motive", ""))),
                _fmt_float(row.get("purpose_weight__vintage_adjusted", 0.0)),
                _fmt_pct_diff_from_lift(row.get("purpose_weight_lift_vs_overall_vintage_adjusted", 0.0)),
            ]
            for row in purpose_region_top.to_dict(orient="records")
        ],
    )
    purpose_level_table = _table_block(
        caption="Top two document-purpose deviations by jurisdiction level on the current sample. Read Adj. share as the adjusted average purpose share within that level and Adj. percent diff exactly as in the regional table. Small numbers are expected because the purpose surface is normalized and multi-label rather than winner-take-all.",
        label="tab:v2-purpose-level-lifts",
        colspec="llrr",
        header=["Level", "Purpose", "Adj. share", "Adj. % diff"],
        rows=[
            [
                _tex_escape(_category_label(row.get("jurisdiction_level", ""))),
                _tex_escape(_category_label(row.get("motive", ""))),
                _fmt_float(row.get("purpose_weight__vintage_adjusted", 0.0)),
                _fmt_pct_diff_from_lift(row.get("purpose_weight_lift_vs_overall_vintage_adjusted", 0.0)),
            ]
            for row in purpose_level_top.to_dict(orient="records")
        ],
    )
    purpose_alignment_rows = []
    for row in purpose_alignment.sort_values("ordinance_count", ascending=False).head(6).to_dict(orient="records"):
        purpose_alignment_rows.append(
            [
                _tex_escape(_category_label(row.get("doc_primary_motive", ""))),
                _fmt_int(row.get("ordinance_count", 0)),
                _fmt_float(row.get("mean_non_other_alignment_share", 0.0)),
                _fmt_float(row.get("filtered_match_rate", 0.0)),
            ]
        )
    purpose_alignment_table = _table_block(
        caption="Alignment between document-level purposes and weighted section-level motives. Construction: a dedicated `gpt-5-mini` document-purpose pass assigns each ordinance one primary purpose, then section-level motives are aggregated within the same ordinance using section word counts. `Mean alignment` is the average filtered section-word share that lands on the document's primary motive. `Exact match share` is the fraction of ordinances in that purpose family whose largest filtered section-motive bucket equals the document-level purpose. Example: a value of `0.535` would mean that about 53.5 percent of the filtered section-word mass lines up with the ordinance's primary document purpose.",
        label="tab:v2-purpose-alignment",
        colspec="lrrr",
        header=["Primary purpose", "Docs", "Mean alignment", "Exact match share"],
        rows=purpose_alignment_rows,
    )
    _plot_purpose_lift_heatmap(
        purpose_region,
        group_col="jurisdiction_region",
        out_path=figures_dir / f"purpose_lift_by_region_heatmap__{target_sample}.png",
        title=f"{target_sample.title()} Document-Purpose Deviations by Region (%)",
    )
    _plot_purpose_lift_heatmap(
        purpose_level,
        group_col="jurisdiction_level",
        out_path=figures_dir / f"purpose_lift_by_level_heatmap__{target_sample}.png",
        title=f"{target_sample.title()} Document-Purpose Deviations by Jurisdiction Level (%)",
    )

    candidate_rel = Path("../research") / candidate_storylines.name if candidate_storylines is not None else Path("")

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{float}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\usepackage{{caption}}
\captionsetup[table]{{position=bottom}}
\captionsetup[figure]{{position=bottom}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\title{{Frontier LLM v2 Pilot Entry Point}}
\author{{Codex research build}}
\date{{March 10, 2026}}
\begin{{document}}
\maketitle

\section*{{What This Is}}
This document is the entry point into the current LLM-native pilot built on the frozen full-code corpus. The current checkpoint is still a pilot, not the full 728-document semantic run. The section-side semantic surface is stable; the clause-side surface is already substantive but remains coverage-limited because full clause enrichment is still running in the background.

{summary_table}

\section*{{Canonical Pilot Design}}
\begin{{itemize}}[leftmargin=1.2em]
\item Canonical semantic inference uses \texttt{{gpt-5-mini}} for section segmentation, document purpose, section motive, clause extraction, clause enrichment, and cluster labeling.
\item Canonical section geometry uses \texttt{{text-embedding-3-large}} with pooled segment embeddings for oversized sections.
\item Cross-sectional claims are vintage-aware: the pilot is treated as a vintage-adjusted cross-section rather than a within-place panel.
\item The strongest current descriptive surfaces are section clusters, retrieval/prototype structure, coherence residuals, regulatory style families, and clause-family regional contrasts.
\end{{itemize}}

\section*{{Stable Findings Now}}
\begin{{itemize}}[leftmargin=1.2em]
\item OpenAI section embeddings currently win over the local baseline on interpretability-oriented diagnostics.
\item Regional section-cluster contrasts remain meaningful after vintage adjustment.
\item The raw complexity/coherence relationship weakens sharply once vintage and jurisdiction level are controlled; the more interesting frontier is the residual distinction between unexpectedly coherent and unexpectedly fragmented ordinances.
\item Clause families are now interpretable enough to support substantive pilot comparisons, even though the enriched-clause surface is still partial.
\end{{itemize}}

{style_table}

\section*{{Document Purposes}}
\begin{{itemize}}[leftmargin=1.2em]
\item High level: the purpose surface records what ordinances say they are trying to do. It is a weighted multi-label object, so one ordinance can load on several purpose families.
\item Exact construction: the LLM can emit multiple purpose rows per ordinance. Those raw purpose weights are normalized within ordinance to sum to one, averaged within group-by-vintage cells, and then reweighted to the sample-wide vintage distribution.
\item Table formula: `Adj. share` is the vintage-adjusted average purpose share for that group. `Adj. \% diff` is `100 * (Adj. share / sample-wide purpose share - 1)`. Small values are expected because the underlying shares already sum to one within each ordinance.
\item Heatmap formula: every cell reports the same percent-difference object. Zero means the group is at the sample-wide baseline for that purpose, positive values indicate overrepresentation, and negative values indicate underrepresentation.
\item Alignment table: this is a different object. It asks whether ordinances whose primary document purpose is, for example, Environmental Protection or Growth Management actually devote more section-level semantic mass to the same motive family.
\end{{itemize}}

{purpose_table}

{purpose_level_table}

{purpose_alignment_table}

{coherence_table}

\section*{{Clause-Side Frontier}}
The clause-side layer now covers {_fmt_int(research_summary["enriched_clause_doc_count"])} pilot ordinances with {_fmt_int(research_summary["enriched_clause_row_count"])} enriched clauses and {_fmt_int(research_summary["clause_cluster_count"])} clause clusters. That is enough to make the governance-style and clause-family surfaces real, but not yet enough to treat enriched-clause typology as fully mature.

{clause_table}

\section*{{Most Useful Current Storylines}}
\begin{{enumerate}}[leftmargin=1.4em]
\item Regional legal architecture rather than raw topic prevalence.
\item Coherent versus fragmented ordinance form as a legal-style distinction.
\item Governance-style families built from direct-rule, discretion, threshold, and exception structure.
\item Purpose-to-instrument slippage as a secondary explanatory frontier.
\end{{enumerate}}

\section*{{Figures}}
{_figure_block("figures/section_semantic_hdbscan_umap.png", "OpenAI section semantic manifold for the current pilot.", "0.88\\textwidth")}

{_figure_block("figures/section_semantic_hdbscan_local_baseline_umap.png", "Local baseline section manifold for comparison.", "0.88\\textwidth")}

{_figure_block("figures/clause_instrument_hdbscan_umap.png", "Clause-instrument manifold on the current enriched-clause surface.", "0.88\\textwidth")}

{_figure_block(f"figures/purpose_lift_by_region_heatmap__{target_sample}.png", "Vintage-adjusted document-purpose deviations by region on the current sample. Construction: normalize LLM purpose weights within ordinance, average them within region-by-vintage cells, and reweight those means to the sample-wide vintage mix. Each cell reports `100 * (group share / sample-wide share - 1)`. Values near zero mean the region is close to the sample-wide baseline for that purpose; `+3.0` means three percent above baseline and `-3.0` means three percent below. Read rows across to see the overall regional purpose profile and columns down to see which regions disproportionately contribute to a given purpose family.", "0.86\\textwidth")}

{_figure_block(f"figures/purpose_lift_by_level_heatmap__{target_sample}.png", "Vintage-adjusted document-purpose deviations by jurisdiction level. This figure is built exactly like the regional heatmap, but across government forms rather than regions. Values near zero indicate near-baseline purpose weights, while larger positive or negative values indicate institutional-form heterogeneity after holding vintage composition fixed.", "0.86\\textwidth")}

{_figure_block("figures/motive_by_region_heatmap.png", "Regional motive shares from the current pilot outputs.", "0.78\\textwidth")}

{_figure_block("figures/instrument_by_region_heatmap.png", "Regional substantive instrument shares from the current pilot outputs.", "0.78\\textwidth")}

\section*{{What Is Still Provisional}}
\begin{{itemize}}[leftmargin=1.2em]
\item The enriched-clause policy-signature layer is still coverage-limited.
\item Clause-side regional differences should be treated as emerging, not yet final.
\item Review packets exist, but the quality-gate file is still pending manual review completion rather than final pass/fail rates.
\end{{itemize}}

\section*{{Where To Read Next}}
\begin{{itemize}}[leftmargin=1.2em]
\item Research summary: \texttt{{../research/research\_summary\_\_pilot.json}}
\item Research memo: \texttt{{../research/research\_memo\_\_pilot.md}}
\item Candidate storylines: \texttt{{{_tex_escape(candidate_rel)}}}
\item Report bundle: \texttt{{report.md}}
\end{{itemize}}

\end{{document}}
"""

    output_path.write_text(tex.strip() + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a v2 pilot entry-point TeX note.")
    parser.add_argument("--run-root", required=True, help="Frontier LLM v2 run root.")
    parser.add_argument("--output-path", default="", help="Optional explicit output path.")
    parser.add_argument("--sample", choices=["pilot", "full"], default="pilot")
    args = parser.parse_args()
    output = build_entrypoint_tex(
        run_root=Path(args.run_root),
        output_path=Path(args.output_path) if args.output_path else None,
        sample=args.sample,
    )
    print(f"built_frontier_llm_v2_entrypoint_tex output={output}")


if __name__ == "__main__":
    main()
