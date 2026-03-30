#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


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


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _titleish(text: str) -> str:
    raw = text.replace("_", " ")
    mapping = {
        "1920 1949": "1920--1949",
        "1950s": "1950s",
        "1960s": "1960s",
        "1970 plus": "1970+",
        "city and county": "City and county",
    }
    lowered = raw.lower()
    if lowered in mapping:
        return mapping[lowered]
    return raw.title()


def _bridge_term_label(term: Any) -> str:
    raw = str(term or "")
    replacements = {
        "structural_complexity_index": "Structural complexity index (1 SD)",
        "dependency_complexity_index": "Dependency complexity index (1 SD)",
        "linguistic_complexity_index": "Linguistic complexity index (1 SD)",
        "section_kind_entropy": "Section-kind entropy",
        "flesch_kincaid_grade_proxy": "Flesch--Kincaid grade proxy",
        "formal_reference_mentions_per_1000_words": "Formal references per 1,000 words",
        "dependency_trigger_mentions_per_1000_words": "Dependency triggers per 1,000 words",
        "map_reference_mentions_per_1000_words": "Map references per 1,000 words",
        "board_reference_mentions_per_1000_words": "Board references per 1,000 words",
    }
    if raw == "Intercept":
        return "Intercept (borough, 1920--1949 baseline)"
    if raw in replacements:
        return replacements[raw]
    if raw.startswith("C(vintage_bin)[T.") and raw.endswith("]"):
        value = raw[len("C(vintage_bin)[T.") : -1]
        return f"Vintage: {_titleish(value)} vs 1920--1949"
    if raw.startswith("C(jurisdiction_level)[T.") and raw.endswith("]"):
        value = raw[len("C(jurisdiction_level)[T.") : -1]
        return f"Jurisdiction type: {_titleish(value)} vs Borough"
    if raw.startswith("C(jurisdiction_region)[T.") and raw.endswith("]"):
        value = raw[len("C(jurisdiction_region)[T.") : -1]
        return f"Region: {_titleish(value)} vs Midwest"
    return _titleish(raw)


def _bridge_model_label(label: Any) -> str:
    raw = str(label or "")
    mapping = {
        "direct_rule": "Direct-rule share",
        "discretion": "Discretion share",
        "exception_or_condition": "Exception/condition share",
        "coherence_subcomponents": "Coherence model",
        "direct_rule_subcomponents": "Direct-rule model",
        "threshold_subcomponents": "Threshold share model",
        "exception_or_condition_subcomponents": "Exception/condition model",
        "discretion_subcomponents": "Discretion model",
    }
    return mapping.get(raw, _titleish(raw))


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
{' & '.join(header)} \\\\
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


def build_combined_writeup(*, complexity_run_root: Path, pilot_run_root: Path, output_path: Path | None = None) -> Path:
    complexity_run_root = complexity_run_root.resolve()
    pilot_run_root = pilot_run_root.resolve()
    reports_dir = complexity_run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_path or (reports_dir / "legal_complexity_frontier_combined.tex")).resolve()

    complexity_summary = _load_json(complexity_run_root / "manifests" / "summary.json")
    pilot_summary = _load_json(pilot_run_root / "research" / "research_summary__pilot.json")

    complexity_region = _read_parquet(complexity_run_root / "complexity" / "complexity_by_region_vintage_adjusted.parquet")
    complexity_level = _read_parquet(complexity_run_root / "complexity" / "complexity_by_level_vintage_adjusted.parquet")
    complexity_effects = _read_parquet(complexity_run_root / "complexity" / "top_complexity_region_level_effects.parquet")
    family_summary = _read_parquet(complexity_run_root / "complexity" / "complexity_family_summary.parquet")
    dependency_profile_summary = _read_parquet(complexity_run_root / "complexity" / "dependency_profile_summary.parquet")
    dependency_effects = _read_parquet(complexity_run_root / "complexity" / "top_dependency_component_effects.parquet")
    dependency_region = _read_parquet(complexity_run_root / "complexity" / "dependency_components_by_region_vintage_adjusted.parquet")
    overall_residuals = _read_parquet(complexity_run_root / "complexity" / "overall_complexity__top_positive_residuals.parquet")
    dependency_residuals = _read_parquet(complexity_run_root / "complexity" / "dependency_complexity__top_positive_residuals.parquet")
    exemplars = _read_parquet(complexity_run_root / "complexity" / "complexity_component_exemplars.parquet")
    bridge_coherence = _read_parquet(complexity_run_root / "bridge" / "pilot_coherence_models.parquet")
    bridge_governance = _read_parquet(complexity_run_root / "bridge" / "pilot_governance_models.parquet")
    bridge_subcomponents = _read_parquet(complexity_run_root / "bridge" / "pilot_subcomponent_bridge_models.parquet")

    section_region = _read_parquet(pilot_run_root / "research" / "section_clusters" / "top_region_contrasts_vintage_adjusted__pilot.parquet")
    style_summary = _read_parquet(pilot_run_root / "research" / "style_families" / "regulatory_style_family_summary__pilot.parquet")
    memo_path = complexity_run_root / "reports" / "legal_complexity_memo.md"
    memo_text = memo_path.read_text(encoding="utf-8") if memo_path.is_file() else ""

    region_rows = []
    for row in complexity_region.sort_values("overall_legal_complexity_index__vintage_adjusted", ascending=False).itertuples(index=False):
        region_rows.append(
            [
                _tex_escape(getattr(row, "jurisdiction_region", "")),
                _fmt_float(getattr(row, "structural_complexity_index__vintage_adjusted", 0.0)),
                _fmt_float(getattr(row, "dependency_complexity_index__vintage_adjusted", 0.0)),
                _fmt_float(getattr(row, "linguistic_complexity_index__vintage_adjusted", 0.0)),
                _fmt_float(getattr(row, "overall_legal_complexity_index__vintage_adjusted", 0.0)),
            ]
        )
    region_table = _table_block(
        caption="Vintage-adjusted legal complexity by Census region on the full 728-document sample. Each component is standardized, so zero is the sample mean, positive values indicate above-average complexity on that component, and negative values indicate below-average complexity. `Overall` is the combined legal-complexity index.",
        label="tab:complexity-region",
        colspec="lrrrr",
        header=["Region", "Structural", "Dependency", "Linguistic", "Overall"],
        rows=region_rows,
    )

    level_rows = []
    for row in complexity_level.sort_values("overall_legal_complexity_index__vintage_adjusted", ascending=False).itertuples(index=False):
        level_rows.append(
            [
                _tex_escape(getattr(row, "jurisdiction_level", "")),
                _fmt_int(getattr(row, "ordinance_count", 0)),
                _fmt_float(getattr(row, "structural_complexity_index__vintage_adjusted", 0.0)),
                _fmt_float(getattr(row, "dependency_complexity_index__vintage_adjusted", 0.0)),
                _fmt_float(getattr(row, "linguistic_complexity_index__vintage_adjusted", 0.0)),
                _fmt_float(getattr(row, "overall_legal_complexity_index__vintage_adjusted", 0.0)),
            ]
        )
    level_table = _table_block(
        caption="Vintage-adjusted legal complexity by jurisdiction type on the full sample. Read these standardized values the same way as Table~\\ref{tab:complexity-region}: positive values indicate more complexity than the sample average on that component after vintage adjustment, and negative values indicate less.",
        label="tab:complexity-level",
        colspec="lrrrrr",
        header=["Level", "Docs", "Structural", "Dependency", "Linguistic", "Overall"],
        rows=level_rows,
    )

    effect_rows = []
    for row in complexity_effects.itertuples(index=False):
        effect_rows.append(
            [
                _tex_escape(getattr(row, "target_col", "").replace("_", " ")),
                _tex_escape(getattr(row, "group_name", "")),
                _tex_escape(getattr(row, "group_value", "")),
                _fmt_float(getattr(row, "coefficient", 0.0)),
                _fmt_float(getattr(row, "pvalue", 0.0), 4),
            ]
        )
    effect_table = _table_block(
        caption="Most informative region, jurisdiction-type, and vintage effects from the decomposed full-sample complexity models. Each row comes from a regression for the named target. The coefficient is relative to an omitted baseline category within that model, so positive values mean the listed group is more complex than the baseline on that dimension and negative values mean less. Rows should be compared within a target, not across different targets.",
        label="tab:complexity-effects",
        colspec=r">{\raggedright\arraybackslash}Xllrr",
        header=["Target", "Group", "Value", "Coef.", "p"],
        rows=effect_rows[:14],
    )

    family_rows = []
    for row in family_summary.sort_values("ordinance_count", ascending=False).itertuples(index=False):
        family_rows.append(
            [
                _tex_escape(_clip(getattr(row, "cluster_label", ""), 48)),
                _fmt_int(getattr(row, "ordinance_count", 0)),
                _fmt_float(getattr(row, "mean_structural_complexity_index", 0.0)),
                _fmt_float(getattr(row, "mean_dependency_complexity_index", 0.0)),
                _fmt_float(getattr(row, "mean_linguistic_complexity_index", 0.0)),
            ]
        )
    family_table = _table_block(
        caption="Exploratory full-sample complexity families built from the decomposed component indices. These are descriptive groupings, not causal categories. The component columns report family means on standardized indices, so the sign and magnitude indicate whether a family is relatively structural, dependency-heavy, or linguistically complex compared with the sample average.",
        label="tab:complexity-families",
        colspec=r">{\raggedright\arraybackslash}Xrrrr",
        header=["Family", "Docs", "Structural", "Dependency", "Linguistic"],
        rows=family_rows,
    )

    overall_residual_rows = []
    for row in overall_residuals.head(10).itertuples(index=False):
        overall_residual_rows.append(
            [
                _tex_escape(_clip(getattr(row, "jurisdiction_display_name", ""), 28)),
                _tex_escape(getattr(row, "jurisdiction_region", "")),
                _tex_escape(getattr(row, "jurisdiction_level", "")),
                _fmt_float(getattr(row, "residual_value", 0.0)),
            ]
        )
    overall_residual_table = _table_block(
        caption="Most unexpectedly complex ordinances after controlling for vintage, region, and jurisdiction type. The residual is the ordinance's observed overall-complexity score minus the model-predicted score, so larger positive values indicate ordinances that are more complex than their vintage, region, and jurisdiction type would predict.",
        label="tab:overall-residuals",
        colspec=r">{\raggedright\arraybackslash}Xllr",
        header=["Jurisdiction", "Region", "Level", "Residual"],
        rows=overall_residual_rows,
    )

    dependency_residual_rows = []
    for row in dependency_residuals.head(10).itertuples(index=False):
        dependency_residual_rows.append(
            [
                _tex_escape(_clip(getattr(row, "jurisdiction_display_name", ""), 28)),
                _tex_escape(getattr(row, "jurisdiction_region", "")),
                _tex_escape(getattr(row, "jurisdiction_level", "")),
                _fmt_float(getattr(row, "residual_value", 0.0)),
            ]
        )
    dependency_residual_table = _table_block(
        caption="Most unexpectedly dependency-complex ordinances after controlling for vintage, region, and jurisdiction type. Larger positive residuals identify ordinances whose internal references, triggers, map citations, and board dependencies are unusually heavy relative to comparable ordinances.",
        label="tab:dependency-residuals",
        colspec=r">{\raggedright\arraybackslash}Xllr",
        header=["Jurisdiction", "Region", "Level", "Residual"],
        rows=dependency_residual_rows,
    )

    exemplar_rows = []
    exemplar_metrics = ["overall_legal_complexity_index", "reference_mentions_per_1000_words", "flesch_kincaid_grade_proxy"]
    for metric_name in exemplar_metrics:
        sub = exemplars[exemplars["metric_name"].astype(str).eq(metric_name)].head(3)
        for row in sub.itertuples(index=False):
            exemplar_rows.append(
                [
                    _tex_escape(metric_name.replace("_", " ")),
                    _tex_escape(_clip(getattr(row, "jurisdiction_display_name", ""), 26)),
                    _tex_escape(getattr(row, "jurisdiction_region", "")),
                    _fmt_float(getattr(row, "metric_value", 0.0)),
                ]
            )
    exemplar_table = _table_block(
        caption="Illustrative high-end exemplars for overall, dependency-burden, and linguistic legal complexity. These are descriptive examples of ordinances near the upper tail of each metric, included to make the abstract complexity dimensions concrete rather than to imply that one ordinance is universally the 'most complex.'",
        label="tab:complexity-exemplars",
        colspec=r">{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}Xlr",
        header=["Metric", "Jurisdiction", "Region", "Value"],
        rows=exemplar_rows,
    )

    dependency_profile_rows = []
    for row in dependency_profile_summary.sort_values("ordinance_count", ascending=False).itertuples(index=False):
        dependency_profile_rows.append(
            [
                _tex_escape(_clip(getattr(row, "cluster_label", ""), 48)),
                _fmt_int(getattr(row, "ordinance_count", 0)),
                _fmt_float(getattr(row, "mean_formal_reference_mentions_per_1000_words", 0.0)),
                _fmt_float(getattr(row, "mean_dependency_trigger_mentions_per_1000_words", 0.0)),
                _fmt_float(getattr(row, "mean_board_reference_mentions_per_1000_words", 0.0)),
            ]
        )
    dependency_profile_table = _table_block(
        caption="Dependency-profile families built from formal references, dependency triggers, board references, map references, and reference-bearing sections. The numeric columns are family means per 1,000 words, so they show which profiles are reference-dense, trigger-dense, or board-heavy relative to one another.",
        label="tab:dependency-profiles",
        colspec=r">{\raggedright\arraybackslash}Xrrrr",
        header=["Profile", "Docs", "Formal", "Triggers", "Boards"],
        rows=dependency_profile_rows,
    )

    dependency_effect_rows = []
    for row in dependency_effects.head(12).itertuples(index=False):
        dependency_effect_rows.append(
            [
                _tex_escape(str(getattr(row, "target_col", "")).replace("_per_1000_words", "").replace("_", " ")),
                _tex_escape(getattr(row, "group_name", "")),
                _tex_escape(getattr(row, "group_value", "")),
                _fmt_float(getattr(row, "coefficient", 0.0)),
                _fmt_float(getattr(row, "pvalue", 0.0), 4),
            ]
        )
    dependency_effect_table = _table_block(
        caption="Most informative vintage, region, and jurisdiction-type effects for the dependency/reference subcomponents. Coefficients are relative to omitted baseline categories within each target model, so they should be read as within-target deviations rather than as directly comparable magnitudes across all dependency measures.",
        label="tab:dependency-effects",
        colspec=r">{\raggedright\arraybackslash}Xllrr",
        header=["Target", "Group", "Value", "Coef.", "p"],
        rows=dependency_effect_rows,
    )

    coherence_terms = bridge_coherence[~bridge_coherence["term"].astype(str).eq("Intercept")].copy()
    coherence_rows = []
    for row in coherence_terms.itertuples(index=False):
        coherence_rows.append(
            [
                _tex_escape(_bridge_term_label(getattr(row, "term", ""))),
                _fmt_float(getattr(row, "coefficient", 0.0)),
                _fmt_float(getattr(row, "pvalue", 0.0), 4),
            ]
        )
    coherence_table = _table_block(
        caption="Bridge model: how the full-sample legal-complexity backbone relates to document-to-section semantic coherence on the 64-document LLM pilot. The omitted categories are the 1920--1949 vintage bin and borough ordinances. Continuous rows such as the structural, dependency, and linguistic indices should be read as one-standard-deviation shifts; vintage and jurisdiction rows are differences relative to those omitted baselines. Positive coefficients indicate higher purpose-to-section alignment and negative coefficients indicate lower alignment.",
        label="tab:bridge-coherence",
        colspec=r">{\raggedright\arraybackslash}Xrr",
        header=["Complexity component", "Coef.", "p"],
        rows=coherence_rows[:10],
    )

    gov_rows = []
    for target in ["direct_rule_share_of_operative", "discretion_share_of_operative", "exception_or_condition_share_of_operative"]:
        sub = bridge_governance[
            (bridge_governance["target_col"].astype(str) == target)
            & (~bridge_governance["term"].astype(str).eq("Intercept"))
        ].copy()
        sub = sub.sort_values("pvalue", ascending=True).head(3)
        for row in sub.itertuples(index=False):
            gov_rows.append(
                [
                    _tex_escape(_bridge_model_label(target.replace("_share_of_operative", ""))),
                    _tex_escape(_bridge_term_label(getattr(row, "term", ""))),
                    _fmt_float(getattr(row, "coefficient", 0.0)),
                    _fmt_float(getattr(row, "pvalue", 0.0), 4),
                ]
            )
    governance_table = _table_block(
        caption="Bridge models: how the full-sample legal-complexity backbone relates to clause-side governance style on the clause-covered pilot subset. The omitted categories are the 1920--1949 vintage bin and borough ordinances. Each model outcome is a clause-share measure, so coefficients describe shifts in regulatory composition rather than raw clause counts. The table reports the three most informative non-intercept terms per governance outcome.",
        label="tab:bridge-governance",
        colspec=r">{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}Xrr",
        header=["Pilot governance outcome", "Complexity component", "Coef.", "p"],
        rows=gov_rows,
    )

    bridge_sub_rows = []
    bridge_subcomponents_filtered = bridge_subcomponents.copy()
    if not bridge_subcomponents_filtered.empty:
        bridge_subcomponents_filtered = bridge_subcomponents_filtered[
            ~bridge_subcomponents_filtered["term"].astype(str).str.startswith(("Intercept", "C("))
        ].copy()
        bridge_subcomponents_filtered = bridge_subcomponents_filtered[bridge_subcomponents_filtered["pvalue"] < 0.10].copy()
        bridge_subcomponents_filtered["target_rank"] = bridge_subcomponents_filtered.groupby("target_col")["pvalue"].rank(method="first")
        bridge_subcomponents_filtered = bridge_subcomponents_filtered[bridge_subcomponents_filtered["target_rank"] <= 2]
        bridge_subcomponents_filtered = bridge_subcomponents_filtered.sort_values(["target_col", "pvalue", "term"])
        for row in bridge_subcomponents_filtered.itertuples(index=False):
            bridge_sub_rows.append(
                [
                    _tex_escape(_bridge_model_label(str(getattr(row, "target_col", "")).replace("_share_of_operative", ""))),
                    _tex_escape(_bridge_term_label(getattr(row, "term", ""))),
                    _fmt_float(getattr(row, "coefficient", 0.0)),
                    _fmt_float(getattr(row, "pvalue", 0.0), 4),
                ]
            )
    bridge_subcomponent_table = _table_block(
        caption="Most informative dependency, architecture, and readability subcomponents linking the full-sample backbone to pilot coherence and governance outcomes. The omitted categories are the 1920--1949 vintage bin and borough ordinances. Read this as a diagnostic table showing which specific legal features, such as section-kind entropy or references per 1,000 words, line up most strongly with coherence or governance style in the pilot.",
        label="tab:bridge-subcomponents",
        colspec=r">{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}Xrr",
        header=["Pilot outcome", "Backbone subcomponent", "Coef.", "p"],
        rows=bridge_sub_rows[:12],
    )

    section_rows = []
    for row in section_region.head(12).itertuples(index=False):
        section_rows.append(
            [
                _tex_escape(getattr(row, "jurisdiction_region", "")),
                _tex_escape(_clip(getattr(row, "cluster_label", ""), 46)),
                _fmt_float(getattr(row, "presence_lift_vs_overall_vintage_adjusted", 0.0)),
            ]
        )
    section_table = _table_block(
        caption="Top vintage-adjusted LLM section-family lifts by region from the refreshed pilot. `Lift` is the region-specific section-family presence rate divided by the pilot-wide baseline after adjusting to a common vintage mix, so values above one indicate section families that are overrepresented in that region relative to the overall pilot baseline.",
        label="tab:pilot-region-sections",
        colspec=r"l>{\raggedright\arraybackslash}Xr",
        header=["Region", "Section family", "Lift"],
        rows=section_rows,
    )

    style_rows = []
    for row in style_summary.sort_values("ordinance_count", ascending=False).itertuples(index=False):
        style_rows.append(
            [
                _tex_escape(_clip(getattr(row, "cluster_label", ""), 46)),
                _fmt_int(getattr(row, "ordinance_count", 0)),
                _fmt_float(getattr(row, "mean_direct_rule_share", 0.0)),
                _fmt_float(getattr(row, "mean_discretion_share", 0.0)),
                _fmt_float(getattr(row, "mean_exception_or_condition_share", 0.0)),
            ]
        )
    style_table = _table_block(
        caption="Clause-informed regulatory style families from the refreshed LLM pilot. `Direct`, `Discretion`, and `Except./Cond.` are mean shares of operative clauses in each family. They need not sum to one because other clause categories such as thresholds and pure standards are omitted from this compact display.",
        label="tab:pilot-style-families",
        colspec=r">{\raggedright\arraybackslash}Xrrrr",
        header=["Family", "Docs", "Direct", "Discretion", "Except./Cond."],
        rows=style_rows,
    )

    full_region_fig = _figure_block(
        str((complexity_run_root / "figures" / "complexity_by_region_vintage_adjusted.png").resolve()),
        "Vintage-adjusted structural, dependency, linguistic, and overall legal complexity by region on the 728-document full sample. Each bar is a standardized complexity score after reweighting the sample to a common vintage mix. Values above zero indicate regions that are above the sample average on that component; values below zero indicate below-average complexity on that dimension.",
    )
    full_level_fig = _figure_block(
        str((complexity_run_root / "figures" / "complexity_by_level_vintage_adjusted.png").resolve()),
        "Vintage-adjusted decomposed legal complexity by jurisdiction type on the full sample. Read this in the same way as the regional figure: positive bars indicate jurisdiction types that are more complex than the sample average on that component after vintage adjustment, while negative bars indicate less complex types.",
    )
    corr_fig = _figure_block(
        str((complexity_run_root / "figures" / "complexity_component_correlation_heatmap.png").resolve()),
        "Correlations across the decomposed full-sample complexity measures. Dark red cells indicate strong positive correlation, blue cells indicate negative correlation, and pale cells indicate weak relationships. The diagonal is mechanically one, so the informative cells are off-diagonal: those show whether structural, dependency, and linguistic complexity move together or capture distinct dimensions of legal burden.",
        width="0.82\\textwidth",
    )
    bridge_fig = _figure_block(
        str((complexity_run_root / "figures" / "pilot_complexity_coherence_scatter.png").resolve()),
        "Pilot semantic coherence against the full-sample overall legal complexity index. Each point is a pilot ordinance. The horizontal axis gives its legal-complexity score from the full-sample backbone and the vertical axis gives semantic coherence from the LLM pilot. The slope shows the raw association, while deviations from the fitted pattern indicate ordinances that are unusually coherent or unusually fragmented relative to their legal complexity.",
        width="0.82\\textwidth",
    )
    section_umap_fig = _figure_block(
        str((pilot_run_root / "reports" / "figures" / "section_semantic_hdbscan_umap.png").resolve()),
        "Section-level OpenAI embedding manifold from the refreshed pilot, labeled by LLM section families. Each point is a section, nearby points are semantically similar in the reduced embedding space, and labeled centroids mark the largest section families. Distinct colored clouds should therefore be read as distinct legal-architecture families rather than literal physical distances in the original text space.",
    )

    memo_excerpt_lines = []
    for line in memo_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "/Users/" in stripped:
            continue
        memo_excerpt_lines.append(line)
        if len(memo_excerpt_lines) >= 16:
            break
    memo_excerpt = "\n".join(memo_excerpt_lines)
    memo_excerpt = _tex_escape(memo_excerpt)

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
\title{{Legal Complexity in Printed Zoning Ordinances: Full-Sample Backbone and LLM-Native Pilot}}
\author{{Codex research build}}
\date{{March 11, 2026}}
\begin{{document}}
\maketitle

\section*{{Goal}}
\begin{{itemize}}[leftmargin=1.2em]
\item This writeup treats legal complexity as a decomposed object rather than a single scalar score.
\item The main dimensions are structural, dependency/reference, linguistic, governance, and semantic coherence.
\end{{itemize}}

\section*{{Data and Design}}
\begin{{itemize}}[leftmargin=1.2em]
\item Full-sample backbone: {_fmt_int(complexity_summary['doc_count'])} census-linked ordinances with clean text.
\item Pilot semantic layer: {_fmt_int(pilot_summary['doc_count'])} ordinances, with {_fmt_int(pilot_summary['enriched_clause_doc_count'])} clause-covered and {_fmt_int(pilot_summary['enriched_clause_row_count'])} enriched clauses.
\item Full-sample backbone: structural size and dispersion, internal references and dependency triggers, and linguistic/readability proxies.
\item LLM-native pilot: document purpose, section motives, section embeddings, clause extraction/enrichment, governance style, and semantic coherence.
\item All cross-sectional comparisons are vintage-aware. The writeup treats the data as a cross-section of ordinance vintages, not as a within-jurisdiction panel.
\end{{itemize}}

\section*{{Complexity Framework}}
\begin{{enumerate}}[leftmargin=1.4em]
\item Structural complexity: section count, section length dispersion, large-section share, and section-kind entropy.
\item Dependency complexity: formal internal references, dependency triggers, map references, board/administrator references, and reference-bearing section share.
\item Linguistic complexity: sentence length, long-word share, and Flesch--Kincaid-style proxy measures.
\item Governance complexity: direct-rule, discretion, exception/condition, threshold, and instrument-diversity structure from clause enrichment.
\item Semantic coherence: how closely document-level purposes line up with weighted section-level motives.
\end{{enumerate}}
\begin{{itemize}}[leftmargin=1.2em]
\item How the indices are built in code: each component index is the mean of several z-scored submeasures, and the overall legal-complexity index is the mean of the structural, dependency, and linguistic component indices.
\item Structural complexity combines log document length, log section count, Shannon-style section-kind entropy (\href{{https://doi.org/10.1002/j.1538-7305.1948.tb01338.x}}{{Shannon, 1948}}), section-length dispersion, the share of sections longer than 750 words, and the log maximum section length.
\item Dependency complexity combines per-1,000-word counts of formal internal references, dependency-trigger phrases, map references, board or administrator references, and the share of sections that contain any such reference. These are custom legal-text measures rather than borrowed off-the-shelf indices.
\item Linguistic complexity combines average sentence length, average word length, the share of words with 7 or more letters, an approximate syllables-per-word measure, and a Flesch--Kincaid-style grade proxy following the readability family introduced by \href{{https://pubmed.ncbi.nlm.nih.gov/18867058/}}{{Flesch (1948)}} and extended by \href{{https://stars.library.ucf.edu/istlibrary/56/}}{{Kincaid et al. (1975)}}.
\item Read the tables dimension by dimension rather than as a single omnibus score. Read the figures as showing whether those dimensions move together or capture distinct legal burdens.
\end{{itemize}}

\begin{{itemize}}[leftmargin=1.2em]
\item How to read the next three tables: Table~\ref{{tab:complexity-region}} and Table~\ref{{tab:complexity-level}} report vintage-adjusted means on standardized indices, so zero is the full-sample average and positive values indicate above-average complexity on that dimension.
\item Table~\ref{{tab:complexity-effects}} comes from separate regressions for each target index. Coefficients are relative to omitted baseline categories inside each model, so comparisons should be made within a target row block rather than across different targets.
\end{{itemize}}

{region_table}

{level_table}

{effect_table}

\begin{{itemize}}[leftmargin=1.2em]
\item How to read the next three figures: the region and jurisdiction-type bar charts show the same standardized indices as the tables, but visually separate which component is driving the gap.
\item The correlation heatmap is not another complexity score. It is a diagnostic showing whether structural, dependency, and linguistic burden move together or capture distinct dimensions. The off-diagonal cells matter; the diagonal is mechanically one.
\end{{itemize}}

{full_region_fig}

{full_level_fig}

{corr_fig}

\section*{{What the Full Sample Shows}}
\begin{{itemize}}[leftmargin=1.2em]
\item The main result is that legal complexity is multi-dimensional rather than scalar.
\item Later ordinance vintages are more complex on every major dimension.
\item After controls, the Northeast remains dependency-complex, the South is highest on the overall index largely through the linguistic component, and townships are the highest-complexity jurisdiction form.
\end{{itemize}}

\begin{{itemize}}[leftmargin=1.2em]
\item The family surface is exploratory rather than definitive.
\item Use it as a supporting typology for case selection and interpretation.
\end{{itemize}}

{family_table}

\begin{{itemize}}[leftmargin=1.2em]
\item Residual cases show that unusual complexity can come from different channels.
\item Some cases are reference-heavy; others are structurally sprawling or linguistically dense.
\item The residual tables are built from models that partial out vintage, region, and jurisdiction type. A large positive residual means the ordinance is more complex than comparable ordinances on that specific dimension.
\end{{itemize}}

{overall_residual_table}

{dependency_residual_table}

{exemplar_table}

\begin{{itemize}}[leftmargin=1.2em]
\item Dependency burden is itself decomposable.
\item Formal references, trigger language, map references, and board references do not always move together.
\item The dependency-profile table is a clustering summary over dependency submeasures. The effects table below it returns to regression coefficients, showing which regions, levels, and vintages drive specific dependency subcomponents once the other covariates are held fixed.
\end{{itemize}}

{dependency_profile_table}

{dependency_effect_table}

\section*{{What the LLM Pilot Adds}}
\begin{{itemize}}[leftmargin=1.2em]
\item The LLM pilot does not replace the full-sample backbone; it adds legal architecture, governance style, clause instruments, and semantic coherence.
\item The OpenAI section surface remains stronger than the local baseline on the pilot.
\end{{itemize}}

{section_table}

{section_umap_fig}

\begin{{itemize}}[leftmargin=1.2em]
\item The clause side is now substantive rather than speculative.
\item The current pilot has {_fmt_int(pilot_summary['regulatory_style_family_count'])} regulatory style families with silhouette {_fmt_float(pilot_summary['regulatory_style_family_silhouette_score'])}.
\item Those families differ in direct-rule share, discretion, exception/condition intensity, and instrument diversity.
\end{{itemize}}

{style_table}

\section*{{Complexity, Governance, and Coherence}}
\begin{{itemize}}[leftmargin=1.2em]
\item The bridge layer links the full-sample complexity backbone to pilot semantic outcomes.
\item This is one of the strongest frontiers because it connects legal burden to how ordinances organize and operationalize regulation.
\item The coherence bridge regresses pilot purpose-to-section alignment on the three full-sample complexity components. The governance bridge regresses pilot clause-share outcomes such as direct-rule and discretion shares on the same components.
\item The subcomponent bridge then replaces the three high-level indices with specific ingredients such as section-kind entropy, reference mentions per 1,000 words, and the readability proxy, so those rows should be read as diagnostic clues about which features are driving the higher-level associations.
\end{{itemize}}

{coherence_table}

\begin{{itemize}}[leftmargin=1.2em]
\item The bridge does not support a simple monotone story in which every complexity component lowers coherence.
\item The better interpretation is residual coherence versus fragmentation after controls.
\end{{itemize}}

{governance_table}

\begin{{itemize}}[leftmargin=1.2em]
\item The governance bridge is suggestive rather than definitive.
\item Higher linguistic complexity is associated with less direct-rule-heavy and more discretion-heavy regulation on the clause-covered subset.
\end{{itemize}}

{bridge_subcomponent_table}

{bridge_fig}

\section*{{Most Defensible Storylines}}
\begin{{itemize}}[leftmargin=1.2em]
\item Printed zoning ordinances differ systematically across region and jurisdiction type in legal architecture, not just in topic.
\item Legal complexity decomposes into structural, dependency, and linguistic dimensions that do not collapse into a single size metric.
\item Later ordinance vintages are more complex on all three dimensions, so vintage adjustment is necessary for cross-sectional comparison.
\item The Northeast remains relatively dependency-complex after controls, while the South is elevated on the overall index through the linguistic component.
\item Townships look especially complex in the full-sample decomposition, while villages look comparatively simple.
\item The semantic/governance pilot suggests that complexity is linked less to simple coherence loss than to differences in regulatory style: direct-rule-heavy versus discretion-heavy and exception-heavy codes.
\end{{itemize}}

\section*{{Limitations}}
The clause-informed pilot is still a pilot. It currently covers {_fmt_int(pilot_summary['enriched_clause_doc_count'])} of {_fmt_int(pilot_summary['doc_count'])} pilot ordinances. That is strong enough to support exploratory governance and coherence claims, but not yet enough to treat the clause-side typology as a fully mature national result. The full-sample backbone is much stronger on structural and dependency complexity than it is on clause-level compliance burden.

\section*{{Working Memo Snapshot}}
\begin{{quote}}
\small
{memo_excerpt}
\end{{quote}}

\section*{{Conclusion}}
\begin{{itemize}}[leftmargin=1.2em]
\item The strongest framing is a decomposed legal-complexity project rather than a pure topic-modeling exercise.
\item The full-sample backbone provides structural and dependency burden.
\item The pilot adds governance style, instrument mix, and semantic coherence.
\end{{itemize}}

\end{{document}}
"""

    output_path.write_text(tex.strip() + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined legal-complexity + LLM pilot writeup.")
    parser.add_argument(
        "--complexity-run-root",
        default="artifacts/runs/legal_complexity_fullsample_v1/full728_20260311_a",
        help="Full-sample legal-complexity run root.",
    )
    parser.add_argument(
        "--pilot-run-root",
        default="artifacts/runs/frontier_llm_v2/pilot64_20260310_a",
        help="LLM-native pilot run root.",
    )
    parser.add_argument("--output-path", default="", help="Optional explicit output path.")
    args = parser.parse_args()
    output = build_combined_writeup(
        complexity_run_root=Path(args.complexity_run_root),
        pilot_run_root=Path(args.pilot_run_root),
        output_path=Path(args.output_path).expanduser().resolve() if str(args.output_path).strip() else None,
    )
    print(f"wrote_combined_legal_complexity_writeup tex_path={output}")


if __name__ == "__main__":
    main()
