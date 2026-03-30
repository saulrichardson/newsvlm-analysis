#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


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


def _clip(text: Any, limit: int) -> str:
    if pd.isna(text):
        return ""
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + "..."


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


def _table_block(*, caption: str, label: str, colspec: str, header: list[str], rows: list[list[str]], size: str = r"\footnotesize") -> str:
    row_text = "\n".join(" & ".join(row) + r" \\" for row in rows) if rows else r"\multicolumn{99}{c}{No rows available.} \\"
    uses_x = "X" in colspec
    env_name = "tabularx" if uses_x else "tabular"
    env_open = rf"\begin{{{env_name}}}{{\textwidth}}{{{colspec}}}" if uses_x else rf"\begin{{{env_name}}}{{{colspec}}}"
    env_close = rf"\end{{{env_name}}}"
    return f"""
\\begin{{table}}[H]
\\centering
{size}
\\setlength{{\\tabcolsep}}{{4pt}}
\\renewcommand{{\\arraystretch}}{{0.98}}
\\caption{{{caption}}}
\\label{{{label}}}
{env_open}
\\toprule
{' & '.join(header)} \\\\
\\midrule
{row_text}
\\bottomrule
{env_close}
\\end{{table}}
""".strip()


def _figure_block(path: str, caption: str, width: str = "0.92\\textwidth") -> str:
    return f"""
\\IfFileExists{{{path}}}{{
\\begin{{figure}}[H]
  \\centering
  \\includegraphics[width={width}]{{{path}}}
  \\caption{{{caption}}}
\\end{{figure}}
}}{{}}
""".strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_entrypoint_tex(*, run_root: Path, output_path: Path | None = None) -> Path:
    run_root = run_root.resolve()
    reports_dir = run_root / "reports"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    output_path = (output_path or (reports_dir / "frontier_entrypoint.tex")).resolve()

    run_manifest = _load_json(run_root / "manifests" / "run_manifest.json")
    corpus_summary = _load_json(run_root / "manifests" / "corpus_summary.json")
    geo_summary = _load_json(run_root / "manifests" / "geo_summary.json")
    structure_summary = _load_json(run_root / "manifests" / "structure_summary.json")
    semantics_summary = _load_json(run_root / "manifests" / "semantics_summary.json")
    embeddings_summary = _load_json(run_root / "manifests" / "embeddings_summary.json")
    clusters_summary = _load_json(run_root / "manifests" / "clusters_summary.json")
    review_summary = _load_json(run_root / "manifests" / "review_summary.json")
    story_summary = _load_json(run_root / "manifests" / "story_summary.json")
    report_summary = _load_json(run_root / "manifests" / "report_summary.json")

    bundle_meta = pd.read_csv(Path(run_manifest["paths"]["bundle_dir"]) / "metadata.csv", low_memory=False)
    total_bundle_issues = int(len(bundle_meta))

    regional_raw = pd.read_csv(tables_dir / "regional_counts_raw.csv")
    regional_dedup = pd.read_csv(tables_dir / "regional_counts_dedup.csv")
    motive_by_region = pd.read_csv(tables_dir / "motive_shares_by_region.csv")
    substantive_instrument_by_region = pd.read_csv(tables_dir / "substantive_instrument_shares_by_region.csv")
    policy_signature_by_region = pd.read_csv(tables_dir / "policy_signature_shares_by_region.csv")
    policy_signature_clusters = pd.read_csv(tables_dir / "policy_signature_clusters.csv")
    mismatch_publications = pd.read_csv(tables_dir / "high_mismatch_publications_top50.csv")
    outliers = pd.read_csv(tables_dir / "outlier_ordinances_top50.csv")
    duplicate_patterns = pd.read_csv(tables_dir / "duplicate_publication_patterns_top100.csv")

    geo_audit = pd.read_parquet(run_root / "geo" / "jurisdiction_audit.parquet")
    doc_sem = pd.read_parquet(run_root / "semantics" / "document_semantics.parquet")
    clause_path = run_root / "semantics" / "operative_clause_semantics.parquet"
    feature_path = run_root / "semantics" / "operative_hard_feature_observations.parquet"
    clause_sem = pd.read_parquet(clause_path if clause_path.is_file() else run_root / "semantics" / "clause_semantics.parquet")
    feature_obs = pd.read_parquet(feature_path if feature_path.is_file() else run_root / "semantics" / "hard_feature_observations.parquet")

    cluster_index = pd.read_parquet(run_root / "clusters" / "experiment_index.parquet")
    section_clusters = pd.read_parquet(run_root / "clusters" / "section_semantic_hdbscan" / "cluster_labels.parquet")
    clause_clusters = pd.read_parquet(run_root / "clusters" / "clause_instrument_hdbscan" / "cluster_labels.parquet")

    motive_counts = doc_sem["primary_motive"].fillna("").astype(str).replace("", "[blank]").value_counts().reset_index()
    motive_counts.columns = ["primary_motive", "ordinance_count"]
    instrument_counts = clause_sem["instrument_type"].fillna("").astype(str).replace("", "[blank]").value_counts().reset_index()
    instrument_counts.columns = ["instrument_type", "clause_count"]
    modality_counts = clause_sem["modality"].fillna("").astype(str).replace("", "[blank]").value_counts().reset_index()
    modality_counts.columns = ["modality", "clause_count"]
    feature_counts = feature_obs["feature_name"].fillna("").astype(str).replace("", "[blank]").value_counts().reset_index()
    feature_counts.columns = ["feature_name", "observation_count"]

    mismatch_examples = geo_audit.loc[
        geo_audit["jurisdiction_match_status"].astype(str) == "mismatch",
        [
            "issue_id",
            "publication_city_name",
            "publication_state_abbr",
            "jurisdiction_city_name",
            "jurisdiction_state_abbr",
            "jurisdiction_level",
        ],
    ].head(8)

    experiment_summaries: list[dict[str, Any]] = []
    for exp in cluster_index.itertuples(index=False):
        summary_path = run_root / "clusters" / str(getattr(exp, "experiment_name")) / "summary.json"
        payload = _load_json(summary_path)
        experiment_summaries.append(
            {
                "experiment_name": str(getattr(exp, "experiment_name")),
                "representation_level": str(getattr(exp, "representation_level")),
                "clusterer": str(getattr(exp, "clusterer")),
                "labeler": str(getattr(exp, "labeler")),
                "item_count": int(payload.get("item_count", 0)),
                "cluster_count": int(payload.get("cluster_count", 0)),
                "noise_count": int(payload.get("noise_count", 0)),
            }
        )
    experiment_df = pd.DataFrame.from_records(experiment_summaries)

    corpus_table = _table_block(
        caption="Run-level corpus summary for the current frontier full-code full-sample run.",
        label="tab:frontier-run-summary",
        colspec=r">{\raggedright\arraybackslash}Xr",
        header=["Metric", "Value"],
        rows=[
            [r"Underlying full-code bundle issues", _fmt_int(total_bundle_issues)],
            [r"Current run issues", _fmt_int(corpus_summary["issue_count"])],
            [r"Current run ordinances", _fmt_int(corpus_summary["ordinance_count"])],
            [r"Current run pages", _fmt_int(corpus_summary["page_count"])],
            [r"Dedup groups", _fmt_int(corpus_summary["dedup_group_count"])],
            [r"Duplicate issue groups (size $>$ 1)", _fmt_int(corpus_summary["duplicate_issue_count"])],
            [r"Jurisdiction matches", _fmt_int(geo_summary["match_count"])],
            [r"Jurisdiction mismatches", _fmt_int(geo_summary["mismatch_count"])],
            [r"Missing jurisdiction", _fmt_int(geo_summary["missing_jurisdiction_count"])],
            [r"Sections", _fmt_int(structure_summary["section_count"])],
            [r"Clauses", _fmt_int(structure_summary["clause_count"])],
            [r"Hard-feature observations", _fmt_int(semantics_summary["feature_rows"])],
        ],
    )

    source_table = _table_block(
        caption="Canonical inputs and provenance surfaces used by the frontier full-code lab.",
        label="tab:frontier-sources",
        colspec=r">{\raggedright\arraybackslash}p{0.24\textwidth}>{\raggedright\arraybackslash}X",
        header=["Input surface", "Path"],
        rows=[
            ["Canonical full-code ordinance bundle", rf"\path{{{run_manifest['paths']['bundle_dir']}}}"],
            ["Archive publication metadata", rf"\path{{{run_manifest['paths']['archive_metadata_csv']}}}"],
            ["Publication location resolver", rf"\path{{{run_manifest['paths']['publication_locations_jsonl']}}}"],
            ["Torch issue manifest", rf"\path{{{run_manifest['paths']['torch_manifest_jsonl']}}}"],
            ["Torch transcription run root", rf"\path{{{run_manifest['paths']['torch_run_root']}}}"],
        ],
        size=r"\scriptsize",
    )

    geography_table = _table_block(
        caption="Regional and jurisdiction-level coverage in the current full-sample run.",
        label="tab:frontier-regional-raw",
        colspec=r">{\raggedright\arraybackslash}l>{\raggedright\arraybackslash}lrrrr",
        header=["Region", "Jurisdiction level", "Issues", "Unique pubs", "Mismatches", "Mismatch share"],
        rows=[
            [
                _tex_escape(row.jurisdiction_region),
                _tex_escape(row.jurisdiction_level),
                _fmt_int(row.raw_issue_count),
                _fmt_int(row.unique_publications),
                _fmt_int(row.mismatch_count),
                _fmt_float(row.mismatch_share, 3),
            ]
            for row in regional_raw.itertuples(index=False)
        ],
        size=r"\scriptsize",
    )

    mismatch_table = _table_block(
        caption="Concrete publication-versus-jurisdiction mismatch examples from the full-sample audit.",
        label="tab:frontier-mismatch-examples",
        colspec=r">{\raggedright\arraybackslash}p{0.28\textwidth}>{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}l",
        header=["Issue", "Publication location", "Jurisdiction", "Level"],
        rows=[
            [
                _tex_escape(_clip(row.issue_id, 52)),
                _tex_escape(f"{row.publication_city_name}, {row.publication_state_abbr}"),
                _tex_escape(f"{row.jurisdiction_city_name}, {row.jurisdiction_state_abbr}"),
                _tex_escape(row.jurisdiction_level),
            ]
            for row in mismatch_examples.itertuples(index=False)
        ],
        size=r"\scriptsize",
    )

    high_mismatch_table = _table_block(
        caption="Publications with the highest full-sample mismatch share.",
        label="tab:frontier-high-mismatch-publications",
        colspec=r">{\raggedright\arraybackslash}Xllrrr",
        header=["Publication", "City", "State", "Issues", "Mismatches", "Mismatch share"],
        rows=[
            [
                _tex_escape(_clip(row.publication_key, 40)),
                _tex_escape(_clip(row.publication_city_name, 18)),
                _tex_escape(row.publication_state_abbr),
                _fmt_int(row.issue_count),
                _fmt_int(row.mismatch_count),
                _fmt_float(row.mismatch_share, 3),
            ]
            for row in mismatch_publications.head(8).itertuples(index=False)
        ],
        size=r"\scriptsize",
    )

    semantics_table = _table_block(
        caption="Document-level motive distribution and operative-clause modality distribution in the full-sample run.",
        label="tab:frontier-semantic-counts",
        colspec=r">{\raggedright\arraybackslash}p{0.28\textwidth}r>{\raggedright\arraybackslash}p{0.22\textwidth}r",
        header=["Primary motive", "Ordinances", "Clause modality", "Clauses"],
        rows=[
            [
                _tex_escape(str(motive_counts.iloc[i]["primary_motive"])) if i < len(motive_counts) else "",
                _fmt_int(motive_counts.iloc[i]["ordinance_count"]) if i < len(motive_counts) else "",
                _tex_escape(str(modality_counts.iloc[i]["modality"])) if i < len(modality_counts) else "",
                _fmt_int(modality_counts.iloc[i]["clause_count"]) if i < len(modality_counts) else "",
            ]
            for i in range(max(len(motive_counts), len(modality_counts)))
        ],
        size=r"\footnotesize",
    )

    feature_table = _table_block(
        caption="Structured hard-feature observations extracted from operative clauses in the full-sample run.",
        label="tab:frontier-feature-counts",
        colspec=r">{\raggedright\arraybackslash}p{0.24\textwidth}r>{\raggedright\arraybackslash}p{0.24\textwidth}r",
        header=["Feature name", "Count", "Instrument type", "Clause count"],
        rows=[
            [
                _tex_escape(str(feature_counts.iloc[i]["feature_name"])) if i < len(feature_counts) else "",
                _fmt_int(feature_counts.iloc[i]["observation_count"]) if i < len(feature_counts) else "",
                _tex_escape(str(instrument_counts.iloc[i]["instrument_type"])) if i < len(instrument_counts) else "",
                _fmt_int(instrument_counts.iloc[i]["clause_count"]) if i < len(instrument_counts) else "",
            ]
            for i in range(max(len(feature_counts), min(len(instrument_counts), 10)))
        ],
        size=r"\footnotesize",
    )

    experiment_table = _table_block(
        caption="Clustering experiments executed in the full-sample run.",
        label="tab:frontier-cluster-experiments",
        colspec=r">{\raggedright\arraybackslash}p{0.28\textwidth}llrrr",
        header=["Experiment", "Level", "Clusterer", "Items", "Clusters", "Noise"],
        rows=[
            [
                _tex_escape(row.experiment_name),
                _tex_escape(row.representation_level),
                _tex_escape(row.clusterer),
                _fmt_int(row.item_count),
                _fmt_int(row.cluster_count),
                _fmt_int(row.noise_count),
            ]
            for row in experiment_df.itertuples(index=False)
        ],
        size=r"\scriptsize",
    )

    section_cluster_table = _table_block(
        caption="Largest section-level semantic clusters from the full-sample run.",
        label="tab:frontier-section-clusters",
        colspec=r"r r >{\raggedright\arraybackslash}X",
        header=["Cluster", "Size", "Label"],
        rows=[
            [
                _fmt_int(row.cluster_id),
                _fmt_int(row.cluster_size),
                _tex_escape(_clip(row.cluster_label, 70)),
            ]
            for row in section_clusters.loc[section_clusters["cluster_id"] >= 0].sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).head(10).itertuples(index=False)
        ],
        size=r"\scriptsize",
    )

    clause_cluster_table = _table_block(
        caption="Largest clause-level instrument clusters from the full-sample run.",
        label="tab:frontier-clause-clusters",
        colspec=r"r r >{\raggedright\arraybackslash}X",
        header=["Cluster", "Size", "Label"],
        rows=[
            [
                _fmt_int(row.cluster_id),
                _fmt_int(row.cluster_size),
                _tex_escape(_clip(row.cluster_label, 70)),
            ]
            for row in clause_clusters.loc[clause_clusters["cluster_id"] >= 0].sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).head(10).itertuples(index=False)
        ],
        size=r"\scriptsize",
    )

    outlier_table = _table_block(
        caption="A few ordinance-level outliers identified by distance from the global embedding centroid.",
        label="tab:frontier-outliers",
        colspec=r">{\raggedright\arraybackslash}p{0.34\textwidth}>{\raggedright\arraybackslash}Xr",
        header=["Ordinance", "Title line", "Similarity"],
        rows=[
            [
                _tex_escape(_clip(row.item_id, 54)),
                _tex_escape(_clip(row.title_line, 78)),
                _fmt_float(row.global_centroid_similarity, 3),
            ]
            for row in outliers.head(8).itertuples(index=False)
        ],
        size=r"\scriptsize",
    )

    substantive_instrument_table = _table_block(
        caption="Most prevalent substantive instrument families by region, measured as ordinance-level presence rather than raw clause counts.",
        label="tab:frontier-substantive-instruments",
        colspec=r">{\raggedright\arraybackslash}p{0.12\textwidth}>{\raggedright\arraybackslash}p{0.38\textwidth}rr",
        header=["Region", "Instrument", "Ordinances", "Share"],
        rows=[
            [
                _tex_escape(row.jurisdiction_region),
                _tex_escape(_clip(row.instrument_type, 36)),
                _fmt_int(row.ordinance_count),
                _fmt_float(row.share_within_region, 3),
            ]
            for row in substantive_instrument_by_region.groupby("jurisdiction_region", dropna=False).head(5).itertuples(index=False)
        ],
        size=r"\scriptsize",
    )

    reportable_policy_clusters = policy_signature_clusters.loc[
        policy_signature_clusters.get("reportable_cluster_flag", 1).astype(int) == 1
    ].copy()
    policy_signature_table = _table_block(
        caption="Reportable policy-signature clusters from the full-sample run.",
        label="tab:frontier-policy-signatures",
        colspec=r"rr>{\raggedright\arraybackslash}p{0.40\textwidth}ll",
        header=["Cluster", "Size", "Label", "Dominant region", "Dominant level"],
        rows=[
            [
                _fmt_int(row.policy_signature_cluster_id),
                _fmt_int(row.cluster_size),
                _tex_escape(_clip(row.cluster_label, 54)),
                _tex_escape(_clip(row.dominant_region, 12)),
                _tex_escape(_clip(row.dominant_level, 14)),
            ]
            for row in reportable_policy_clusters.itertuples(index=False)
        ],
        size=r"\scriptsize",
    )

    policy_region_table = _table_block(
        caption="Largest reportable policy-signature clusters within each region.",
        label="tab:frontier-policy-by-region",
        colspec=r">{\raggedright\arraybackslash}p{0.12\textwidth}>{\raggedright\arraybackslash}p{0.42\textwidth}rr",
        header=["Region", "Policy signature", "Ordinances", "Share"],
        rows=[
            [
                _tex_escape(row.jurisdiction_region),
                _tex_escape(_clip(row.cluster_label, 52)),
                _fmt_int(row.ordinance_count),
                _fmt_float(row.share_within_region, 3),
            ]
            for row in policy_signature_by_region.loc[
                policy_signature_by_region["policy_signature_cluster_id"].isin(
                    set(reportable_policy_clusters["policy_signature_cluster_id"].tolist())
                )
            ]
            .groupby("jurisdiction_region", dropna=False)
            .head(4)
            .itertuples(index=False)
        ],
        size=r"\scriptsize",
    )

    duplicate_note = "The duplicate-publication layer is structurally implemented even when duplicate groups are rare."
    if not duplicate_patterns.empty and int(duplicate_patterns["issue_count"].max()) > 1:
        duplicate_note = "The duplicate-publication table below shows repeated ordinance texts published across issues or newspapers."

    report_tex = f"""
\\documentclass[11pt]{{article}}

\\usepackage[margin=1in]{{geometry}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{caption}}
\\usepackage{{enumitem}}
\\usepackage{{float}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{longtable}}
\\usepackage{{tabularx}}
\\usepackage{{xcolor}}
\\usepackage{{xurl}}

\\hypersetup{{hidelinks}}
\\graphicspath{{{{figures/}}}}

\\title{{Frontier Full-Code Research Lab v1: Entry-Point Writeup}}
\\author{{}}
\\date{{Generated from run \\texttt{{{_tex_escape(run_manifest['run_id'])}}} on {_tex_escape(run_manifest['created_at_utc'])}}}

\\begin{{document}}
\\maketitle

\\section{{What this document is}}

This note is the entry point into the newly implemented frontier full-code lab. It is meant to do two jobs at once:
first, to summarize the empirical approach and artifact layout of the lab itself; second, to summarize what the current full-sample run actually produced.
The substantive framing is descriptive and cross-sectional. The operative unit is \\textbf{{ordinance by governing jurisdiction}}, not newspaper by time.

The current report is grounded in the full linked run at:
\\begin{{itemize}}[leftmargin=*]
  \\item Run root: \\path{{{run_manifest['run_root']}}}
  \\item Scope label: \\texttt{{{_tex_escape(run_manifest['scope_label'])}}}
  \\item Schema version: \\texttt{{{_tex_escape(run_manifest['schema_version'])}}}
\\end{{itemize}}

The underlying full-code bundle contains {_fmt_int(total_bundle_issues)} issues, and the current run processes the full bundle shell of {_fmt_int(corpus_summary['issue_count'])} issues. Within that shell, {_fmt_int(corpus_summary['jurisdiction_status_counts'].get('resolved_with_census', 0))} issues resolve to a one-to-one census-linked jurisdiction and {_fmt_int(semantics_summary['document_rows'])} ordinances have non-empty linked text available for the downstream semantic and clustering layers.

\\section{{Canonical inputs and operating model}}

The canonical input text for this lab is the clean ordinance text reconstructed from the full-code post-gate transcription bundle, with Torch retained as the provenance and execution surface. Publication geography and governing jurisdiction geography are preserved separately in every canonical table.

{source_table}

The operating model is:
\\begin{{enumerate}}[leftmargin=*]
  \\item Treat the per-issue full-code ordinance text as the canonical legal-text surface.
  \\item Carry both publication geography and extracted ordinance jurisdiction through the corpus.
  \\item Use governing jurisdiction as the default substantive geography for analysis.
  \\item Preserve publication-versus-jurisdiction mismatches as an explicit audit object rather than silently resolving them away.
\\end{{enumerate}}

\\section{{Empirical approach}}

The implemented empirical workflow has eight layers:
\\begin{{enumerate}}[leftmargin=*]
  \\item \\textbf{{Corpus layer.}} Build canonical ordinance records, page records, geography links, dedup groups, and a minimal longitudinal scaffold.
  \\item \\textbf{{Geography audit.}} Classify each issue as a publication/jurisdiction match, mismatch, or missing-jurisdiction case.
  \\item \\textbf{{Structure layer.}} Segment the clean ordinance text into legal-text-aware sections and then clauses.
  \\item \\textbf{{Semantics layer.}} Assign document-level motives and purposes, clause-level modality and instrument hints, and structured hard-feature observations.
  \\item \\textbf{{Embeddings layer.}} Build ordinance-, section-, and clause-level text representations; compute PCA and UMAP manifolds for each.
  \\item \\textbf{{Cluster layer.}} Run multiple clustering views rather than one single partition: ordinance semantic HDBSCAN, section semantic HDBSCAN, section mechanics graph clustering, clause instrument HDBSCAN, and clause hierarchical clustering.
  \\item \\textbf{{Review layer.}} Generate packetized review surfaces for jurisdiction, section boundaries, clauses, and cluster interpretation.
  \\item \\textbf{{Story/report layer.}} Aggregate into region-oriented tables, mismatch packets, outlier packets, and a static report bundle.
\\end{{enumerate}}

The aim is not only to ``get clusters,'' but to build an auditable descriptive platform that can support a paper-level story about what early full-code zoning ordinances look like, how they vary across jurisdictions, and how publication geography can differ from governing jurisdiction.

\\section{{Run-level corpus summary}}

{corpus_table}

Publication geography in the full run is fully resolved ({_fmt_int(geo_summary['publication_geo_coverage'])} of {_fmt_int(corpus_summary['issue_count'])} issues), using a combination of archive metadata and the scraper-side location resolver. Archive metadata accounts for {_fmt_int(corpus_summary['publication_geo_sources'].get('archive_v2', 0))} issues and the scraper-side fallback accounts for {_fmt_int(corpus_summary['publication_geo_sources'].get('locations_headful', 0))} issues.

The canonical text status counts in the full run are:
\\begin{{itemize}}[leftmargin=*]
  \\item \\texttt{{ok}}: {_fmt_int(corpus_summary['canonical_status_counts'].get('ok', 0))}
  \\item \\texttt{{repaired\\_from\\_raw}}: {_fmt_int(corpus_summary['canonical_status_counts'].get('repaired_from_raw', 0))}
  \\item \\texttt{{raw\\_only}}: {_fmt_int(corpus_summary['canonical_status_counts'].get('raw_only', 0))}
  \\item \\texttt{{empty\\_output}}: {_fmt_int(corpus_summary['canonical_status_counts'].get('empty_output', 0))}
\\end{{itemize}}

For reference, the full underlying bundle contains 716 \\texttt{{ok}}, 14 \\texttt{{repaired\\_from\\_raw}}, 1 \\texttt{{raw\\_only}}, and 2 \\texttt{{empty\\_output}} issues.

\\section{{Jurisdiction audit and regional framing}}

The full run confirms that publication geography and ordinance jurisdiction cannot be treated as interchangeable.
Out of {_fmt_int(corpus_summary['issue_count'])} issues, {_fmt_int(geo_summary['mismatch_count'])} are mismatches and {_fmt_int(geo_summary['missing_jurisdiction_count'])} are missing-jurisdiction cases.

{_figure_block('figures/jurisdiction_match_status.png', 'Jurisdiction audit for the full-sample run. The central design point is that publication geography and ordinance jurisdiction are carried separately and audited directly.')}

{geography_table}

{mismatch_table}

{high_mismatch_table}

Some mismatch cases are clearly substantive cross-jurisdiction publications, such as Evanston publishing a Wilmette ordinance or Oak Park publishing a River Forest ordinance.
Other cases appear to reflect normalization problems rather than substantive geographic differences, such as \\emph{{Provo}} versus \\emph{{Provo City}}. The point of the audit layer is precisely to make those different kinds of cases visible instead of burying them.

\\section{{Structure, semantics, and hard features}}

The structure builder produced {_fmt_int(structure_summary['section_count'])} sections and {_fmt_int(structure_summary['clause_count'])} clauses from {_fmt_int(corpus_summary['issue_count'])} issues. The semantic layer then assigned document-level motives, document-purpose snippets, clause modalities, instrument hints, and {_fmt_int(semantics_summary['feature_rows'])} hard-feature observations.

At full scale, document-level primary motives remain concentrated in infrastructure coordination, public health and safety, and externality control:
\\begin{{itemize}}[leftmargin=*]
  \\item \\texttt{{public\\_health\\_safety}}: {_fmt_int((motive_counts.loc[motive_counts['primary_motive'] == 'public_health_safety', 'ordinance_count'].iloc[0] if (motive_counts['primary_motive'] == 'public_health_safety').any() else 0))}
  \\item \\texttt{{externality\\_control}}: {_fmt_int((motive_counts.loc[motive_counts['primary_motive'] == 'externality_control', 'ordinance_count'].iloc[0] if (motive_counts['primary_motive'] == 'externality_control').any() else 0))}
  \\item all other motive categories combined: {_fmt_int(int(corpus_summary['issue_count']) - int((motive_counts.loc[motive_counts['primary_motive'] == 'public_health_safety', 'ordinance_count'].iloc[0] if (motive_counts['primary_motive'] == 'public_health_safety').any() else 0)) - int((motive_counts.loc[motive_counts['primary_motive'] == 'externality_control', 'ordinance_count'].iloc[0] if (motive_counts['primary_motive'] == 'externality_control').any() else 0)))}
\\end{{itemize}}

Clause modalities are spread across mandatory, prohibitory, discretionary, and definitional forms, with \\texttt{{must}} and \\texttt{{must\\_not}} together accounting for the bulk of non-residual operative language.

{semantics_table}

{feature_table}

The hard-feature extraction layer is already surfacing quantities that matter for a descriptive paper: height limits, rear setbacks, side setbacks, front setbacks, minimum lot area thresholds, and penalties. In the full run, the operative-clause filter reduces the clause universe from {_fmt_int(semantics_summary['clause_rows'])} total clauses to {_fmt_int(semantics_summary['operative_clause_rows'])} operative clauses, which materially sharpens the downstream instrument summaries.

{_figure_block('figures/motive_by_region_heatmap.png', 'Heatmap of document-level primary motives by jurisdiction region in the full linked run. These are now full-sample descriptive estimates for the linked corpus, not just a smoke-test diagnostic.')}

{substantive_instrument_table}

\\section{{Embeddings, manifolds, and clustering}}

The embeddings layer ran at three levels:
\\begin{{itemize}}[leftmargin=*]
  \\item Ordinance level: {_fmt_int(embeddings_summary['ordinance']['item_count'])} texts
  \\item Section level: {_fmt_int(embeddings_summary['section']['item_count'])} texts
  \\item Clause level: {_fmt_int(embeddings_summary['clause']['item_count'])} texts
\\end{{itemize}}

The experiment registry for the full run contains {_fmt_int(clusters_summary['experiment_count'])} clustering experiments.

{experiment_table}

The main interpretive lesson from the full run is that raw ordinance-text clustering is no longer empty, but it is still less informative than the more structured layers: ordinance HDBSCAN yields 36 clusters with 115 noise cases, section HDBSCAN yields 817 clusters with 11,916 noise sections, clause HDBSCAN yields 1,082 operative-clause instrument clusters with 19,465 noise clauses, and hierarchical clause clustering compresses those operative clauses into 16 broader families.

{section_cluster_table}

{clause_cluster_table}

{_figure_block('figures/ordinance_semantic_hdbscan_umap.png', 'Ordinance-level UMAP colored by ordinance semantic HDBSCAN. At full scale this view recovers non-noise clusters, but the content remains semantically coarse relative to the more structured policy-signature and operative-clause views.')}

{_figure_block('figures/section_semantic_hdbscan_umap.png', 'Section-level UMAP colored by section semantic HDBSCAN. Section-level semantic structure is much richer than whole-document clustering in the full run.')}

{_figure_block('figures/section_graph_mechanics_umap.png', 'Section-level manifold with mechanics graph clustering. In the full run this view resolves into three graph communities, but it is still less interpretable than the operative-clause and policy-signature layers.')}

{_figure_block('figures/clause_instrument_hdbscan_umap.png', 'Clause-level UMAP colored by operative-clause instrument HDBSCAN. This remains the most informative density-clustering surface in the full run.')}

{_figure_block('figures/clause_hierarchical_instrument_umap.png', 'Clause-level manifold viewed through hierarchical instrument clustering, which gives a coarser partition than HDBSCAN and helps summarize broad instrument families.')}

{policy_signature_table}

{policy_region_table}

\\section{{Outliers, duplicates, and review infrastructure}}

The story layer produces outlier packets, mismatch packets, representative-region packets, and cluster-exemplar packets. The review layer produces structured review surfaces for jurisdictions, section boundaries, clauses, and cluster interpretation.

{outlier_table}

{_figure_block('figures/duplicate_group_sizes.png', 'Duplicate-group size distribution for the full linked run. Most ordinances are unique, but the small set of duplicate groups confirms that republication and near-republication are real empirical features of the corpus.')}

{_table_block(
        caption="Review infrastructure generated by the full-sample run.",
        label="tab:frontier-review-summary",
        colspec=r">{\raggedright\arraybackslash}Xr",
        header=["Review artifact", "Rows / packets"],
        rows=[
            ["Jurisdiction review rows", _fmt_int(review_summary["jurisdiction_review_count"])],
            ["Section-boundary review rows", _fmt_int(review_summary["section_review_count"])],
            ["Clause review rows", _fmt_int(review_summary["clause_review_count"])],
            ["Cluster review packets", _fmt_int(review_summary["cluster_packet_count"])],
        ],
    )}

{duplicate_note}

\\section{{What we learned from the current full run}}

The full run already supports several concrete takeaways.

\\begin{{enumerate}}[leftmargin=*]
  \\item \\textbf{{Jurisdiction is not publication, but it is now mostly solved.}} The linked sample reaches {_fmt_int(corpus_summary['resolver_pid6_coverage'])} one-to-one census-linked jurisdictions out of {_fmt_int(corpus_summary['issue_count'])} bundle issues, which is enough to move the project past the metadata bottleneck and into substantive analysis.
  \\item \\textbf{{Operative-clause filtering is a real frontier gain.}} The semantic layer trims the clause universe from {_fmt_int(semantics_summary['clause_rows'])} to {_fmt_int(semantics_summary['operative_clause_rows'])} operative clauses, which sharply reduces preamble and boilerplate contamination in the instrument summaries.
  \\item \\textbf{{Regional differences are visible in both motives and regulatory structure.}} The Midwest is relatively infrastructure-oriented, the South and West lean more toward public health and safety, the South is overrepresented in setback-heavy policy signatures, the West is overrepresented in district-token / parking / conditional-use and permit / plat signatures, and the Northeast is overrepresented in appeal / variance / special-exception signatures.
  \\item \\textbf{{Structured ordinance typology is more informative than raw document clustering.}} The policy-signature layer recovers six reportable ordinance families that map onto recognizable regulatory styles, while raw ordinance embeddings remain semantically coarser.
\\end{{enumerate}}

\\section{{Current limitations and next steps}}

This entry-point report should be read with three current limitations in mind.

\\begin{{enumerate}}[leftmargin=*]
  \\item The current frontier embeddings are still local TF--IDF + SVD representations rather than external semantic embeddings. A targeted section-level embedding comparison remains an open extension.
  \\item The policy-signature probe currently produces 8 clusters, but only 6 are reportable; the built-in sweep recommends a 6-cluster solution for a cleaner paper-facing typology.
  \\item Some mismatch cases are genuine cross-jurisdiction publications, while others are name-normalization problems. The audit infrastructure exposes both, but mismatch counts should still be interpreted as an audit object rather than the headline estimand.
\\end{{enumerate}}

The next practical step is no longer basic corpus construction. It is interpretive and methodological refinement: tighten the policy-signature solution around the recommended 6-cluster regime, compare section-level embedding alternatives, and decide which region-by-typology contrasts belong in the paper's main descriptive results.

\\section{{Artifact map}}

The generated artifact bundle for this full-sample run is organized as follows:
\\begin{{itemize}}[leftmargin=*]
  \\item Corpus outputs: \\path{{{run_root / 'corpus'}}}
  \\item Geography outputs: \\path{{{run_root / 'geo'}}}
  \\item Structure outputs: \\path{{{run_root / 'structure'}}}
  \\item Semantic outputs: \\path{{{run_root / 'semantics'}}}
  \\item Embeddings: \\path{{{run_root / 'embeddings'}}}
  \\item Clusters: \\path{{{run_root / 'clusters'}}}
  \\item Review packets: \\path{{{run_root / 'review'}}}
  \\item Story packets: \\path{{{run_root / 'story'}}}
  \\item Markdown report bundle: \\path{{{report_summary['report_path']}}}
\\end{{itemize}}

\\end{{document}}
""".strip() + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_tex, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a LaTeX entry-point writeup for a frontier full-code run.")
    parser.add_argument("--run-root", required=True, help="Frontier full-code run root.")
    parser.add_argument("--output", default="", help="Optional output .tex path.")
    args = parser.parse_args()

    output_path = build_entrypoint_tex(
        run_root=Path(args.run_root),
        output_path=Path(args.output).expanduser() if args.output else None,
    )
    print(f"built_frontier_entrypoint_tex output={output_path}")


if __name__ == "__main__":
    main()
