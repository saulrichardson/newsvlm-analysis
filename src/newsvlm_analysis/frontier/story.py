from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import clean_optional_str, write_json, write_parquet

GENERIC_STORY_INSTRUMENTS = {"general_rule", "purpose", "definition", "zoning_map"}


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"story matrix contains non-finite values with shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    normalized = arr / norms
    if not np.isfinite(normalized).all():
        raise RuntimeError(f"story normalized matrix contains non-finite values with shape {normalized.shape}")
    return normalized


def _cosine_similarity_to_centroid(matrix: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sims = _row_normalize(matrix) @ _row_normalize(centroid).T
    if not np.isfinite(sims).all():
        raise RuntimeError(f"story cosine similarity produced non-finite values with shape {sims.shape}")
    return sims.reshape(-1)


def _primary_cluster_experiment(run_root: Path, representation: str, labeler: str) -> str:
    index_df = pd.read_parquet(run_root / "clusters" / "experiment_index.parquet")
    subset = index_df[
        (index_df["representation_level"].astype(str) == representation) & (index_df["labeler"].astype(str) == labeler)
    ].sort_values(["cluster_count", "item_count"], ascending=[False, False])
    if subset.empty:
        return ""
    return str(subset.iloc[0]["experiment_name"])


def build_story_outputs(*, run_root: Path) -> dict[str, Any]:
    story_dir = run_root / "story"
    story_dir.mkdir(parents=True, exist_ok=True)

    ordinances = pd.read_parquet(run_root / "corpus" / "ordinances.parquet")
    if "jurisdiction_resolution_status" in ordinances.columns:
        ordinances = ordinances[ordinances["jurisdiction_resolution_status"].astype(str) == "resolved_with_census"].copy()
    geo_audit = pd.read_parquet(run_root / "geo" / "jurisdiction_audit.parquet")
    doc_sem = pd.read_parquet(run_root / "semantics" / "document_semantics.parquet")
    clause_path = run_root / "semantics" / "operative_clause_semantics.parquet"
    clause_sem = pd.read_parquet(clause_path if clause_path.is_file() else run_root / "semantics" / "clause_semantics.parquet")
    dedup = pd.read_parquet(run_root / "corpus" / "dedup_groups.parquet")
    emb_meta = pd.read_parquet(run_root / "embeddings" / "ordinance" / "metadata.parquet")
    emb = np.load(run_root / "embeddings" / "ordinance" / "embedding_matrix.npy")

    base = ordinances.merge(
        geo_audit[["issue_id", "jurisdiction_match_status"]], on="issue_id", how="left"
    ).merge(
        doc_sem[
            [
                "ordinance_doc_id",
                "primary_motive",
                "dominant_instrument",
                "clause_count",
                "operative_clause_count",
                "operative_dominant_instrument",
            ]
        ],
        on="ordinance_doc_id",
        how="left",
    )

    raw_region = (
        base.groupby(["jurisdiction_region", "jurisdiction_level"], dropna=False)
        .agg(
            raw_issue_count=("issue_id", "count"),
            unique_publications=("publication_key", "nunique"),
            mismatch_count=("jurisdiction_match_status", lambda s: int((s == "mismatch").sum())),
        )
        .reset_index()
        .sort_values(["jurisdiction_region", "jurisdiction_level"])
        .reset_index(drop=True)
    )
    raw_region["mismatch_share"] = raw_region["mismatch_count"] / raw_region["raw_issue_count"].clip(lower=1)

    dedup_base = base.sort_values(["issue_date", "ordinance_doc_id"]).drop_duplicates("dedup_group_id", keep="first")
    dedup_region = (
        dedup_base.groupby(["jurisdiction_region", "jurisdiction_level"], dropna=False)
        .agg(
            dedup_ordinance_count=("dedup_group_id", "count"),
            primary_motive_mode=("primary_motive", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
        )
        .reset_index()
        .sort_values(["jurisdiction_region", "jurisdiction_level"])
        .reset_index(drop=True)
    )

    motive_by_region = (
        dedup_base.groupby(["jurisdiction_region", "primary_motive"], dropna=False)["ordinance_doc_id"]
        .count()
        .reset_index(name="ordinance_count")
        .sort_values(["jurisdiction_region", "ordinance_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    motive_by_region["share_within_region"] = motive_by_region.groupby("jurisdiction_region")["ordinance_count"].transform(
        lambda s: s / s.sum()
    )

    analysis_docs = dedup_base[
        ["ordinance_doc_id", "jurisdiction_region", "publication_region", "jurisdiction_level"]
    ].drop_duplicates("ordinance_doc_id").copy()
    clause_merge_cols = ["ordinance_doc_id", "publication_region"]
    for col in ("jurisdiction_region", "jurisdiction_level"):
        if col not in clause_sem.columns:
            clause_merge_cols.append(col)
    clause_region = clause_sem.merge(analysis_docs[clause_merge_cols], on="ordinance_doc_id", how="inner")
    instrument_clause_shares_by_region = (
        clause_region.groupby(["jurisdiction_region", "instrument_type"], dropna=False)["clause_id"]
        .count()
        .reset_index(name="clause_count")
        .sort_values(["jurisdiction_region", "clause_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    instrument_clause_shares_by_region["share_within_region"] = instrument_clause_shares_by_region.groupby("jurisdiction_region")["clause_count"].transform(
        lambda s: s / s.sum()
    )
    ordinance_instrument_presence = (
        clause_region.groupby(["ordinance_doc_id", "jurisdiction_region", "jurisdiction_level", "instrument_type"], dropna=False)
        .size()
        .reset_index(name="clause_count")
    )
    region_totals = analysis_docs.groupby("jurisdiction_region", dropna=False)["ordinance_doc_id"].nunique().reset_index(name="region_ordinance_total")
    level_totals = analysis_docs.groupby("jurisdiction_level", dropna=False)["ordinance_doc_id"].nunique().reset_index(name="level_ordinance_total")
    instrument_presence_by_region = (
        ordinance_instrument_presence.groupby(["jurisdiction_region", "instrument_type"], dropna=False)["ordinance_doc_id"]
        .nunique()
        .reset_index(name="ordinance_count")
        .merge(region_totals, on="jurisdiction_region", how="left")
        .sort_values(["jurisdiction_region", "ordinance_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    instrument_presence_by_region["share_within_region"] = instrument_presence_by_region["ordinance_count"] / instrument_presence_by_region[
        "region_ordinance_total"
    ].clip(lower=1)
    substantive_instrument_presence_by_region = instrument_presence_by_region[
        ~instrument_presence_by_region["instrument_type"].astype(str).isin(GENERIC_STORY_INSTRUMENTS)
    ].reset_index(drop=True)
    overall_instrument_share = (
        ordinance_instrument_presence.groupby("instrument_type", dropna=False)["ordinance_doc_id"]
        .nunique()
        .reset_index(name="overall_ordinance_count")
    )
    overall_instrument_share["overall_share"] = overall_instrument_share["overall_ordinance_count"] / analysis_docs["ordinance_doc_id"].nunique()
    substantive_instrument_presence_by_region = substantive_instrument_presence_by_region.merge(
        overall_instrument_share[["instrument_type", "overall_share"]],
        on="instrument_type",
        how="left",
    )
    substantive_instrument_presence_by_region["share_lift_vs_overall"] = substantive_instrument_presence_by_region["share_within_region"] / substantive_instrument_presence_by_region[
        "overall_share"
    ].clip(lower=1e-9)
    substantive_instrument_presence_by_region["share_diff_vs_overall"] = (
        substantive_instrument_presence_by_region["share_within_region"] - substantive_instrument_presence_by_region["overall_share"]
    )
    instrument_presence_by_level = (
        ordinance_instrument_presence.groupby(["jurisdiction_level", "instrument_type"], dropna=False)["ordinance_doc_id"]
        .nunique()
        .reset_index(name="ordinance_count")
        .merge(level_totals, on="jurisdiction_level", how="left")
        .sort_values(["jurisdiction_level", "ordinance_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    instrument_presence_by_level["share_within_level"] = instrument_presence_by_level["ordinance_count"] / instrument_presence_by_level[
        "level_ordinance_total"
    ].clip(lower=1)
    substantive_instrument_presence_by_level = instrument_presence_by_level[
        ~instrument_presence_by_level["instrument_type"].astype(str).isin(GENERIC_STORY_INSTRUMENTS)
    ].reset_index(drop=True)

    high_mismatch = (
        base.groupby(["publication_key", "publication_city_name", "publication_state_abbr"], dropna=False)
        .agg(
            issue_count=("issue_id", "count"),
            mismatch_count=("jurisdiction_match_status", lambda s: int((s == "mismatch").sum())),
        )
        .reset_index()
    )
    high_mismatch["mismatch_share"] = high_mismatch["mismatch_count"] / high_mismatch["issue_count"].clip(lower=1)
    high_mismatch = high_mismatch.sort_values(["mismatch_share", "issue_count"], ascending=[False, False]).reset_index(drop=True)

    duplicate_patterns = (
        ordinances.groupby("dedup_group_id", dropna=False)
        .agg(
            issue_count=("issue_id", "count"),
            unique_publications=("publication_key", "nunique"),
            unique_jurisdictions=("jurisdiction_key", "nunique"),
        )
        .reset_index()
        .merge(dedup[["dedup_group_id", "representative_ordinance_doc_id"]], on="dedup_group_id", how="left")
        .sort_values(["issue_count", "unique_publications"], ascending=[False, False])
        .reset_index(drop=True)
    )

    if len(emb_meta) and emb.shape[0] == len(emb_meta):
        centroid = emb.mean(axis=0, keepdims=True)
        global_sims = _cosine_similarity_to_centroid(emb, centroid)
        outliers = emb_meta[["item_id", "jurisdiction_key", "publication_key", "issue_date", "title_line"]].copy()
        outliers["global_centroid_similarity"] = global_sims
        outliers = outliers.sort_values(["global_centroid_similarity", "item_id"], ascending=[True, True]).reset_index(drop=True)
    else:
        outliers = pd.DataFrame(columns=["item_id", "jurisdiction_key", "publication_key", "issue_date", "title_line", "global_centroid_similarity"])

    representative_rows: list[dict[str, Any]] = []
    if len(emb_meta) and emb.shape[0] == len(emb_meta):
        region_frame = emb_meta.copy()
        needed = [col for col in ["jurisdiction_region", "title_line"] if col not in region_frame.columns]
        if needed:
            region_frame = region_frame.merge(
                ordinances[["ordinance_doc_id", *needed]],
                left_on="item_id",
                right_on="ordinance_doc_id",
                how="left",
            )
        for region, group in region_frame.groupby("jurisdiction_region", dropna=False):
            idx = group["embedding_row_index"].to_numpy(dtype=int)
            region_emb = emb[idx]
            region_centroid = region_emb.mean(axis=0, keepdims=True)
            sims = _cosine_similarity_to_centroid(region_emb, region_centroid)
            best = group.assign(region_centroid_similarity=sims).sort_values(
                ["region_centroid_similarity", "item_id"], ascending=[False, True]
            ).iloc[0]
            representative_rows.append(best.to_dict())
    representative_df = pd.DataFrame.from_records(representative_rows)

    ordinance_cluster_exp = _primary_cluster_experiment(run_root, "ordinance", "semantic")
    cluster_exemplar_df = pd.DataFrame()
    if ordinance_cluster_exp:
        cluster_exemplar_path = run_root / "clusters" / ordinance_cluster_exp / "cluster_exemplars.parquet"
        if cluster_exemplar_path.is_file():
            cluster_exemplar_df = pd.read_parquet(cluster_exemplar_path)

    policy_dir = run_root / "clusters" / "policy_signature_kmeans_probe"
    policy_cluster_df = pd.DataFrame()
    policy_regional_df = pd.DataFrame()
    policy_exemplars_df = pd.DataFrame()
    policy_region_mix_df = pd.DataFrame()
    policy_level_mix_df = pd.DataFrame()
    if (policy_dir / "cluster_labels.parquet").is_file():
        policy_cluster_df = pd.read_parquet(policy_dir / "cluster_labels.parquet")
    if (policy_dir / "regional_shares.parquet").is_file():
        policy_regional_df = pd.read_parquet(policy_dir / "regional_shares.parquet")
    if (policy_dir / "cluster_exemplars.parquet").is_file():
        policy_exemplars_df = pd.read_parquet(policy_dir / "cluster_exemplars.parquet")
    if (policy_dir / "assignments.parquet").is_file():
        policy_assignments = pd.read_parquet(policy_dir / "assignments.parquet")
        policy_region_mix_df = (
            policy_assignments.groupby(["jurisdiction_region", "policy_signature_cluster_id", "cluster_label"], dropna=False)["ordinance_doc_id"]
            .count()
            .reset_index(name="ordinance_count")
            .merge(region_totals, on="jurisdiction_region", how="left")
            .sort_values(["jurisdiction_region", "ordinance_count"], ascending=[True, False])
            .reset_index(drop=True)
        )
        policy_region_mix_df["share_within_region"] = policy_region_mix_df["ordinance_count"] / policy_region_mix_df[
            "region_ordinance_total"
        ].clip(lower=1)
        overall_policy_share = (
            policy_assignments.groupby(["policy_signature_cluster_id", "cluster_label"], dropna=False)["ordinance_doc_id"]
            .count()
            .reset_index(name="overall_ordinance_count")
        )
        overall_policy_share["overall_share"] = overall_policy_share["overall_ordinance_count"] / policy_assignments["ordinance_doc_id"].nunique()
        policy_region_mix_df = policy_region_mix_df.merge(
            overall_policy_share[["policy_signature_cluster_id", "cluster_label", "overall_share"]],
            on=["policy_signature_cluster_id", "cluster_label"],
            how="left",
        )
        policy_region_mix_df["share_lift_vs_overall"] = policy_region_mix_df["share_within_region"] / policy_region_mix_df[
            "overall_share"
        ].clip(lower=1e-9)
        policy_region_mix_df["share_diff_vs_overall"] = policy_region_mix_df["share_within_region"] - policy_region_mix_df["overall_share"]
        policy_level_mix_df = (
            policy_assignments.groupby(["jurisdiction_level", "policy_signature_cluster_id", "cluster_label"], dropna=False)["ordinance_doc_id"]
            .count()
            .reset_index(name="ordinance_count")
            .merge(level_totals, on="jurisdiction_level", how="left")
            .sort_values(["jurisdiction_level", "ordinance_count"], ascending=[True, False])
            .reset_index(drop=True)
        )
        policy_level_mix_df["share_within_level"] = policy_level_mix_df["ordinance_count"] / policy_level_mix_df[
            "level_ordinance_total"
        ].clip(lower=1)

    write_parquet(raw_region, story_dir / "regional_counts_raw.parquet")
    write_parquet(dedup_region, story_dir / "regional_counts_dedup.parquet")
    write_parquet(motive_by_region, story_dir / "motive_shares_by_region.parquet")
    write_parquet(instrument_clause_shares_by_region, story_dir / "instrument_clause_shares_by_region.parquet")
    write_parquet(instrument_presence_by_region, story_dir / "instrument_shares_by_region.parquet")
    write_parquet(substantive_instrument_presence_by_region, story_dir / "substantive_instrument_shares_by_region.parquet")
    write_parquet(instrument_presence_by_level, story_dir / "instrument_shares_by_level.parquet")
    write_parquet(substantive_instrument_presence_by_level, story_dir / "substantive_instrument_shares_by_level.parquet")
    write_parquet(high_mismatch, story_dir / "high_mismatch_publications.parquet")
    write_parquet(outliers, story_dir / "outlier_ordinances.parquet")
    write_parquet(representative_df, story_dir / "representative_ordinances_by_region.parquet")
    write_parquet(duplicate_patterns, story_dir / "duplicate_publication_patterns.parquet")
    if not cluster_exemplar_df.empty:
        write_parquet(cluster_exemplar_df, story_dir / "cluster_exemplars.parquet")
    if not policy_cluster_df.empty:
        write_parquet(policy_cluster_df, story_dir / "policy_signature_clusters.parquet")
    if not policy_region_mix_df.empty:
        write_parquet(policy_region_mix_df, story_dir / "policy_signature_shares_by_region.parquet")
    if not policy_regional_df.empty:
        write_parquet(policy_regional_df, story_dir / "policy_signature_cluster_composition_by_region.parquet")
    if not policy_level_mix_df.empty:
        write_parquet(policy_level_mix_df, story_dir / "policy_signature_shares_by_level.parquet")
    if not policy_exemplars_df.empty:
        write_parquet(policy_exemplars_df, story_dir / "policy_signature_cluster_exemplars.parquet")

    packets_dir = story_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    for _, row in high_mismatch.head(10).iterrows():
        key = str(row["publication_key"])
        subset = base[base["publication_key"] == key].sort_values(["issue_date", "issue_id"]).head(5)
        lines = [
            f"# High mismatch publication: {key}",
            "",
            f"- publication_city: {clean_optional_str(row['publication_city_name'])}, {clean_optional_str(row['publication_state_abbr'])}",
            f"- issue_count: {int(row['issue_count'])}",
            f"- mismatch_share: {float(row['mismatch_share']):.3f}",
            "",
            "## Example ordinances",
            "",
        ]
        for ex in subset.itertuples(index=False):
            lines.append(f"- `{clean_optional_str(getattr(ex, 'issue_id', ''))}` -> `{clean_optional_str(getattr(ex, 'jurisdiction_key', ''))}`")
        (packets_dir / f"high_mismatch__{key}.md").write_text("\n".join(lines), encoding="utf-8")

    for _, row in outliers.head(10).iterrows():
        item_id = clean_optional_str(row["item_id"])
        text = clean_optional_str(ordinances.loc[ordinances["ordinance_doc_id"] == item_id, "clean_text"].head(1).squeeze())
        lines = [
            f"# Outlier ordinance: {item_id}",
            "",
            f"- jurisdiction_key: `{clean_optional_str(row['jurisdiction_key'])}`",
            f"- publication_key: `{clean_optional_str(row['publication_key'])}`",
            f"- issue_date: `{clean_optional_str(row['issue_date'])}`",
            f"- global_centroid_similarity: {float(row['global_centroid_similarity']):.4f}",
            "",
            "```text",
            text[:3000],
            "```",
        ]
        (packets_dir / f"outlier__{item_id}.md").write_text("\n".join(lines), encoding="utf-8")

    for _, row in representative_df.iterrows():
        region = clean_optional_str(row.get("jurisdiction_region"))
        item_id = clean_optional_str(row.get("item_id"))
        lines = [
            f"# Representative ordinance for region: {region}",
            "",
            f"- ordinance_doc_id: `{item_id}`",
            f"- similarity: {float(row.get('region_centroid_similarity', 0.0)):.4f}",
            f"- title: {clean_optional_str(row.get('title_line'))}",
        ]
        (packets_dir / f"representative_region__{region}.md").write_text("\n".join(lines), encoding="utf-8")

    if not cluster_exemplar_df.empty and ordinance_cluster_exp:
        for cluster_id, group in cluster_exemplar_df.groupby("cluster_id", dropna=False):
            if int(cluster_id) < 0:
                continue
            top = group.head(3)
            lines = [f"# Cluster exemplars: {ordinance_cluster_exp} cluster {int(cluster_id)}", ""]
            for ex in top.itertuples(index=False):
                lines.append(f"## {clean_optional_str(getattr(ex, 'item_id', ''))}")
                lines.append("")
                lines.append("```text")
                lines.append(clean_optional_str(getattr(ex, "text_for_embedding", ""))[:2000])
                lines.append("```")
                lines.append("")
            (packets_dir / f"cluster_exemplars__{ordinance_cluster_exp}__{int(cluster_id):04d}.md").write_text(
                "\n".join(lines), encoding="utf-8"
            )

    if not policy_exemplars_df.empty:
        policy_assignments = pd.read_parquet(policy_dir / "assignments.parquet")
        for cluster_id, group in policy_exemplars_df.groupby("policy_signature_cluster_id", dropna=False):
            label = clean_optional_str(group["cluster_label"].iloc[0]) if "cluster_label" in group.columns and not group.empty else ""
            lines = [f"# Policy signature cluster {int(cluster_id)}", "", f"- cluster_label: {label}", ""]
            for ex in group.head(4).itertuples(index=False):
                ordinance_doc_id = clean_optional_str(getattr(ex, "ordinance_doc_id", ""))
                full_row = policy_assignments.loc[policy_assignments["ordinance_doc_id"] == ordinance_doc_id].head(1)
                title_line = clean_optional_str(full_row["title_line"].squeeze()) if not full_row.empty else ""
                lines.append(f"## {ordinance_doc_id}")
                lines.append("")
                lines.append(f"- jurisdiction_key: `{clean_optional_str(getattr(ex, 'jurisdiction_key', ''))}`")
                lines.append(f"- region: `{clean_optional_str(getattr(ex, 'jurisdiction_region', ''))}`")
                lines.append(f"- title: {title_line}")
                lines.append("")
            (packets_dir / f"policy_signature_cluster__{int(cluster_id):02d}.md").write_text("\n".join(lines), encoding="utf-8")

    write_json(
        story_dir / "story_summary.json",
        {
            "regional_rows": int(len(raw_region)),
            "high_mismatch_publications": int(len(high_mismatch)),
            "outlier_count": int(len(outliers)),
            "representative_region_count": int(len(representative_df)),
            "duplicate_group_count": int((duplicate_patterns["issue_count"] > 1).sum()) if not duplicate_patterns.empty else 0,
            "policy_signature_cluster_count": int(len(policy_cluster_df)),
            "reportable_policy_signature_cluster_count": int(
                policy_cluster_df["reportable_cluster_flag"].astype(int).sum()
            )
            if "reportable_cluster_flag" in policy_cluster_df.columns and not policy_cluster_df.empty
            else 0,
            "clause_source": "operative_clause_semantics" if clause_path.is_file() else "clause_semantics",
        },
    )
    return {
        "regional_rows": int(len(raw_region)),
        "outlier_count": int(len(outliers)),
        "high_mismatch_publications": int(len(high_mismatch)),
        "policy_signature_cluster_count": int(len(policy_cluster_df)),
        "reportable_policy_signature_cluster_count": int(
            policy_cluster_df["reportable_cluster_flag"].astype(int).sum()
        )
        if "reportable_cluster_flag" in policy_cluster_df.columns and not policy_cluster_df.empty
        else 0,
    }
