from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .io_utils import clean_optional_str, write_json, write_parquet

GENERIC_LABEL_FEATURES = {
    "document_purpose_present",
    "clause_count_log",
    "must_share",
    "must_not_share",
    "definition_share",
    "inst_share__general_rule",
}
REPORTABLE_POLICY_CLUSTER_MIN_SIZE = 10
POLICY_HARD_FEATURE_PREFIXES = ("instrument_attr__",)
POLICY_HARD_FEATURE_NAMES = {
    "has_condition",
    "has_exception",
    "has_threshold",
    "has_spatial_scope",
}


def _feature_label(feature_name: str) -> str:
    cleaned = str(feature_name)
    for prefix in ("inst_share__", "has_feat__", "instrument_attr__"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned.replace("_", " ")


def _label_feature_names(feature_diffs: pd.Series) -> list[str]:
    positive = [name for name, value in feature_diffs.items() if float(value) > 0]
    filtered = [name for name in positive if name not in GENERIC_LABEL_FEATURES]
    if filtered:
        return filtered[:6]
    if positive:
        return positive[:6]
    return list(feature_diffs.head(6).index)


def _cluster_label_from_diffs(feature_diffs: pd.Series) -> str:
    names = _label_feature_names(feature_diffs)
    labels: list[str] = []
    seen: set[str] = set()
    for name in names:
        label = _feature_label(name)
        if label in seen:
            continue
        labels.append(label)
        seen.add(label)
        if len(labels) >= 3:
            break
    return " | ".join(labels)


def _k_sweep_summary(scaled: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    item_count = int(scaled.shape[0])
    if item_count < 20:
        return pd.DataFrame(columns=["cluster_count", "silhouette_score", "min_cluster_size", "median_cluster_size", "max_cluster_size"])
    for cluster_count in range(4, min(11, item_count)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model = KMeans(n_clusters=int(cluster_count), random_state=0, n_init=20)
            labels = model.fit_predict(scaled)
            silhouette = float(silhouette_score(scaled, labels)) if item_count > int(cluster_count) else 0.0
        cluster_sizes = np.bincount(labels)
        rows.append(
            {
                "cluster_count": int(cluster_count),
                "silhouette_score": float(silhouette),
                "min_cluster_size": int(cluster_sizes.min()) if len(cluster_sizes) else 0,
                "median_cluster_size": float(np.median(cluster_sizes)) if len(cluster_sizes) else 0.0,
                "max_cluster_size": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
            }
        )
    return pd.DataFrame.from_records(rows)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator.astype(float).div(denom).fillna(0.0)


def _load_policy_signature_frame(
    run_root: Path,
    *,
    sample: str | None = None,
    aggregation_mode: str = "raw_clause_share",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordinance_df = pd.read_parquet(run_root / "corpus" / "ordinances.parquet")
    ordinance_df = ordinance_df[
        ordinance_df["jurisdiction_resolution_status"].astype(str).eq("resolved_with_census")
        & ordinance_df["clean_text"].astype(str).str.strip().astype(bool)
    ].copy()
    if sample:
        sample_path = run_root / "corpus" / f"{sample}_sample.parquet"
        if sample_path.is_file():
            sample_docs = pd.read_parquet(sample_path)
            sample_doc_ids = set(sample_docs["ordinance_doc_id"].astype(str))
            ordinance_df = ordinance_df[ordinance_df["ordinance_doc_id"].astype(str).isin(sample_doc_ids)].copy()

    document_df = pd.read_parquet(run_root / "semantics" / "document_semantics.parquet")
    clause_semantics_path = run_root / "semantics" / "operative_clause_semantics.parquet"
    feature_obs_path = run_root / "semantics" / "operative_hard_feature_observations.parquet"
    clause_df = pd.read_parquet(clause_semantics_path if clause_semantics_path.is_file() else run_root / "semantics" / "clause_semantics.parquet")
    feature_df = pd.read_parquet(feature_obs_path if feature_obs_path.is_file() else run_root / "semantics" / "hard_feature_observations.parquet")

    semantic_doc_ids = set(document_df["ordinance_doc_id"].astype(str))
    clause_doc_ids = set(clause_df["ordinance_doc_id"].astype(str))
    feature_doc_ids = set(feature_df["ordinance_doc_id"].astype(str))
    available_doc_ids = semantic_doc_ids & clause_doc_ids
    if feature_doc_ids:
        available_doc_ids = available_doc_ids & feature_doc_ids
    ordinance_df = ordinance_df[ordinance_df["ordinance_doc_id"].astype(str).isin(available_doc_ids)].copy()
    document_df = document_df[document_df["ordinance_doc_id"].astype(str).isin(available_doc_ids)].copy()
    clause_df = clause_df[clause_df["ordinance_doc_id"].astype(str).isin(available_doc_ids)].copy()
    feature_df = feature_df[feature_df["ordinance_doc_id"].astype(str).isin(available_doc_ids)].copy()

    feature_df = feature_df[
        feature_df["feature_name"].astype(str).map(
            lambda name: any(str(name).startswith(prefix) for prefix in POLICY_HARD_FEATURE_PREFIXES)
            or str(name) in POLICY_HARD_FEATURE_NAMES
        )
    ].copy()

    if aggregation_mode == "raw_clause_share":
        instrument_shares = clause_df.groupby(["ordinance_doc_id", "instrument_type"]).size().unstack(fill_value=0)
        instrument_shares = instrument_shares.div(instrument_shares.sum(axis=1), axis=0).add_prefix("inst_share__")

        hard_feature_presence = (
            feature_df.assign(feature_present=1)
            .pivot_table(index="ordinance_doc_id", columns="feature_name", values="feature_present", aggfunc="max", fill_value=0)
            .add_prefix("has_feat__")
        )

        doc_cols = [
            "ordinance_doc_id",
            "clause_count",
            "must_clause_count",
            "must_not_clause_count",
            "definition_clause_count",
            "document_purpose_present",
        ]
        if "operative_clause_count" in document_df.columns:
            doc_cols.extend(["operative_clause_count", "operative_must_clause_count", "operative_must_not_clause_count"])
        doc_summary = document_df[doc_cols].drop_duplicates("ordinance_doc_id").set_index("ordinance_doc_id")

        if "operative_clause_count" in doc_summary.columns:
            clause_count = doc_summary["operative_clause_count"].clip(lower=1)
            doc_summary = doc_summary.assign(
                clause_count_log=np.log1p(doc_summary["operative_clause_count"]),
                must_share=doc_summary["operative_must_clause_count"] / clause_count,
                must_not_share=doc_summary["operative_must_not_clause_count"] / clause_count,
                definition_share=doc_summary["definition_clause_count"] / doc_summary["clause_count"].clip(lower=1),
            ).drop(
                columns=[
                    "clause_count",
                    "must_clause_count",
                    "must_not_clause_count",
                    "operative_clause_count",
                    "operative_must_clause_count",
                    "operative_must_not_clause_count",
                    "definition_clause_count",
                ],
                errors="ignore",
            )
        else:
            clause_count = doc_summary["clause_count"].clip(lower=1)
            doc_summary = doc_summary.assign(
                clause_count_log=np.log1p(doc_summary["clause_count"]),
                must_share=doc_summary["must_clause_count"] / clause_count,
                must_not_share=doc_summary["must_not_clause_count"] / clause_count,
                definition_share=doc_summary["definition_clause_count"] / clause_count,
            ).drop(columns=["clause_count", "must_clause_count", "must_not_clause_count", "definition_clause_count"], errors="ignore")
    elif aggregation_mode == "section_presence":
        clause_section_presence = (
            clause_df[["ordinance_doc_id", "section_id", "instrument_type"]]
            .drop_duplicates()
            .assign(section_present=1.0)
        )
        instrument_section_counts = clause_section_presence.pivot_table(
            index="ordinance_doc_id",
            columns="instrument_type",
            values="section_present",
            aggfunc="sum",
            fill_value=0.0,
        )
        section_totals = clause_section_presence.groupby("ordinance_doc_id")["section_id"].nunique()
        instrument_shares = instrument_section_counts.apply(
            lambda col: _safe_divide(col, section_totals.reindex(col.index).fillna(0.0))
        ).add_prefix("inst_share__")

        feature_section_presence = (
            feature_df[["ordinance_doc_id", "section_id", "feature_name"]]
            .drop_duplicates()
            .assign(feature_present=1.0)
        )
        hard_feature_counts = feature_section_presence.pivot_table(
            index="ordinance_doc_id",
            columns="feature_name",
            values="feature_present",
            aggfunc="sum",
            fill_value=0.0,
        )
        feature_section_totals = feature_section_presence.groupby("ordinance_doc_id")["section_id"].nunique()
        hard_feature_presence = hard_feature_counts.apply(
            lambda col: _safe_divide(col, feature_section_totals.reindex(col.index).fillna(0.0))
        ).add_prefix("has_feat__")

        operative_sections = clause_df[["ordinance_doc_id", "section_id"]].drop_duplicates()
        operative_section_counts = operative_sections.groupby("ordinance_doc_id")["section_id"].nunique()
        must_section_counts = (
            clause_df[clause_df["modality"].astype(str).eq("must")][["ordinance_doc_id", "section_id"]]
            .drop_duplicates()
            .groupby("ordinance_doc_id")["section_id"]
            .nunique()
        )
        must_not_section_counts = (
            clause_df[clause_df["modality"].astype(str).eq("must_not")][["ordinance_doc_id", "section_id"]]
            .drop_duplicates()
            .groupby("ordinance_doc_id")["section_id"]
            .nunique()
        )
        definition_section_counts = (
            clause_df[clause_df["instrument_type"].astype(str).eq("definition")][["ordinance_doc_id", "section_id"]]
            .drop_duplicates()
            .groupby("ordinance_doc_id")["section_id"]
            .nunique()
        )
        doc_summary_base = document_df[["ordinance_doc_id", "document_purpose_present"]].drop_duplicates("ordinance_doc_id").set_index("ordinance_doc_id")
        doc_summary = doc_summary_base.assign(
            clause_count_log=np.log1p(operative_section_counts.reindex(doc_summary_base.index).fillna(0.0)),
            must_share=_safe_divide(must_section_counts.reindex(doc_summary_base.index).fillna(0.0), operative_section_counts.reindex(doc_summary_base.index).fillna(0.0)),
            must_not_share=_safe_divide(must_not_section_counts.reindex(doc_summary_base.index).fillna(0.0), operative_section_counts.reindex(doc_summary_base.index).fillna(0.0)),
            definition_share=_safe_divide(definition_section_counts.reindex(doc_summary_base.index).fillna(0.0), operative_section_counts.reindex(doc_summary_base.index).fillna(0.0)),
        )
    elif aggregation_mode == "section_normalized_clause_share":
        section_instrument_counts = (
            clause_df.groupby(["ordinance_doc_id", "section_id", "instrument_type"], dropna=False)
            .size()
            .rename("clause_count")
            .reset_index()
        )
        section_totals = section_instrument_counts.groupby(["ordinance_doc_id", "section_id"], dropna=False)["clause_count"].sum().rename("section_clause_total")
        section_instrument_counts = section_instrument_counts.merge(
            section_totals.reset_index(),
            on=["ordinance_doc_id", "section_id"],
            how="left",
        )
        section_instrument_counts["section_share"] = _safe_divide(
            section_instrument_counts["clause_count"],
            section_instrument_counts["section_clause_total"],
        )
        instrument_shares = (
            section_instrument_counts.groupby(["ordinance_doc_id", "instrument_type"], dropna=False)["section_share"]
            .mean()
            .unstack(fill_value=0.0)
            .add_prefix("inst_share__")
        )

        feature_section_counts = (
            feature_df[["ordinance_doc_id", "section_id", "feature_name", "clause_id"]]
            .drop_duplicates()
            .groupby(["ordinance_doc_id", "section_id", "feature_name"], dropna=False)
            .size()
            .rename("feature_clause_count")
            .reset_index()
        )
        if not feature_section_counts.empty:
            feature_section_totals = feature_section_counts.groupby(["ordinance_doc_id", "section_id"], dropna=False)["feature_clause_count"].sum().rename("feature_section_total")
            feature_section_counts = feature_section_counts.merge(
                feature_section_totals.reset_index(),
                on=["ordinance_doc_id", "section_id"],
                how="left",
            )
            feature_section_counts["section_share"] = _safe_divide(
                feature_section_counts["feature_clause_count"],
                feature_section_counts["feature_section_total"],
            )
            hard_feature_presence = (
                feature_section_counts.groupby(["ordinance_doc_id", "feature_name"], dropna=False)["section_share"]
                .mean()
                .unstack(fill_value=0.0)
                .add_prefix("has_feat__")
            )
        else:
            hard_feature_presence = pd.DataFrame()

        operative_sections = clause_df[["ordinance_doc_id", "section_id"]].drop_duplicates()
        operative_section_counts = operative_sections.groupby("ordinance_doc_id")["section_id"].nunique()
        must_section_counts = (
            clause_df[clause_df["modality"].astype(str).eq("must")][["ordinance_doc_id", "section_id"]]
            .drop_duplicates()
            .groupby("ordinance_doc_id")["section_id"]
            .nunique()
        )
        must_not_section_counts = (
            clause_df[clause_df["modality"].astype(str).eq("must_not")][["ordinance_doc_id", "section_id"]]
            .drop_duplicates()
            .groupby("ordinance_doc_id")["section_id"]
            .nunique()
        )
        definition_section_counts = (
            clause_df[clause_df["instrument_type"].astype(str).eq("definition")][["ordinance_doc_id", "section_id"]]
            .drop_duplicates()
            .groupby("ordinance_doc_id")["section_id"]
            .nunique()
        )
        doc_summary_base = document_df[["ordinance_doc_id", "document_purpose_present"]].drop_duplicates("ordinance_doc_id").set_index("ordinance_doc_id")
        doc_summary = doc_summary_base.assign(
            clause_count_log=np.log1p(operative_section_counts.reindex(doc_summary_base.index).fillna(0.0)),
            must_share=_safe_divide(must_section_counts.reindex(doc_summary_base.index).fillna(0.0), operative_section_counts.reindex(doc_summary_base.index).fillna(0.0)),
            must_not_share=_safe_divide(must_not_section_counts.reindex(doc_summary_base.index).fillna(0.0), operative_section_counts.reindex(doc_summary_base.index).fillna(0.0)),
            definition_share=_safe_divide(definition_section_counts.reindex(doc_summary_base.index).fillna(0.0), operative_section_counts.reindex(doc_summary_base.index).fillna(0.0)),
        )
    else:
        raise ValueError(f"Unsupported policy signature aggregation_mode={aggregation_mode!r}")

    base_cols = [
        "ordinance_doc_id",
        "issue_id",
        "jurisdiction_key",
        "jurisdiction_region",
        "jurisdiction_level",
        "census_id_pid6",
        "title_line",
        "dedup_group_id",
    ]
    base_df = ordinance_df[base_cols].copy()
    matrix_df = (
        base_df[["ordinance_doc_id"]]
        .merge(instrument_shares, on="ordinance_doc_id", how="left")
        .merge(hard_feature_presence, on="ordinance_doc_id", how="left")
        .merge(doc_summary, on="ordinance_doc_id", how="left")
        .fillna(0.0)
    )
    return base_df, matrix_df


def build_policy_signature_outputs(
    *,
    run_root: Path,
    cluster_count: int = 8,
    sample: str | None = None,
    aggregation_mode: str = "raw_clause_share",
) -> dict[str, Any]:
    out_dir_name = "policy_signature_kmeans_probe" if aggregation_mode == "raw_clause_share" else f"policy_signature_{aggregation_mode}_probe"
    out_dir = run_root / "clusters" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df, matrix_df = _load_policy_signature_frame(run_root, sample=sample, aggregation_mode=aggregation_mode)
    feature_columns = [c for c in matrix_df.columns if c != "ordinance_doc_id"]
    numeric = matrix_df[feature_columns].astype(float)

    feature_matrix = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(feature_matrix).all():
        raise RuntimeError(f"policy signature matrix contains non-finite values with shape {feature_matrix.shape}")

    scaled = StandardScaler().fit_transform(feature_matrix)
    if not np.isfinite(scaled).all():
        raise RuntimeError(f"policy signature scaled matrix contains non-finite values with shape {scaled.shape}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model = KMeans(n_clusters=int(cluster_count), random_state=0, n_init=30)
        labels = model.fit_predict(scaled)

    assignments = base_df.merge(matrix_df, on="ordinance_doc_id", how="left")
    assignments["policy_signature_cluster_id"] = labels

    global_mean = numeric.mean(axis=0)
    cluster_rows: list[dict[str, Any]] = []
    exemplar_rows: list[dict[str, Any]] = []

    for cluster_id in range(int(cluster_count)):
        member_mask = assignments["policy_signature_cluster_id"].astype(int).eq(cluster_id)
        member_rows = assignments.loc[member_mask].copy()
        member_features = numeric.loc[member_mask].copy()
        diffs = (member_features.mean(axis=0) - global_mean).sort_values(ascending=False)
        cluster_label = _cluster_label_from_diffs(diffs)

        center = model.cluster_centers_[cluster_id]
        member_scaled = scaled[member_mask.to_numpy()]
        distances = np.linalg.norm(member_scaled - center, axis=1)
        member_rows = member_rows.assign(centroid_distance=distances, cluster_label=cluster_label)
        exemplar_rows.extend(
            member_rows.sort_values(["centroid_distance", "ordinance_doc_id"], ascending=[True, True])
            .head(5)
            .drop(columns=["centroid_distance"])
            .to_dict(orient="records")
        )

        label_features = _label_feature_names(diffs)
        cluster_rows.append(
            {
                "policy_signature_cluster_id": cluster_id,
                "cluster_size": int(member_mask.sum()),
                "cluster_label": cluster_label,
                "top_features": "|".join(_feature_label(name) for name in label_features[:8]),
                "dominant_region": clean_optional_str(member_rows["jurisdiction_region"].astype(str).value_counts().index[0]) if not member_rows.empty else "",
                "dominant_level": clean_optional_str(member_rows["jurisdiction_level"].astype(str).value_counts().index[0]) if not member_rows.empty else "",
                "reportable_cluster_flag": int(int(member_mask.sum()) >= REPORTABLE_POLICY_CLUSTER_MIN_SIZE),
            }
        )

    cluster_df = pd.DataFrame.from_records(cluster_rows).sort_values("policy_signature_cluster_id").reset_index(drop=True)
    assignments = assignments.merge(cluster_df, on="policy_signature_cluster_id", how="left")
    exemplars_df = pd.DataFrame.from_records(exemplar_rows)

    regional_shares = (
        assignments.groupby(["policy_signature_cluster_id", "cluster_label", "jurisdiction_region"], as_index=False)
        .size()
        .rename(columns={"size": "ordinance_count"})
    )
    regional_shares["share_within_cluster"] = regional_shares["ordinance_count"] / regional_shares.groupby(
        "policy_signature_cluster_id"
    )["ordinance_count"].transform("sum")

    feature_means = (
        assignments.groupby(["policy_signature_cluster_id", "cluster_label"], as_index=False)[feature_columns]
        .mean()
        .sort_values("policy_signature_cluster_id")
        .reset_index(drop=True)
    )

    silhouette = float(silhouette_score(scaled, labels)) if len(assignments) > int(cluster_count) else 0.0
    k_sweep = _k_sweep_summary(scaled)
    eligible = k_sweep.loc[k_sweep["min_cluster_size"].astype(int) >= REPORTABLE_POLICY_CLUSTER_MIN_SIZE].copy()
    recommended_cluster_count = 0
    if not eligible.empty:
        recommended_cluster_count = int(
            eligible.sort_values(["silhouette_score", "cluster_count"], ascending=[False, True]).iloc[0]["cluster_count"]
        )

    write_parquet(assignments, out_dir / "assignments.parquet")
    write_parquet(cluster_df, out_dir / "cluster_labels.parquet")
    write_parquet(exemplars_df, out_dir / "cluster_exemplars.parquet")
    write_parquet(regional_shares, out_dir / "regional_shares.parquet")
    write_parquet(feature_means, out_dir / "feature_means.parquet")
    write_parquet(k_sweep, out_dir / "k_sweep.parquet")
    write_json(
        out_dir / "summary.json",
        {
            "aggregation_mode": aggregation_mode,
            "item_count": int(len(assignments)),
            "cluster_count": int(cluster_count),
            "feature_count": int(len(feature_columns)),
            "silhouette_score": silhouette,
            "reportable_cluster_count": int(cluster_df["reportable_cluster_flag"].sum()) if not cluster_df.empty else 0,
            "rare_cluster_count": int((cluster_df["reportable_cluster_flag"].astype(int) == 0).sum()) if not cluster_df.empty else 0,
            "recommended_cluster_count": int(recommended_cluster_count),
        },
    )
    return {
        "aggregation_mode": aggregation_mode,
        "item_count": int(len(assignments)),
        "cluster_count": int(cluster_count),
        "feature_count": int(len(feature_columns)),
        "silhouette_score": silhouette,
        "reportable_cluster_count": int(cluster_df["reportable_cluster_flag"].sum()) if not cluster_df.empty else 0,
        "rare_cluster_count": int((cluster_df["reportable_cluster_flag"].astype(int) == 0).sum()) if not cluster_df.empty else 0,
        "recommended_cluster_count": int(recommended_cluster_count),
    }
