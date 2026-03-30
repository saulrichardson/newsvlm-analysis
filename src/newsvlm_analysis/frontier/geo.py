from __future__ import annotations

import pandas as pd

from .io_utils import make_city_key, normalize_city_name, write_json, write_parquet


def build_geo_outputs(*, run_root) -> dict[str, object]:
    corpus_dir = run_root / "corpus"
    geo_dir = run_root / "geo"
    geo_df = pd.read_parquet(corpus_dir / "geo_links.parquet")
    publication_df = pd.read_parquet(corpus_dir / "publication_issues.parquet")
    dedup_df = pd.read_parquet(corpus_dir / "dedup_groups.parquet")

    geo_df = geo_df.copy()
    geo_df["publication_city_key"] = geo_df.apply(
        lambda r: make_city_key(str(r.get("publication_city_name", "")), str(r.get("publication_state_abbr", "")))
        if not str(r.get("publication_city_key", "")).strip()
        else str(r.get("publication_city_key", "")).strip(),
        axis=1,
    )
    geo_df["jurisdiction_city_key_norm"] = geo_df.apply(
        lambda r: make_city_key(str(r.get("jurisdiction_city_name", "")), str(r.get("jurisdiction_state_abbr", ""))),
        axis=1,
    )

    def classify(row: pd.Series) -> str:
        if not str(row.get("jurisdiction_city_name", "")).strip() or not str(row.get("jurisdiction_state_abbr", "")).strip():
            return "missing_jurisdiction"
        if str(row.get("publication_city_key", "")).strip() == str(row.get("jurisdiction_city_key_norm", "")).strip():
            return "match"
        return "mismatch"

    geo_df["jurisdiction_match_status"] = geo_df.apply(classify, axis=1)
    geo_df["jurisdiction_match_reason"] = geo_df["jurisdiction_match_status"].map(
        {
            "match": "same_city_key",
            "mismatch": "different_city_key",
            "missing_jurisdiction": "missing_jurisdiction_fields",
        }
    )

    pub_agg = (
        geo_df.groupby(["publication_region", "publication_state_abbr", "publication_city_key", "publication_key"], dropna=False)
        .agg(
            issue_count=("issue_id", "count"),
            mismatch_count=("jurisdiction_match_status", lambda s: int((s == "mismatch").sum())),
            missing_jurisdiction_count=("jurisdiction_match_status", lambda s: int((s == "missing_jurisdiction").sum())),
        )
        .reset_index()
        .sort_values(["mismatch_count", "issue_count", "publication_key"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    pub_agg["mismatch_share"] = pub_agg["mismatch_count"] / pub_agg["issue_count"].clip(lower=1)

    publication_with_dedup = publication_df.merge(
        geo_df[["issue_id", "jurisdiction_match_status"]], on="issue_id", how="left"
    ).merge(
        dedup_df[["dedup_group_id", "group_size"]], on="dedup_group_id", how="left"
    )
    jurisdiction_counts = (
        publication_with_dedup.groupby(["jurisdiction_region", "jurisdiction_level"], dropna=False)
        .agg(
            raw_issue_count=("issue_id", "count"),
            dedup_ordinance_count=("dedup_group_id", "nunique"),
            mismatch_count=("jurisdiction_match_status", lambda s: int((s == "mismatch").sum())),
        )
        .reset_index()
        .sort_values(["jurisdiction_region", "jurisdiction_level"])
        .reset_index(drop=True)
    )
    jurisdiction_counts["mismatch_share"] = jurisdiction_counts["mismatch_count"] / jurisdiction_counts["raw_issue_count"].clip(lower=1)

    publication_counts = (
        publication_with_dedup.groupby(["publication_region", "publication_state_abbr"], dropna=False)
        .agg(
            raw_issue_count=("issue_id", "count"),
            dedup_ordinance_count=("dedup_group_id", "nunique"),
            mismatch_count=("jurisdiction_match_status", lambda s: int((s == "mismatch").sum())),
        )
        .reset_index()
        .sort_values(["publication_region", "publication_state_abbr"])
        .reset_index(drop=True)
    )
    publication_counts["mismatch_share"] = publication_counts["mismatch_count"] / publication_counts["raw_issue_count"].clip(lower=1)

    write_parquet(geo_df, geo_dir / "jurisdiction_audit.parquet")
    write_parquet(jurisdiction_counts, geo_dir / "by_jurisdiction_geo.parquet")
    write_parquet(publication_counts, geo_dir / "by_publication_geo.parquet")
    write_parquet(pub_agg, geo_dir / "high_mismatch_publications.parquet")

    summary = {
        "issue_count": int(len(geo_df)),
        "match_count": int((geo_df["jurisdiction_match_status"] == "match").sum()),
        "mismatch_count": int((geo_df["jurisdiction_match_status"] == "mismatch").sum()),
        "missing_jurisdiction_count": int((geo_df["jurisdiction_match_status"] == "missing_jurisdiction").sum()),
        "publication_geo_coverage": int((geo_df["publication_city_key"].astype(str).str.len() > 0).sum()),
        "resolved_with_census_count": int((geo_df["census_id_pid6"].astype(str).str.len() > 0).sum())
        if "census_id_pid6" in geo_df
        else 0,
        "state_backfill_count": int(geo_df["jurisdiction_state_backfill_used"].fillna(0).astype(int).sum())
        if "jurisdiction_state_backfill_used" in geo_df
        else 0,
    }
    write_json(geo_dir / "geo_summary.json", summary)
    return summary
