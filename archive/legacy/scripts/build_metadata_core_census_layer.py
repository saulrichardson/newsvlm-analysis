#!/usr/bin/env python3
"""
Build census-ready merge keys for a metadata_core bundle.

Inputs (under --metadata-core-dir):
  - newspapers.csv
  - issues.csv

Outputs (under --metadata-core-dir/census by default):
  - city_place_crosswalk.csv
  - city_place_match_candidates.csv
  - city_census_keys.csv
  - newspaper_census_keys.csv
  - city_county20_weights.csv
  - city_puma20_weights.csv
  - city_cbsa23_weights.csv
  - unmatched_city_keys.csv
  - build_summary.json

Also writes:
  - newspapers_enriched.csv (same folder as newspapers.csv)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


def _slug(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unknown"


def _norm_state(x: Any) -> str:
    return str(x or "").strip().lower()


def _norm_geoid7(x: Any) -> str:
    s = str(x or "").strip()
    s = re.sub(r"\.0$", "", s)
    return s.zfill(7) if s.isdigit() else s


def _norm_geoid5(x: Any) -> str:
    s = str(x or "").strip()
    s = re.sub(r"\.0$", "", s)
    return s.zfill(5) if s.isdigit() else s


def _to_str_or_empty(x: Any) -> str:
    s = str(x or "").strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _load_inputs(metadata_core_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    newspapers_path = metadata_core_dir / "newspapers.csv"
    issues_path = metadata_core_dir / "issues.csv"
    if not newspapers_path.is_file():
        raise SystemExit(f"Missing required file: {newspapers_path}")
    if not issues_path.is_file():
        raise SystemExit(f"Missing required file: {issues_path}")
    newspapers = pd.read_csv(newspapers_path, low_memory=False)
    issues = pd.read_csv(issues_path, low_memory=False)
    need_news = {"newspaper_slug", "city_name", "state_abbr"}
    need_issues = {"issue_id", "newspaper_slug", "issue_date"}
    miss_news = sorted(c for c in need_news if c not in newspapers.columns)
    miss_issues = sorted(c for c in need_issues if c not in issues.columns)
    if miss_news:
        raise SystemExit(f"newspapers.csv missing columns: {miss_news}")
    if miss_issues:
        raise SystemExit(f"issues.csv missing columns: {miss_issues}")
    return newspapers, issues


def _build_city_refs(newspapers: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    news = newspapers.copy()
    news["state_abbr"] = news["state_abbr"].map(_norm_state)
    news["city_name"] = news["city_name"].astype(str).str.strip()
    news["city_key"] = news.apply(lambda r: f"{_slug(r['city_name'])}__{r['state_abbr']}", axis=1)

    issue = issues.copy()
    issue["issue_year"] = pd.to_datetime(issue["issue_date"], errors="coerce").dt.year
    issue = issue.dropna(subset=["issue_year"]).copy()
    issue["issue_year"] = issue["issue_year"].astype(int)

    issue_city = issue.merge(
        news[["newspaper_slug", "city_key", "city_name", "state_abbr"]].drop_duplicates(),
        on="newspaper_slug",
        how="left",
    )
    issue_city = issue_city.dropna(subset=["city_key"]).copy()

    by_city = (
        issue_city.groupby(["city_key", "city_name", "state_abbr"], as_index=False)
        .agg(min_issue_year=("issue_year", "min"), max_issue_year=("issue_year", "max"))
    )
    by_city["region"] = "unknown"
    by_city["urbanicity_proxy"] = "unknown"
    by_city = by_city.sort_values(["state_abbr", "city_name", "city_key"]).reset_index(drop=True)
    return by_city


def _run_place_matching(
    city_refs: pd.DataFrame,
    output_dir: Path,
    *,
    fuzzy_threshold: float,
    candidate_top_n: int,
    refresh_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Reuse existing matching logic from the covariate builder.
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import build_external_covariate_panel as bec  # type: ignore

    refs: list[Any] = []
    for r in city_refs.itertuples(index=False):
        refs.append(
            bec.CityRef(
                city_key=str(r.city_key),
                city_name=str(r.city_name),
                state_abbr=str(r.state_abbr),
                region="unknown",
                urbanicity_proxy="unknown",
                min_issue_year=int(r.min_issue_year),
                max_issue_year=int(r.max_issue_year),
            )
        )

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gaz = bec._load_gazetteer_places(cache_dir, refresh=bool(refresh_cache))
    place_xw, candidates = bec._match_city_to_place(
        refs,
        gaz,
        manual_overrides={},
        fuzzy_threshold=float(fuzzy_threshold),
        candidate_top_n=int(candidate_top_n),
    )
    return place_xw, candidates


def _run_geo_backbone(place_xw_path: Path, output_dir: Path, *, refresh_cache: bool) -> Path:
    scripts_dir = Path(__file__).resolve().parent
    geo_script = scripts_dir / "build_geo_backbone_crosswalk.py"
    if not geo_script.is_file():
        raise SystemExit(f"Missing geo backbone script: {geo_script}")
    tmp_run = output_dir / "_tmp_run"
    tmp_run.mkdir(parents=True, exist_ok=True)
    geo_out = output_dir / "geo_backbone"
    cmd = [
        sys.executable,
        str(geo_script),
        "--run-dir",
        str(tmp_run),
        "--city-place-crosswalk",
        str(place_xw_path),
        "--output-dir",
        str(geo_out),
    ]
    if refresh_cache:
        cmd.append("--refresh-cache")
    subprocess.run(cmd, check=True)
    return geo_out


def _load_primary_ids(geo_out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    county = pd.read_csv(geo_out / "city_place20_county20_crosswalk.csv", low_memory=False)
    puma = pd.read_csv(geo_out / "city_place20_puma20_crosswalk.csv", low_memory=False)
    cbsa = pd.read_csv(geo_out / "city_place20_cbsa23_crosswalk.csv", low_memory=False)

    county = county.sort_values(["city_key", "weight_place20_to_county20"], ascending=[True, False]).copy()
    county_top = county.groupby("city_key", as_index=False).first()
    county_top = county_top.rename(
        columns={
            "county20_fips": "county20_fips_primary",
            "weight_place20_to_county20": "county20_primary_weight",
        }
    )[["city_key", "county20_fips_primary", "county20_primary_weight"]]

    puma = puma.sort_values(["city_key", "weight_place20_to_puma20"], ascending=[True, False]).copy()
    puma_top = puma.groupby("city_key", as_index=False).first()
    puma_top = puma_top.rename(
        columns={
            "puma20_geoid": "puma20_geoid_primary",
            "weight_place20_to_puma20": "puma20_primary_weight",
        }
    )[["city_key", "puma20_geoid_primary", "puma20_primary_weight"]]

    cbsa = cbsa.sort_values(["city_key", "weight_place20_to_cbsa23"], ascending=[True, False]).copy()
    cbsa_top = cbsa.groupby("city_key", as_index=False).first()
    cbsa_top = cbsa_top.rename(
        columns={
            "cbsa23_geoid": "cbsa23_geoid_primary",
            "cbsa23_name": "cbsa23_name_primary",
            "weight_place20_to_cbsa23": "cbsa23_primary_weight",
        }
    )[["city_key", "cbsa23_geoid_primary", "cbsa23_name_primary", "cbsa23_primary_weight"]]

    return county_top, puma_top, cbsa_top


def main() -> None:
    ap = argparse.ArgumentParser(description="Build census merge layer for metadata_core.")
    ap.add_argument("--metadata-core-dir", required=True, help="Path to metadata_core folder.")
    ap.add_argument("--output-subdir", default="census", help="Subdir name under metadata_core (default: census).")
    ap.add_argument("--fuzzy-match-threshold", type=float, default=0.92, help="Place matching threshold.")
    ap.add_argument("--candidate-top-n", type=int, default=8, help="Top-N city-place candidates to keep.")
    ap.add_argument("--refresh-cache", action="store_true", help="Refresh census cache files.")
    args = ap.parse_args()

    metadata_core_dir = Path(args.metadata_core_dir).expanduser().resolve()
    output_dir = metadata_core_dir / str(args.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    newspapers, issues = _load_inputs(metadata_core_dir)
    city_refs = _build_city_refs(newspapers, issues)

    place_xw, candidates = _run_place_matching(
        city_refs,
        output_dir,
        fuzzy_threshold=float(args.fuzzy_match_threshold),
        candidate_top_n=int(args.candidate_top_n),
        refresh_cache=bool(args.refresh_cache),
    )
    place_xw["place_geoid"] = place_xw["place_geoid"].map(_norm_geoid7)
    place_xw_path = output_dir / "city_place_crosswalk.csv"
    place_xw.to_csv(place_xw_path, index=False)
    candidates.to_csv(output_dir / "city_place_match_candidates.csv", index=False)

    unmatched = place_xw[place_xw["place_geoid"].astype(str).str.len() != 7][["city_key", "city_name", "state_abbr", "match_method", "match_confidence"]].copy()
    unmatched.to_csv(output_dir / "unmatched_city_keys.csv", index=False)

    # Build weighted geography crosswalks off matched places only.
    place_xw_matched = place_xw[place_xw["place_geoid"].astype(str).str.len() == 7].copy()
    place_xw_matched.to_csv(place_xw_path, index=False)
    geo_out = _run_geo_backbone(place_xw_path, output_dir, refresh_cache=bool(args.refresh_cache))

    county_w = pd.read_csv(geo_out / "city_place20_county20_crosswalk.csv", low_memory=False)
    puma_w = pd.read_csv(geo_out / "city_place20_puma20_crosswalk.csv", low_memory=False)
    cbsa_w = pd.read_csv(geo_out / "city_place20_cbsa23_crosswalk.csv", low_memory=False)
    quality = pd.read_csv(geo_out / "crosswalk_city_quality.csv", low_memory=False)

    county_w.to_csv(output_dir / "city_county20_weights.csv", index=False)
    puma_w.to_csv(output_dir / "city_puma20_weights.csv", index=False)
    cbsa_w.to_csv(output_dir / "city_cbsa23_weights.csv", index=False)

    county_top, puma_top, cbsa_top = _load_primary_ids(geo_out)
    city_keys = place_xw_matched[
        ["city_key", "city_name", "state_abbr", "place_geoid", "match_method", "match_confidence", "match_score"]
    ].copy()
    city_keys = city_keys.rename(
        columns={
            "place_geoid": "place20_geoid",
            "match_method": "place_match_method",
            "match_confidence": "place_match_confidence",
            "match_score": "place_match_score",
        }
    )
    for c in ["place20_geoid"]:
        if c in city_keys.columns:
            city_keys[c] = city_keys[c].map(_norm_geoid7)

    quality_keep = [
        "city_key",
        "place10_count",
        "zcta20_count",
        "tract20_count",
        "county20_count",
        "puma20_count",
        "cbsa23_count",
        "county20_split_flag",
        "puma20_split_flag",
        "cbsa23_split_flag",
    ]
    quality_keep = [c for c in quality_keep if c in quality.columns]
    quality_lean = quality[quality_keep].copy()

    city_keys = (
        city_keys.merge(quality_lean, on="city_key", how="left")
        .merge(county_top, on="city_key", how="left")
        .merge(puma_top, on="city_key", how="left")
        .merge(cbsa_top, on="city_key", how="left")
    )
    if "county20_fips_primary" in city_keys.columns:
        city_keys["county20_fips_primary"] = city_keys["county20_fips_primary"].map(_norm_geoid5)
    if "puma20_geoid_primary" in city_keys.columns:
        city_keys["puma20_geoid_primary"] = city_keys["puma20_geoid_primary"].map(_norm_geoid7)
    if "cbsa23_geoid_primary" in city_keys.columns:
        city_keys["cbsa23_geoid_primary"] = city_keys["cbsa23_geoid_primary"].map(_norm_geoid5)
    for c in ["place_match_method", "place_match_confidence"]:
        if c in city_keys.columns:
            city_keys[c] = city_keys[c].map(_to_str_or_empty)
    city_keys.to_csv(output_dir / "city_census_keys.csv", index=False)

    # Newspaper-level easy-join table and enriched newspapers.
    news = newspapers.copy()
    news["state_abbr"] = news["state_abbr"].map(_norm_state)
    news["city_name"] = news["city_name"].astype(str).str.strip()
    news["city_key"] = news.apply(lambda r: f"{_slug(r['city_name'])}__{r['state_abbr']}", axis=1)

    news_keys = news.merge(city_keys, on=["city_key", "city_name", "state_abbr"], how="left")
    for c in ["place20_geoid", "county20_fips_primary", "puma20_geoid_primary", "cbsa23_geoid_primary"]:
        if c in news_keys.columns:
            news_keys[c] = news_keys[c].map(_to_str_or_empty)
    for c in ["place_match_method", "place_match_confidence"]:
        if c in news_keys.columns:
            news_keys[c] = news_keys[c].map(_to_str_or_empty)
    news_keys.to_csv(output_dir / "newspaper_census_keys.csv", index=False)
    news_keys.to_csv(metadata_core_dir / "newspapers_enriched.csv", index=False)

    place_nonempty = (
        news_keys["place20_geoid"].astype(str).str.strip().str.len().gt(0).sum()
        if "place20_geoid" in news_keys.columns
        else 0
    )
    county_nonempty = (
        news_keys["county20_fips_primary"].astype(str).str.strip().str.len().gt(0).sum()
        if "county20_fips_primary" in news_keys.columns
        else 0
    )
    puma_nonempty = (
        news_keys["puma20_geoid_primary"].astype(str).str.strip().str.len().gt(0).sum()
        if "puma20_geoid_primary" in news_keys.columns
        else 0
    )
    cbsa_nonempty = (
        news_keys["cbsa23_geoid_primary"].astype(str).str.strip().str.len().gt(0).sum()
        if "cbsa23_geoid_primary" in news_keys.columns
        else 0
    )

    summary = {
        "metadata_core_dir": str(metadata_core_dir),
        "output_dir": str(output_dir),
        "newspaper_rows": int(len(news)),
        "city_rows": int(len(city_refs)),
        "matched_city_rows": int(len(place_xw_matched)),
        "unmatched_city_rows": int(len(unmatched)),
        "newspaper_rows_with_place20": int(place_nonempty),
        "newspaper_rows_with_county20": int(county_nonempty),
        "newspaper_rows_with_puma20": int(puma_nonempty),
        "newspaper_rows_with_cbsa23": int(cbsa_nonempty),
        "fuzzy_match_threshold": float(args.fuzzy_match_threshold),
        "candidate_top_n": int(args.candidate_top_n),
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
