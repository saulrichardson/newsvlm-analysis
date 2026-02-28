#!/usr/bin/env python3
"""
Build a robust city geospatial backbone from Census relationship files.

This script is designed to make downstream joins (tabular and shapefile) stable when
city boundaries and Census geographies change over time.

Primary anchor:
  - city -> PLACE (2020 vintage) from city_place_crosswalk.csv

Derived weighted crosswalks:
  - place20 -> place10
  - place20 -> zcta20
  - place20 -> tract20 (composed via zcta20)
  - place20 -> county20 (derived from tract20)
  - place20 -> puma20 (via tract20 -> puma20 lookup)
  - place20 -> ua20
  - place20 -> cbsa23 (via ua20)
  - tract20 -> tract10 (city-scoped)
  - puma20 -> puma10 (city-scoped)
  - zcta20 -> zcta10 (city-scoped)

Outputs (under <output-dir>):
  - city_geo_backbone.csv
  - city_place20_place10_crosswalk.csv
  - city_place20_zcta20_crosswalk.csv
  - city_place20_tract20_crosswalk.csv
  - city_place20_county20_crosswalk.csv
  - city_place20_puma20_crosswalk.csv
  - city_place20_ua20_crosswalk.csv
  - city_place20_cbsa23_crosswalk.csv
  - city_tract20_tract10_crosswalk.csv
  - city_puma20_puma10_crosswalk.csv
  - city_zcta20_zcta10_crosswalk.csv
  - crosswalk_city_quality.csv
  - crosswalk_quality_summary.csv
  - provenance.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests


REL_BASE = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"

REL_FILES = {
    "place20_place10": "place/tab20_place20_place10_natl.txt",
    "zcta20_place20": "zcta520/tab20_zcta520_place20_natl.txt",
    "zcta20_tract20": "zcta520/tab20_zcta520_tract20_natl.txt",
    "zcta10_zcta20": "zcta520/tab20_zcta510_zcta520_natl.txt",
    "tract20_tract10": "tract/tab20_tract20_tract10_natl.txt",
    "puma20_puma10": "puma520/tab20_puma520_puma510_natl.txt",
    "tract20_puma20": "2020_Census_Tract_to_2020_PUMA.txt",
    "ua20_place20": "ua/tab20_ua20_place20_natl.txt",
    "ua20_cbsa23": "cbsa/acs23_ua20_cbsa23_natl.txt",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_id(v: Any, width: int | None = None) -> str:
    s = str(v or "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    s = re.sub(r"\.0$", "", s)
    s = s.strip()
    if width and s.isdigit():
        s = s.zfill(width)
    return s


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_share(num: pd.Series, den: pd.Series) -> pd.Series:
    out = pd.Series([math.nan] * len(num), index=num.index, dtype="float64")
    mask = den.notna() & (den > 0) & num.notna() & (num >= 0)
    out.loc[mask] = num.loc[mask] / den.loc[mask]
    return out


def _download(url: str, dst: Path, *, refresh: bool) -> Path:
    if dst.exists() and not refresh:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dst)
    return dst


def _read_filtered(
    path: Path,
    *,
    sep: str,
    usecols: list[str] | None,
    filter_col: str | None = None,
    allowed: set[str] | None = None,
    transform: Callable[[Any], str] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    if not filter_col or not allowed:
        return pd.read_csv(path, sep=sep, dtype=str, usecols=usecols, low_memory=False)
    out: list[pd.DataFrame] = []
    xf = transform or (lambda x: str(x or "").strip())
    for chunk in pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        usecols=usecols,
        low_memory=False,
        chunksize=chunksize,
    ):
        vals = chunk[filter_col].map(xf)
        mask = vals.isin(allowed)
        if mask.any():
            out.append(chunk.loc[mask].copy())
    if not out:
        return pd.DataFrame(columns=usecols if usecols is not None else [])
    return pd.concat(out, ignore_index=True)


def _rank_desc(df: pd.DataFrame, by: list[str], score: str, rank_col: str) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out[rank_col] = []
        return out
    out = df.sort_values(by + [score], ascending=[True] * len(by) + [False]).copy()
    out[rank_col] = out.groupby(by).cumcount() + 1
    return out


def _normalize_within(df: pd.DataFrame, group_col: str, value_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    denom = out.groupby(group_col)[value_col].transform("sum")
    out[out_col] = _safe_share(out[value_col], denom)
    return out


def _compose_place_bridge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_key: str,
    mid_key: str,
    right_key: str,
    left_w: str,
    right_w: str,
    out_weight: str,
    min_weight: float,
) -> pd.DataFrame:
    if left.empty or right.empty:
        return pd.DataFrame(columns=[left_key, right_key, out_weight, "raw_weight", "left_raw_sum"])
    m = left[[left_key, mid_key, left_w]].merge(
        right[[mid_key, right_key, right_w]],
        on=mid_key,
        how="inner",
    )
    if m.empty:
        return pd.DataFrame(columns=[left_key, right_key, out_weight, "raw_weight", "left_raw_sum"])
    m["raw_weight"] = _to_num(m[left_w]) * _to_num(m[right_w])
    m = m[m["raw_weight"].notna() & (m["raw_weight"] > 0)].copy()
    if m.empty:
        return pd.DataFrame(columns=[left_key, right_key, out_weight, "raw_weight", "left_raw_sum"])
    g = (
        m.groupby([left_key, right_key], as_index=False)["raw_weight"]
        .sum()
        .sort_values([left_key, "raw_weight"], ascending=[True, False])
    )
    g["left_raw_sum"] = g.groupby(left_key)["raw_weight"].transform("sum")
    g[out_weight] = _safe_share(g["raw_weight"], g["left_raw_sum"])
    g = g[g[out_weight].fillna(0.0) >= float(min_weight)].copy()
    # Re-normalize after thresholding.
    g = _normalize_within(g, group_col=left_key, value_col=out_weight, out_col=out_weight)
    return g


def _collapse_primary(
    df: pd.DataFrame,
    *,
    group_col: str,
    id_col: str,
    weight_col: str,
    out_id: str,
    out_weight: str,
    out_count: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, out_id, out_weight, out_count])
    g = df[[group_col, id_col, weight_col]].copy()
    g[weight_col] = _to_num(g[weight_col])
    g = g[(g[id_col].astype(str).str.len() > 0) & g[weight_col].notna()].copy()
    if g.empty:
        return pd.DataFrame(columns=[group_col, out_id, out_weight, out_count])
    count_df = g.groupby(group_col, as_index=False)[id_col].nunique().rename(columns={id_col: out_count})
    g = g.sort_values([group_col, weight_col, id_col], ascending=[True, False, True]).copy()
    top = g.groupby(group_col, as_index=False).first().rename(columns={id_col: out_id, weight_col: out_weight})
    out = count_df.merge(top, on=group_col, how="left")
    return out[[group_col, out_id, out_weight, out_count]]


def _load_city_anchor(city_place_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(city_place_path, dtype=str)
    if raw.empty:
        raise SystemExit(f"No rows in {city_place_path}")
    need = {"city_key", "city_name", "state_abbr", "place_geoid"}
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise SystemExit(f"Missing columns in {city_place_path}: {missing}")

    out = raw.copy()
    if "census_place_geoid" in out.columns:
        out["place_geoid"] = out["place_geoid"].fillna("").astype(str)
        miss_mask = out["place_geoid"].str.strip() == ""
        out.loc[miss_mask, "place_geoid"] = out.loc[miss_mask, "census_place_geoid"].fillna("").astype(str)
    out["place_geoid"] = out["place_geoid"].map(lambda x: _norm_id(x, 7))
    out = out[out["place_geoid"].str.len() == 7].copy()
    if out.empty:
        raise SystemExit(f"No valid place GEOIDs in {city_place_path}")

    if "state_fips" not in out.columns:
        out["state_fips"] = out["place_geoid"].str.slice(0, 2)
    else:
        out["state_fips"] = out["state_fips"].fillna("").astype(str)
        miss_mask = out["state_fips"].str.strip() == ""
        out.loc[miss_mask, "state_fips"] = out.loc[miss_mask, "place_geoid"].str.slice(0, 2)
    if "place_fips" not in out.columns:
        out["place_fips"] = out["place_geoid"].str.slice(2, 7)
    else:
        out["place_fips"] = out["place_fips"].fillna("").astype(str)
        miss_mask = out["place_fips"].str.strip() == ""
        out.loc[miss_mask, "place_fips"] = out.loc[miss_mask, "place_geoid"].str.slice(2, 7)

    for c in ["region", "urbanicity_proxy", "min_issue_year", "max_issue_year", "place_name_gazetteer", "place_pop10", "intptlat", "intptlong"]:
        if c not in out.columns:
            out[c] = ""

    # One anchor row per city_key.
    out = out.sort_values(["city_key", "match_confidence", "match_score"], ascending=[True, True, False], na_position="last")
    out = out.drop_duplicates(subset=["city_key"], keep="first").reset_index(drop=True)
    return out


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build robust Census geospatial backbone crosswalks.")
    ap.add_argument("--run-dir", required=True, help="Run directory (contains covariates outputs).")
    ap.add_argument(
        "--covariate-dir",
        default="covariates_v2",
        help="Subdirectory under run-dir containing city_place_crosswalk.csv (default: covariates_v2).",
    )
    ap.add_argument(
        "--city-place-crosswalk",
        default="",
        help="Optional explicit path to city_place_crosswalk.csv (overrides --covariate-dir lookup).",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-dir>/geo_backbone).",
    )
    ap.add_argument("--refresh-cache", action="store_true", help="Re-download Census relationship files.")
    ap.add_argument(
        "--min-weight",
        type=float,
        default=1e-4,
        help="Drop crosswalk rows below this share weight before final re-normalization.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cov_dir = Path(args.covariate_dir)
    if not cov_dir.is_absolute():
        cov_dir = run_dir / cov_dir
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "geo_backbone")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache" / "rel2020"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if str(args.city_place_crosswalk).strip():
        city_place_path = Path(args.city_place_crosswalk).expanduser().resolve()
    else:
        city_place_path = cov_dir / "city_place_crosswalk.csv"
        if not city_place_path.exists():
            fallback = run_dir / "covariates" / "city_place_crosswalk.csv"
            if fallback.exists():
                city_place_path = fallback
    if not city_place_path.exists():
        raise SystemExit(f"Could not find city_place_crosswalk.csv: {city_place_path}")

    anchor = _load_city_anchor(city_place_path)
    city_count = int(anchor["city_key"].nunique())
    place20_set = set(anchor["place_geoid"].astype(str))

    rel_local: dict[str, Path] = {}
    for key, rel_path in REL_FILES.items():
        url = REL_BASE + rel_path
        dst = cache_dir / Path(rel_path).name
        rel_local[key] = _download(url, dst, refresh=bool(args.refresh_cache))

    # 1) place20 -> place10
    p20p10_cols = [
        "GEOID_PLACE_20",
        "NAMELSAD_PLACE_20",
        "AREALAND_PLACE_20",
        "GEOID_PLACE_10",
        "NAMELSAD_PLACE_10",
        "AREALAND_PLACE_10",
        "AREALAND_PART",
    ]
    place20_place10 = _read_filtered(
        rel_local["place20_place10"],
        sep="|",
        usecols=p20p10_cols,
        filter_col="GEOID_PLACE_20",
        allowed=place20_set,
        transform=lambda x: _norm_id(x, 7),
    )
    place20_place10["GEOID_PLACE_20"] = place20_place10["GEOID_PLACE_20"].map(lambda x: _norm_id(x, 7))
    place20_place10["GEOID_PLACE_10"] = place20_place10["GEOID_PLACE_10"].map(lambda x: _norm_id(x, 7))
    place20_place10 = place20_place10[
        (place20_place10["GEOID_PLACE_20"].str.len() == 7) & (place20_place10["GEOID_PLACE_10"].str.len() == 7)
    ].copy()
    place20_place10["AREALAND_PART"] = _to_num(place20_place10["AREALAND_PART"])
    place20_place10["AREALAND_PLACE_20"] = _to_num(place20_place10["AREALAND_PLACE_20"])
    place20_place10["AREALAND_PLACE_10"] = _to_num(place20_place10["AREALAND_PLACE_10"])
    place20_place10["weight_place20_to_place10"] = _safe_share(
        place20_place10["AREALAND_PART"],
        place20_place10["AREALAND_PLACE_20"],
    )
    place20_place10["weight_place10_to_place20"] = _safe_share(
        place20_place10["AREALAND_PART"],
        place20_place10["AREALAND_PLACE_10"],
    )
    place20_place10 = place20_place10[place20_place10["weight_place20_to_place10"].fillna(0.0) >= float(args.min_weight)].copy()
    place20_place10 = _normalize_within(
        place20_place10,
        group_col="GEOID_PLACE_20",
        value_col="weight_place20_to_place10",
        out_col="weight_place20_to_place10",
    )
    place20_place10 = _rank_desc(
        place20_place10,
        by=["GEOID_PLACE_20"],
        score="weight_place20_to_place10",
        rank_col="rank_within_place20",
    )
    city_place20_place10 = anchor.merge(
        place20_place10,
        left_on="place_geoid",
        right_on="GEOID_PLACE_20",
        how="left",
    )
    city_place20_place10 = city_place20_place10.rename(
        columns={
            "place_geoid": "place20_geoid",
            "GEOID_PLACE_10": "place10_geoid",
            "NAMELSAD_PLACE_20": "place20_name",
            "NAMELSAD_PLACE_10": "place10_name",
        }
    )
    city_place20_place10 = city_place20_place10[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "place20_name",
            "place10_geoid",
            "place10_name",
            "AREALAND_PART",
            "weight_place20_to_place10",
            "weight_place10_to_place20",
            "rank_within_place20",
        ]
    ].copy()
    city_place20_place10 = city_place20_place10[city_place20_place10["place10_geoid"].astype(str).str.len() == 7].reset_index(drop=True)

    # 2) place20 -> zcta20
    z20p20_cols = [
        "GEOID_ZCTA5_20",
        "GEOID_PLACE_20",
        "AREALAND_ZCTA5_20",
        "AREALAND_PLACE_20",
        "AREALAND_PART",
    ]
    zcta20_place20 = _read_filtered(
        rel_local["zcta20_place20"],
        sep="|",
        usecols=z20p20_cols,
        filter_col="GEOID_PLACE_20",
        allowed=place20_set,
        transform=lambda x: _norm_id(x, 7),
    )
    zcta20_place20["GEOID_ZCTA5_20"] = zcta20_place20["GEOID_ZCTA5_20"].map(lambda x: _norm_id(x, 5))
    zcta20_place20["GEOID_PLACE_20"] = zcta20_place20["GEOID_PLACE_20"].map(lambda x: _norm_id(x, 7))
    zcta20_place20 = zcta20_place20[
        (zcta20_place20["GEOID_ZCTA5_20"].str.len() == 5) & (zcta20_place20["GEOID_PLACE_20"].str.len() == 7)
    ].copy()
    zcta20_place20["AREALAND_PART"] = _to_num(zcta20_place20["AREALAND_PART"])
    zcta20_place20["AREALAND_ZCTA5_20"] = _to_num(zcta20_place20["AREALAND_ZCTA5_20"])
    zcta20_place20["AREALAND_PLACE_20"] = _to_num(zcta20_place20["AREALAND_PLACE_20"])
    zcta20_place20["weight_place20_to_zcta20"] = _safe_share(
        zcta20_place20["AREALAND_PART"],
        zcta20_place20["AREALAND_PLACE_20"],
    )
    zcta20_place20["weight_zcta20_to_place20"] = _safe_share(
        zcta20_place20["AREALAND_PART"],
        zcta20_place20["AREALAND_ZCTA5_20"],
    )
    zcta20_place20 = zcta20_place20[zcta20_place20["weight_place20_to_zcta20"].fillna(0.0) >= float(args.min_weight)].copy()
    zcta20_place20 = _normalize_within(
        zcta20_place20,
        group_col="GEOID_PLACE_20",
        value_col="weight_place20_to_zcta20",
        out_col="weight_place20_to_zcta20",
    )
    zcta20_place20 = _rank_desc(
        zcta20_place20,
        by=["GEOID_PLACE_20"],
        score="weight_place20_to_zcta20",
        rank_col="rank_within_place20",
    )
    city_place20_zcta20 = anchor.merge(
        zcta20_place20,
        left_on="place_geoid",
        right_on="GEOID_PLACE_20",
        how="left",
    )
    city_place20_zcta20 = city_place20_zcta20.rename(
        columns={
            "place_geoid": "place20_geoid",
            "GEOID_ZCTA5_20": "zcta20",
        }
    )
    city_place20_zcta20 = city_place20_zcta20[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "zcta20",
            "AREALAND_PART",
            "weight_place20_to_zcta20",
            "weight_zcta20_to_place20",
            "rank_within_place20",
        ]
    ].copy()
    city_place20_zcta20 = city_place20_zcta20[city_place20_zcta20["zcta20"].astype(str).str.len() == 5].reset_index(drop=True)

    # 3) zcta20 -> tract20 (filtered to zctas touching selected places)
    zcta20_set = set(city_place20_zcta20["zcta20"].dropna().astype(str))
    z20t20_cols = [
        "GEOID_ZCTA5_20",
        "GEOID_TRACT_20",
        "AREALAND_ZCTA5_20",
        "AREALAND_TRACT_20",
        "AREALAND_PART",
    ]
    zcta20_tract20 = _read_filtered(
        rel_local["zcta20_tract20"],
        sep="|",
        usecols=z20t20_cols,
        filter_col="GEOID_ZCTA5_20",
        allowed=zcta20_set,
        transform=lambda x: _norm_id(x, 5),
    )
    zcta20_tract20["GEOID_ZCTA5_20"] = zcta20_tract20["GEOID_ZCTA5_20"].map(lambda x: _norm_id(x, 5))
    zcta20_tract20["GEOID_TRACT_20"] = zcta20_tract20["GEOID_TRACT_20"].map(lambda x: _norm_id(x, 11))
    zcta20_tract20 = zcta20_tract20[
        (zcta20_tract20["GEOID_ZCTA5_20"].str.len() == 5) & (zcta20_tract20["GEOID_TRACT_20"].str.len() == 11)
    ].copy()
    zcta20_tract20["AREALAND_PART"] = _to_num(zcta20_tract20["AREALAND_PART"])
    zcta20_tract20["AREALAND_ZCTA5_20"] = _to_num(zcta20_tract20["AREALAND_ZCTA5_20"])
    zcta20_tract20["weight_zcta20_to_tract20"] = _safe_share(
        zcta20_tract20["AREALAND_PART"],
        zcta20_tract20["AREALAND_ZCTA5_20"],
    )
    zcta20_tract20 = zcta20_tract20[zcta20_tract20["weight_zcta20_to_tract20"].fillna(0.0) >= float(args.min_weight)].copy()
    zcta20_tract20 = _normalize_within(
        zcta20_tract20,
        group_col="GEOID_ZCTA5_20",
        value_col="weight_zcta20_to_tract20",
        out_col="weight_zcta20_to_tract20",
    )

    # 4) place20 -> tract20 composition via zcta20
    place_zcta_for_bridge = city_place20_zcta20[
        ["place20_geoid", "zcta20", "weight_place20_to_zcta20"]
    ].drop_duplicates()
    place20_tract20 = _compose_place_bridge(
        left=place_zcta_for_bridge.rename(columns={"place20_geoid": "place20", "zcta20": "zcta20"}),
        right=zcta20_tract20.rename(columns={"GEOID_ZCTA5_20": "zcta20", "GEOID_TRACT_20": "tract20"}),
        left_key="place20",
        mid_key="zcta20",
        right_key="tract20",
        left_w="weight_place20_to_zcta20",
        right_w="weight_zcta20_to_tract20",
        out_weight="weight_place20_to_tract20",
        min_weight=float(args.min_weight),
    )
    place20_tract20 = place20_tract20.rename(columns={"place20": "place20_geoid", "tract20": "tract20_geoid"})
    anchor2 = anchor.rename(columns={"place_geoid": "place20_geoid"})
    city_place20_tract20 = anchor2.merge(place20_tract20, on="place20_geoid", how="left")
    city_place20_tract20 = city_place20_tract20[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "tract20_geoid",
            "weight_place20_to_tract20",
            "raw_weight",
            "left_raw_sum",
        ]
    ].copy()
    city_place20_tract20 = city_place20_tract20[city_place20_tract20["tract20_geoid"].astype(str).str.len() == 11].reset_index(drop=True)

    # 5) place20 -> county20 from tract20
    place20_county20 = city_place20_tract20[
        ["place20_geoid", "tract20_geoid", "weight_place20_to_tract20"]
    ].drop_duplicates()
    place20_county20["county20_fips"] = place20_county20["tract20_geoid"].astype(str).str.slice(0, 5)
    place20_county20["weight_place20_to_county20_raw"] = _to_num(place20_county20["weight_place20_to_tract20"])
    place20_county20 = (
        place20_county20.groupby(["place20_geoid", "county20_fips"], as_index=False)["weight_place20_to_county20_raw"].sum()
    )
    place20_county20 = _normalize_within(
        place20_county20,
        group_col="place20_geoid",
        value_col="weight_place20_to_county20_raw",
        out_col="weight_place20_to_county20",
    )
    place20_county20 = place20_county20[
        place20_county20["weight_place20_to_county20"].fillna(0.0) >= float(args.min_weight)
    ].copy()
    place20_county20 = _normalize_within(
        place20_county20,
        group_col="place20_geoid",
        value_col="weight_place20_to_county20",
        out_col="weight_place20_to_county20",
    )
    place20_county20 = _rank_desc(
        place20_county20,
        by=["place20_geoid"],
        score="weight_place20_to_county20",
        rank_col="rank_within_place20",
    )
    city_place20_county20 = anchor.rename(columns={"place_geoid": "place20_geoid"}).merge(
        place20_county20,
        on="place20_geoid",
        how="left",
    )
    city_place20_county20 = city_place20_county20[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "county20_fips",
            "weight_place20_to_county20",
            "rank_within_place20",
        ]
    ].copy()
    city_place20_county20 = city_place20_county20[
        city_place20_county20["county20_fips"].astype(str).str.len() == 5
    ].reset_index(drop=True)

    # 6) tract20 -> puma20 (lookup) and place20 -> puma20
    t20p20 = pd.read_csv(rel_local["tract20_puma20"], dtype=str, low_memory=False)
    need_t20p20 = {"STATEFP", "COUNTYFP", "TRACTCE", "PUMA5CE"}
    miss = [c for c in need_t20p20 if c not in t20p20.columns]
    if miss:
        raise SystemExit(f"Missing columns in tract->puma relation: {miss}")
    t20p20["tract20_geoid"] = (
        t20p20["STATEFP"].map(lambda x: _norm_id(x, 2))
        + t20p20["COUNTYFP"].map(lambda x: _norm_id(x, 3))
        + t20p20["TRACTCE"].map(lambda x: _norm_id(x, 6))
    )
    t20p20["puma20_geoid"] = (
        t20p20["STATEFP"].map(lambda x: _norm_id(x, 2)) + t20p20["PUMA5CE"].map(lambda x: _norm_id(x, 5))
    )
    t20p20 = t20p20[
        (t20p20["tract20_geoid"].str.len() == 11) & (t20p20["puma20_geoid"].str.len() == 7)
    ][["tract20_geoid", "puma20_geoid"]].drop_duplicates()

    place20_puma20 = city_place20_tract20[
        ["place20_geoid", "tract20_geoid", "weight_place20_to_tract20"]
    ].drop_duplicates()
    place20_puma20 = place20_puma20.merge(t20p20, on="tract20_geoid", how="left")
    place20_puma20 = place20_puma20[place20_puma20["puma20_geoid"].astype(str).str.len() == 7].copy()
    place20_puma20["weight_place20_to_puma20_raw"] = _to_num(place20_puma20["weight_place20_to_tract20"])
    place20_puma20 = (
        place20_puma20.groupby(["place20_geoid", "puma20_geoid"], as_index=False)["weight_place20_to_puma20_raw"].sum()
    )
    place20_puma20 = _normalize_within(
        place20_puma20,
        group_col="place20_geoid",
        value_col="weight_place20_to_puma20_raw",
        out_col="weight_place20_to_puma20",
    )
    place20_puma20 = place20_puma20[
        place20_puma20["weight_place20_to_puma20"].fillna(0.0) >= float(args.min_weight)
    ].copy()
    place20_puma20 = _normalize_within(
        place20_puma20,
        group_col="place20_geoid",
        value_col="weight_place20_to_puma20",
        out_col="weight_place20_to_puma20",
    )
    place20_puma20 = _rank_desc(
        place20_puma20,
        by=["place20_geoid"],
        score="weight_place20_to_puma20",
        rank_col="rank_within_place20",
    )
    city_place20_puma20 = anchor.rename(columns={"place_geoid": "place20_geoid"}).merge(
        place20_puma20,
        on="place20_geoid",
        how="left",
    )
    city_place20_puma20 = city_place20_puma20[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "puma20_geoid",
            "weight_place20_to_puma20",
            "rank_within_place20",
        ]
    ].copy()
    city_place20_puma20 = city_place20_puma20[
        city_place20_puma20["puma20_geoid"].astype(str).str.len() == 7
    ].reset_index(drop=True)

    # 7) tract20 -> tract10 (city-scoped by tracts touched by selected places)
    tract20_set = set(city_place20_tract20["tract20_geoid"].dropna().astype(str))
    t20t10_cols = [
        "GEOID_TRACT_20",
        "GEOID_TRACT_10",
        "AREALAND_TRACT_20",
        "AREALAND_TRACT_10",
        "AREALAND_PART",
    ]
    tract20_tract10 = _read_filtered(
        rel_local["tract20_tract10"],
        sep="|",
        usecols=t20t10_cols,
        filter_col="GEOID_TRACT_20",
        allowed=tract20_set,
        transform=lambda x: _norm_id(x, 11),
    )
    tract20_tract10["GEOID_TRACT_20"] = tract20_tract10["GEOID_TRACT_20"].map(lambda x: _norm_id(x, 11))
    tract20_tract10["GEOID_TRACT_10"] = tract20_tract10["GEOID_TRACT_10"].map(lambda x: _norm_id(x, 11))
    tract20_tract10 = tract20_tract10[
        (tract20_tract10["GEOID_TRACT_20"].str.len() == 11) & (tract20_tract10["GEOID_TRACT_10"].str.len() == 11)
    ].copy()
    tract20_tract10["AREALAND_PART"] = _to_num(tract20_tract10["AREALAND_PART"])
    tract20_tract10["AREALAND_TRACT_20"] = _to_num(tract20_tract10["AREALAND_TRACT_20"])
    tract20_tract10["AREALAND_TRACT_10"] = _to_num(tract20_tract10["AREALAND_TRACT_10"])
    tract20_tract10["weight_tract20_to_tract10"] = _safe_share(
        tract20_tract10["AREALAND_PART"],
        tract20_tract10["AREALAND_TRACT_20"],
    )
    tract20_tract10["weight_tract10_to_tract20"] = _safe_share(
        tract20_tract10["AREALAND_PART"],
        tract20_tract10["AREALAND_TRACT_10"],
    )
    tract20_tract10 = tract20_tract10[tract20_tract10["weight_tract20_to_tract10"].fillna(0.0) >= float(args.min_weight)].copy()
    tract20_tract10 = _normalize_within(
        tract20_tract10,
        group_col="GEOID_TRACT_20",
        value_col="weight_tract20_to_tract10",
        out_col="weight_tract20_to_tract10",
    )
    city_tract20_tract10 = city_place20_tract20[
        ["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "place20_geoid", "tract20_geoid"]
    ].drop_duplicates()
    city_tract20_tract10 = city_tract20_tract10.merge(
        tract20_tract10.rename(columns={"GEOID_TRACT_20": "tract20_geoid", "GEOID_TRACT_10": "tract10_geoid"}),
        on="tract20_geoid",
        how="left",
    )
    city_tract20_tract10 = city_tract20_tract10[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "tract20_geoid",
            "tract10_geoid",
            "AREALAND_PART",
            "weight_tract20_to_tract10",
            "weight_tract10_to_tract20",
        ]
    ].copy()
    city_tract20_tract10 = city_tract20_tract10[city_tract20_tract10["tract10_geoid"].astype(str).str.len() == 11].reset_index(drop=True)

    # 8) puma20 -> puma10 (city-scoped by pumas touched by selected places)
    puma20_set = set(city_place20_puma20["puma20_geoid"].dropna().astype(str))
    p20p10_cols = [
        "GEOID_PUMA5_20",
        "GEOID_PUMA5_10",
        "AREALAND_PUMA5_20",
        "AREALAND_PUMA5_10",
        "AREALAND_PART",
    ]
    puma20_puma10 = _read_filtered(
        rel_local["puma20_puma10"],
        sep="|",
        usecols=p20p10_cols,
        filter_col="GEOID_PUMA5_20",
        allowed=puma20_set,
        transform=lambda x: _norm_id(x, 7),
    )
    puma20_puma10["GEOID_PUMA5_20"] = puma20_puma10["GEOID_PUMA5_20"].map(lambda x: _norm_id(x, 7))
    puma20_puma10["GEOID_PUMA5_10"] = puma20_puma10["GEOID_PUMA5_10"].map(lambda x: _norm_id(x, 7))
    puma20_puma10 = puma20_puma10[
        (puma20_puma10["GEOID_PUMA5_20"].str.len() == 7) & (puma20_puma10["GEOID_PUMA5_10"].str.len() == 7)
    ].copy()
    puma20_puma10["AREALAND_PART"] = _to_num(puma20_puma10["AREALAND_PART"])
    puma20_puma10["AREALAND_PUMA5_20"] = _to_num(puma20_puma10["AREALAND_PUMA5_20"])
    puma20_puma10["AREALAND_PUMA5_10"] = _to_num(puma20_puma10["AREALAND_PUMA5_10"])
    puma20_puma10["weight_puma20_to_puma10"] = _safe_share(
        puma20_puma10["AREALAND_PART"],
        puma20_puma10["AREALAND_PUMA5_20"],
    )
    puma20_puma10["weight_puma10_to_puma20"] = _safe_share(
        puma20_puma10["AREALAND_PART"],
        puma20_puma10["AREALAND_PUMA5_10"],
    )
    puma20_puma10 = puma20_puma10[puma20_puma10["weight_puma20_to_puma10"].fillna(0.0) >= float(args.min_weight)].copy()
    puma20_puma10 = _normalize_within(
        puma20_puma10,
        group_col="GEOID_PUMA5_20",
        value_col="weight_puma20_to_puma10",
        out_col="weight_puma20_to_puma10",
    )
    city_puma20_puma10 = city_place20_puma20[
        ["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "place20_geoid", "puma20_geoid"]
    ].drop_duplicates()
    city_puma20_puma10 = city_puma20_puma10.merge(
        puma20_puma10.rename(columns={"GEOID_PUMA5_20": "puma20_geoid", "GEOID_PUMA5_10": "puma10_geoid"}),
        on="puma20_geoid",
        how="left",
    )
    city_puma20_puma10 = city_puma20_puma10[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "puma20_geoid",
            "puma10_geoid",
            "AREALAND_PART",
            "weight_puma20_to_puma10",
            "weight_puma10_to_puma20",
        ]
    ].copy()
    city_puma20_puma10 = city_puma20_puma10[city_puma20_puma10["puma10_geoid"].astype(str).str.len() == 7].reset_index(drop=True)

    # 9) zcta20 -> zcta10 (city-scoped by zctas touching selected places)
    z20z10_cols = [
        "GEOID_ZCTA5_20",
        "GEOID_ZCTA5_10",
        "AREALAND_ZCTA5_20",
        "AREALAND_ZCTA5_10",
        "AREALAND_PART",
    ]
    zcta20_zcta10 = _read_filtered(
        rel_local["zcta10_zcta20"],
        sep="|",
        usecols=z20z10_cols,
        filter_col="GEOID_ZCTA5_20",
        allowed=zcta20_set,
        transform=lambda x: _norm_id(x, 5),
    )
    zcta20_zcta10["GEOID_ZCTA5_20"] = zcta20_zcta10["GEOID_ZCTA5_20"].map(lambda x: _norm_id(x, 5))
    zcta20_zcta10["GEOID_ZCTA5_10"] = zcta20_zcta10["GEOID_ZCTA5_10"].map(lambda x: _norm_id(x, 5))
    zcta20_zcta10 = zcta20_zcta10[
        (zcta20_zcta10["GEOID_ZCTA5_20"].str.len() == 5) & (zcta20_zcta10["GEOID_ZCTA5_10"].str.len() == 5)
    ].copy()
    zcta20_zcta10["AREALAND_PART"] = _to_num(zcta20_zcta10["AREALAND_PART"])
    zcta20_zcta10["AREALAND_ZCTA5_20"] = _to_num(zcta20_zcta10["AREALAND_ZCTA5_20"])
    zcta20_zcta10["AREALAND_ZCTA5_10"] = _to_num(zcta20_zcta10["AREALAND_ZCTA5_10"])
    zcta20_zcta10["weight_zcta20_to_zcta10"] = _safe_share(
        zcta20_zcta10["AREALAND_PART"],
        zcta20_zcta10["AREALAND_ZCTA5_20"],
    )
    zcta20_zcta10["weight_zcta10_to_zcta20"] = _safe_share(
        zcta20_zcta10["AREALAND_PART"],
        zcta20_zcta10["AREALAND_ZCTA5_10"],
    )
    zcta20_zcta10 = zcta20_zcta10[zcta20_zcta10["weight_zcta20_to_zcta10"].fillna(0.0) >= float(args.min_weight)].copy()
    zcta20_zcta10 = _normalize_within(
        zcta20_zcta10,
        group_col="GEOID_ZCTA5_20",
        value_col="weight_zcta20_to_zcta10",
        out_col="weight_zcta20_to_zcta10",
    )
    city_zcta20_zcta10 = city_place20_zcta20[
        ["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "place20_geoid", "zcta20"]
    ].drop_duplicates()
    city_zcta20_zcta10 = city_zcta20_zcta10.merge(
        zcta20_zcta10.rename(columns={"GEOID_ZCTA5_20": "zcta20", "GEOID_ZCTA5_10": "zcta10"}),
        on="zcta20",
        how="left",
    )
    city_zcta20_zcta10 = city_zcta20_zcta10[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "zcta20",
            "zcta10",
            "AREALAND_PART",
            "weight_zcta20_to_zcta10",
            "weight_zcta10_to_zcta20",
        ]
    ].copy()
    city_zcta20_zcta10 = city_zcta20_zcta10[city_zcta20_zcta10["zcta10"].astype(str).str.len() == 5].reset_index(drop=True)

    # 10) ua20 -> place20 and place20 -> cbsa23
    ua20_place20_cols = [
        "GEOID_UA_20",
        "NAMELSAD_UA_20",
        "GEOID_PLACE_20",
        "AREALAND_UA_20",
        "AREALAND_PLACE_20",
        "AREALAND_PART",
    ]
    ua20_place20 = _read_filtered(
        rel_local["ua20_place20"],
        sep="|",
        usecols=ua20_place20_cols,
        filter_col="GEOID_PLACE_20",
        allowed=place20_set,
        transform=lambda x: _norm_id(x, 7),
    )
    ua20_place20["GEOID_UA_20"] = ua20_place20["GEOID_UA_20"].map(lambda x: _norm_id(x, 5))
    ua20_place20["GEOID_PLACE_20"] = ua20_place20["GEOID_PLACE_20"].map(lambda x: _norm_id(x, 7))
    ua20_place20 = ua20_place20[
        (ua20_place20["GEOID_UA_20"].str.len() == 5) & (ua20_place20["GEOID_PLACE_20"].str.len() == 7)
    ].copy()
    ua20_place20["AREALAND_PART"] = _to_num(ua20_place20["AREALAND_PART"])
    ua20_place20["AREALAND_UA_20"] = _to_num(ua20_place20["AREALAND_UA_20"])
    ua20_place20["AREALAND_PLACE_20"] = _to_num(ua20_place20["AREALAND_PLACE_20"])
    ua20_place20["weight_place20_to_ua20"] = _safe_share(
        ua20_place20["AREALAND_PART"],
        ua20_place20["AREALAND_PLACE_20"],
    )
    ua20_place20["weight_ua20_to_place20"] = _safe_share(
        ua20_place20["AREALAND_PART"],
        ua20_place20["AREALAND_UA_20"],
    )
    ua20_place20 = ua20_place20[ua20_place20["weight_place20_to_ua20"].fillna(0.0) >= float(args.min_weight)].copy()
    ua20_place20 = _normalize_within(
        ua20_place20,
        group_col="GEOID_PLACE_20",
        value_col="weight_place20_to_ua20",
        out_col="weight_place20_to_ua20",
    )
    ua20_place20 = _rank_desc(
        ua20_place20,
        by=["GEOID_PLACE_20"],
        score="weight_place20_to_ua20",
        rank_col="rank_within_place20",
    )
    city_place20_ua20 = anchor.rename(columns={"place_geoid": "place20_geoid"}).merge(
        ua20_place20.rename(columns={"GEOID_PLACE_20": "place20_geoid", "GEOID_UA_20": "ua20_geoid", "NAMELSAD_UA_20": "ua20_name"}),
        on="place20_geoid",
        how="left",
    )
    city_place20_ua20 = city_place20_ua20[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "ua20_geoid",
            "ua20_name",
            "weight_place20_to_ua20",
            "weight_ua20_to_place20",
            "rank_within_place20",
        ]
    ].copy()
    city_place20_ua20 = city_place20_ua20[city_place20_ua20["ua20_geoid"].astype(str).str.len() == 5].reset_index(drop=True)

    ua20_set = set(city_place20_ua20["ua20_geoid"].dropna().astype(str))
    ua20_cbsa23_cols = [
        "GEOID_UA_20",
        "GEOID_CBSA_23",
        "NAMELSAD_CBSA_23",
        "AREALAND_UA_20",
        "AREALAND_CBSA_23",
        "AREALAND_PART",
    ]
    ua20_cbsa23 = _read_filtered(
        rel_local["ua20_cbsa23"],
        sep="|",
        usecols=ua20_cbsa23_cols,
        filter_col="GEOID_UA_20",
        allowed=ua20_set,
        transform=lambda x: _norm_id(x, 5),
    )
    ua20_cbsa23["GEOID_UA_20"] = ua20_cbsa23["GEOID_UA_20"].map(lambda x: _norm_id(x, 5))
    ua20_cbsa23["GEOID_CBSA_23"] = ua20_cbsa23["GEOID_CBSA_23"].map(lambda x: _norm_id(x, 5))
    ua20_cbsa23 = ua20_cbsa23[
        (ua20_cbsa23["GEOID_UA_20"].str.len() == 5) & (ua20_cbsa23["GEOID_CBSA_23"].str.len() == 5)
    ].copy()
    ua20_cbsa23["AREALAND_PART"] = _to_num(ua20_cbsa23["AREALAND_PART"])
    ua20_cbsa23["AREALAND_UA_20"] = _to_num(ua20_cbsa23["AREALAND_UA_20"])
    ua20_cbsa23["weight_ua20_to_cbsa23"] = _safe_share(
        ua20_cbsa23["AREALAND_PART"],
        ua20_cbsa23["AREALAND_UA_20"],
    )
    ua20_cbsa23 = ua20_cbsa23[ua20_cbsa23["weight_ua20_to_cbsa23"].fillna(0.0) >= float(args.min_weight)].copy()
    ua20_cbsa23 = _normalize_within(
        ua20_cbsa23,
        group_col="GEOID_UA_20",
        value_col="weight_ua20_to_cbsa23",
        out_col="weight_ua20_to_cbsa23",
    )

    place20_ua20 = city_place20_ua20[
        ["place20_geoid", "ua20_geoid", "weight_place20_to_ua20"]
    ].drop_duplicates()
    place20_cbsa23 = _compose_place_bridge(
        left=place20_ua20.rename(columns={"place20_geoid": "place20", "ua20_geoid": "ua20"}),
        right=ua20_cbsa23.rename(columns={"GEOID_UA_20": "ua20", "GEOID_CBSA_23": "cbsa23"}),
        left_key="place20",
        mid_key="ua20",
        right_key="cbsa23",
        left_w="weight_place20_to_ua20",
        right_w="weight_ua20_to_cbsa23",
        out_weight="weight_place20_to_cbsa23",
        min_weight=float(args.min_weight),
    )
    place20_cbsa23 = place20_cbsa23.rename(columns={"place20": "place20_geoid", "cbsa23": "cbsa23_geoid"})
    place20_cbsa23 = place20_cbsa23.merge(
        ua20_cbsa23[["GEOID_CBSA_23", "NAMELSAD_CBSA_23"]].drop_duplicates().rename(
            columns={"GEOID_CBSA_23": "cbsa23_geoid", "NAMELSAD_CBSA_23": "cbsa23_name"}
        ),
        on="cbsa23_geoid",
        how="left",
    )
    city_place20_cbsa23 = anchor.rename(columns={"place_geoid": "place20_geoid"}).merge(
        place20_cbsa23,
        on="place20_geoid",
        how="left",
    )
    city_place20_cbsa23 = city_place20_cbsa23[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "cbsa23_geoid",
            "cbsa23_name",
            "weight_place20_to_cbsa23",
        ]
    ].copy()
    city_place20_cbsa23 = city_place20_cbsa23[city_place20_cbsa23["cbsa23_geoid"].astype(str).str.len() == 5].reset_index(drop=True)

    # Build city-level backbone summary (primary IDs + multiplicity).
    city_backbone = anchor.rename(columns={"place_geoid": "place20_geoid"}).copy()
    city_backbone = city_backbone[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "state_fips",
            "place_fips",
            "min_issue_year",
            "max_issue_year",
            "place_name_gazetteer",
            "place_pop10",
            "intptlat",
            "intptlong",
        ]
    ].copy()

    p10_primary = _collapse_primary(
        city_place20_place10,
        group_col="city_key",
        id_col="place10_geoid",
        weight_col="weight_place20_to_place10",
        out_id="place10_primary_geoid",
        out_weight="place10_primary_weight",
        out_count="place10_count",
    )
    county_primary = _collapse_primary(
        city_place20_county20,
        group_col="city_key",
        id_col="county20_fips",
        weight_col="weight_place20_to_county20",
        out_id="county20_primary_fips",
        out_weight="county20_primary_weight",
        out_count="county20_count",
    )
    tract_count = (
        city_place20_tract20.groupby("city_key", as_index=False)["tract20_geoid"].nunique().rename(columns={"tract20_geoid": "tract20_count"})
    )
    puma_primary = _collapse_primary(
        city_place20_puma20,
        group_col="city_key",
        id_col="puma20_geoid",
        weight_col="weight_place20_to_puma20",
        out_id="puma20_primary_geoid",
        out_weight="puma20_primary_weight",
        out_count="puma20_count",
    )
    zcta_count = (
        city_place20_zcta20.groupby("city_key", as_index=False)["zcta20"].nunique().rename(columns={"zcta20": "zcta20_count"})
    )
    ua_primary = _collapse_primary(
        city_place20_ua20,
        group_col="city_key",
        id_col="ua20_geoid",
        weight_col="weight_place20_to_ua20",
        out_id="ua20_primary_geoid",
        out_weight="ua20_primary_weight",
        out_count="ua20_count",
    )
    cbsa_primary = _collapse_primary(
        city_place20_cbsa23,
        group_col="city_key",
        id_col="cbsa23_geoid",
        weight_col="weight_place20_to_cbsa23",
        out_id="cbsa23_primary_geoid",
        out_weight="cbsa23_primary_weight",
        out_count="cbsa23_count",
    )
    city_backbone = city_backbone.merge(p10_primary, on="city_key", how="left")
    city_backbone = city_backbone.merge(county_primary, on="city_key", how="left")
    city_backbone = city_backbone.merge(tract_count, on="city_key", how="left")
    city_backbone = city_backbone.merge(puma_primary, on="city_key", how="left")
    city_backbone = city_backbone.merge(zcta_count, on="city_key", how="left")
    city_backbone = city_backbone.merge(ua_primary, on="city_key", how="left")
    city_backbone = city_backbone.merge(cbsa_primary, on="city_key", how="left")
    city_backbone["county20_split_flag"] = (pd.to_numeric(city_backbone["county20_count"], errors="coerce").fillna(0) > 1).astype(int)
    city_backbone["puma20_split_flag"] = (pd.to_numeric(city_backbone["puma20_count"], errors="coerce").fillna(0) > 1).astype(int)
    city_backbone["cbsa23_split_flag"] = (pd.to_numeric(city_backbone["cbsa23_count"], errors="coerce").fillna(0) > 1).astype(int)

    # Per-city quality diagnostics.
    city_quality = city_backbone[
        [
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "place20_geoid",
            "place10_count",
            "place10_primary_weight",
            "zcta20_count",
            "tract20_count",
            "county20_count",
            "county20_primary_weight",
            "puma20_count",
            "puma20_primary_weight",
            "ua20_count",
            "ua20_primary_weight",
            "cbsa23_count",
            "cbsa23_primary_weight",
            "county20_split_flag",
            "puma20_split_flag",
            "cbsa23_split_flag",
        ]
    ].copy()

    quality_summary_rows: list[dict[str, Any]] = []
    def _add_metric(name: str, value: Any) -> None:
        quality_summary_rows.append({"metric": name, "value": value})

    _add_metric("city_count", city_count)
    _add_metric("place20_anchor_count", int(anchor["place_geoid"].nunique()))
    _add_metric("mean_place10_primary_weight", float(pd.to_numeric(city_backbone["place10_primary_weight"], errors="coerce").mean()))
    _add_metric("mean_county20_primary_weight", float(pd.to_numeric(city_backbone["county20_primary_weight"], errors="coerce").mean()))
    _add_metric("mean_puma20_primary_weight", float(pd.to_numeric(city_backbone["puma20_primary_weight"], errors="coerce").mean()))
    _add_metric("mean_cbsa23_primary_weight", float(pd.to_numeric(city_backbone["cbsa23_primary_weight"], errors="coerce").mean()))
    _add_metric("share_split_county20", float(pd.to_numeric(city_backbone["county20_split_flag"], errors="coerce").mean()))
    _add_metric("share_split_puma20", float(pd.to_numeric(city_backbone["puma20_split_flag"], errors="coerce").mean()))
    _add_metric("share_split_cbsa23", float(pd.to_numeric(city_backbone["cbsa23_split_flag"], errors="coerce").mean()))
    _add_metric("mean_tract20_count_per_city", float(pd.to_numeric(city_backbone["tract20_count"], errors="coerce").mean()))
    _add_metric("mean_zcta20_count_per_city", float(pd.to_numeric(city_backbone["zcta20_count"], errors="coerce").mean()))
    quality_summary = pd.DataFrame.from_records(quality_summary_rows)

    # Persist outputs.
    _write(city_backbone, out_dir / "city_geo_backbone.csv")
    _write(city_place20_place10, out_dir / "city_place20_place10_crosswalk.csv")
    _write(city_place20_zcta20, out_dir / "city_place20_zcta20_crosswalk.csv")
    _write(city_place20_tract20, out_dir / "city_place20_tract20_crosswalk.csv")
    _write(city_place20_county20, out_dir / "city_place20_county20_crosswalk.csv")
    _write(city_place20_puma20, out_dir / "city_place20_puma20_crosswalk.csv")
    _write(city_place20_ua20, out_dir / "city_place20_ua20_crosswalk.csv")
    _write(city_place20_cbsa23, out_dir / "city_place20_cbsa23_crosswalk.csv")
    _write(city_tract20_tract10, out_dir / "city_tract20_tract10_crosswalk.csv")
    _write(city_puma20_puma10, out_dir / "city_puma20_puma10_crosswalk.csv")
    _write(city_zcta20_zcta10, out_dir / "city_zcta20_zcta10_crosswalk.csv")
    _write(city_quality, out_dir / "crosswalk_city_quality.csv")
    _write(quality_summary, out_dir / "crosswalk_quality_summary.csv")

    prov = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(run_dir),
        "input_city_place_crosswalk": str(city_place_path),
        "output_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "min_weight": float(args.min_weight),
        "refresh_cache": bool(args.refresh_cache),
        "files": {k: str(v) for k, v in rel_local.items()},
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "city_count": city_count,
        "output_rows": {
            "city_geo_backbone": int(len(city_backbone)),
            "city_place20_place10_crosswalk": int(len(city_place20_place10)),
            "city_place20_zcta20_crosswalk": int(len(city_place20_zcta20)),
            "city_place20_tract20_crosswalk": int(len(city_place20_tract20)),
            "city_place20_county20_crosswalk": int(len(city_place20_county20)),
            "city_place20_puma20_crosswalk": int(len(city_place20_puma20)),
            "city_place20_ua20_crosswalk": int(len(city_place20_ua20)),
            "city_place20_cbsa23_crosswalk": int(len(city_place20_cbsa23)),
            "city_tract20_tract10_crosswalk": int(len(city_tract20_tract10)),
            "city_puma20_puma10_crosswalk": int(len(city_puma20_puma10)),
            "city_zcta20_zcta10_crosswalk": int(len(city_zcta20_zcta10)),
        },
    }
    with (out_dir / "provenance.json").open("w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, sort_keys=True)

    print(
        "built_geo_backbone "
        f"cities={city_count} "
        f"place10_rows={len(city_place20_place10)} "
        f"tract_rows={len(city_place20_tract20)} "
        f"county_rows={len(city_place20_county20)} "
        f"puma_rows={len(city_place20_puma20)} "
        f"cbsa_rows={len(city_place20_cbsa23)}"
    )
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
