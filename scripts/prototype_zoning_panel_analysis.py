#!/usr/bin/env python3
"""
Prototype analysis workflow for zoning-text measurement.

This script does four things in one run:

1) Builds baseline summary tables from the v4 issue-level mechanics panel:
   - national coverage by year
   - initial full-ordinance adoption timing
   - region and urbanicity splits
   - length + mechanics-complexity profiles over time

2) Reconstructs issue-level zoning text panels for selected city-state units using:
   - merged issue manifest (issue_id -> pages_with_zoning)
   - local per-page *.vlm.json OCR outputs

3) Runs an LLM experiment via agent-gateway:
   - panel-level prompt (multiple issues in one request)
   - issue-level prompt (one issue per request)
   - compares consistency and measurement outputs

4) Writes a compact markdown + CSV artifact bundle under --output-dir.

Notes:
- Urbanicity is a transparent proxy based on 2020 Census place population.
- Text reconstruction is limited by locally available page files.
- This is intentionally "0 -> 1": practical and auditable, not final.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# Census four-region convention.
_REGION_MAP: dict[str, str] = {
    "ct": "Northeast",
    "me": "Northeast",
    "ma": "Northeast",
    "nh": "Northeast",
    "ri": "Northeast",
    "vt": "Northeast",
    "nj": "Northeast",
    "ny": "Northeast",
    "pa": "Northeast",
    "il": "Midwest",
    "in": "Midwest",
    "mi": "Midwest",
    "oh": "Midwest",
    "wi": "Midwest",
    "ia": "Midwest",
    "ks": "Midwest",
    "mn": "Midwest",
    "mo": "Midwest",
    "ne": "Midwest",
    "nd": "Midwest",
    "sd": "Midwest",
    "de": "South",
    "fl": "South",
    "ga": "South",
    "md": "South",
    "nc": "South",
    "sc": "South",
    "va": "South",
    "dc": "South",
    "wv": "South",
    "al": "South",
    "ky": "South",
    "ms": "South",
    "tn": "South",
    "ar": "South",
    "la": "South",
    "ok": "South",
    "tx": "South",
    "az": "West",
    "co": "West",
    "id": "West",
    "mt": "West",
    "nv": "West",
    "nm": "West",
    "ut": "West",
    "wy": "West",
    "ak": "West",
    "ca": "West",
    "hi": "West",
    "or": "West",
    "wa": "West",
}


_STATE_FIPS_TO_ABBR: dict[str, str] = {
    "01": "al",
    "02": "ak",
    "04": "az",
    "05": "ar",
    "06": "ca",
    "08": "co",
    "09": "ct",
    "10": "de",
    "11": "dc",
    "12": "fl",
    "13": "ga",
    "15": "hi",
    "16": "id",
    "17": "il",
    "18": "in",
    "19": "ia",
    "20": "ks",
    "21": "ky",
    "22": "la",
    "23": "me",
    "24": "md",
    "25": "ma",
    "26": "mi",
    "27": "mn",
    "28": "ms",
    "29": "mo",
    "30": "mt",
    "31": "ne",
    "32": "nv",
    "33": "nh",
    "34": "nj",
    "35": "nm",
    "36": "ny",
    "37": "nc",
    "38": "nd",
    "39": "oh",
    "40": "ok",
    "41": "or",
    "42": "pa",
    "44": "ri",
    "45": "sc",
    "46": "sd",
    "47": "tn",
    "48": "tx",
    "49": "ut",
    "50": "vt",
    "51": "va",
    "53": "wa",
    "54": "wv",
    "55": "wi",
    "56": "wy",
}


_PLACE_SUFFIXES = (
    "census designated place",
    "unified government balance",
    "metropolitan government balance",
    "municipio",
    "municipality",
    "city and borough",
    "borough",
    "village",
    "town",
    "city",
    "cdp",
)

_FIXED_CATEGORY_KEYS = [
    "land_use_restrictions",
    "bulk_dimensional_standards",
    "procedural_governance",
    "nonconforming_use",
    "definitions",
    "parking_transport",
    "environmental_protection",
    "historic_preservation",
    "flexible_zoning",
    "inclusionary_affordable_housing",
    "enforcement_penalties",
    "miscellaneous_site_rules",
]

_DEFAULT_REGION_TARGETS = {
    "Midwest": 21,
    "South": 14,
    "Northeast": 13,
    "West": 12,
}

_DEFAULT_URBANICITY_MINIMUMS = {
    "urban": 10,
    "suburban": 20,
    "rural": 25,
}

_NOISE_PATTERNS = {
    "classified_ad": re.compile(r"\b(classified|for sale|call today|realtor|garage sale|help wanted)\b", flags=re.I),
    "foreclosure_legal": re.compile(
        r"\b(foreclosure|sheriff sale|case no\.?|public notice|notice of sale|circuit court|estate of)\b",
        flags=re.I,
    ),
    "election_ballot": re.compile(
        r"\b(general election|ballot|polls will be open|proposal [a-z]|qualified electors)\b",
        flags=re.I,
    ),
    "non_zoning_public_notice": re.compile(
        r"\b(library trustees|public library district|treasurer'?s statement|condominium sale|domestic relations)\b",
        flags=re.I,
    ),
}

_NOISE_NOTE_PATTERN = re.compile(
    r"\b(unrelated|non-zoning|non zoning|ads|advertis|foreclosure|election|noise|ignored)\b",
    flags=re.I,
)


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _safe_slug(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] or "x"


def _parse_csv_paths(raw: str) -> list[Path]:
    out: list[Path] = []
    for p in str(raw).split(","):
        q = p.strip()
        if not q:
            continue
        out.append(Path(q).expanduser().resolve())
    return out


def _parse_named_int_targets(raw: str, *, normalize_lower: bool = False) -> dict[str, int]:
    """
    Parse strings like:
      "Midwest:21,South:14,Northeast:13,West:12"
      "urban:10,suburban:20,rural:25"
    """
    out: dict[str, int] = {}
    for token in str(raw or "").split(","):
        t = token.strip()
        if not t:
            continue
        if ":" not in t:
            continue
        k, v = t.split(":", 1)
        key = str(k).strip()
        if normalize_lower:
            key = key.lower()
        try:
            iv = int(str(v).strip())
        except Exception:
            continue
        if iv <= 0:
            continue
        out[key] = iv
    return out


def _mode_with_fallback(values: pd.Series, *, fallback: str = "unknown", prefer_order: list[str] | None = None) -> str:
    s = values.astype(str).str.strip().str.lower()
    s = s[s != ""]
    if s.empty:
        return fallback
    if prefer_order:
        pref = [str(x).strip().lower() for x in prefer_order]
        non_unknown = s[s.isin(pref)]
        if not non_unknown.empty:
            vc = non_unknown.value_counts()
            mx = int(vc.max())
            tied = [k for k, v in vc.items() if int(v) == mx]
            for p in pref:
                if p in tied:
                    return p
    vc = s.value_counts()
    return str(vc.index[0]) if not vc.empty else fallback


def _load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported input format: {path} (expected .parquet or .csv)")


def _canonical_place_name(name: str) -> str:
    s = str(name or "").strip().lower()
    if not s:
        return ""
    # Keep only part before comma in Census NAME field ("X city, State").
    if "," in s:
        s = s.split(",", 1)[0].strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove known administrative suffixes (longest first).
    for suf in sorted(_PLACE_SUFFIXES, key=len, reverse=True):
        if s.endswith(" " + suf):
            s = s[: -len(suf) - 1].strip()
    # Common apostrophe normalization.
    s = s.replace("saint ", "st ")
    return s.strip()


def _region_from_state(state_abbr: str) -> str:
    return _REGION_MAP.get(str(state_abbr or "").strip().lower(), "Unknown")


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _fetch_census_place_population(*, cache_path: Path, refresh: bool) -> dict[tuple[str, str], int]:
    """
    Return mapping:
      (state_abbr_lower, canonical_place_name) -> max(2020 population)
    """
    if cache_path.is_file() and not refresh:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        out: dict[tuple[str, str], int] = {}
        for k, v in raw.items():
            st, nm = k.split("||", 1)
            out[(st, nm)] = int(v)
        return out

    url = "https://api.census.gov/data/2020/dec/pl?get=NAME,P1_001N&for=place:*&in=state:*"
    with urllib.request.urlopen(url, timeout=90) as r:
        arr = json.loads(r.read().decode("utf-8"))
    if not isinstance(arr, list) or len(arr) < 2:
        raise SystemExit("Unexpected Census API response shape")

    header = arr[0]
    hix = {k: i for i, k in enumerate(header)}
    need = {"NAME", "P1_001N", "state"}
    if any(k not in hix for k in need):
        raise SystemExit(f"Census API response missing expected fields: {need}")

    out: dict[tuple[str, str], int] = {}
    for row in arr[1:]:
        try:
            state_fips = str(row[hix["state"]]).zfill(2)
            st = _STATE_FIPS_TO_ABBR.get(state_fips)
            if not st:
                continue
            nm = _canonical_place_name(str(row[hix["NAME"]]))
            if not nm:
                continue
            pop = int(float(row[hix["P1_001N"]]))
        except Exception:
            continue
        key = (st, nm)
        prev = out.get(key)
        if prev is None or pop > prev:
            out[key] = pop

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serial = {f"{k[0]}||{k[1]}": int(v) for k, v in out.items()}
    cache_path.write_text(json.dumps(serial, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _urbanicity_from_population(pop: int | None, *, pub_slug: str) -> str:
    if pop is not None:
        if pop >= 100000:
            return "urban"
        if pop >= 25000:
            return "suburban"
        return "rural"

    # Fallback heuristic if place-name match failed.
    s = str(pub_slug or "").lower()
    if "suburban" in s:
        return "suburban"
    if any(tok in s for tok in ("rural", "township", "county", "village")):
        return "rural"
    return "unknown"


def _attach_geo_proxies(issue_df: pd.DataFrame, *, place_pop: dict[tuple[str, str], int]) -> pd.DataFrame:
    df = issue_df.copy()
    df["state_abbr"] = df["state_abbr"].astype(str).str.lower()
    df["region"] = df["state_abbr"].map(_REGION_MAP).fillna("Unknown")

    def _lookup_pop(row: pd.Series) -> int | None:
        st = str(row.get("state_abbr") or "").strip().lower()
        city = _canonical_place_name(str(row.get("city_name") or ""))
        if not st or not city:
            return None
        v = place_pop.get((st, city))
        if v is not None:
            return int(v)
        # Try a second pass without "st"/"saint" normalization.
        city2 = re.sub(r"\bst\b", "saint", city)
        if city2 != city:
            v2 = place_pop.get((st, city2))
            if v2 is not None:
                return int(v2)
        return None

    df["population_2020_place"] = df.apply(_lookup_pop, axis=1)
    df["urbanicity_proxy"] = df.apply(
        lambda r: _urbanicity_from_population(
            int(r["population_2020_place"]) if pd.notna(r["population_2020_place"]) else None,
            pub_slug=str(r.get("pub_slug") or ""),
        ),
        axis=1,
    )
    return df


def _compute_issue_complexity(issue_df: pd.DataFrame) -> pd.DataFrame:
    df = issue_df.copy()
    action_cols = sorted([c for c in df.columns if c.startswith("action__")])
    dim_cols = sorted([c for c in df.columns if c.startswith("dim__")])
    body_cols = sorted([c for c in df.columns if c.startswith("body__")])
    instr_cols = sorted([c for c in df.columns if c.startswith("instr__")])

    for cols in (action_cols, dim_cols, body_cols, instr_cols):
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["n_action_active"] = (df[action_cols] > 0).sum(axis=1) if action_cols else 0
    df["n_dim_active"] = (df[dim_cols] > 0).sum(axis=1) if dim_cols else 0
    df["n_body_active"] = (df[body_cols] > 0).sum(axis=1) if body_cols else 0
    df["n_instr_active"] = (df[instr_cols] > 0).sum(axis=1) if instr_cols else 0
    df["n_mechanics_active"] = df["n_action_active"] + df["n_dim_active"] + df["n_body_active"] + df["n_instr_active"]

    chars = pd.to_numeric(df.get("chunk_chars_total"), errors="coerce").fillna(0.0).clip(lower=0.0)
    df["length_log1p"] = (chars + 1.0).map(math.log)
    # A simple continuous proxy combining structural breadth + text length.
    df["complexity_proxy"] = df["n_mechanics_active"] + 0.75 * df["length_log1p"]
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _write_baseline_tables(issue_df: pd.DataFrame, *, out_dir: Path) -> dict[str, Path]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    df = issue_df.copy()
    df["issue_year"] = pd.to_datetime(df["issue_date"], errors="coerce").dt.year

    # 1) National coverage by year.
    cov = (
        df.groupby("issue_year", dropna=True)
        .agg(
            n_issues=("issue_id", "nunique"),
            n_city_states=("city_name", lambda s: int(pd.DataFrame({"city_name": s, "state_abbr": df.loc[s.index, "state_abbr"]}).drop_duplicates().shape[0])),
            n_cities=("city_name", "nunique"),
            n_states=("state_abbr", "nunique"),
            n_pub_slugs=("pub_slug", "nunique"),
        )
        .reset_index()
        .sort_values("issue_year")
    )
    lab_counts = (
        df.pivot_table(index="issue_year", columns="classification_label", values="issue_id", aggfunc="nunique", fill_value=0)
        .reset_index()
    )
    cov = cov.merge(lab_counts, on="issue_year", how="left")
    p = tables_dir / "national_coverage_by_year.csv"
    _write_csv(cov, p)
    out["national_coverage_by_year"] = p

    # 2) Initial full-ordinance adoption by city-state.
    full = df[df["classification_label"] == "full_ordinance"].copy()
    full["issue_date"] = pd.to_datetime(full["issue_date"], errors="coerce")
    first = (
        full.dropna(subset=["issue_date"])
        .sort_values("issue_date")
        .groupby(["city_name", "state_abbr"], as_index=False)
        .first()[["city_name", "state_abbr", "region", "urbanicity_proxy", "population_2020_place", "issue_date", "pub_slug"]]
        .rename(columns={"issue_date": "first_full_ordinance_date", "pub_slug": "example_pub_slug"})
    )
    first["first_full_ordinance_year"] = first["first_full_ordinance_date"].dt.year
    p = tables_dir / "initial_full_ordinance_by_city.csv"
    _write_csv(first, p)
    out["initial_full_ordinance_by_city"] = p

    by_region = (
        first.groupby("region", dropna=False)
        .agg(
            n_city_states=("city_name", "count"),
            first_year_min=("first_full_ordinance_year", "min"),
            first_year_p25=("first_full_ordinance_year", lambda s: float(pd.Series(s).quantile(0.25))),
            first_year_median=("first_full_ordinance_year", "median"),
            first_year_p75=("first_full_ordinance_year", lambda s: float(pd.Series(s).quantile(0.75))),
            first_year_max=("first_full_ordinance_year", "max"),
        )
        .reset_index()
        .sort_values("region")
    )
    p = tables_dir / "initial_full_ordinance_by_region.csv"
    _write_csv(by_region, p)
    out["initial_full_ordinance_by_region"] = p

    by_urb = (
        first.groupby("urbanicity_proxy", dropna=False)
        .agg(
            n_city_states=("city_name", "count"),
            first_year_min=("first_full_ordinance_year", "min"),
            first_year_median=("first_full_ordinance_year", "median"),
            first_year_max=("first_full_ordinance_year", "max"),
        )
        .reset_index()
        .sort_values("urbanicity_proxy")
    )
    p = tables_dir / "initial_full_ordinance_by_urbanicity_proxy.csv"
    _write_csv(by_urb, p)
    out["initial_full_ordinance_by_urbanicity_proxy"] = p

    # 3) Distribution by region x urbanicity x label.
    dist = (
        df.groupby(["region", "urbanicity_proxy", "classification_label"], dropna=False)
        .agg(n_issues=("issue_id", "nunique"), n_city_states=("city_name", lambda s: int(pd.DataFrame({"city_name": s, "state_abbr": df.loc[s.index, "state_abbr"]}).drop_duplicates().shape[0])))
        .reset_index()
        .sort_values(["region", "urbanicity_proxy", "classification_label"])
    )
    p = tables_dir / "distribution_region_urbanicity_label.csv"
    _write_csv(dist, p)
    out["distribution_region_urbanicity_label"] = p

    # 4) Length over time and by urbanicity.
    for c in ("chunk_chars_total", "chunk_count", "n_mechanics_active", "complexity_proxy"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    length_y = (
        df.groupby("issue_year", dropna=True)
        .agg(
            n_issues=("issue_id", "nunique"),
            chunk_chars_median=("chunk_chars_total", "median"),
            chunk_chars_mean=("chunk_chars_total", "mean"),
            chunk_count_median=("chunk_count", "median"),
            mechanics_active_mean=("n_mechanics_active", "mean"),
            complexity_proxy_mean=("complexity_proxy", "mean"),
        )
        .reset_index()
        .sort_values("issue_year")
    )
    p = tables_dir / "length_complexity_over_time.csv"
    _write_csv(length_y, p)
    out["length_complexity_over_time"] = p

    length_urb = (
        df.groupby(["issue_year", "urbanicity_proxy"], dropna=False)
        .agg(
            n_issues=("issue_id", "nunique"),
            chunk_chars_median=("chunk_chars_total", "median"),
            chunk_count_median=("chunk_count", "median"),
            mechanics_active_mean=("n_mechanics_active", "mean"),
            complexity_proxy_mean=("complexity_proxy", "mean"),
        )
        .reset_index()
        .sort_values(["urbanicity_proxy", "issue_year"])
    )
    p = tables_dir / "length_complexity_over_time_by_urbanicity_proxy.csv"
    _write_csv(length_urb, p)
    out["length_complexity_over_time_by_urbanicity_proxy"] = p

    # 5) Topic composition proxies (mechanics tags).
    mech_cols = [c for c in df.columns if c.startswith(("action__", "dim__", "body__", "instr__"))]
    sums = df[mech_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=0)
    top = (
        pd.DataFrame({"feature": sums.index, "weight_sum": sums.values})
        .sort_values("weight_sum", ascending=False)
        .head(80)
        .reset_index(drop=True)
    )
    total = float(top["weight_sum"].sum()) or 1.0
    top["weight_share_within_top80"] = top["weight_sum"] / total
    p = tables_dir / "topic_mechanics_weight_share_top80.csv"
    _write_csv(top, p)
    out["topic_mechanics_weight_share_top80"] = p

    return out


def _text_noise_hits(text: str) -> list[str]:
    hits: list[str] = []
    t = str(text or "")
    for name, pat in _NOISE_PATTERNS.items():
        if pat.search(t):
            hits.append(name)
    return hits


def _has_noise_note(parsed_obj: dict[str, Any] | None) -> bool:
    if not isinstance(parsed_obj, dict):
        return False
    notes = parsed_obj.get("notes")
    txt = ""
    if isinstance(notes, list):
        txt = " ".join(str(x) for x in notes)
    elif isinstance(notes, str):
        txt = notes
    return bool(_NOISE_NOTE_PATTERN.search(txt))


def _load_issue_noise_map(parsed_issue_path: Path | None) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if parsed_issue_path is None or not parsed_issue_path.is_file():
        return out
    for raw in parsed_issue_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        cid = str(obj.get("custom_id") or "")
        parts = cid.split("::", 2)
        if len(parts) < 3:
            continue
        issue_id = parts[2]
        out[issue_id] = _has_noise_note(obj.get("parsed_json") if isinstance(obj.get("parsed_json"), dict) else None)
    return out


def _load_panel_noise_map(parsed_panel_path: Path | None) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if parsed_panel_path is None or not parsed_panel_path.is_file():
        return out
    for raw in parsed_panel_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        cid = str(obj.get("custom_id") or "")
        parts = cid.split("::", 1)
        city_key = parts[1] if len(parts) > 1 else cid
        out[city_key] = _has_noise_note(obj.get("parsed_json") if isinstance(obj.get("parsed_json"), dict) else None)
    return out


def _write_contamination_tables(
    *,
    panel_rows: list[PanelIssue],
    out_dir: Path,
    llm_paths: dict[str, Path] | None,
) -> dict[str, Path]:
    issue_noise_map = _load_issue_noise_map(None if not llm_paths else llm_paths.get("parsed_issue_outputs"))
    panel_noise_map = _load_panel_noise_map(None if not llm_paths else llm_paths.get("parsed_panel_outputs"))
    rows: list[dict[str, Any]] = []
    for r in panel_rows:
        hits = _text_noise_hits(r.text)
        lexical_flag = bool(hits)
        llm_issue_noise = bool(issue_noise_map.get(r.issue_id, False))
        llm_panel_noise = bool(panel_noise_map.get(r.city_key, False))
        llm_noise_any = bool(llm_issue_noise or llm_panel_noise)
        rows.append(
            {
                "city_key": r.city_key,
                "issue_id": r.issue_id,
                "issue_date": r.issue_date,
                "classification_label": r.classification_label,
                "region": r.region,
                "urbanicity_proxy": r.urbanicity_proxy,
                "text_chars": int(r.text_chars),
                "page_count": int(r.page_count),
                "lexical_noise_flag": int(lexical_flag),
                "noise_hit_terms": "|".join(hits),
                "llm_issue_noise_note_flag": int(llm_issue_noise),
                "llm_panel_noise_note_flag": int(llm_panel_noise),
                "llm_noise_flag_any": int(llm_noise_any),
                "noise_flag_any": int(bool(lexical_flag or llm_noise_any)),
            }
        )
    issue_df = pd.DataFrame.from_records(rows)
    if issue_df.empty:
        issue_df = pd.DataFrame(
            columns=[
                "city_key",
                "issue_id",
                "issue_date",
                "classification_label",
                "region",
                "urbanicity_proxy",
                "text_chars",
                "page_count",
                "lexical_noise_flag",
                "noise_hit_terms",
                "llm_issue_noise_note_flag",
                "llm_panel_noise_note_flag",
                "llm_noise_flag_any",
                "noise_flag_any",
            ]
        )
    issue_out = out_dir / "tables" / "contamination_issue_flags.csv"
    _write_csv(issue_df, issue_out)

    if issue_df.empty:
        panel_summary = pd.DataFrame(
            columns=[
                "city_key",
                "classification_label",
                "n_issues",
                "lexical_noise_share",
                "llm_noise_share",
                "pre_filter_noise_share",
                "post_filter_noise_share",
            ]
        )
    else:
        grp = issue_df.groupby(["city_key", "classification_label"], dropna=False)
        panel_summary = (
            grp.agg(
                n_issues=("issue_id", "nunique"),
                lexical_noise_share=("lexical_noise_flag", "mean"),
                llm_noise_share=("llm_noise_flag_any", "mean"),
                pre_filter_noise_share=("noise_flag_any", "mean"),
            )
            .reset_index()
            .sort_values(["city_key", "classification_label"])
        )
        post_vals: list[float] = []
        for r in panel_summary.itertuples(index=False):
            sub = issue_df[
                (issue_df["city_key"] == r.city_key)
                & (issue_df["classification_label"] == r.classification_label)
                & (issue_df["lexical_noise_flag"] == 0)
            ]
            post_vals.append(float(sub["llm_noise_flag_any"].mean()) if not sub.empty else math.nan)
        panel_summary["post_filter_noise_share"] = post_vals
    panel_out = out_dir / "tables" / "contamination_panel_summary.csv"
    _write_csv(panel_summary, panel_out)
    return {
        "contamination_issue_flags": issue_out,
        "contamination_panel_summary": panel_out,
    }


def _load_issue_numeric_constraints(parsed_issue_path: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if parsed_issue_path is None or not parsed_issue_path.is_file():
        return out
    for raw in parsed_issue_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        cid = str(obj.get("custom_id") or "")
        parts = cid.split("::", 2)
        if len(parts) < 3:
            continue
        issue_id = parts[2]
        parsed = obj.get("parsed_json")
        if not isinstance(parsed, dict):
            continue
        nc = parsed.get("numeric_constraints")
        if isinstance(nc, dict):
            out[issue_id] = nc
    return out


def _write_numeric_constraint_tables(
    *,
    panel_rows: list[PanelIssue],
    out_dir: Path,
    llm_paths: dict[str, Path] | None,
) -> dict[str, Path]:
    nc_map = _load_issue_numeric_constraints(None if not llm_paths else llm_paths.get("parsed_issue_outputs"))
    rows: list[dict[str, Any]] = []
    for r in panel_rows:
        nums = r.numeric_mentions or {}
        llm_nc = nc_map.get(r.issue_id, {})
        year = pd.to_datetime(r.issue_date, errors="coerce").year
        rows.append(
            {
                "city_key": r.city_key,
                "city_name": r.city_name,
                "state_abbr": r.state_abbr,
                "region": r.region,
                "urbanicity_proxy": r.urbanicity_proxy,
                "issue_id": r.issue_id,
                "issue_date": r.issue_date,
                "issue_year": int(year) if pd.notna(year) else math.nan,
                "classification_label": r.classification_label,
                "distinct_zone_code_count": int(nums.get("distinct_zone_code_count") or 0),
                "max_min_lot_size_sqft": nums.get("max_min_lot_size_sqft"),
                "max_sqft_mention": nums.get("max_sqft_mention"),
                "max_height_ft_mention": nums.get("max_height_ft_mention"),
                "max_front_setback_ft": llm_nc.get("max_front_setback_ft"),
                "max_parking_spaces_per_unit": llm_nc.get("max_parking_spaces_per_unit"),
            }
        )
    issue_num_df = pd.DataFrame.from_records(rows)

    city_year = (
        issue_num_df.groupby(["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "issue_year"], dropna=False)
        .agg(
            n_issues=("issue_id", "nunique"),
            distinct_zone_code_count_max=("distinct_zone_code_count", "max"),
            max_min_lot_size_sqft=("max_min_lot_size_sqft", "max"),
            max_height_ft_mention=("max_height_ft_mention", "max"),
            max_front_setback_ft=("max_front_setback_ft", "max"),
            max_parking_spaces_per_unit=("max_parking_spaces_per_unit", "max"),
        )
        .reset_index()
        .sort_values(["city_key", "issue_year"])
    )
    city_year_path = out_dir / "tables" / "distinct_zone_metrics_by_city_year.csv"
    _write_csv(city_year, city_year_path)

    ext = (
        city_year.groupby(["region", "urbanicity_proxy"], dropna=False)
        .agg(
            n_city_year_rows=("city_key", "count"),
            distinct_zone_code_count_p50=("distinct_zone_code_count_max", "median"),
            distinct_zone_code_count_p90=("distinct_zone_code_count_max", lambda s: float(pd.Series(s).quantile(0.9))),
            max_min_lot_size_sqft_p90=("max_min_lot_size_sqft", lambda s: float(pd.Series(s).quantile(0.9))),
            max_height_ft_p90=("max_height_ft_mention", lambda s: float(pd.Series(s).quantile(0.9))),
            max_front_setback_ft_p90=("max_front_setback_ft", lambda s: float(pd.Series(s).quantile(0.9))),
        )
        .reset_index()
        .sort_values(["region", "urbanicity_proxy"])
    )
    ext_path = out_dir / "tables" / "numeric_constraints_extremes_by_region_urbanicity.csv"
    _write_csv(ext, ext_path)

    lot_ext = issue_num_df[issue_num_df["max_min_lot_size_sqft"].notna()].copy()
    lot_ext = lot_ext.sort_values("max_min_lot_size_sqft", ascending=False).head(50)
    lot_path = out_dir / "tables" / "single_family_lot_size_extremes.csv"
    _write_csv(lot_ext, lot_path)

    qc_rows = [
        {
            "metric": "distinct_zone_code_count",
            "non_null_share": float(issue_num_df["distinct_zone_code_count"].notna().mean()) if not issue_num_df.empty else math.nan,
            "positive_share": float((issue_num_df["distinct_zone_code_count"] > 0).mean()) if not issue_num_df.empty else math.nan,
            "extreme_flag_share": float((issue_num_df["distinct_zone_code_count"] > 200).mean()) if not issue_num_df.empty else math.nan,
        },
        {
            "metric": "max_min_lot_size_sqft",
            "non_null_share": float(issue_num_df["max_min_lot_size_sqft"].notna().mean()) if not issue_num_df.empty else math.nan,
            "positive_share": float((pd.to_numeric(issue_num_df["max_min_lot_size_sqft"], errors="coerce").fillna(0) > 0).mean())
            if not issue_num_df.empty
            else math.nan,
            "extreme_flag_share": float((pd.to_numeric(issue_num_df["max_min_lot_size_sqft"], errors="coerce").fillna(0) > 200000).mean())
            if not issue_num_df.empty
            else math.nan,
        },
        {
            "metric": "max_height_ft_mention",
            "non_null_share": float(issue_num_df["max_height_ft_mention"].notna().mean()) if not issue_num_df.empty else math.nan,
            "positive_share": float((pd.to_numeric(issue_num_df["max_height_ft_mention"], errors="coerce").fillna(0) > 0).mean())
            if not issue_num_df.empty
            else math.nan,
            "extreme_flag_share": float((pd.to_numeric(issue_num_df["max_height_ft_mention"], errors="coerce").fillna(0) > 500).mean())
            if not issue_num_df.empty
            else math.nan,
        },
        {
            "metric": "max_front_setback_ft",
            "non_null_share": float(issue_num_df["max_front_setback_ft"].notna().mean()) if not issue_num_df.empty else math.nan,
            "positive_share": float((pd.to_numeric(issue_num_df["max_front_setback_ft"], errors="coerce").fillna(0) > 0).mean())
            if not issue_num_df.empty
            else math.nan,
            "extreme_flag_share": float((pd.to_numeric(issue_num_df["max_front_setback_ft"], errors="coerce").fillna(0) > 300).mean())
            if not issue_num_df.empty
            else math.nan,
        },
    ]
    qc = pd.DataFrame.from_records(qc_rows)
    qc_path = out_dir / "tables" / "numeric_extraction_quality_checks.csv"
    _write_csv(qc, qc_path)

    return {
        "distinct_zone_metrics_by_city_year": city_year_path,
        "numeric_constraints_extremes_by_region_urbanicity": ext_path,
        "single_family_lot_size_extremes": lot_path,
        "numeric_extraction_quality_checks": qc_path,
    }


def _load_issue_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing merged manifest: {path}")
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            issue_id = obj.get("issue_id")
            if isinstance(issue_id, str) and issue_id:
                out[issue_id] = obj
    if not out:
        raise SystemExit(f"No issue rows loaded from {path}")
    return out


def _page_sort_key(page_id: str) -> tuple[int, str]:
    m = re.search(r"-p-(\d+)$", str(page_id))
    if m:
        return (int(m.group(1)), str(page_id))
    return (10**9, str(page_id))


def _box_sort_key(box: dict[str, Any]) -> tuple[int, str]:
    bid = str(box.get("id") or "")
    m = re.search(r"(\d+)$", bid)
    if m:
        return (int(m.group(1)), bid)
    return (10**9, bid)


def _extract_text_from_vlm_page(path: Path) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))
    boxes = obj.get("boxes")
    if not isinstance(boxes, list):
        return ""
    parts: list[str] = []
    for b in sorted([x for x in boxes if isinstance(x, dict)], key=_box_sort_key):
        status = str(b.get("status") or "").strip().lower()
        txt = str(b.get("transcript") or "").strip()
        if not txt:
            continue
        if status in {"unreadable", "error"}:
            continue
        parts.append(txt)
    text = "\n\n".join(parts).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _find_page_file(page_id: str, roots: list[Path]) -> Path | None:
    fn = f"{page_id}.vlm.json"
    for root in roots:
        p = root / fn
        if p.is_file():
            return p
    return None


def _extract_numeric_mentions(text: str) -> dict[str, Any]:
    t = str(text or "")

    zone_codes = sorted(set(re.findall(r"\b([A-Z]{1,3}-\d{1,3}[A-Z]?)\b", t)))
    lot_sqft_vals = [int(x) for x in re.findall(r"\b(\d{3,7})\s*(?:square\s*feet|sq\.?\s*ft|sqft)\b", t, flags=re.I)]
    height_vals = [int(x) for x in re.findall(r"\b(\d{1,3})\s*(?:feet|foot|ft)\b", t, flags=re.I)]

    # Naive "min lot size" anchors.
    min_lot_vals: list[int] = []
    for m in re.finditer(r"(minimum|min\.)\s+lot(?:\s+area|\s+size)?[^0-9]{0,24}(\d{3,7})", t, flags=re.I):
        try:
            min_lot_vals.append(int(m.group(2)))
        except Exception:
            continue

    return {
        "distinct_zone_code_count": int(len(zone_codes)),
        "distinct_zone_codes_sample": zone_codes[:30],
        "max_min_lot_size_sqft": int(max(min_lot_vals)) if min_lot_vals else None,
        "max_sqft_mention": int(max(lot_sqft_vals)) if lot_sqft_vals else None,
        "max_height_ft_mention": int(max(height_vals)) if height_vals else None,
    }


@dataclass(frozen=True)
class PanelIssue:
    issue_id: str
    city_name: str
    state_abbr: str
    city_key: str
    region: str
    urbanicity_proxy: str
    issue_date: str
    classification_label: str
    text: str
    text_chars: int
    page_count: int
    numeric_mentions: dict[str, Any]


def _choose_evenly_spaced(rows: pd.DataFrame, k: int) -> pd.DataFrame:
    if rows.empty:
        return rows
    if len(rows) <= k:
        return rows.copy()
    rows = rows.sort_values("issue_date").reset_index(drop=True)
    idxs = sorted(set(int(round(i)) for i in pd.Series(range(k)).map(lambda j: j * (len(rows) - 1) / max(1, (k - 1)))))
    return rows.iloc[idxs].copy()


def _build_panels(
    *,
    issue_df: pd.DataFrame,
    issue_manifest: dict[str, dict[str, Any]],
    vlm_roots: list[Path],
    out_dir: Path,
    panel_count: int,
    panel_issues: int,
    min_panel_issues: int,
    selection_mode: str,
    region_targets: dict[str, int],
    urbanicity_minimums: dict[str, int],
    write_selected_issues: bool,
) -> tuple[pd.DataFrame, list[PanelIssue], pd.DataFrame, dict[str, Path]]:
    """
    Select high-coverage city-state panels, reconstruct issue text, and write panel artifacts.
    """
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    df = issue_df.copy()
    df["state_abbr"] = df["state_abbr"].astype(str).str.lower()
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df = df.dropna(subset=["issue_date"])

    # Mark text availability at issue level.
    has_text_flags: list[bool] = []
    for iid in df["issue_id"].astype(str).tolist():
        man = issue_manifest.get(iid)
        if not man:
            has_text_flags.append(False)
            continue
        pages = [p for p in (man.get("pages_with_zoning") or []) if isinstance(p, dict) and p.get("page_id")]
        if not pages:
            has_text_flags.append(False)
            continue
        ok = True
        for p in pages:
            pid = str(p.get("page_id"))
            if _find_page_file(pid, vlm_roots) is None:
                ok = False
                break
        has_text_flags.append(ok)
    df["has_local_text"] = has_text_flags

    city = (
        df.groupby(["city_name", "state_abbr"], as_index=False)
        .agg(
            n_issues=("issue_id", "nunique"),
            n_full=("classification_label", lambda s: int((s == "full_ordinance").sum())),
            n_with_text=("has_local_text", "sum"),
            issue_date_min=("issue_date", "min"),
            issue_date_max=("issue_date", "max"),
            region=("region", lambda s: str(s.value_counts().index[0]) if not s.empty else "Unknown"),
            urbanicity_proxy=(
                "urbanicity_proxy",
                lambda s: _mode_with_fallback(
                    s,
                    fallback="unknown",
                    prefer_order=["urban", "suburban", "rural", "unknown"],
                ),
            ),
        )
        .sort_values(["n_with_text", "n_issues", "n_full"], ascending=False)
        .reset_index(drop=True)
    )
    city["span_years"] = ((city["issue_date_max"] - city["issue_date_min"]).dt.days / 365.25).fillna(0.0)
    city["city_key"] = city.apply(lambda r: f"{_safe_slug(r.city_name)}__{_safe_slug(r.state_abbr)}", axis=1)
    city = city[city["n_with_text"] >= int(min_panel_issues)].copy()
    city = city.sort_values(["n_with_text", "n_full", "span_years"], ascending=False).reset_index(drop=True)
    city["rank_order"] = range(len(city))

    if city.empty:
        raise SystemExit(
            "No city-state units met panel construction thresholds. "
            "Lower --min-panel-issues or adjust VLM roots."
        )

    selected: pd.DataFrame
    mode = str(selection_mode or "top").strip().lower()
    if mode == "stratified":
        wanted_regions = {k: int(v) for k, v in region_targets.items() if int(v) > 0}
        selected_rows: list[pd.DataFrame] = []
        used_keys: set[str] = set()

        for region_name, target in wanted_regions.items():
            g = city[(city["region"] == region_name) & (~city["city_key"].isin(used_keys))].head(int(target)).copy()
            if not g.empty:
                selected_rows.append(g)
                used_keys |= set(g["city_key"].astype(str).tolist())

        selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else city.head(0).copy()
        remain = city[~city["city_key"].isin(set(selected["city_key"].astype(str).tolist()))].copy()

        if len(selected) < int(panel_count):
            selected = pd.concat([selected, remain.head(int(panel_count) - len(selected))], ignore_index=True)
            remain = remain.iloc[max(0, int(panel_count) - len(selected)) :]

        selected = selected.sort_values(["rank_order"]).head(int(panel_count)).reset_index(drop=True)
        selected_keys = set(selected["city_key"].astype(str).tolist())

        # Urbanicity minimums: swap within same region when possible.
        if urbanicity_minimums:
            selected_u = selected["urbanicity_proxy"].astype(str).str.lower().value_counts().to_dict()
            remain = city[~city["city_key"].isin(selected_keys)].copy()
            for urb, need in urbanicity_minimums.items():
                u = str(urb).strip().lower()
                target = int(need)
                while int(selected_u.get(u, 0)) < target:
                    cand_pool = remain[remain["urbanicity_proxy"].astype(str).str.lower() == u]
                    if cand_pool.empty:
                        break
                    cand = cand_pool.sort_values(["rank_order"]).iloc[0]
                    cand_region = str(cand["region"])
                    rem_pool = selected[selected["region"] == cand_region].copy()
                    if rem_pool.empty:
                        break
                    rem_pool["urb_lower"] = rem_pool["urbanicity_proxy"].astype(str).str.lower()
                    # Only remove from urbanicity buckets above minimum requirements.
                    removable = rem_pool[
                        rem_pool["urb_lower"].map(lambda x: int(selected_u.get(x, 0)) > int(urbanicity_minimums.get(x, 0)))
                    ]
                    if removable.empty:
                        break
                    out_row = removable.sort_values(["rank_order"], ascending=False).iloc[0]
                    out_key = str(out_row["city_key"])
                    in_key = str(cand["city_key"])

                    selected = selected[selected["city_key"] != out_key]
                    selected = pd.concat([selected, cand.to_frame().T], ignore_index=True)
                    selected = selected.sort_values(["rank_order"]).head(int(panel_count)).reset_index(drop=True)

                    selected_keys = set(selected["city_key"].astype(str).tolist())
                    remain = city[~city["city_key"].isin(selected_keys)].copy()
                    selected_u = selected["urbanicity_proxy"].astype(str).str.lower().value_counts().to_dict()
                    if in_key in set(remain["city_key"].astype(str).tolist()):
                        remain = remain[remain["city_key"] != in_key]
                    if out_key not in selected_keys:
                        pass
    else:
        selected = city.head(int(panel_count)).copy()

    selected = selected.sort_values(["rank_order"]).reset_index(drop=True)

    # Selection diagnostics.
    diag_rows: list[dict[str, Any]] = []
    for dim in ("region", "urbanicity_proxy"):
        elig_counts = city[dim].astype(str).value_counts()
        sel_counts = selected[dim].astype(str).value_counts()
        keys = sorted(set(elig_counts.index.tolist()) | set(sel_counts.index.tolist()))
        for k in keys:
            e = int(elig_counts.get(k, 0))
            s = int(sel_counts.get(k, 0))
            diag_rows.append(
                {
                    "dimension": dim,
                    "group": k,
                    "eligible_n": e,
                    "selected_n": s,
                    "eligible_share": (float(e / max(1, len(city)))),
                    "selected_share": (float(s / max(1, len(selected)))),
                }
            )
    diag_df = pd.DataFrame.from_records(diag_rows).sort_values(["dimension", "group"])
    diag_path = out_dir / "tables" / "panel_selection_coverage_bias.csv"
    _write_csv(diag_df, diag_path)

    target_rows: list[dict[str, Any]] = []
    for k, v in region_targets.items():
        target_rows.append(
            {
                "target_type": "region",
                "group": k,
                "target_n": int(v),
                "selected_n": int((selected["region"] == k).sum()),
            }
        )
    for k, v in urbanicity_minimums.items():
        kk = str(k).lower()
        target_rows.append(
            {
                "target_type": "urbanicity_minimum",
                "group": kk,
                "target_n": int(v),
                "selected_n": int((selected["urbanicity_proxy"].astype(str).str.lower() == kk).sum()),
            }
        )
    target_df = pd.DataFrame.from_records(target_rows)
    target_path = out_dir / "tables" / "panel_selection_targets.csv"
    _write_csv(target_df, target_path)

    eligible_path = panel_dir / "eligible_city_states.csv"
    _write_csv(city.sort_values(["rank_order"]).drop(columns=["rank_order"]), eligible_path)

    panel_issues_out: list[PanelIssue] = []
    selected_issue_rows: list[dict[str, Any]] = []

    for crow in selected.itertuples(index=False):
        city_name = str(crow.city_name)
        st = str(crow.state_abbr).lower()
        key = f"{_safe_slug(city_name)}__{_safe_slug(st)}"
        cdir = panel_dir / key
        cdir.mkdir(parents=True, exist_ok=True)

        # Prefer full ordinances first, then substantial, then targeted.
        g = df[(df["city_name"] == city_name) & (df["state_abbr"] == st) & (df["has_local_text"] == True)].copy()
        g["label_rank"] = g["classification_label"].map(
            {"full_ordinance": 0, "amendment_substantial": 1, "amendment_targeted": 2}
        ).fillna(9)
        g = g.sort_values(["label_rank", "issue_date", "issue_id"]).reset_index(drop=True)
        chosen = _choose_evenly_spaced(g, int(panel_issues))

        rows_for_jsonl: list[dict[str, Any]] = []
        for r in chosen.itertuples(index=False):
            iid = str(r.issue_id)
            man = issue_manifest.get(iid)
            if not man:
                continue
            pages = [p for p in (man.get("pages_with_zoning") or []) if isinstance(p, dict) and p.get("page_id")]
            pages = sorted(pages, key=lambda x: _page_sort_key(str(x.get("page_id"))))
            page_texts: list[str] = []
            page_ids: list[str] = []
            for p in pages:
                pid = str(p.get("page_id"))
                fp = _find_page_file(pid, vlm_roots)
                if fp is None:
                    continue
                txt = _extract_text_from_vlm_page(fp)
                if txt:
                    page_texts.append(txt)
                    page_ids.append(pid)

            text = "\n\n".join(page_texts).strip()
            if not text:
                continue
            nums = _extract_numeric_mentions(text)
            row = PanelIssue(
                issue_id=iid,
                city_name=city_name,
                state_abbr=st,
                city_key=key,
                region=str(crow.region),
                urbanicity_proxy=str(crow.urbanicity_proxy),
                issue_date=str(pd.Timestamp(r.issue_date).date()),
                classification_label=str(r.classification_label),
                text=text,
                text_chars=len(text),
                page_count=len(page_ids),
                numeric_mentions=nums,
            )
            panel_issues_out.append(row)
            selected_issue_rows.append(
                {
                    "city_key": key,
                    "city_name": city_name,
                    "state_abbr": st,
                    "region": str(crow.region),
                    "urbanicity_proxy": str(crow.urbanicity_proxy),
                    "issue_id": row.issue_id,
                    "issue_date": row.issue_date,
                    "classification_label": row.classification_label,
                    "text_chars": row.text_chars,
                    "page_count": row.page_count,
                }
            )
            rows_for_jsonl.append(
                {
                    "issue_id": row.issue_id,
                    "city_name": row.city_name,
                    "state_abbr": row.state_abbr,
                    "city_key": row.city_key,
                    "region": row.region,
                    "urbanicity_proxy": row.urbanicity_proxy,
                    "issue_date": row.issue_date,
                    "classification_label": row.classification_label,
                    "text_chars": row.text_chars,
                    "page_count": row.page_count,
                    "page_ids": page_ids,
                    "numeric_mentions": row.numeric_mentions,
                    "text": row.text,
                }
            )

        with (cdir / "issue_texts.jsonl").open("w", encoding="utf-8") as f:
            for obj in rows_for_jsonl:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    city_path = panel_dir / "selected_panels.csv"
    _write_csv(selected.drop(columns=["rank_order"], errors="ignore"), city_path)

    selected_issue_df = pd.DataFrame.from_records(selected_issue_rows)
    if not selected_issue_df.empty:
        selected_issue_df = selected_issue_df.sort_values(["city_key", "issue_date", "issue_id"]).reset_index(drop=True)
    if write_selected_issues:
        _write_csv(selected_issue_df, panel_dir / "selected_panel_issues.csv")

    diag_paths = {
        "panel_selection_coverage_bias": diag_path,
        "panel_selection_targets": target_path,
        "eligible_city_states": eligible_path,
        "selected_panels": city_path,
    }
    return selected.drop(columns=["rank_order"], errors="ignore"), panel_issues_out, selected_issue_df, diag_paths


def _panel_prompt(city_name: str, state_abbr: str, rows: list[PanelIssue], *, max_issue_chars: int) -> str:
    lines: list[str] = []
    lines.append(
        "You are coding zoning regulations from historical newspaper transcripts. "
        "Return ONLY JSON (no markdown, no code fences)."
    )
    lines.append("")
    lines.append("Task:")
    lines.append("1) Infer category composition and regulatory complexity for this city panel.")
    lines.append("2) Use the fixed taxonomy exactly; do not rename categories.")
    lines.append("3) Capture key numeric constraints if present (lot size, height, setback, parking).")
    lines.append("4) Report trend direction over time (increase/decrease/mixed/flat).")
    lines.append("")
    lines.append("Fixed category keys (must all appear in category_shares):")
    lines.append(", ".join(_FIXED_CATEGORY_KEYS))
    lines.append("")
    lines.append("Output JSON schema:")
    lines.append(
        '{"city":"", "state_abbr":"", "category_shares":{"land_use_restrictions":0.0,"bulk_dimensional_standards":0.0,'
        '"procedural_governance":0.0,"nonconforming_use":0.0,"definitions":0.0,"parking_transport":0.0,'
        '"environmental_protection":0.0,"historic_preservation":0.0,"flexible_zoning":0.0,'
        '"inclusionary_affordable_housing":0.0,"enforcement_penalties":0.0,"miscellaneous_site_rules":0.0}, '
        '"topics":[{"name":"","share_0_to_1":0.0,"evidence":""}], '
        '"open_topics":[{"name":"","share_0_to_1":0.0,"evidence":""}], '
        '"open_to_fixed_map":[{"open_topic":"","fixed_category":"","confidence_0_to_1":0.0}], '
        '"measures":{"regulatory_complexity_0_100":0,"procedural_intensity_0_100":0,'
        '"dimensional_specificity_0_100":0,"nonconforming_use_signal_0_100":0,'
        '"flexible_zoning_signal_0_100":0,"inclusionary_signal_0_100":0}, '
        '"numeric_constraints":{"max_min_lot_size_sqft":null,"max_height_ft":null,'
        '"max_front_setback_ft":null,"max_parking_spaces_per_unit":null}, '
        '"time_trend":{"direction":"","evidence":""}, '
        '"notes":[""]}'
    )
    lines.append("Rules: category_shares must be non-negative, sum to 1.0 (+/- 0.02), and use all keys.")
    lines.append("")
    lines.append(f"City: {city_name}, {state_abbr.upper()}")
    lines.append("Panel issues:")
    for i, r in enumerate(sorted(rows, key=lambda x: x.issue_date), start=1):
        txt = str(r.text or "").strip()
        cap = int(max_issue_chars)
        if cap > 0 and len(txt) > cap:
            raise SystemExit(
                f"issue_id={r.issue_id} chars={len(txt)} exceeds hard cap {cap}; no clipping allowed. "
                "Increase the cap or set it to 0 for no local cap."
            )
        excerpt = txt
        lines.append("")
        lines.append(
            f"[ISSUE {i}] issue_id={r.issue_id} date={r.issue_date} label={r.classification_label} chars={r.text_chars}"
        )
        lines.append(excerpt)
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _issue_prompt(row: PanelIssue, *, max_issue_chars: int) -> str:
    txt = str(row.text or "").strip()
    cap = int(max_issue_chars)
    if cap > 0 and len(txt) > cap:
        raise SystemExit(
            f"issue_id={row.issue_id} chars={len(txt)} exceeds hard cap {cap}; no clipping allowed. "
            "Increase the cap or set it to 0 for no local cap."
        )
    excerpt = txt
    return (
        "You are coding a single zoning-related newspaper regulation transcript.\n"
        "Return ONLY JSON (no markdown, no code fences).\n\n"
        "Use this fixed taxonomy exactly for category_shares keys:\n"
        + ", ".join(_FIXED_CATEGORY_KEYS)
        + "\n\n"
        "Output JSON schema:\n"
        '{"issue_id":"", "category_shares":{"land_use_restrictions":0.0,"bulk_dimensional_standards":0.0,'
        '"procedural_governance":0.0,"nonconforming_use":0.0,"definitions":0.0,"parking_transport":0.0,'
        '"environmental_protection":0.0,"historic_preservation":0.0,"flexible_zoning":0.0,'
        '"inclusionary_affordable_housing":0.0,"enforcement_penalties":0.0,"miscellaneous_site_rules":0.0}, '
        '"topics":[{"name":"","share_0_to_1":0.0}], '
        '"open_topics":[{"name":"","share_0_to_1":0.0}], '
        '"open_to_fixed_map":[{"open_topic":"","fixed_category":"","confidence_0_to_1":0.0}], '
        '"measures":{"regulatory_complexity_0_100":0,"procedural_intensity_0_100":0,'
        '"dimensional_specificity_0_100":0}, '
        '"numeric_constraints":{"max_min_lot_size_sqft":null,"max_height_ft":null,'
        '"max_front_setback_ft":null,"max_parking_spaces_per_unit":null}, '
        '"notes":[""]}\n\n'
        "Rules: category_shares must be non-negative, sum to 1.0 (+/- 0.02), and use all keys.\n\n"
        f"Issue metadata: issue_id={row.issue_id} city={row.city_name} state={row.state_abbr.upper()} "
        f"date={row.issue_date} label={row.classification_label}\n\n"
        "Transcript:\n"
        f"{excerpt}\n\n"
        "Return JSON now."
    )


def _write_openai_requests(path: Path, rows: list[tuple[str, str]], *, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cid, prompt in rows:
            body = {
                "model": str(model),
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                "stream": False,
            }
            req = {"custom_id": cid, "method": "POST", "url": "/v1/responses", "body": body}
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def _extract_openai_output_text(body: dict[str, Any]) -> str:
    out = body.get("output")
    if not isinstance(out, list):
        return ""
    parts: list[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "output_text":
                txt = c.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
    return "\n".join(parts).strip()


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    def _try_parse(raw: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    def _repair_common_json_issues(raw: str) -> str:
        # Some model outputs omit the closing brace for an object in "topics",
        # resulting in: "share_0_to_1": 0.55, { ... }.
        fixed = re.sub(
            r'("share_0_to_1"\s*:\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*\n\s*\{',
            r"\1\n    },\n    {",
            raw,
        )
        # Remove trailing commas before object/array close.
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        return fixed

    s = str(text or "").strip()
    if not s:
        return None
    # Drop code fences if present.
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    # Direct parse first.
    obj = _try_parse(s)
    if obj is not None:
        return obj
    # Try lightweight repair for common malformed JSON patterns.
    s_fixed = _repair_common_json_issues(s)
    obj = _try_parse(s_fixed)
    if obj is not None:
        return obj
    # Fallback: extract first {...} block greedily.
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    obj = _try_parse(m.group(0))
    if obj is not None:
        return obj
    obj = _try_parse(_repair_common_json_issues(m.group(0)))
    if obj is not None:
        return obj
    return None


def _run_gateway_requests(
    *,
    request_dir: Path,
    output_dir: Path,
    runner_path: Path,
    model: str,
    gateway_pythonpath: Path | None,
    timeout_s: float,
    concurrency: int,
    gov_env_path: Path,
    skip_existing: bool,
) -> None:
    if not runner_path.is_file():
        raise SystemExit(f"Gateway runner not found: {runner_path}")

    env = dict(os.environ)
    if gateway_pythonpath is not None:
        pp = str(gateway_pythonpath)
        prior = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = pp if not prior else f"{pp}:{prior}"

    parsed = _parse_env_file(gov_env_path)

    def _first_nonempty(keys: list[str]) -> str | None:
        for k in keys:
            v = str(parsed.get(k) or "").strip()
            if v:
                return v
        return None

    # Primary keys (direct).
    for key in ("OPENAI_KEY", "GEMINI_KEY", "CLAUDE_KEY"):
        if not env.get(key) and str(parsed.get(key) or "").strip():
            env[key] = str(parsed[key]).strip()

    # Back-compat aliases that appear in some local env files.
    if not env.get("OPENAI_KEY"):
        alias = str(parsed.get("CODEX_API_KEY") or parsed.get("OPENAI_API_KEY") or "").strip()
        if alias:
            env["OPENAI_KEY"] = alias

    provider = str(model).split(":", 1)[0].lower() if ":" in str(model) else "openai"
    need_key = {
        "openai": "OPENAI_KEY",
        "gemini": "GEMINI_KEY",
        "claude": "CLAUDE_KEY",
    }.get(provider)

    # Build provider-key candidates in priority order:
    # current env -> explicit primary in .env -> backups 1..5.
    provider_key_candidates: list[str] = []
    if need_key:
        seen: set[str] = set()
        for cand in (
            str(env.get(need_key) or "").strip(),
            str(parsed.get(need_key) or "").strip(),
            *[str(parsed.get(f"{need_key}_BACKUP_{i}") or "").strip() for i in range(1, 6)],
        ):
            if not cand or cand in seen:
                continue
            seen.add(cand)
            provider_key_candidates.append(cand)
        if not provider_key_candidates:
            raise SystemExit(f"{need_key} is not set and was not found in --gov-env-path for provider '{provider}'.")

    cmd = [
        sys.executable,
        str(runner_path),
        "--request-dir",
        str(request_dir),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model),
        "--max-concurrency",
        str(int(concurrency)),
        "--timeout",
        str(float(timeout_s)),
        "--max-retries",
        "1",
    ]
    if bool(skip_existing):
        cmd.append("--skip-existing")
    if not need_key:
        subprocess.run(cmd, check=True, env=env)
        return

    last_err: Exception | None = None
    for i, key_val in enumerate(provider_key_candidates, start=1):
        env_try = dict(env)
        env_try[need_key] = key_val
        try:
            _eprint(f"Gateway request run with {need_key} candidate {i}/{len(provider_key_candidates)}.")
            subprocess.run(cmd, check=True, env=env_try)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            continue
    raise SystemExit(f"Gateway request run failed for all {need_key} candidates. last_error={last_err}")


def _read_result_jsonl(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(results_dir.glob("openai_results_shard*.jsonl")):
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            if isinstance(cid, str) and cid:
                out[cid] = obj
    return out


def _llm_experiment(
    *,
    panel_issues: list[PanelIssue],
    out_dir: Path,
    runner_path: Path,
    gateway_pythonpath: Path | None,
    model: str,
    timeout_s: float,
    concurrency: int,
    gov_env_path: Path,
    max_issue_chars: int,
    skip_existing: bool,
) -> dict[str, Path]:
    """
    Build panel-level and issue-level request sets, run via gateway, and compare outputs.
    """
    out: dict[str, Path] = {}
    llm_dir = out_dir / "llm_panel_vs_issue"
    req_panel = llm_dir / "requests_panel"
    req_issue = llm_dir / "requests_issue"
    res_panel = llm_dir / "results_panel"
    res_issue = llm_dir / "results_issue"
    llm_dir.mkdir(parents=True, exist_ok=True)

    # Group panel issues by city-state.
    by_city: dict[tuple[str, str], list[PanelIssue]] = {}
    for r in panel_issues:
        by_city.setdefault((r.city_name, r.state_abbr), []).append(r)
    if not by_city:
        raise SystemExit("No panel issues available for LLM experiment.")

    panel_reqs: list[tuple[str, str]] = []
    issue_reqs: list[tuple[str, str]] = []
    for (city, st), rows in sorted(by_city.items()):
        key = f"{_safe_slug(city)}__{_safe_slug(st)}"
        panel_reqs.append((f"panel::{key}", _panel_prompt(city, st, rows, max_issue_chars=max_issue_chars)))
        for r in rows:
            cid = f"issue::{key}::{r.issue_id}"
            issue_reqs.append((cid, _issue_prompt(r, max_issue_chars=max_issue_chars)))

    _write_openai_requests(req_panel / "openai_requests_shard000.jsonl", panel_reqs, model=model)
    _write_openai_requests(req_issue / "openai_requests_shard000.jsonl", issue_reqs, model=model)

    _eprint(f"LLM experiment: panel_requests={len(panel_reqs)} issue_requests={len(issue_reqs)} model={model}")

    _run_gateway_requests(
        request_dir=req_panel,
        output_dir=res_panel,
        runner_path=runner_path,
        model=model,
        gateway_pythonpath=gateway_pythonpath,
        timeout_s=timeout_s,
        concurrency=concurrency,
        gov_env_path=gov_env_path,
        skip_existing=skip_existing,
    )
    _run_gateway_requests(
        request_dir=req_issue,
        output_dir=res_issue,
        runner_path=runner_path,
        model=model,
        gateway_pythonpath=gateway_pythonpath,
        timeout_s=timeout_s,
        concurrency=concurrency,
        gov_env_path=gov_env_path,
        skip_existing=skip_existing,
    )

    panel_raw = _read_result_jsonl(res_panel)
    issue_raw = _read_result_jsonl(res_issue)

    parsed_panel_rows: list[dict[str, Any]] = []
    for cid, obj in sorted(panel_raw.items()):
        body = (((obj.get("response") or {}).get("body")) or {})
        txt = _extract_openai_output_text(body if isinstance(body, dict) else {})
        payload = _parse_json_from_text(txt)
        parsed_panel_rows.append(
            {
                "custom_id": cid,
                "status_code": int(((obj.get("response") or {}).get("status_code")) or 0),
                "output_text": txt,
                "parsed_json": payload,
            }
        )
    parsed_issue_rows: list[dict[str, Any]] = []
    for cid, obj in sorted(issue_raw.items()):
        body = (((obj.get("response") or {}).get("body")) or {})
        txt = _extract_openai_output_text(body if isinstance(body, dict) else {})
        payload = _parse_json_from_text(txt)
        parsed_issue_rows.append(
            {
                "custom_id": cid,
                "status_code": int(((obj.get("response") or {}).get("status_code")) or 0),
                "output_text": txt,
                "parsed_json": payload,
            }
        )

    panel_path = llm_dir / "parsed_panel_outputs.jsonl"
    with panel_path.open("w", encoding="utf-8") as f:
        for r in parsed_panel_rows:
            z = dict(r)
            f.write(json.dumps(z, ensure_ascii=False) + "\n")
    out["parsed_panel_outputs"] = panel_path

    issue_path = llm_dir / "parsed_issue_outputs.jsonl"
    with issue_path.open("w", encoding="utf-8") as f:
        for r in parsed_issue_rows:
            z = dict(r)
            f.write(json.dumps(z, ensure_ascii=False) + "\n")
    out["parsed_issue_outputs"] = issue_path

    def _extract_open_topics(obj: dict[str, Any]) -> dict[str, tuple[float, str]]:
        raw = obj.get("open_topics")
        if not isinstance(raw, list):
            raw = obj.get("topics")
        out_open: dict[str, tuple[float, str]] = {}
        if not isinstance(raw, list):
            return out_open
        for t in raw:
            if not isinstance(t, dict):
                continue
            nm = str(t.get("name") or "").strip()
            if not nm:
                continue
            key = nm.lower()
            try:
                sh = float(t.get("share_0_to_1") or 0.0)
            except Exception:
                sh = 0.0
            if not math.isfinite(sh):
                sh = 0.0
            ev = str(t.get("evidence") or "").strip()
            prev = out_open.get(key)
            if prev is None or sh > prev[0]:
                out_open[key] = (max(0.0, sh), ev)
        return out_open

    def _extract_open_map(obj: dict[str, Any]) -> list[tuple[str, str, float]]:
        raw = obj.get("open_to_fixed_map")
        if not isinstance(raw, list):
            return []
        out_map: list[tuple[str, str, float]] = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            open_topic = str(r.get("open_topic") or "").strip().lower()
            fixed = str(r.get("fixed_category") or "").strip()
            if not open_topic:
                continue
            if fixed not in _FIXED_CATEGORY_KEYS:
                fixed = ""
            try:
                c = float(r.get("confidence_0_to_1") or 0.0)
            except Exception:
                c = 0.0
            if not math.isfinite(c):
                c = 0.0
            c = min(1.0, max(0.0, c))
            out_map.append((open_topic, fixed, c))
        return out_map

    def _map_open_to_fixed(
        open_topics: dict[str, tuple[float, str]],
        open_map: list[tuple[str, str, float]],
    ) -> tuple[dict[str, float], list[dict[str, Any]], float]:
        best_map: dict[str, tuple[str, float]] = {}
        for open_topic, fixed, conf in open_map:
            prev = best_map.get(open_topic)
            if prev is None or conf > prev[1]:
                best_map[open_topic] = (fixed, conf)
        fixed_sum = {k: 0.0 for k in _FIXED_CATEGORY_KEYS}
        diverged_rows: list[dict[str, Any]] = []
        low_conf_mass = 0.0
        for open_topic, (share, _) in open_topics.items():
            fixed, conf = best_map.get(open_topic, ("", 0.0))
            if fixed and conf >= 0.5:
                fixed_sum[fixed] += float(share * conf)
            else:
                diverged_rows.append(
                    {
                        "open_topic": open_topic,
                        "open_topic_share": float(share),
                        "mapped_fixed_category": fixed,
                        "mapping_confidence": float(conf),
                        "divergence_reason": ("low_confidence" if fixed else "unmapped"),
                    }
                )
                low_conf_mass += float(share)
        return fixed_sum, diverged_rows, low_conf_mass

    # Compare panel vs issue outputs by city.
    issue_map: dict[str, list[dict[str, Any]]] = {}
    issue_id_map: dict[str, str] = {}
    for r in parsed_issue_rows:
        cid = str(r["custom_id"])
        # issue::<city_key>::<issue_id>
        parts = cid.split("::", 2)
        if len(parts) < 3:
            continue
        city_key = parts[1]
        issue_id = parts[2]
        issue_map.setdefault(city_key, []).append(r)
        issue_id_map[cid] = issue_id

    cmp_rows: list[dict[str, Any]] = []
    cat_cmp_rows: list[dict[str, Any]] = []
    open_panel_rows: list[dict[str, Any]] = []
    open_issue_rows: list[dict[str, Any]] = []
    open_crosswalk_rows: list[dict[str, Any]] = []
    open_divergence_rows: list[dict[str, Any]] = []
    open_rel_rows: list[dict[str, Any]] = []
    for prow in parsed_panel_rows:
        cid = str(prow["custom_id"])
        # panel::<city_key>
        city_key = cid.split("::", 1)[1] if "::" in cid else cid
        pobj = prow.get("parsed_json") if isinstance(prow.get("parsed_json"), dict) else {}
        irows = issue_map.get(city_key, [])
        iobjs = [x.get("parsed_json") for x in irows if isinstance(x.get("parsed_json"), dict)]

        panel_topics = {
            str(t.get("name")).strip().lower()
            for t in (pobj.get("topics") or [])
            if isinstance(t, dict) and str(t.get("name") or "").strip()
        }
        issue_topics_union: set[str] = set()
        issue_complexities: list[float] = []
        for io in iobjs:
            tops = io.get("topics") or []
            if isinstance(tops, list):
                for t in tops:
                    if isinstance(t, dict) and str(t.get("name") or "").strip():
                        issue_topics_union.add(str(t.get("name")).strip().lower())
            m = io.get("measures")
            if isinstance(m, dict):
                v = m.get("regulatory_complexity_0_100")
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    issue_complexities.append(float(v))

        inter = panel_topics & issue_topics_union
        union = panel_topics | issue_topics_union
        jaccard = float(len(inter) / len(union)) if union else math.nan

        # Category shares (fixed taxonomy) - comparable across panel vs issue.
        panel_shares = pobj.get("category_shares") if isinstance(pobj.get("category_shares"), dict) else {}
        panel_share_sum = float(sum(float(panel_shares.get(k) or 0.0) for k in _FIXED_CATEGORY_KEYS)) if panel_shares else math.nan
        issue_share_sums: list[float] = []
        panel_active = {k for k in _FIXED_CATEGORY_KEYS if float(panel_shares.get(k) or 0.0) >= 0.05} if panel_shares else set()
        issue_active_union: set[str] = set()
        issue_share_values: dict[str, list[float]] = {k: [] for k in _FIXED_CATEGORY_KEYS}
        for io in iobjs:
            cs = io.get("category_shares") if isinstance(io.get("category_shares"), dict) else {}
            if not cs:
                continue
            issue_share_sums.append(float(sum(float(cs.get(k) or 0.0) for k in _FIXED_CATEGORY_KEYS)))
            for k in _FIXED_CATEGORY_KEYS:
                v = float(cs.get(k) or 0.0)
                issue_share_values[k].append(v)
                if v >= 0.05:
                    issue_active_union.add(k)
        issue_share_sum_mean = float(sum(issue_share_sums) / len(issue_share_sums)) if issue_share_sums else math.nan
        cat_union = panel_active | issue_active_union
        cat_jaccard = float(len(panel_active & issue_active_union) / len(cat_union)) if cat_union else math.nan

        for k in _FIXED_CATEGORY_KEYS:
            cat_cmp_rows.append(
                {
                    "city_key": city_key,
                    "category": k,
                    "panel_share": float(panel_shares.get(k) or 0.0),
                    "issue_share_mean": (
                        float(sum(issue_share_values[k]) / len(issue_share_values[k])) if issue_share_values[k] else math.nan
                    ),
                    "issue_share_std": (
                        float(pd.Series(issue_share_values[k]).std(ddof=0)) if issue_share_values[k] else math.nan
                    ),
                }
            )

        # Open topic bridge rows.
        panel_open = _extract_open_topics(pobj)
        panel_map = _extract_open_map(pobj)
        panel_mapped_fixed, panel_diverged, panel_low_conf_mass = _map_open_to_fixed(panel_open, panel_map)

        for open_topic, (share, evidence) in panel_open.items():
            open_panel_rows.append(
                {
                    "city_key": city_key,
                    "open_topic": open_topic,
                    "panel_open_share": float(share),
                    "evidence": evidence,
                }
            )
        for open_topic, fixed, conf in panel_map:
            open_crosswalk_rows.append(
                {
                    "city_key": city_key,
                    "source_level": "panel",
                    "issue_id": "",
                    "open_topic": open_topic,
                    "fixed_category": fixed,
                    "mapping_confidence": float(conf),
                }
            )
        for d in panel_diverged:
            dd = dict(d)
            dd.update({"city_key": city_key, "source_level": "panel", "issue_id": ""})
            open_divergence_rows.append(dd)

        issue_mapped_fixed_rows: list[dict[str, float]] = []
        for ir in irows:
            issue_id = issue_id_map.get(str(ir.get("custom_id") or ""), "")
            io = ir.get("parsed_json")
            if not isinstance(io, dict):
                continue
            iopen = _extract_open_topics(io)
            imap = _extract_open_map(io)
            mapped, diverged, _ = _map_open_to_fixed(iopen, imap)
            issue_mapped_fixed_rows.append(mapped)
            for open_topic, (share, _) in iopen.items():
                open_issue_rows.append(
                    {
                        "city_key": city_key,
                        "issue_id": issue_id,
                        "open_topic": open_topic,
                        "issue_open_share": float(share),
                    }
                )
            for open_topic, fixed, conf in imap:
                open_crosswalk_rows.append(
                    {
                        "city_key": city_key,
                        "source_level": "issue",
                        "issue_id": issue_id,
                        "open_topic": open_topic,
                        "fixed_category": fixed,
                        "mapping_confidence": float(conf),
                    }
                )
            for d in diverged:
                dd = dict(d)
                dd.update({"city_key": city_key, "source_level": "issue", "issue_id": issue_id})
                open_divergence_rows.append(dd)

        issue_mapped_fixed_mean: dict[str, float] = {k: math.nan for k in _FIXED_CATEGORY_KEYS}
        if issue_mapped_fixed_rows:
            for k in _FIXED_CATEGORY_KEYS:
                vals = [float(x.get(k) or 0.0) for x in issue_mapped_fixed_rows]
                issue_mapped_fixed_mean[k] = float(sum(vals) / len(vals))

        panel_fixed_vs_open_abs = []
        issue_fixed_vs_open_abs = []
        for k in _FIXED_CATEGORY_KEYS:
            panel_fixed = float(panel_shares.get(k) or 0.0)
            panel_open_m = float(panel_mapped_fixed.get(k) or 0.0)
            panel_fixed_vs_open_abs.append(abs(panel_fixed - panel_open_m))
            issue_fixed_mean = float(sum(issue_share_values[k]) / len(issue_share_values[k])) if issue_share_values[k] else math.nan
            issue_open_mean = issue_mapped_fixed_mean.get(k, math.nan)
            if math.isfinite(issue_fixed_mean) and math.isfinite(issue_open_mean):
                issue_fixed_vs_open_abs.append(abs(float(issue_fixed_mean) - float(issue_open_mean)))

        open_rel_rows.append(
            {
                "city_key": city_key,
                "panel_fixed_vs_open_mean_abs_gap": float(sum(panel_fixed_vs_open_abs) / len(panel_fixed_vs_open_abs))
                if panel_fixed_vs_open_abs
                else math.nan,
                "issue_fixed_vs_open_mean_abs_gap": float(sum(issue_fixed_vs_open_abs) / len(issue_fixed_vs_open_abs))
                if issue_fixed_vs_open_abs
                else math.nan,
                "panel_low_conf_or_unmapped_open_mass": float(panel_low_conf_mass),
                "n_panel_open_topics": int(len(panel_open)),
            }
        )

        panel_complexity = None
        pm = pobj.get("measures")
        if isinstance(pm, dict):
            vv = pm.get("regulatory_complexity_0_100")
            if isinstance(vv, (int, float)) and math.isfinite(float(vv)):
                panel_complexity = float(vv)

        cmp_rows.append(
            {
                "city_key": city_key,
                "n_issue_outputs": len(irows),
                "panel_topics_n": len(panel_topics),
                "issue_topics_union_n": len(issue_topics_union),
                "topic_jaccard_panel_vs_issue_union": jaccard,
                "category_jaccard_panel_vs_issue_union": cat_jaccard,
                "panel_category_share_sum": panel_share_sum,
                "issue_category_share_sum_mean": issue_share_sum_mean,
                "panel_complexity_0_100": panel_complexity,
                "issue_complexity_mean_0_100": (float(sum(issue_complexities) / len(issue_complexities)) if issue_complexities else math.nan),
                "issue_complexity_std_0_100": (float(pd.Series(issue_complexities).std(ddof=0)) if issue_complexities else math.nan),
            }
        )

    cmp_df = pd.DataFrame.from_records(cmp_rows)
    if not cmp_df.empty and "city_key" in cmp_df.columns:
        cmp_df = cmp_df.sort_values("city_key")
    cmp_path = llm_dir / "panel_vs_issue_consistency.csv"
    _write_csv(cmp_df, cmp_path)
    out["panel_vs_issue_consistency"] = cmp_path

    cat_df = pd.DataFrame.from_records(cat_cmp_rows)
    if not cat_df.empty:
        cat_df = cat_df.sort_values(["city_key", "category"])
    cat_path = llm_dir / "panel_vs_issue_category_shares.csv"
    _write_csv(cat_df, cat_path)
    out["panel_vs_issue_category_shares"] = cat_path

    open_panel_df = pd.DataFrame.from_records(open_panel_rows)
    if not open_panel_df.empty:
        open_panel_df = open_panel_df.sort_values(["city_key", "open_topic"])
    open_panel_path = llm_dir / "open_topic_shares_panel.csv"
    _write_csv(open_panel_df, open_panel_path)
    out["open_topic_shares_panel"] = open_panel_path

    open_issue_df = pd.DataFrame.from_records(open_issue_rows)
    if not open_issue_df.empty:
        open_issue_df = open_issue_df.sort_values(["city_key", "issue_id", "open_topic"])
    open_issue_path = llm_dir / "open_topic_shares_issue.csv"
    _write_csv(open_issue_df, open_issue_path)
    out["open_topic_shares_issue"] = open_issue_path

    open_cross_df = pd.DataFrame.from_records(open_crosswalk_rows)
    if not open_cross_df.empty:
        open_cross_df = open_cross_df.sort_values(["city_key", "source_level", "issue_id", "open_topic"])
    open_cross_path = llm_dir / "open_to_fixed_crosswalk.csv"
    _write_csv(open_cross_df, open_cross_path)
    out["open_to_fixed_crosswalk"] = open_cross_path

    open_div_df = pd.DataFrame.from_records(open_divergence_rows)
    if not open_div_df.empty:
        open_div_df = open_div_df.sort_values(["city_key", "source_level", "issue_id", "open_topic_share"], ascending=[True, True, True, False])
    open_div_path = llm_dir / "open_fixed_divergence.csv"
    _write_csv(open_div_df, open_div_path)
    out["open_fixed_divergence"] = open_div_path

    open_rel_df = pd.DataFrame.from_records(open_rel_rows)
    if not open_rel_df.empty:
        open_rel_df = open_rel_df.sort_values(["city_key"])
    open_rel_path = llm_dir / "open_fixed_reliability_summary.csv"
    _write_csv(open_rel_df, open_rel_path)
    out["open_fixed_reliability_summary"] = open_rel_path

    return out


def _write_summary_md(
    *,
    out_dir: Path,
    issue_df: pd.DataFrame,
    selected_panels: pd.DataFrame,
    table_paths: dict[str, Path],
    llm_paths: dict[str, Path] | None,
) -> Path:
    md = out_dir / "prototype_summary.md"
    n_issues = int(issue_df["issue_id"].nunique())
    n_city_states = int(issue_df[["city_name", "state_abbr"]].drop_duplicates().shape[0])
    n_states = int(issue_df["state_abbr"].nunique())
    min_date = str(pd.to_datetime(issue_df["issue_date"], errors="coerce").min().date())
    max_date = str(pd.to_datetime(issue_df["issue_date"], errors="coerce").max().date())

    lines: list[str] = []
    lines.append("# Zoning Prototype (0->1) Summary")
    lines.append("")
    lines.append(f"Created: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Dataset scope")
    lines.append(f"- issues: {n_issues}")
    lines.append(f"- city-state units: {n_city_states}")
    lines.append(f"- states: {n_states}")
    lines.append(f"- issue_date span: {min_date} to {max_date}")
    lines.append("")
    lines.append("## Baseline outputs")
    for k, p in sorted(table_paths.items()):
        lines.append(f"- {k}: `{p}`")
    lines.append("")
    lines.append("## Selected text panels")
    if selected_panels.empty:
        lines.append("- none")
    else:
        for r in selected_panels.itertuples(index=False):
            lines.append(
                f"- {r.city_name}, {str(r.state_abbr).upper()}: n_issues={int(r.n_issues)} "
                f"n_with_text={int(r.n_with_text)} n_full={int(r.n_full)} span_years={float(r.span_years):.1f}"
            )
    if llm_paths:
        lines.append("")
        lines.append("## LLM panel-vs-issue outputs")
        for k, p in sorted(llm_paths.items()):
            lines.append(f"- {k}: `{p}`")
        lines.append("")
        lines.append("Interpretation note: urbanicity is a proxy and LLM-derived measures are exploratory.")

    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prototype zoning panel analysis + panel-level LLM experiment.")
    ap.add_argument(
        "--issue-panel",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/rhs_panel_mechanics_v4_gpt5nano_20260130_013333_citytime/rhs_issue_panel_20260129_233814.parquet",
        help="Issue-level mechanics panel (.parquet or .csv).",
    )
    ap.add_argument(
        "--merged-manifest",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/vast_snap_20260130_041623/corpus_union_20260123_100834/merged_manifest.jsonl",
        help="Merged issue manifest JSONL with pages_with_zoning.",
    )
    ap.add_argument(
        "--vlm-roots",
        default=(
            "/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/vlm_out_openai_gpt52_reasoning_medium_split_pages_with_ok,"
            "/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/vlm_out_manifest_gemini,"
            "/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/vlm_out_smoke,"
            "/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/vlm_out_test,"
            "/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/data/vlm_out_test_gemini2"
        ),
        help="Comma-separated roots containing <page_id>.vlm.json files.",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/prototype_zoning_panel_analysis",
        help="Output directory for tables and experiment artifacts.",
    )
    ap.add_argument("--panel-count", type=int, default=60, help="Number of city panels to build.")
    ap.add_argument("--panel-issues", type=int, default=10, help="Issues per selected city panel.")
    ap.add_argument("--min-panel-issues", type=int, default=8, help="Minimum local-text issue count to qualify for panel selection.")
    ap.add_argument(
        "--selection-mode",
        choices=["top", "stratified"],
        default="top",
        help="Panel selection strategy. 'top' is coverage-ranked. 'stratified' enforces region targets + urbanicity minima.",
    )
    ap.add_argument(
        "--region-targets",
        default="Midwest:21,South:14,Northeast:13,West:12",
        help="Region target counts used when --selection-mode stratified.",
    )
    ap.add_argument(
        "--urbanicity-minimums",
        default="urban:10,suburban:20,rural:25",
        help="Urbanicity minimum counts used when --selection-mode stratified.",
    )
    ap.add_argument(
        "--write-selected-issues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write panels/selected_panel_issues.csv manifest.",
    )
    ap.add_argument(
        "--max-issue-chars",
        type=int,
        default=12000,
        help="Hard cap for issue transcript text included in LLM prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )

    ap.add_argument("--run-llm", action="store_true", help="Run panel-vs-issue LLM comparison via agent-gateway.")
    ap.add_argument("--llm-model", default="gemini:gemini-2.5-flash", help="Gateway model string for LLM experiment.")
    ap.add_argument("--llm-timeout", type=float, default=180.0, help="Per-request gateway timeout seconds.")
    ap.add_argument("--llm-concurrency", type=int, default=3, help="Gateway request concurrency.")
    ap.add_argument(
        "--llm-skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When true, skip custom_ids already present in gateway result shards (supports resume).",
    )
    ap.add_argument(
        "--gateway-runner",
        default=str(Path(__file__).resolve().parents[1] / "scripts" / "run_openai_requests_via_gateway.py"),
        help="Path to run_openai_requests_via_gateway.py",
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default=str(Path(__file__).resolve().parents[1] / "agent-gateway" / "src"),
        help="Path to add to PYTHONPATH for importing gateway package.",
    )
    ap.add_argument(
        "--gov-env-path",
        default=".env",
        help="Env file path used to source OPENAI_KEY/GEMINI_KEY/CLAUDE_KEY (and backups).",
    )
    ap.add_argument("--refresh-census-cache", action="store_true", help="Refresh cached Census place population map.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    issue_panel_path = Path(args.issue_panel).expanduser().resolve()
    merged_manifest_path = Path(args.merged_manifest).expanduser().resolve()
    vlm_roots = _parse_csv_paths(args.vlm_roots)
    runner_path = Path(args.gateway_runner).expanduser().resolve()
    gateway_pythonpath = Path(args.gateway_pythonpath).expanduser().resolve() if str(args.gateway_pythonpath).strip() else None
    gov_env_path = Path(args.gov_env_path).expanduser().resolve()

    _eprint(f"Loading issue panel: {issue_panel_path}")
    issue_df = _load_table(issue_panel_path)

    required = {"issue_id", "issue_date", "classification_label", "city_name", "state_abbr", "pub_slug"}
    miss = sorted(c for c in required if c not in issue_df.columns)
    if miss:
        raise SystemExit(f"Issue panel missing required columns: {miss}")

    _eprint("Fetching/reading Census place population map (2020 Decennial).")
    place_pop = _fetch_census_place_population(
        cache_path=out_dir / "cache" / "census_place_pop_2020.json",
        refresh=bool(args.refresh_census_cache),
    )
    _eprint(f"Loaded place population entries: {len(place_pop)}")

    issue_df = _attach_geo_proxies(issue_df, place_pop=place_pop)
    issue_df = _compute_issue_complexity(issue_df)

    _eprint("Writing baseline summary tables.")
    table_paths = _write_baseline_tables(issue_df, out_dir=out_dir)

    _eprint(f"Loading issue manifest: {merged_manifest_path}")
    issue_manifest = _load_issue_manifest(merged_manifest_path)

    _eprint("Building same-city panels with reconstructed zoning transcript text.")
    region_targets = _parse_named_int_targets(str(args.region_targets))
    if not region_targets:
        region_targets = dict(_DEFAULT_REGION_TARGETS)
    urbanicity_minimums = _parse_named_int_targets(str(args.urbanicity_minimums), normalize_lower=True)
    if not urbanicity_minimums:
        urbanicity_minimums = dict(_DEFAULT_URBANICITY_MINIMUMS)

    selected_panels, panel_rows, selected_panel_issues, panel_diag_paths = _build_panels(
        issue_df=issue_df,
        issue_manifest=issue_manifest,
        vlm_roots=vlm_roots,
        out_dir=out_dir,
        panel_count=int(args.panel_count),
        panel_issues=int(args.panel_issues),
        min_panel_issues=int(args.min_panel_issues),
        selection_mode=str(args.selection_mode),
        region_targets=region_targets,
        urbanicity_minimums=urbanicity_minimums,
        write_selected_issues=bool(args.write_selected_issues),
    )
    _eprint(f"Selected panels: {len(selected_panels)}; reconstructed issues: {len(panel_rows)}")
    table_paths.update(panel_diag_paths)

    _eprint("Writing numeric constraint / distinct zone summary tables.")
    numeric_paths = _write_numeric_constraint_tables(panel_rows=panel_rows, out_dir=out_dir, llm_paths=None)
    table_paths.update(numeric_paths)

    llm_paths: dict[str, Path] | None = None
    if args.run_llm:
        _eprint("Running panel-vs-issue LLM experiment via agent-gateway.")
        llm_paths = _llm_experiment(
            panel_issues=panel_rows,
            out_dir=out_dir,
            runner_path=runner_path,
            gateway_pythonpath=gateway_pythonpath,
            model=str(args.llm_model),
            timeout_s=float(args.llm_timeout),
            concurrency=int(args.llm_concurrency),
            gov_env_path=gov_env_path,
            max_issue_chars=int(args.max_issue_chars),
            skip_existing=bool(args.llm_skip_existing),
        )
    else:
        _eprint("Skipping LLM experiment (pass --run-llm to enable).")

    _eprint("Writing contamination diagnostics.")
    contam_paths = _write_contamination_tables(panel_rows=panel_rows, out_dir=out_dir, llm_paths=llm_paths)
    table_paths.update(contam_paths)
    # Recompute numeric tables with LLM numeric constraints when available.
    numeric_paths = _write_numeric_constraint_tables(panel_rows=panel_rows, out_dir=out_dir, llm_paths=llm_paths)
    table_paths.update(numeric_paths)

    summary_path = _write_summary_md(
        out_dir=out_dir,
        issue_df=issue_df,
        selected_panels=selected_panels,
        table_paths=table_paths,
        llm_paths=llm_paths,
    )

    # Snapshot provenance.
    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "issue_panel": str(issue_panel_path),
        "merged_manifest": str(merged_manifest_path),
        "vlm_roots": [str(p) for p in vlm_roots],
        "panel_count": int(args.panel_count),
        "panel_issues": int(args.panel_issues),
        "min_panel_issues": int(args.min_panel_issues),
        "selection_mode": str(args.selection_mode),
        "region_targets": region_targets,
        "urbanicity_minimums": urbanicity_minimums,
        "selected_panel_issues_rows": int(len(selected_panel_issues)),
        "run_llm": bool(args.run_llm),
        "llm_model": str(args.llm_model),
        "summary_md": str(summary_path),
        "table_paths": {k: str(v) for k, v in table_paths.items()},
        "llm_paths": None if llm_paths is None else {k: str(v) for k, v in llm_paths.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Done. wrote={out_dir}")


if __name__ == "__main__":
    main()
