#!/usr/bin/env python3
"""
Build place/county covariate panel aligned to selected city panels.

Outputs:
  - covariates/city_place_crosswalk.csv
  - covariates/city_place_match_candidates.csv
  - covariates/city_place_match_summary.csv
  - covariates/city_place_unmatched.csv
  - covariates/manual_place_overrides_template.csv
  - covariates/city_county_crosswalk.csv
  - covariates/city_year_external_covariates.csv
  - covariates/covariate_missingness_report.csv
  - covariates/provenance.json

Notes:
  - Place-level covariates: ACS 5-year API.
  - County-level covariates: ACS profile API.
  - Building permits: Census BPS annual place files (where available).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import re
import time
import zipfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests


_STATE_ABBR_TO_FIPS: dict[str, str] = {
    "al": "01",
    "ak": "02",
    "az": "04",
    "ar": "05",
    "ca": "06",
    "co": "08",
    "ct": "09",
    "de": "10",
    "dc": "11",
    "fl": "12",
    "ga": "13",
    "hi": "15",
    "id": "16",
    "il": "17",
    "in": "18",
    "ia": "19",
    "ks": "20",
    "ky": "21",
    "la": "22",
    "me": "23",
    "md": "24",
    "ma": "25",
    "mi": "26",
    "mn": "27",
    "ms": "28",
    "mo": "29",
    "mt": "30",
    "ne": "31",
    "nv": "32",
    "nh": "33",
    "nj": "34",
    "nm": "35",
    "ny": "36",
    "nc": "37",
    "nd": "38",
    "oh": "39",
    "ok": "40",
    "or": "41",
    "pa": "42",
    "ri": "44",
    "sc": "45",
    "sd": "46",
    "tn": "47",
    "tx": "48",
    "ut": "49",
    "vt": "50",
    "va": "51",
    "wa": "53",
    "wv": "54",
    "wi": "55",
    "wy": "56",
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

_MATCH_TOKEN_EQUIV = (
    ("saint", "st"),
    ("st", "saint"),
    ("mount", "mt"),
    ("mt", "mount"),
    ("fort", "ft"),
    ("ft", "fort"),
)

_ACS_PLACE_VARS = [
    "B01003_001E",  # population
    "B25001_001E",  # housing units
    "B19013_001E",  # median household income
    "B25077_001E",  # median home value
    "B25002_001E",  # occupancy status total
    "B25002_003E",  # vacancy count
]

_ACS_COUNTY_PROFILE_VARS = [
    "DP03_0009PE",  # unemployment rate
    "DP03_0088E",  # per-capita income
    "DP03_0033PE",  # agriculture share
    "DP03_0034PE",  # construction share
    "DP03_0035PE",  # manufacturing share
    "DP03_0037PE",  # retail share
    "DP03_0041PE",  # professional services share
]

_BPS_REGION_SPEC = {
    "Midwest": ("Midwest%20Region", "mw"),
    "Northeast": ("Northeast%20Region", "ne"),
    "South": ("South%20Region", "so"),
    "West": ("West%20Region", "we"),
}


def _to_float(v: Any) -> float:
    try:
        x = float(str(v).strip())
    except Exception:
        return math.nan
    if not math.isfinite(x):
        return math.nan
    return x


def _to_int(v: Any) -> int | None:
    x = _to_float(v)
    if not math.isfinite(x):
        return None
    try:
        return int(round(x))
    except Exception:
        return None


def _normalize_place_text(name: str) -> str:
    s = str(name or "").strip().lower()
    if not s:
        return ""
    if "," in s:
        s = s.split(",", 1)[0].strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_place_suffix_once(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    for suf in sorted(_PLACE_SUFFIXES, key=len, reverse=True):
        if s.endswith(" " + suf):
            return s[: -len(suf) - 1].strip()
    return s


def _strip_place_suffix_all(name: str) -> str:
    prev = ""
    cur = str(name or "").strip()
    while cur and cur != prev:
        prev = cur
        cur = _strip_place_suffix_once(cur)
    return cur


def _name_alias_variants(name: str) -> list[str]:
    s = str(name or "").strip()
    if not s:
        return []
    out: list[str] = []
    for src, dst in _MATCH_TOKEN_EQUIV:
        pat = re.compile(rf"\b{re.escape(src)}\b")
        if pat.search(s):
            alt = pat.sub(dst, s)
            if alt and alt != s:
                out.append(alt)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _build_name_variants(name: str) -> dict[str, Any]:
    raw = _normalize_place_text(name)
    strip1 = _strip_place_suffix_once(raw)
    stripall = _strip_place_suffix_all(raw)

    keys = [raw, strip1, stripall]
    keys.extend(_name_alias_variants(raw))
    keys.extend(_name_alias_variants(strip1))
    keys.extend(_name_alias_variants(stripall))

    seen: set[str] = set()
    ordered: list[str] = []
    for k in keys:
        kk = str(k or "").strip()
        if not kk or kk in seen:
            continue
        seen.add(kk)
        ordered.append(kk)

    token_sets = [set(x.split()) for x in ordered if x]
    return {
        "raw": raw,
        "strip1": strip1,
        "stripall": stripall,
        "keys": ordered,
        "token_sets": token_sets,
    }


def _best_name_similarity(ref_keys: list[str], cand_keys: list[str]) -> float:
    best = 0.0
    for a in ref_keys:
        for b in cand_keys:
            if not a or not b:
                continue
            best = max(best, float(SequenceMatcher(None, a, b).ratio()))
    return float(best)


def _best_token_overlap(ref_token_sets: list[set[str]], cand_token_sets: list[set[str]]) -> float:
    best = 0.0
    for a in ref_token_sets:
        if not a:
            continue
        for b in cand_token_sets:
            if not b:
                continue
            inter = len(a & b)
            union = len(a | b)
            if union <= 0:
                continue
            best = max(best, float(inter / union))
    return float(best)


def _canonical_place_name(name: str) -> str:
    # Backward-compatible canonical form used in a few places.
    return _strip_place_suffix_once(_normalize_place_text(name))


def _request_json_with_retry(url: str, *, timeout: float = 90.0, max_retries: int = 3) -> Any:
    last_err: Exception | None = None
    for i in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if i + 1 < max_retries:
                time.sleep(1.25 * (i + 1))
                continue
    raise RuntimeError(f"Request failed after retries: {url} err={last_err}")


def _request_text_with_retry(url: str, *, timeout: float = 90.0, max_retries: int = 3) -> str:
    last_err: Exception | None = None
    for i in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if i + 1 < max_retries:
                time.sleep(1.25 * (i + 1))
                continue
    raise RuntimeError(f"Request failed after retries: {url} err={last_err}")


@dataclass(frozen=True)
class CityRef:
    city_key: str
    city_name: str
    state_abbr: str
    region: str
    urbanicity_proxy: str
    min_issue_year: int
    max_issue_year: int


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build external covariate panel for selected city panels.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30",
        help="Prototype run directory with panels/selected_panel_issues.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory for covariate artifacts (default: <run-dir>/covariates).",
    )
    ap.add_argument(
        "--acs-start-year",
        type=int,
        default=2009,
        help="Earliest ACS year to request.",
    )
    ap.add_argument(
        "--acs-end-year",
        type=int,
        default=0,
        help="Latest ACS year to request (0 => current_year - 1).",
    )
    ap.add_argument(
        "--permits-start-year",
        type=int,
        default=1980,
        help="Earliest year for BPS annual place permit files.",
    )
    ap.add_argument(
        "--year-min",
        type=int,
        default=0,
        help="Optional global minimum year override (0 => infer from selected issues).",
    )
    ap.add_argument(
        "--year-max",
        type=int,
        default=0,
        help="Optional global maximum year override (0 => infer from selected issues).",
    )
    ap.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached API/file pulls and re-fetch.",
    )
    ap.add_argument(
        "--manual-place-overrides",
        default="",
        help="Optional CSV with manual place IDs: city_key + place_geoid (or state_fips + place_fips).",
    )
    ap.add_argument(
        "--fuzzy-match-threshold",
        type=float,
        default=0.92,
        help="Minimum combined score for fuzzy place matching.",
    )
    ap.add_argument(
        "--candidate-top-n",
        type=int,
        default=8,
        help="How many place candidates per city to persist for diagnostics.",
    )
    return ap.parse_args()


def _load_city_refs(run_dir: Path) -> list[CityRef]:
    p = run_dir / "panels" / "selected_panel_issues.csv"
    if p.is_file():
        df = pd.read_csv(p)
    else:
        rows: list[dict[str, Any]] = []
        panels_dir = run_dir / "panels"
        if not panels_dir.is_dir():
            raise SystemExit(f"Missing required directory: {panels_dir}")
        for city_dir in sorted(panels_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            issue_path = city_dir / "issue_texts.jsonl"
            if not issue_path.is_file():
                continue
            for raw in issue_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rows.append(
                    {
                        "city_key": str(obj.get("city_key") or city_dir.name),
                        "city_name": str(obj.get("city_name") or ""),
                        "state_abbr": str(obj.get("state_abbr") or "").lower(),
                        "region": str(obj.get("region") or ""),
                        "urbanicity_proxy": str(obj.get("urbanicity_proxy") or ""),
                        "issue_date": str(obj.get("issue_date") or ""),
                    }
                )
        if not rows:
            raise SystemExit(f"Missing required file: {p} and could not reconstruct from panels/issue_texts.jsonl")
        df = pd.DataFrame.from_records(rows)
    need = {"city_key", "city_name", "state_abbr", "region", "issue_date"}
    miss = sorted(c for c in need if c not in df.columns)
    if miss:
        raise SystemExit(f"{p} missing columns: {miss}")
    if "urbanicity_proxy" not in df.columns:
        df["urbanicity_proxy"] = "unknown"

    df["state_abbr"] = df["state_abbr"].astype(str).str.lower()
    df["issue_year"] = pd.to_datetime(df["issue_date"], errors="coerce").dt.year
    df = df.dropna(subset=["issue_year"]).copy()
    g = (
        df.groupby(["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy"], dropna=False)
        .agg(min_issue_year=("issue_year", "min"), max_issue_year=("issue_year", "max"))
        .reset_index()
        .sort_values(["state_abbr", "city_name"])
    )
    refs: list[CityRef] = []
    for r in g.itertuples(index=False):
        refs.append(
            CityRef(
                city_key=str(r.city_key),
                city_name=str(r.city_name),
                state_abbr=str(r.state_abbr),
                region=str(r.region),
                urbanicity_proxy=str(r.urbanicity_proxy),
                min_issue_year=int(r.min_issue_year),
                max_issue_year=int(r.max_issue_year),
            )
        )
    if not refs:
        raise SystemExit("No city refs discovered from selected_panel_issues.csv")
    return refs


def _prepare_gazetteer_places(gaz_raw: pd.DataFrame) -> pd.DataFrame:
    gaz = gaz_raw.copy()
    gaz.columns = [str(c).strip() for c in gaz.columns]

    has_raw = {"USPS", "GEOID", "NAME", "POP10", "INTPTLAT", "INTPTLONG"}.issubset(set(gaz.columns))
    if has_raw:
        gaz = gaz[["USPS", "GEOID", "NAME", "POP10", "INTPTLAT", "INTPTLONG"]].copy()
        gaz["state_abbr"] = gaz["USPS"].astype(str).str.lower()
        gaz["state_fips"] = gaz["GEOID"].astype(str).str.zfill(7).str.slice(0, 2)
        gaz["place_fips"] = gaz["GEOID"].astype(str).str.zfill(7).str.slice(2, 7)
        gaz["place_geoid"] = gaz["GEOID"].astype(str).str.zfill(7)
        gaz["place_name_gazetteer"] = gaz["NAME"].astype(str)
        gaz["pop10"] = pd.to_numeric(gaz["POP10"], errors="coerce")
        gaz["intptlat"] = pd.to_numeric(gaz["INTPTLAT"], errors="coerce")
        gaz["intptlong"] = pd.to_numeric(gaz["INTPTLONG"], errors="coerce")
    else:
        need = {"state_abbr", "state_fips", "place_fips", "place_geoid", "place_name_gazetteer"}
        miss = sorted(c for c in need if c not in gaz.columns)
        if miss:
            raise SystemExit(f"Gazetteer cache missing required columns: {miss}")
        if "pop10" not in gaz.columns:
            gaz["pop10"] = math.nan
        if "intptlat" not in gaz.columns:
            gaz["intptlat"] = math.nan
        if "intptlong" not in gaz.columns:
            gaz["intptlong"] = math.nan

    gaz["state_abbr"] = gaz["state_abbr"].astype(str).str.lower()
    gaz["state_fips"] = gaz["state_fips"].astype(str).str.zfill(2)
    gaz["place_fips"] = gaz["place_fips"].astype(str).str.zfill(5)
    gaz["place_geoid"] = gaz["place_geoid"].astype(str).str.zfill(7)
    gaz["place_name_gazetteer"] = gaz["place_name_gazetteer"].astype(str)
    gaz["pop10"] = pd.to_numeric(gaz["pop10"], errors="coerce")
    gaz["intptlat"] = pd.to_numeric(gaz["intptlat"], errors="coerce")
    gaz["intptlong"] = pd.to_numeric(gaz["intptlong"], errors="coerce")

    var = gaz["place_name_gazetteer"].astype(str).map(_build_name_variants)
    gaz["name_norm_raw"] = var.map(lambda d: str(d.get("raw") or ""))
    gaz["name_norm_strip1"] = var.map(lambda d: str(d.get("strip1") or ""))
    gaz["name_norm_stripall"] = var.map(lambda d: str(d.get("stripall") or ""))
    gaz["canonical_name"] = gaz["name_norm_strip1"]
    gaz["name_token_count"] = gaz["name_norm_strip1"].astype(str).str.split().str.len().fillna(0).astype(int)

    keep_cols = [
        "state_abbr",
        "state_fips",
        "place_fips",
        "place_geoid",
        "place_name_gazetteer",
        "name_norm_raw",
        "name_norm_strip1",
        "name_norm_stripall",
        "canonical_name",
        "name_token_count",
        "pop10",
        "intptlat",
        "intptlong",
    ]
    return gaz[keep_cols].drop_duplicates(subset=["place_geoid"], keep="first").sort_values(["state_abbr", "place_geoid"]).reset_index(drop=True)


def _load_gazetteer_places(cache_dir: Path, *, refresh: bool) -> pd.DataFrame:
    cache_json = cache_dir / "gazetteer_places_national.json"
    if cache_json.is_file() and not refresh:
        arr = json.loads(cache_json.read_text(encoding="utf-8"))
        gaz = pd.DataFrame.from_records(arr)
        return _prepare_gazetteer_places(gaz)

    url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/Gaz_places_national.zip"
    raw = requests.get(url, timeout=90)
    raw.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(raw.content))
    names = zf.namelist()
    if not names:
        raise SystemExit("Unexpected empty gazetteer zip")
    with zf.open(names[0]) as f:
        gaz_raw = pd.read_csv(f, sep="\t", dtype=str, encoding="latin1")
    gaz = _prepare_gazetteer_places(gaz_raw)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_json.write_text(gaz.to_json(orient="records"), encoding="utf-8")
    return gaz


def _load_manual_place_overrides(path: Path, gaz: pd.DataFrame) -> dict[str, str]:
    if not str(path or "").strip():
        return {}
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"manual place overrides file not found: {p}")
    df = pd.read_csv(p, dtype=str).fillna("")
    if "city_key" not in df.columns:
        raise SystemExit(f"manual place overrides must include city_key column: {p}")
    known_geoids = set(gaz["place_geoid"].astype(str))
    out: dict[str, str] = {}
    bad_rows: list[str] = []
    for r in df.itertuples(index=False):
        city_key = str(getattr(r, "city_key", "") or "").strip()
        if not city_key:
            continue
        geoid = str(getattr(r, "place_geoid", "") or getattr(r, "census_place_geoid", "") or "").strip()
        if not geoid:
            st = str(getattr(r, "state_fips", "") or getattr(r, "census_state_fips", "") or "").strip()
            pl = str(getattr(r, "place_fips", "") or getattr(r, "census_place_fips", "") or "").strip()
            if st and pl:
                geoid = st.zfill(2) + pl.zfill(5)
        geoid = re.sub(r"[^0-9]", "", geoid)
        if len(geoid) == 5:
            st = str(getattr(r, "state_fips", "") or getattr(r, "census_state_fips", "") or "").strip()
            if st:
                geoid = st.zfill(2) + geoid
        if len(geoid) != 7:
            bad_rows.append(f"{city_key}: invalid geoid")
            continue
        if geoid not in known_geoids:
            bad_rows.append(f"{city_key}: geoid {geoid} not in gazetteer")
            continue
        out[city_key] = geoid
    if bad_rows:
        sample = "; ".join(bad_rows[:8])
        extra = f" (+{len(bad_rows)-8} more)" if len(bad_rows) > 8 else ""
        print(f"Warning: skipped invalid manual overrides: {sample}{extra}", flush=True)
    return out


def _resolve_exact_method(ref_var: dict[str, Any], cand_var: dict[str, Any]) -> tuple[str, float]:
    ref_raw = str(ref_var.get("raw") or "")
    ref_strip1 = str(ref_var.get("strip1") or "")
    ref_stripall = str(ref_var.get("stripall") or "")
    cand_raw = str(cand_var.get("raw") or "")
    cand_strip1 = str(cand_var.get("strip1") or "")
    cand_stripall = str(cand_var.get("stripall") or "")

    if ref_raw and cand_raw and ref_raw == cand_raw:
        return "exact_raw", 1.0
    if ref_raw and ref_raw in {cand_strip1, cand_stripall}:
        return "exact_gazetteer_suffix_strip", 0.998
    if ref_strip1 and ref_strip1 in {cand_strip1, cand_stripall}:
        return "exact_both_suffix_strip", 0.996
    if ref_stripall and ref_stripall in {cand_stripall, cand_strip1}:
        return "exact_full_suffix_strip", 0.995

    ref_alias = set(ref_var.get("keys") or [])
    cand_alias = set(cand_var.get("keys") or [])
    if ref_alias and cand_alias and (ref_alias & cand_alias):
        return "exact_alias", 0.994
    return "", math.nan


def _match_city_to_place(
    refs: list[CityRef],
    gaz: pd.DataFrame,
    *,
    manual_overrides: dict[str, str] | None = None,
    fuzzy_threshold: float = 0.92,
    candidate_top_n: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manual = dict(manual_overrides or {})
    geoid_to_row = {str(r.place_geoid): r for r in gaz.itertuples(index=False)}

    rows: list[dict[str, Any]] = []
    cand_rows: list[dict[str, Any]] = []
    for r in refs:
        st = r.state_abbr
        ref_var = _build_name_variants(r.city_name)
        g = gaz[gaz["state_abbr"] == st].copy()

        scored: list[dict[str, Any]] = []
        for gr in g.itertuples(index=False):
            cand_var = {
                "raw": str(getattr(gr, "name_norm_raw", "") or ""),
                "strip1": str(getattr(gr, "name_norm_strip1", "") or ""),
                "stripall": str(getattr(gr, "name_norm_stripall", "") or ""),
                "keys": [
                    str(getattr(gr, "name_norm_raw", "") or ""),
                    str(getattr(gr, "name_norm_strip1", "") or ""),
                    str(getattr(gr, "name_norm_stripall", "") or ""),
                ]
                + _name_alias_variants(str(getattr(gr, "name_norm_raw", "") or ""))
                + _name_alias_variants(str(getattr(gr, "name_norm_strip1", "") or ""))
                + _name_alias_variants(str(getattr(gr, "name_norm_stripall", "") or "")),
                "token_sets": [],
            }
            # Deduplicate candidate keys and build token sets.
            seen: set[str] = set()
            ck: list[str] = []
            for k in cand_var["keys"]:
                kk = str(k or "").strip()
                if not kk or kk in seen:
                    continue
                seen.add(kk)
                ck.append(kk)
            cand_var["keys"] = ck
            cand_var["token_sets"] = [set(x.split()) for x in ck if x]

            exact_method, exact_score = _resolve_exact_method(ref_var, cand_var)
            similarity = _best_name_similarity(ref_var["keys"], cand_var["keys"])
            token_overlap = _best_token_overlap(ref_var["token_sets"], cand_var["token_sets"])
            combined = float(exact_score) if exact_method else float(max(similarity, 0.75 * similarity + 0.25 * token_overlap))
            fuzzy_ok = bool((not exact_method) and combined >= float(fuzzy_threshold) and token_overlap >= 0.34)

            scored.append(
                {
                    "place_geoid": str(gr.place_geoid),
                    "state_fips": str(gr.state_fips),
                    "place_fips": str(gr.place_fips),
                    "place_name_gazetteer": str(gr.place_name_gazetteer),
                    "place_pop10": _to_int(gr.pop10),
                    "intptlat": _to_float(gr.intptlat),
                    "intptlong": _to_float(gr.intptlong),
                    "name_norm_raw": str(getattr(gr, "name_norm_raw", "") or ""),
                    "name_norm_strip1": str(getattr(gr, "name_norm_strip1", "") or ""),
                    "name_norm_stripall": str(getattr(gr, "name_norm_stripall", "") or ""),
                    "exact_method": str(exact_method or ""),
                    "similarity": float(similarity),
                    "token_overlap": float(token_overlap),
                    "score": float(combined),
                    "fuzzy_ok": int(fuzzy_ok),
                }
            )

        scored = sorted(
            scored,
            key=lambda x: (
                1 if str(x["exact_method"]) else 0,
                float(x["score"]),
                float(x["token_overlap"]),
                float(x["place_pop10"] or -1),
                str(x["place_geoid"]),
            ),
            reverse=True,
        )

        selected: dict[str, Any] | None = None
        selected_method = "unmatched"
        selected_conf = "unmatched"
        selected_score = math.nan
        selected_similarity = math.nan
        selected_token_overlap = math.nan
        selected_detail = ""
        candidate_count = 0
        manual_used = 0

        manual_geoid = manual.get(str(r.city_key))
        if manual_geoid and manual_geoid in geoid_to_row:
            gg = geoid_to_row[manual_geoid]
            selected = {
                "place_geoid": str(gg.place_geoid),
                "state_fips": str(gg.state_fips),
                "place_fips": str(gg.place_fips),
                "place_name_gazetteer": str(gg.place_name_gazetteer),
                "place_pop10": _to_int(gg.pop10),
                "intptlat": _to_float(gg.intptlat),
                "intptlong": _to_float(gg.intptlong),
                "name_norm_raw": str(getattr(gg, "name_norm_raw", "") or ""),
                "name_norm_strip1": str(getattr(gg, "name_norm_strip1", "") or ""),
            }
            selected_method = "manual_override"
            selected_conf = "high"
            selected_score = 1.0
            selected_similarity = 1.0
            selected_token_overlap = 1.0
            selected_detail = "manual_override"
            candidate_count = 1
            manual_used = 1
        elif scored:
            top = scored[0]
            if str(top["exact_method"]):
                selected = top
                selected_method = "exact"
                selected_conf = "high"
                selected_score = float(top["score"])
                selected_similarity = float(top["similarity"])
                selected_token_overlap = float(top["token_overlap"])
                selected_detail = str(top["exact_method"])
                candidate_count = int(sum(1 for x in scored if str(x["exact_method"]) == str(top["exact_method"])))
            elif int(top.get("fuzzy_ok") or 0) == 1:
                selected = top
                selected_method = "fuzzy"
                selected_score = float(top["score"])
                selected_similarity = float(top["similarity"])
                selected_token_overlap = float(top["token_overlap"])
                selected_detail = "fuzzy_combined"
                selected_conf = "medium" if selected_score >= 0.96 and selected_token_overlap >= 0.5 else "low"
                candidate_count = int(sum(1 for x in scored if int(x.get("fuzzy_ok") or 0) == 1))

        selected_geoid = str(selected.get("place_geoid") or "") if isinstance(selected, dict) else ""
        top_n = max(1, int(candidate_top_n))
        for rank, x in enumerate(scored[:top_n], start=1):
            cand_rows.append(
                {
                    "city_key": str(r.city_key),
                    "city_name": str(r.city_name),
                    "state_abbr": st,
                    "region": str(r.region),
                    "rank": int(rank),
                    "selected_place_geoid": selected_geoid,
                    "is_selected": int(bool(selected_geoid and str(x["place_geoid"]) == selected_geoid)),
                    "is_manual_override": int(manual_used),
                    "place_geoid": str(x["place_geoid"]),
                    "state_fips": str(x["state_fips"]),
                    "place_fips": str(x["place_fips"]),
                    "place_name_gazetteer": str(x["place_name_gazetteer"]),
                    "place_pop10": x["place_pop10"],
                    "score": float(x["score"]),
                    "similarity": float(x["similarity"]),
                    "token_overlap": float(x["token_overlap"]),
                    "exact_method": str(x["exact_method"]),
                    "fuzzy_ok": int(x["fuzzy_ok"]),
                }
            )

        if selected is not None:
            rows.append(
                {
                    "city_key": r.city_key,
                    "city_name": r.city_name,
                    "state_abbr": st,
                    "region": r.region,
                    "urbanicity_proxy": r.urbanicity_proxy,
                    "min_issue_year": r.min_issue_year,
                    "max_issue_year": r.max_issue_year,
                    "place_geoid": str(selected.get("place_geoid") or ""),
                    "state_fips": str(selected.get("state_fips") or ""),
                    "place_fips": str(selected.get("place_fips") or ""),
                    "census_place_geoid": str(selected.get("place_geoid") or ""),
                    "census_state_fips": str(selected.get("state_fips") or ""),
                    "census_place_fips": str(selected.get("place_fips") or ""),
                    "place_name_gazetteer": str(selected.get("place_name_gazetteer") or ""),
                    "place_pop10": _to_int(selected.get("place_pop10")),
                    "intptlat": _to_float(selected.get("intptlat")),
                    "intptlong": _to_float(selected.get("intptlong")),
                    "ref_name_norm_raw": str(ref_var.get("raw") or ""),
                    "ref_name_norm_strip1": str(ref_var.get("strip1") or ""),
                    "match_name_norm_raw": str(selected.get("name_norm_raw") or ""),
                    "match_name_norm_strip1": str(selected.get("name_norm_strip1") or ""),
                    "match_method": str(selected_method),
                    "match_method_detail": str(selected_detail),
                    "match_confidence": str(selected_conf),
                    "match_score": float(selected_score),
                    "name_similarity": float(selected_similarity),
                    "token_overlap": float(selected_token_overlap),
                    "candidate_count": int(candidate_count),
                    "candidate_count_state": int(len(g)),
                }
            )
            continue

        rows.append(
            {
                "city_key": r.city_key,
                "city_name": r.city_name,
                "state_abbr": st,
                "region": r.region,
                "urbanicity_proxy": r.urbanicity_proxy,
                "min_issue_year": r.min_issue_year,
                "max_issue_year": r.max_issue_year,
                "place_geoid": "",
                "state_fips": _STATE_ABBR_TO_FIPS.get(st, ""),
                "place_fips": "",
                "census_place_geoid": "",
                "census_state_fips": _STATE_ABBR_TO_FIPS.get(st, ""),
                "census_place_fips": "",
                "place_name_gazetteer": "",
                "place_pop10": math.nan,
                "intptlat": math.nan,
                "intptlong": math.nan,
                "ref_name_norm_raw": str(ref_var.get("raw") or ""),
                "ref_name_norm_strip1": str(ref_var.get("strip1") or ""),
                "match_name_norm_raw": "",
                "match_name_norm_strip1": "",
                "match_method": "unmatched",
                "match_method_detail": "unmatched",
                "match_confidence": "unmatched",
                "match_score": math.nan,
                "name_similarity": math.nan,
                "token_overlap": math.nan,
                "candidate_count": 0,
                "candidate_count_state": int(len(g)),
            }
        )

    out = pd.DataFrame.from_records(rows).sort_values(["state_abbr", "city_name"]).reset_index(drop=True)
    cands = pd.DataFrame.from_records(cand_rows).sort_values(["state_abbr", "city_name", "rank"]).reset_index(drop=True)
    return out, cands


def _load_fcc_county_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): (v if isinstance(v, dict) else {}) for k, v in raw.items()}


def _save_fcc_county_cache(path: Path, payload: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attach_county_crosswalk(place_xw: pd.DataFrame, cache_dir: Path, *, refresh: bool) -> pd.DataFrame:
    cache_path = cache_dir / "fcc_place_to_county_cache.json"
    cache = {} if refresh else _load_fcc_county_cache(cache_path)
    rows: list[dict[str, Any]] = []
    for r in place_xw.itertuples(index=False):
        place_geoid = str(r.place_geoid or "").strip()
        lat = _to_float(r.intptlat)
        lon = _to_float(r.intptlong)
        payload: dict[str, Any] = {}
        if place_geoid and math.isfinite(lat) and math.isfinite(lon):
            if place_geoid in cache:
                payload = cache[place_geoid]
            else:
                url = (
                    "https://geo.fcc.gov/api/census/block/find"
                    f"?format=json&latitude={lat}&longitude={lon}&showall=false"
                )
                try:
                    raw = _request_json_with_retry(url, timeout=45.0, max_retries=3)
                    county = raw.get("County") if isinstance(raw, dict) else {}
                    payload = {
                        "county_fips": str((county or {}).get("FIPS") or ""),
                        "county_name": str((county or {}).get("name") or ""),
                        "status": str((raw or {}).get("status") or ""),
                    }
                except Exception as e:
                    payload = {"county_fips": "", "county_name": "", "status": f"error:{e}"}
                cache[place_geoid] = payload

        rows.append(
            {
                "city_key": str(r.city_key),
                "city_name": str(r.city_name),
                "state_abbr": str(r.state_abbr),
                "state_fips": str(r.state_fips or ""),
                "place_geoid": place_geoid,
                "county_fips": str(payload.get("county_fips") or ""),
                "county_name": str(payload.get("county_name") or ""),
                "county_match_status": str(payload.get("status") or ""),
            }
        )

    _save_fcc_county_cache(cache_path, cache)
    out = pd.DataFrame.from_records(rows).sort_values(["state_abbr", "city_name"]).reset_index(drop=True)
    return out


def _load_acs_place_state_year(
    *,
    state_fips: str,
    year: int,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    cache_file = cache_dir / f"acs_place_{year}_{state_fips}.json"
    if cache_file.is_file() and not refresh:
        arr = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        vars_csv = ",".join(_ACS_PLACE_VARS)
        url = (
            f"https://api.census.gov/data/{int(year)}/acs/acs5"
            f"?get=NAME,{vars_csv}&for=place:*&in=state:{state_fips}"
        )
        arr = _request_json_with_retry(url, timeout=90.0, max_retries=3)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(arr), encoding="utf-8")

    if not isinstance(arr, list) or len(arr) < 2:
        return pd.DataFrame()
    hdr = arr[0]
    body = arr[1:]
    df = pd.DataFrame(body, columns=hdr)
    df["year"] = int(year)
    return df


def _build_place_covariates(
    place_xw: pd.DataFrame,
    *,
    years: list[int],
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    mapped = place_xw[place_xw["place_fips"].astype(str).str.len() > 0].copy()
    states = sorted(set(mapped["state_fips"].astype(str).tolist()))
    pulls: list[pd.DataFrame] = []
    for year in years:
        for state_fips in states:
            try:
                df = _load_acs_place_state_year(
                    state_fips=state_fips,
                    year=year,
                    cache_dir=cache_dir,
                    refresh=refresh,
                )
            except Exception:
                continue
            if df.empty:
                continue
            pulls.append(df)
    if not pulls:
        return pd.DataFrame()
    acs = pd.concat(pulls, ignore_index=True)
    acs["state_fips"] = acs["state"].astype(str).str.zfill(2)
    acs["place_fips"] = acs["place"].astype(str).str.zfill(5)
    keep = mapped[["state_fips", "place_fips"]].drop_duplicates()
    acs = acs.merge(keep, on=["state_fips", "place_fips"], how="inner")
    acs["population_place"] = pd.to_numeric(acs["B01003_001E"], errors="coerce")
    acs["housing_units_place"] = pd.to_numeric(acs["B25001_001E"], errors="coerce")
    acs["median_household_income_place"] = pd.to_numeric(acs["B19013_001E"], errors="coerce")
    acs["median_home_value_place"] = pd.to_numeric(acs["B25077_001E"], errors="coerce")
    occ_total = pd.to_numeric(acs["B25002_001E"], errors="coerce")
    vacant = pd.to_numeric(acs["B25002_003E"], errors="coerce")
    acs["vacancy_rate_place"] = (vacant / occ_total).replace([math.inf, -math.inf], math.nan)
    acs = acs[
        [
            "year",
            "state_fips",
            "place_fips",
            "population_place",
            "housing_units_place",
            "median_household_income_place",
            "median_home_value_place",
            "vacancy_rate_place",
        ]
    ].copy()
    return acs.drop_duplicates(subset=["year", "state_fips", "place_fips"], keep="last")


def _load_acs_county_state_year(
    *,
    state_fips: str,
    year: int,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    cache_file = cache_dir / f"acs_county_profile_{year}_{state_fips}.json"
    if cache_file.is_file() and not refresh:
        arr = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        vars_csv = ",".join(_ACS_COUNTY_PROFILE_VARS)
        url = (
            f"https://api.census.gov/data/{int(year)}/acs/acs5/profile"
            f"?get=NAME,{vars_csv}&for=county:*&in=state:{state_fips}"
        )
        arr = _request_json_with_retry(url, timeout=90.0, max_retries=3)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(arr), encoding="utf-8")
    if not isinstance(arr, list) or len(arr) < 2:
        return pd.DataFrame()
    hdr = arr[0]
    body = arr[1:]
    df = pd.DataFrame(body, columns=hdr)
    df["year"] = int(year)
    return df


def _build_county_covariates(
    county_xw: pd.DataFrame,
    *,
    years: list[int],
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    xw = county_xw[county_xw["county_fips"].astype(str).str.len() == 5].copy()
    states = sorted(set(xw["state_fips"].astype(str).tolist()))
    pulls: list[pd.DataFrame] = []
    for year in years:
        for state_fips in states:
            try:
                df = _load_acs_county_state_year(
                    state_fips=state_fips,
                    year=year,
                    cache_dir=cache_dir,
                    refresh=refresh,
                )
            except Exception:
                continue
            if df.empty:
                continue
            pulls.append(df)
    if not pulls:
        return pd.DataFrame()
    acs = pd.concat(pulls, ignore_index=True)
    acs["state_fips"] = acs["state"].astype(str).str.zfill(2)
    acs["county_3"] = acs["county"].astype(str).str.zfill(3)
    acs["county_fips"] = acs["state_fips"] + acs["county_3"]
    keep = xw[["county_fips"]].drop_duplicates()
    acs = acs.merge(keep, on="county_fips", how="inner")

    acs["unemployment_rate_county_pct"] = pd.to_numeric(acs["DP03_0009PE"], errors="coerce")
    acs["per_capita_income_county"] = pd.to_numeric(acs["DP03_0088E"], errors="coerce")
    acs["industry_agriculture_share_county_pct"] = pd.to_numeric(acs["DP03_0033PE"], errors="coerce")
    acs["industry_construction_share_county_pct"] = pd.to_numeric(acs["DP03_0034PE"], errors="coerce")
    acs["industry_manufacturing_share_county_pct"] = pd.to_numeric(acs["DP03_0035PE"], errors="coerce")
    acs["industry_retail_share_county_pct"] = pd.to_numeric(acs["DP03_0037PE"], errors="coerce")
    acs["industry_prof_services_share_county_pct"] = pd.to_numeric(acs["DP03_0041PE"], errors="coerce")

    acs = acs[
        [
            "year",
            "county_fips",
            "unemployment_rate_county_pct",
            "per_capita_income_county",
            "industry_agriculture_share_county_pct",
            "industry_construction_share_county_pct",
            "industry_manufacturing_share_county_pct",
            "industry_retail_share_county_pct",
            "industry_prof_services_share_county_pct",
        ]
    ].copy()
    return acs.drop_duplicates(subset=["year", "county_fips"], keep="last")


def _parse_bps_annual_file(text: str, *, year: int) -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return pd.DataFrame()
    data_rows = rows[2:]
    out_rows: list[dict[str, Any]] = []
    for r in data_rows:
        if len(r) < 41:
            continue
        state_fips = str(r[1]).strip().zfill(2)
        place_fips = str(r[5]).strip().zfill(5)
        if not state_fips or not place_fips:
            continue
        # 99990 appears to be county/other aggregate records.
        if place_fips == "99990":
            continue
        units_1 = _to_float(r[18])
        units_2 = _to_float(r[21])
        units_3_4 = _to_float(r[24])
        units_5_plus = _to_float(r[27])
        vals = [x for x in (units_1, units_2, units_3_4, units_5_plus) if math.isfinite(x)]
        permits_total = float(sum(vals)) if vals else math.nan
        out_rows.append(
            {
                "year": int(year),
                "state_fips": state_fips,
                "place_fips": place_fips,
                "permits_units_1": units_1,
                "permits_units_2": units_2,
                "permits_units_3_4": units_3_4,
                "permits_units_5_plus": units_5_plus,
                "permits_units_total": permits_total,
            }
        )
    out = pd.DataFrame.from_records(out_rows)
    if out.empty:
        return out
    out = (
        out.groupby(["year", "state_fips", "place_fips"], dropna=False)
        .agg(
            permits_units_1=("permits_units_1", "sum"),
            permits_units_2=("permits_units_2", "sum"),
            permits_units_3_4=("permits_units_3_4", "sum"),
            permits_units_5_plus=("permits_units_5_plus", "sum"),
            permits_units_total=("permits_units_total", "sum"),
        )
        .reset_index()
    )
    return out


def _build_permits_covariates(
    place_xw: pd.DataFrame,
    *,
    years: list[int],
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    mapped = place_xw[
        (place_xw["place_fips"].astype(str).str.len() == 5)
        & (place_xw["region"].astype(str).isin(_BPS_REGION_SPEC.keys()))
    ].copy()
    if mapped.empty:
        return pd.DataFrame()
    keep = mapped[["state_fips", "place_fips"]].drop_duplicates()
    regions = sorted(set(mapped["region"].astype(str).tolist()))
    out_rows: list[pd.DataFrame] = []
    for year in years:
        for region in regions:
            reg_path, reg_prefix = _BPS_REGION_SPEC[region]
            filename = f"{reg_prefix}{int(year)}a.txt"
            cache_file = cache_dir / f"bps_place_{region}_{year}.txt"
            text: str | None = None
            if cache_file.is_file() and not refresh:
                text = cache_file.read_text(encoding="utf-8", errors="ignore")
            else:
                url = f"https://www2.census.gov/econ/bps/Place/{reg_path}/{filename}"
                try:
                    text = _request_text_with_retry(url, timeout=90.0, max_retries=2)
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(text, encoding="utf-8")
                except Exception:
                    continue
            if not text:
                continue
            p = _parse_bps_annual_file(text, year=int(year))
            if p.empty:
                continue
            out_rows.append(p)
    if not out_rows:
        return pd.DataFrame()
    permits = pd.concat(out_rows, ignore_index=True)
    permits = permits.merge(keep, on=["state_fips", "place_fips"], how="inner")
    permits = permits.drop_duplicates(subset=["year", "state_fips", "place_fips"], keep="last")
    return permits


def _make_city_year_grid(city_xw: pd.DataFrame, *, year_min: int, year_max: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in city_xw.itertuples(index=False):
        lo = max(int(year_min), int(r.min_issue_year))
        hi = min(int(year_max), int(r.max_issue_year))
        if lo > hi:
            continue
        for y in range(lo, hi + 1):
            rows.append(
                {
                    "city_key": str(r.city_key),
                    "city_name": str(r.city_name),
                    "state_abbr": str(r.state_abbr),
                    "region": str(r.region),
                    "urbanicity_proxy": str(r.urbanicity_proxy),
                    "state_fips": str(r.state_fips or ""),
                    "place_fips": str(r.place_fips or ""),
                    "place_geoid": str(r.place_geoid or ""),
                    "county_fips": str(getattr(r, "county_fips", "") or ""),
                    "county_name": str(getattr(r, "county_name", "") or ""),
                    "year": int(y),
                }
            )
    return pd.DataFrame.from_records(rows)


def _build_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dimension", "key", "variable", "n_rows", "non_missing_n", "coverage_share"])
    vars_cov = [
        "population_place",
        "housing_units_place",
        "median_household_income_place",
        "median_home_value_place",
        "vacancy_rate_place",
        "permits_units_total",
        "permits_per_1000_pop",
        "unemployment_rate_county_pct",
        "per_capita_income_county",
        "industry_agriculture_share_county_pct",
        "industry_construction_share_county_pct",
        "industry_manufacturing_share_county_pct",
        "industry_retail_share_county_pct",
        "industry_prof_services_share_county_pct",
    ]
    rows: list[dict[str, Any]] = []
    n_all = len(df)
    for v in vars_cov:
        nn = int(df[v].notna().sum()) if v in df.columns else 0
        rows.append(
            {
                "dimension": "variable",
                "key": "all_years",
                "variable": v,
                "n_rows": int(n_all),
                "non_missing_n": int(nn),
                "coverage_share": float(nn / max(1, n_all)),
            }
        )
    for y, g in df.groupby("year", dropna=False):
        n = int(len(g))
        for v in vars_cov:
            nn = int(g[v].notna().sum()) if v in g.columns else 0
            rows.append(
                {
                    "dimension": "year",
                    "key": str(int(y) if pd.notna(y) else "NA"),
                    "variable": v,
                    "n_rows": n,
                    "non_missing_n": nn,
                    "coverage_share": float(nn / max(1, n)),
                }
            )
    return pd.DataFrame.from_records(rows)


def _build_place_match_summary(place_xw: pd.DataFrame) -> pd.DataFrame:
    if place_xw.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "value",
                "count",
                "share",
            ]
        )

    rows: list[dict[str, Any]] = []
    n = int(len(place_xw))
    for metric, col in (
        ("match_method", "match_method"),
        ("match_confidence", "match_confidence"),
    ):
        vc = place_xw[col].astype(str).fillna("").replace({"": "missing"}).value_counts(dropna=False)
        for k, c in vc.items():
            rows.append(
                {
                    "metric": metric,
                    "value": str(k),
                    "count": int(c),
                    "share": float(c / max(1, n)),
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(["metric", "count", "value"], ascending=[True, False, True]).reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "covariates")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    refs = _load_city_refs(run_dir)
    inferred_year_min = min(r.min_issue_year for r in refs)
    inferred_year_max = max(r.max_issue_year for r in refs)
    year_min = int(args.year_min) if int(args.year_min) > 0 else int(inferred_year_min)
    year_max = int(args.year_max) if int(args.year_max) > 0 else int(inferred_year_max)
    if year_min > year_max:
        raise SystemExit(f"Invalid year bounds: min={year_min} max={year_max}")

    acs_end_default = dt.date.today().year - 1
    acs_end_year = int(args.acs_end_year) if int(args.acs_end_year) > 0 else int(acs_end_default)
    acs_years = [y for y in range(max(int(args.acs_start_year), year_min), min(acs_end_year, year_max) + 1)]
    permit_years = [y for y in range(max(int(args.permits_start_year), year_min), year_max + 1)]

    gaz = _load_gazetteer_places(cache_dir, refresh=bool(args.refresh_cache))
    manual_overrides = _load_manual_place_overrides(Path(args.manual_place_overrides), gaz) if str(args.manual_place_overrides).strip() else {}
    place_xw, place_candidates = _match_city_to_place(
        refs,
        gaz,
        manual_overrides=manual_overrides,
        fuzzy_threshold=float(args.fuzzy_match_threshold),
        candidate_top_n=int(args.candidate_top_n),
    )
    place_xw_path = out_dir / "city_place_crosswalk.csv"
    place_xw.to_csv(place_xw_path, index=False, quoting=csv.QUOTE_MINIMAL)
    place_candidates_path = out_dir / "city_place_match_candidates.csv"
    place_candidates.to_csv(place_candidates_path, index=False, quoting=csv.QUOTE_MINIMAL)
    place_match_summary = _build_place_match_summary(place_xw)
    place_match_summary_path = out_dir / "city_place_match_summary.csv"
    place_match_summary.to_csv(place_match_summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    unmatched = place_xw[place_xw["match_method"].astype(str) == "unmatched"].copy()
    unmatched_path = out_dir / "city_place_unmatched.csv"
    unmatched.to_csv(unmatched_path, index=False, quoting=csv.QUOTE_MINIMAL)
    manual_template = unmatched[["city_key", "city_name", "state_abbr"]].copy()
    manual_template["place_geoid"] = ""
    manual_template["state_fips"] = manual_template["state_abbr"].astype(str).map(_STATE_ABBR_TO_FIPS).fillna("")
    manual_template["place_fips"] = ""
    manual_template["notes"] = ""
    manual_template_path = out_dir / "manual_place_overrides_template.csv"
    manual_template.to_csv(manual_template_path, index=False, quoting=csv.QUOTE_MINIMAL)

    county_xw = _attach_county_crosswalk(place_xw, cache_dir=cache_dir, refresh=bool(args.refresh_cache))
    county_xw_path = out_dir / "city_county_crosswalk.csv"
    county_xw.to_csv(county_xw_path, index=False, quoting=csv.QUOTE_MINIMAL)

    place_cov = _build_place_covariates(
        place_xw,
        years=acs_years,
        cache_dir=cache_dir / "acs_place",
        refresh=bool(args.refresh_cache),
    )
    county_cov = _build_county_covariates(
        county_xw,
        years=acs_years,
        cache_dir=cache_dir / "acs_county",
        refresh=bool(args.refresh_cache),
    )
    permits_cov = _build_permits_covariates(
        place_xw,
        years=permit_years,
        cache_dir=cache_dir / "bps_place",
        refresh=bool(args.refresh_cache),
    )

    city_meta = place_xw.merge(
        county_xw[["city_key", "county_fips", "county_name"]],
        on="city_key",
        how="left",
    )
    city_year = _make_city_year_grid(city_meta, year_min=year_min, year_max=year_max)
    if city_year.empty:
        raise SystemExit("City-year grid is empty; check selected_panel_issues.csv and year bounds.")

    if not place_cov.empty:
        city_year = city_year.merge(place_cov, on=["year", "state_fips", "place_fips"], how="left")
    else:
        for c in (
            "population_place",
            "housing_units_place",
            "median_household_income_place",
            "median_home_value_place",
            "vacancy_rate_place",
        ):
            city_year[c] = math.nan

    if not permits_cov.empty:
        city_year = city_year.merge(permits_cov, on=["year", "state_fips", "place_fips"], how="left")
    else:
        for c in ("permits_units_1", "permits_units_2", "permits_units_3_4", "permits_units_5_plus", "permits_units_total"):
            city_year[c] = math.nan

    if not county_cov.empty:
        city_year = city_year.merge(county_cov, on=["year", "county_fips"], how="left")
    else:
        for c in (
            "unemployment_rate_county_pct",
            "per_capita_income_county",
            "industry_agriculture_share_county_pct",
            "industry_construction_share_county_pct",
            "industry_manufacturing_share_county_pct",
            "industry_retail_share_county_pct",
            "industry_prof_services_share_county_pct",
        ):
            city_year[c] = math.nan

    city_year["permits_per_1000_pop"] = (
        pd.to_numeric(city_year["permits_units_total"], errors="coerce")
        / pd.to_numeric(city_year["population_place"], errors="coerce")
        * 1000.0
    ).replace([math.inf, -math.inf], math.nan)

    cov_path = out_dir / "city_year_external_covariates.csv"
    city_year = city_year.sort_values(["city_key", "year"]).reset_index(drop=True)
    city_year.to_csv(cov_path, index=False, quoting=csv.QUOTE_MINIMAL)

    miss = _build_missingness_report(city_year)
    miss_path = out_dir / "covariate_missingness_report.csv"
    miss.to_csv(miss_path, index=False, quoting=csv.QUOTE_MINIMAL)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "n_city_refs": int(len(refs)),
        "year_min": int(year_min),
        "year_max": int(year_max),
        "acs_start_year": int(args.acs_start_year),
        "acs_end_year": int(acs_end_year),
        "permits_start_year": int(args.permits_start_year),
        "manual_place_overrides": str(Path(args.manual_place_overrides).expanduser().resolve()) if str(args.manual_place_overrides).strip() else "",
        "fuzzy_match_threshold": float(args.fuzzy_match_threshold),
        "candidate_top_n": int(args.candidate_top_n),
        "acs_years_requested": acs_years,
        "permit_years_requested": permit_years,
        "n_city_year_rows": int(len(city_year)),
        "exact_match_rate": float((place_xw["match_method"] == "exact").mean()) if not place_xw.empty else math.nan,
        "high_confidence_place_match_rate": float((place_xw["match_confidence"] == "high").mean()) if not place_xw.empty else math.nan,
        "place_match_rate": float((place_xw["match_method"] != "unmatched").mean()) if not place_xw.empty else math.nan,
        "county_match_rate": float((county_xw["county_fips"].astype(str).str.len() == 5).mean()) if not county_xw.empty else math.nan,
        "outputs": {
            "city_place_crosswalk": str(place_xw_path),
            "city_place_match_candidates": str(place_candidates_path),
            "city_place_match_summary": str(place_match_summary_path),
            "city_place_unmatched": str(unmatched_path),
            "manual_place_overrides_template": str(manual_template_path),
            "city_county_crosswalk": str(county_xw_path),
            "city_year_external_covariates": str(cov_path),
            "covariate_missingness_report": str(miss_path),
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"city_refs={len(refs)} "
        f"city_year_rows={len(city_year)} "
        f"exact_match_rate={prov['exact_match_rate']:.3f} "
        f"place_match_rate={prov['place_match_rate']:.3f} "
        f"county_match_rate={prov['county_match_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
