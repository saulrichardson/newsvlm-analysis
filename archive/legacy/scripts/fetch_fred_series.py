#!/usr/bin/env python3
"""
Fetch small outcome series from FRED and store them as local artifacts.

Motivation:
  - We want to sanity-check candidate text-derived RHS measures against plausible outcomes
    (prices, housing, etc.) without committing to a causal design yet.
  - FRED provides lightweight public series we can overlay visually.

Inputs:
  - A JSON city/series mapping (see scripts/fixtures/fred_outcomes_city_map.json):
      {
        "madison__wi": {
          "city_name": "Madison",
          "state_abbr": "wi",
          "series": { "fhfa_atnhpi_msa_q": "ATNHPIUS31540Q" }
        }
      }

Outputs (output-dir):
  - series/<SERIES_ID>.csv (raw values with DATE + value)
  - fred_city_outcomes_<stamp>.csv/.parquet (tidy long panel)
  - provenance_<stamp>.json

Notes:
  - This script is deterministic and does not require an API key; it uses the
    public `fredgraph.csv` endpoint.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch FRED series for a small set of cities and cache locally.")
    ap.add_argument("--city-map", required=True, help="JSON mapping of cities to FRED series IDs.")
    ap.add_argument("--output-dir", required=True, help="Directory to write cached series + tidy panel.")
    ap.add_argument("--force", action="store_true", help="Redownload even if cached series files exist.")
    return ap.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing --city-map: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"--city-map must be a JSON object: {path}")
    return obj


def _fred_csv_url(series_id: str) -> str:
    # Public endpoint; does not require API key.
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _fetch_one_series(*, series_id: str, cache_csv: Path, force: bool) -> pd.DataFrame:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    if cache_csv.is_file() and not force:
        raw = pd.read_csv(cache_csv)
    else:
        url = _fred_csv_url(series_id)
        try:
            raw = pd.read_csv(url)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Failed to download FRED series {series_id} from {url}: {e}") from e
        raw.to_csv(cache_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    # Expected columns vary across endpoints:
    # - fredgraph.csv often uses `observation_date`
    # - other exports use `DATE`
    date_col = "DATE" if "DATE" in raw.columns else "observation_date" if "observation_date" in raw.columns else ""
    if not date_col or series_id not in raw.columns:
        raise SystemExit(f"Unexpected FRED CSV schema for {series_id}: columns={list(raw.columns)}")

    df = raw.rename(columns={date_col: "date", series_id: "value"}).copy()
    df["series_id"] = series_id
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    return df[["series_id", "date", "value"]]


def main() -> None:
    args = _parse_args()
    city_map_path = Path(args.city_map).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "series"
    cache_dir.mkdir(parents=True, exist_ok=True)

    city_map = _load_json(city_map_path)

    rows: list[dict[str, Any]] = []
    for city_key, spec in city_map.items():
        if not isinstance(spec, dict):
            raise SystemExit(f"Bad city spec for key={city_key}: expected object, got {type(spec)}")
        city_name = str(spec.get("city_name", "")).strip()
        state_abbr = str(spec.get("state_abbr", "")).strip().lower()
        series_map = spec.get("series")
        if not city_name or not state_abbr:
            raise SystemExit(f"Bad city spec for key={city_key}: missing city_name/state_abbr")
        if not isinstance(series_map, dict) or not series_map:
            raise SystemExit(f"Bad city spec for key={city_key}: missing non-empty series dict")

        for series_key, series_id in series_map.items():
            sid = str(series_id).strip()
            skey = str(series_key).strip()
            if not sid or not skey:
                continue
            df = _fetch_one_series(series_id=sid, cache_csv=cache_dir / f"{sid}.csv", force=bool(args.force))
            df["city_key"] = city_key
            df["city_name"] = city_name
            df["state_abbr"] = state_abbr
            df["series_key"] = skey
            rows.append(df)

    if not rows:
        raise SystemExit("No series were fetched (check --city-map).")

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["series_id", "city_key", "date"]).reset_index(drop=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"fred_city_outcomes_{stamp}.csv"
    out_pq = out_dir / f"fred_city_outcomes_{stamp}.parquet"
    panel.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    try:
        panel.to_parquet(out_pq, index=False)
    except Exception:
        # Parquet is optional; CSV is the portable fallback.
        pass

    provenance = {
        "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city_map": str(city_map_path),
        "output_dir": str(out_dir),
        "force": bool(args.force),
        "n_rows": int(len(panel)),
        "n_series": int(panel["series_id"].nunique()),
        "series_ids": sorted(panel["series_id"].dropna().unique().tolist()),
    }
    (out_dir / f"provenance_{stamp}.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Done. wrote={out_dir} rows={len(panel)} series={panel['series_id'].nunique()}")


if __name__ == "__main__":
    main()
