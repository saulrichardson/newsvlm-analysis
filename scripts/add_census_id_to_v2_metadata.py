#!/usr/bin/env python3
"""
Add a Census-of-Governments ID from an Excel town table into v2 metadata.csv.

Typical usage:
  python scripts/add_census_id_to_v2_metadata.py \
    --metadata-csv "/path/to/v2/metadata.csv" \
    --census-xlsx "/path/to/census_of_gov_22.xlsx" \
    --output-csv "/path/to/v2/metadata_with_census_id.csv"

If workbook column names are unknown, run:
  python scripts/add_census_id_to_v2_metadata.py \
    --census-xlsx "/path/to/census_of_gov_22.xlsx" \
    --preview-columns
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_METADATA_CSV = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/saul/newspapers/"
    "page_artifact_packets_v5_full_20260225/v2/metadata.csv"
)
DEFAULT_CENSUS_XLSX = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/"
    "raw_data/census_data/census_of_gov_22.xlsx"
)


KEY_COL_CANDIDATES = [
    "town_id",
    "city_id",
    "townid",
    "cityid",
    "place20_geoid",
    "place_geoid",
    "geoid",
    "geoid_place",
    "place_fips",
]


ID_COL_CANDIDATES = [
    "census_id",
    "censusid",
    "census_of_gov_id",
    "census_of_governments_id",
    "census_of_gov_22_id",
    "gov_id",
    "government_id",
    "id_census_of_gov",
    "fips",
]


def _norm_text(x: object) -> str:
    s = str(x or "").strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _norm_key(x: object) -> str:
    s = _norm_text(x)
    if not s:
        return ""
    s = re.sub(r"\.0+$", "", s)
    return s


def _slug_col(name: str) -> str:
    s = _norm_text(name).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _pick_column(columns: Iterable[str], candidates: list[str], label: str) -> str:
    raw = [str(c) for c in columns]
    by_slug = {_slug_col(c): c for c in raw}
    for c in candidates:
        hit = by_slug.get(_slug_col(c))
        if hit:
            return hit
    raise SystemExit(
        f"Could not auto-detect {label}. "
        f"Pass it explicitly with --{label}. "
        f"Available columns: {raw}"
    )


def _pick_sheet(xlsx: Path, sheet: str | None) -> str:
    xl = pd.ExcelFile(xlsx, engine="openpyxl")
    if sheet:
        if sheet not in xl.sheet_names:
            raise SystemExit(f"--sheet '{sheet}' not found. Available: {xl.sheet_names}")
        return sheet

    # Pick first sheet containing at least one column.
    for s in xl.sheet_names:
        d = xl.parse(s, nrows=5)
        if len(d.columns) > 0:
            return s
    raise SystemExit(f"No usable sheets found in workbook: {xlsx}")


def _preview_columns(xlsx: Path, sheet: str | None) -> None:
    use_sheet = _pick_sheet(xlsx, sheet)
    df = pd.read_excel(xlsx, sheet_name=use_sheet, engine="openpyxl", nrows=5)
    print(f"workbook={xlsx}")
    print(f"sheet={use_sheet}")
    print(f"columns={list(df.columns)}")
    if not df.empty:
        print("sample_rows:")
        print(df.head(5).to_string(index=False))


def _validate_readable_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(
            f"{label} is empty (0 bytes): {path}\n"
            "If this is a Dropbox placeholder, mark it as available offline first."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    ap.add_argument("--census-xlsx", type=Path, default=DEFAULT_CENSUS_XLSX)
    ap.add_argument("--sheet", default="", help="Optional sheet name in the census workbook.")
    ap.add_argument(
        "--preview-columns",
        action="store_true",
        help="Print workbook columns/sample and exit.",
    )
    ap.add_argument(
        "--metadata-key-col",
        default="town_id",
        help="Join key column in metadata CSV (default: town_id).",
    )
    ap.add_argument(
        "--census-key-col",
        default="",
        help="Join key column in workbook. If omitted, auto-detect.",
    )
    ap.add_argument(
        "--census-id-col",
        default="",
        help="Census ID column in workbook. If omitted, auto-detect.",
    )
    ap.add_argument(
        "--new-col",
        default="census_of_gov_id",
        help="Output column name for merged census id.",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <metadata stem>_with_census_id.csv).",
    )
    ap.add_argument(
        "--unmatched-csv",
        type=Path,
        default=None,
        help="Optional output CSV of unmatched metadata keys.",
    )
    args = ap.parse_args()

    metadata_csv = args.metadata_csv.expanduser().resolve()
    census_xlsx = args.census_xlsx.expanduser().resolve()
    sheet = _norm_text(args.sheet) or None
    metadata_key_col = _norm_text(args.metadata_key_col)
    census_key_col = _norm_text(args.census_key_col)
    census_id_col = _norm_text(args.census_id_col)
    new_col = _norm_text(args.new_col) or "census_of_gov_id"

    _validate_readable_file(census_xlsx, "census workbook")
    if args.preview_columns:
        _preview_columns(census_xlsx, sheet)
        return

    _validate_readable_file(metadata_csv, "metadata CSV")
    use_sheet = _pick_sheet(census_xlsx, sheet)

    md = pd.read_csv(metadata_csv, low_memory=False)
    if metadata_key_col not in md.columns:
        raise SystemExit(
            f"metadata key column '{metadata_key_col}' not found in {metadata_csv}. "
            f"Available: {list(md.columns)}"
        )

    towns = pd.read_excel(census_xlsx, sheet_name=use_sheet, engine="openpyxl")
    if towns.empty:
        raise SystemExit(f"Workbook sheet '{use_sheet}' is empty: {census_xlsx}")

    if not census_key_col:
        census_key_col = _pick_column(towns.columns, KEY_COL_CANDIDATES, "census-key-col")
    if not census_id_col:
        census_id_col = _pick_column(towns.columns, ID_COL_CANDIDATES, "census-id-col")

    keep = towns[[census_key_col, census_id_col]].copy()
    keep["__join_key"] = keep[census_key_col].map(_norm_key)
    keep[new_col] = keep[census_id_col].map(_norm_text)
    keep = keep[["__join_key", new_col]]
    keep = keep[keep["__join_key"] != ""].copy()

    # Deterministic dedupe: keep first non-empty census id per key.
    keep = keep.sort_values([new_col], ascending=[False]).drop_duplicates("__join_key", keep="first")

    md_out = md.copy()
    md_out["__join_key"] = md_out[metadata_key_col].map(_norm_key)
    md_out = md_out.merge(keep, on="__join_key", how="left")
    md_out[new_col] = md_out[new_col].fillna("").map(_norm_text)
    md_out = md_out.drop(columns=["__join_key"])

    out_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else metadata_csv.with_name(f"{metadata_csv.stem}_with_census_id.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    md_out.to_csv(out_csv, index=False)

    matched = int(md_out[new_col].astype(str).str.strip().ne("").sum())
    total = int(len(md_out))
    matched_pct = (matched / total * 100.0) if total else 0.0

    unique_md_keys = md_out[metadata_key_col].map(_norm_key)
    unique_md_keys = unique_md_keys[unique_md_keys != ""]
    unique_md_key_n = int(unique_md_keys.nunique())

    matched_key_n = int(
        md_out.loc[
            md_out[new_col].astype(str).str.strip().ne(""),
            metadata_key_col,
        ]
        .map(_norm_key)
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )

    print(f"metadata_csv={metadata_csv}")
    print(f"census_xlsx={census_xlsx}")
    print(f"sheet={use_sheet}")
    print(f"metadata_key_col={metadata_key_col}")
    print(f"census_key_col={census_key_col}")
    print(f"census_id_col={census_id_col}")
    print(f"new_col={new_col}")
    print(f"rows_total={total}")
    print(f"rows_with_{new_col}={matched} ({matched_pct:.2f}%)")
    print(f"unique_metadata_keys={unique_md_key_n}")
    print(f"unique_metadata_keys_matched={matched_key_n}")
    print(f"output_csv={out_csv}")

    if args.unmatched_csv is not None:
        unmatched = md_out.loc[
            md_out[new_col].astype(str).str.strip().eq(""),
            [metadata_key_col],
        ].copy()
        unmatched["norm_key"] = unmatched[metadata_key_col].map(_norm_key)
        unmatched = unmatched[unmatched["norm_key"] != ""].drop_duplicates("norm_key")
        unmatched = unmatched.rename(columns={metadata_key_col: "metadata_key"})
        unmatched_out = args.unmatched_csv.expanduser().resolve()
        unmatched_out.parent.mkdir(parents=True, exist_ok=True)
        unmatched.to_csv(unmatched_out, index=False)
        print(f"unmatched_csv={unmatched_out} rows={len(unmatched)}")


if __name__ == "__main__":
    main()

