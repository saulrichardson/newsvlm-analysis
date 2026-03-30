#!/usr/bin/env python3
"""
Merge Census of Governments 2022 IDs into v2 metadata.

This script is tailored to:
  - v2 metadata.csv with columns: city_name, state_abbr, place20_geoid
  - census_of_gov_22.xlsx "General Purpose" sheet with columns:
      CENSUS_ID_PID6, CENSUS_ID_GIDID, UNIT_NAME, UNIT_TYPE, FIPS_STATE, FIPS_PLACE, CITY, STATE

Join strategy:
  1) Exact place geoid join (metadata.place20_geoid -> FIPS_STATE+FIPS_PLACE) for MUNICIPAL rows.
  2) Fallback by city/state only where the candidate is unique in (MUNICIPAL + TOWNSHIP).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METADATA_CSV = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/saul/newspapers/"
    "page_artifact_packets_v5_full_20260225/v2/metadata.csv"
)
DEFAULT_CENSUS_XLSX = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/"
    "raw_data/census_data/census_of_gov_22.xlsx"
)


def _norm_text(x: object) -> str:
    s = str(x or "").strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _norm_geoid(x: object) -> str:
    s = _norm_text(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _validate(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(
            f"{label} is empty (0 bytes): {path}\n"
            "If this is a cloud placeholder, open/download it first."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    ap.add_argument("--census-xlsx", type=Path, default=DEFAULT_CENSUS_XLSX)
    ap.add_argument("--sheet", default="General Purpose")
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Write back to --metadata-csv (creates backup if missing).",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output file when not --inplace (default: <metadata stem>_with_census.csv).",
    )
    ap.add_argument(
        "--backup-csv",
        type=Path,
        default=None,
        help="Optional backup path when --inplace (default: <metadata stem>.pre_census_backup_YYYYMMDD.csv).",
    )
    ap.add_argument(
        "--unmatched-csv",
        type=Path,
        default=None,
        help="Optional CSV of unmatched unique towns.",
    )
    args = ap.parse_args()

    metadata_csv = args.metadata_csv.expanduser().resolve()
    census_xlsx = args.census_xlsx.expanduser().resolve()
    _validate(metadata_csv, "metadata CSV")
    _validate(census_xlsx, "census workbook")

    md = pd.read_csv(metadata_csv, low_memory=False)
    gp = pd.read_excel(census_xlsx, sheet_name=args.sheet, engine="openpyxl")

    need_md = {"city_name", "state_abbr", "place20_geoid"}
    miss_md = sorted(c for c in need_md if c not in md.columns)
    if miss_md:
        raise SystemExit(f"metadata CSV missing columns: {miss_md}")

    need_gp = {
        "CENSUS_ID_PID6",
        "CENSUS_ID_GIDID",
        "UNIT_NAME",
        "UNIT_TYPE",
        "FIPS_STATE",
        "FIPS_PLACE",
        "CITY",
        "STATE",
    }
    miss_gp = sorted(c for c in need_gp if c not in gp.columns)
    if miss_gp:
        raise SystemExit(f"census sheet missing columns: {miss_gp}")

    # Normalize metadata fields.
    md["_place20_geoid_n"] = md["place20_geoid"].map(_norm_geoid)
    md["_city_n"] = md["city_name"].map(_norm_text).str.lower()
    md["_state_n"] = md["state_abbr"].map(_norm_text).str.lower()

    # Normalize census workbook and derive place geoid.
    gp = gp.copy()
    gp["FIPS_STATE_n"] = pd.to_numeric(gp["FIPS_STATE"], errors="coerce").astype("Int64")
    gp["FIPS_PLACE_n"] = pd.to_numeric(gp["FIPS_PLACE"], errors="coerce").astype("Int64")
    gp["_state2"] = gp["FIPS_STATE_n"].astype(str).replace("<NA>", "").str.zfill(2)
    gp["_place5"] = gp["FIPS_PLACE_n"].astype(str).replace("<NA>", "").str.zfill(5)
    gp["_place_geoid"] = np.where(
        gp["_state2"].str.len().eq(2) & gp["_place5"].str.len().eq(5),
        gp["_state2"] + gp["_place5"],
        "",
    )
    gp["_city_n"] = gp["CITY"].map(_norm_text).str.lower()
    gp["_state_n"] = gp["STATE"].map(_norm_text).str.lower()

    # Stage 1: place20 geoid to MUNICIPAL.
    gp_m = gp[gp["UNIT_TYPE"].astype(str).eq("2 - MUNICIPAL")].copy()
    geo_tbl = (
        gp_m.sort_values(["CENSUS_ID_PID6", "CENSUS_ID_GIDID"])
        .drop_duplicates("_place_geoid", keep="first")[
            ["_place_geoid", "CENSUS_ID_PID6", "CENSUS_ID_GIDID", "UNIT_NAME", "UNIT_TYPE"]
        ]
        .rename(
            columns={
                "CENSUS_ID_PID6": "_pid6_geo",
                "CENSUS_ID_GIDID": "_gidid_geo",
                "UNIT_NAME": "_unit_name_geo",
                "UNIT_TYPE": "_unit_type_geo",
            }
        )
    )
    out = md.merge(geo_tbl, left_on="_place20_geoid_n", right_on="_place_geoid", how="left")

    # Stage 2 fallback: unique city/state candidate in (MUNICIPAL + TOWNSHIP).
    gp_mt = gp[gp["UNIT_TYPE"].astype(str).isin(["2 - MUNICIPAL", "3 - TOWNSHIP"])].copy()
    city_counts = (
        gp_mt.groupby(["_city_n", "_state_n"])["CENSUS_ID_PID6"]
        .nunique()
        .reset_index(name="_n_ids")
    )
    city_unique = (
        gp_mt.merge(city_counts, on=["_city_n", "_state_n"], how="left")
        .query("_n_ids == 1")
        .sort_values(["CENSUS_ID_PID6", "CENSUS_ID_GIDID"])
        .drop_duplicates(["_city_n", "_state_n"], keep="first")[
            ["_city_n", "_state_n", "CENSUS_ID_PID6", "CENSUS_ID_GIDID", "UNIT_NAME", "UNIT_TYPE"]
        ]
        .rename(
            columns={
                "CENSUS_ID_PID6": "_pid6_city",
                "CENSUS_ID_GIDID": "_gidid_city",
                "UNIT_NAME": "_unit_name_city",
                "UNIT_TYPE": "_unit_type_city",
            }
        )
    )
    out = out.merge(city_unique, on=["_city_n", "_state_n"], how="left")

    use_geo = out["_pid6_geo"].notna()
    use_city = (~use_geo) & out["_pid6_city"].notna()

    out["census_id_pid6"] = np.where(use_geo, out["_pid6_geo"], out["_pid6_city"])
    out["census_id_gidid"] = np.where(use_geo, out["_gidid_geo"], out["_gidid_city"])
    out["census_unit_name"] = np.where(use_geo, out["_unit_name_geo"], out["_unit_name_city"])
    out["census_unit_type"] = np.where(use_geo, out["_unit_type_geo"], out["_unit_type_city"])
    out["census_join_method"] = np.where(
        use_geo,
        "place20_geoid_to_municipal",
        np.where(use_city, "city_state_unique_muni_or_township", "unmatched"),
    )

    # Canonical output formatting.
    out["census_id_pid6"] = (
        pd.to_numeric(out["census_id_pid6"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "")
    )
    out["census_id_gidid"] = (
        pd.to_numeric(out["census_id_gidid"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "")
    )
    for c in ["census_unit_name", "census_unit_type", "census_join_method"]:
        out[c] = out[c].map(_norm_text)

    drop_cols = [c for c in out.columns if c.startswith("_") or c in {"FIPS_STATE_n", "FIPS_PLACE_n"}]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    if args.inplace:
        backup = (
            args.backup_csv.expanduser().resolve()
            if args.backup_csv is not None
            else metadata_csv.with_name(f"{metadata_csv.stem}.pre_census_backup_20260227.csv")
        )
        if not backup.exists():
            backup.write_bytes(metadata_csv.read_bytes())
        out_path = metadata_csv
    else:
        out_path = (
            args.output_csv.expanduser().resolve()
            if args.output_csv is not None
            else metadata_csv.with_name(f"{metadata_csv.stem}_with_census.csv")
        )
        backup = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    total = int(len(out))
    matched = int((out["census_id_pid6"] != "").sum())
    matched_pct = (matched / total * 100.0) if total else 0.0

    print(f"metadata_csv={metadata_csv}")
    print(f"census_xlsx={census_xlsx}")
    print(f"sheet={args.sheet}")
    if backup is not None:
        print(f"backup_csv={backup}")
    print(f"output_csv={out_path}")
    print(f"rows_total={total}")
    print(f"rows_with_census_id={matched} ({matched_pct:.2f}%)")
    print("join_method_counts=")
    print(out["census_join_method"].value_counts(dropna=False).to_string())

    if args.unmatched_csv is not None:
        unmatched = (
            out[out["census_join_method"] == "unmatched"][
                ["town_id", "city_name", "state_abbr", "city_key", "place20_geoid"]
            ]
            .drop_duplicates()
            .sort_values(["state_abbr", "city_name"])
        )
        u_path = args.unmatched_csv.expanduser().resolve()
        u_path.parent.mkdir(parents=True, exist_ok=True)
        unmatched.to_csv(u_path, index=False)
        print(f"unmatched_csv={u_path} rows={len(unmatched)}")


if __name__ == "__main__":
    main()

