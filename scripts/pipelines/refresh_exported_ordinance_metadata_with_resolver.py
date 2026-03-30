#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from export_full_ordinance_project_bundle import (
    DEFAULT_ARCHIVE_METADATA_CSV,
    DEFAULT_PUBLICATION_LOCATIONS_JSONL,
    clean,
    load_publication_indexes,
    select_publication_geo,
)
from export_jurisdiction_utils import (
    build_jurisdiction_resolver,
    derive_geo_target_fields,
    metadata_namespace,
    resolve_issue_with_preferred_public_fallback,
    stringify_resolution_fields,
)

DEFAULT_CENSUS_OF_GOV_XLSX = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/census_data/census_of_gov_22.xlsx"
)

RESOLVER_FIELDS = [
    "jurisdiction_key",
    "jurisdiction_input_source",
    "jurisdiction_input_city_name",
    "jurisdiction_input_city_name_normalized",
    "jurisdiction_input_state_abbr",
    "jurisdiction_input_level",
    "jurisdiction_input_confidence_0_to_1",
    "jurisdiction_evidence_quote",
    "jurisdiction_resolution_status",
    "jurisdiction_resolution_method",
    "jurisdiction_resolution_score",
    "jurisdiction_candidate_count",
    "jurisdiction_candidate_pid6s",
    "jurisdiction_state_source",
    "jurisdiction_state_backfill_used",
    "jurisdiction_city_name",
    "jurisdiction_city_name_normalized",
    "jurisdiction_state_abbr",
    "jurisdiction_level",
    "jurisdiction_display_name",
    "place20_geoid",
    "county20_fips_primary",
    "puma20_geoid_primary",
    "cbsa23_geoid_primary",
    "cbsa23_name_primary",
    "census_id_pid6",
    "census_id_gidid",
    "census_unit_name",
    "census_unit_type",
    "census_join_method",
    "manual_override_note",
    "legacy_bundle_city_key",
    "legacy_bundle_place20_geoid",
    "legacy_bundle_census_id_pid6",
]

PUBLICATION_FIELDS = [
    "publication_city_name",
    "publication_state_abbr",
    "publication_city_key",
    "publication_cbsa_name",
    "publication_geo_source",
    "publication_census_join_method",
    "publication_place20_geoid",
    "publication_census_id_pid6",
    "publication_census_id_gidid",
    "publication_census_unit_name",
    "publication_census_unit_type",
]

GEO_TARGET_FIELDS = [
    "geo_target_city_name",
    "geo_target_state_abbr",
    "geo_target_level",
    "geo_target_basis",
    "geo_target_state_backfill_used",
    "geo_target_overrode_publication_location",
    "geo_target_match_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh an existing Desmond-style ordinance metadata.csv in place using "
            "the workbook-backed jurisdiction resolver and publication-location fallback."
        )
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument(
        "--archive-metadata-csv",
        type=Path,
        default=DEFAULT_ARCHIVE_METADATA_CSV,
    )
    parser.add_argument(
        "--publication-locations-jsonl",
        type=Path,
        default=DEFAULT_PUBLICATION_LOCATIONS_JSONL,
    )
    parser.add_argument(
        "--census-of-gov-xlsx",
        type=Path,
        default=DEFAULT_CENSUS_OF_GOV_XLSX,
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak copy of metadata.csv before overwriting it.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise SystemExit(f"No CSV header found in {path}")
        return list(reader.fieldnames), rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def publication_geo_from_row(
    row: dict[str, str],
    *,
    issue_id: str,
    newspaper_slug: str,
    archive_issue_index: dict[str, dict[str, str]],
    archive_slug_index: dict[str, dict[str, str]],
    locations_slug_index: dict[str, dict[str, str]],
) -> dict[str, str]:
    row_geo = {field: clean(row.get(field)) for field in PUBLICATION_FIELDS}
    if row_geo["publication_city_name"]:
        return row_geo
    return select_publication_geo(
        issue_id=issue_id,
        newspaper_slug=newspaper_slug,
        archive_issue_index=archive_issue_index,
        archive_slug_index=archive_slug_index,
        locations_slug_index=locations_slug_index,
    )


def resolver_metadata_fields(row: dict[str, str]) -> dict[str, str]:
    fields = dict(row)
    if not clean(fields.get("jurisdiction_city_name")):
        fields["jurisdiction_city_name"] = clean(
            row.get("extracted_jurisdiction_city_name")
        )
    if not clean(fields.get("jurisdiction_state_abbr")):
        fields["jurisdiction_state_abbr"] = clean(
            row.get("extracted_jurisdiction_state_abbr")
        )
    return fields


def main() -> None:
    args = parse_args()
    fieldnames, rows = load_rows(args.metadata_csv)
    if args.backup:
        backup_path = args.metadata_csv.with_suffix(args.metadata_csv.suffix + ".bak")
        shutil.copy2(args.metadata_csv, backup_path)

    archive_issue_index, archive_slug_index, locations_slug_index = load_publication_indexes(
        args.archive_metadata_csv,
        args.publication_locations_jsonl,
    )
    resolver = build_jurisdiction_resolver(
        archive_metadata_csv=args.archive_metadata_csv,
        census_of_gov_xlsx=args.census_of_gov_xlsx,
    )

    before_pid6 = sum(1 for row in rows if clean(row.get("census_id_pid6")))
    before_place = sum(1 for row in rows if clean(row.get("place20_geoid")))

    refreshed_rows: list[dict[str, str]] = []
    for row in rows:
        issue_id = clean(row.get("issue_id"))
        newspaper_slug = clean(row.get("newspaper_slug"))
        publication_geo = publication_geo_from_row(
            row,
            issue_id=issue_id,
            newspaper_slug=newspaper_slug,
            archive_issue_index=archive_issue_index,
            archive_slug_index=archive_slug_index,
            locations_slug_index=locations_slug_index,
        )
        resolver_row = resolver_metadata_fields(row)
        resolution = resolve_issue_with_preferred_public_fallback(
            resolver=resolver,
            issue_id=issue_id,
            issue_json={"classification": {}},
            metadata_row=metadata_namespace(resolver_row),
            publication_geo=publication_geo,
        )
        geo_target = derive_geo_target_fields(
            publication_geo=publication_geo,
            resolution=resolution,
        )
        new_row = dict(row)
        new_row.update(publication_geo)
        new_row.update(stringify_resolution_fields(resolution))
        new_row.update(geo_target)
        refreshed_rows.append(new_row)

    for field in PUBLICATION_FIELDS + RESOLVER_FIELDS + GEO_TARGET_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    write_rows(args.metadata_csv, fieldnames, refreshed_rows)

    after_pid6 = sum(1 for row in refreshed_rows if clean(row.get("census_id_pid6")))
    after_place = sum(1 for row in refreshed_rows if clean(row.get("place20_geoid")))
    print(
        {
            "metadata_csv": str(args.metadata_csv),
            "row_count": len(refreshed_rows),
            "pid6_before": before_pid6,
            "pid6_after": after_pid6,
            "place20_before": before_place,
            "place20_after": after_place,
        }
    )


if __name__ == "__main__":
    main()
