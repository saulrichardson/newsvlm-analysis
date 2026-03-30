#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from export_jurisdiction_utils import (
    build_jurisdiction_resolver,
    derive_geo_target_fields,
    metadata_namespace,
    resolve_issue_with_preferred_public_fallback,
    stringify_resolution_fields,
)


DEFAULT_ARCHIVE_METADATA_CSV = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/newspapers/archive/v2/metadata.csv"
)
DEFAULT_PUBLICATION_LOCATIONS_JSONL = Path(
    "/Users/saulrichardson/projects/newspapers/newspaper-scrapping/data/locations_headful.jsonl"
)
DEFAULT_CENSUS_OF_GOV_XLSX = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/census_data/census_of_gov_22.xlsx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the full-ordinance Dropbox bundle into the project-folder layout "
            "with metadata.csv at root, clean ordinances in parsed/, page PNGs in "
            "raw/<issue_id>/, and issue transcripts in transcript/."
        )
    )
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
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
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing export.",
    )
    return parser.parse_args()


def ensure_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    if not path.is_dir():
        raise SystemExit(f"Target path is not a directory: {path}")
    if overwrite:
        return
    if any(path.iterdir()):
        raise SystemExit(
            f"Target directory is not empty and --overwrite was not set: {path}"
        )


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_state(value: object) -> str:
    return clean(value).upper()


def normalize_geo_name(value: object) -> str:
    text = clean(value).casefold()
    text = re.sub(
        r"^(city|village|town|township|borough|county)\s+of\s+",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


RESOLVER_FIELDS = [
    "jurisdiction_key",
    "jurisdiction_input_source",
    "jurisdiction_input_city_name",
    "jurisdiction_input_city_name_normalized",
    "jurisdiction_input_state_abbr",
    "jurisdiction_input_level",
    "jurisdiction_input_confidence_0_to_1",
    "jurisdiction_resolution_status",
    "jurisdiction_resolution_method",
    "jurisdiction_resolution_score",
    "jurisdiction_candidate_count",
    "jurisdiction_candidate_pid6s",
    "jurisdiction_city_name_normalized",
    "jurisdiction_state_backfill_used",
    "jurisdiction_state_source",
    "jurisdiction_display_name",
    "manual_override_note",
    "legacy_bundle_city_key",
    "legacy_bundle_place20_geoid",
    "legacy_bundle_census_id_pid6",
]


def load_publication_indexes(
    archive_metadata_csv: Path, publication_locations_jsonl: Path
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    archive_issue_index: dict[str, dict[str, str]] = {}
    archive_slug_index: dict[str, dict[str, str]] = {}
    locations_slug_index: dict[str, dict[str, str]] = {}

    if archive_metadata_csv.is_file():
        with archive_metadata_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                issue_id = clean(row.get("issue_id"))
                slug = clean(row.get("newspaper_slug"))
                city = clean(row.get("city_name"))
                state = normalize_state(row.get("state_abbr"))
                record = {
                    "publication_city_name": city,
                    "publication_state_abbr": state,
                    "publication_city_key": clean(row.get("city_key")),
                    "publication_cbsa_name": clean(row.get("cbsa23_name_primary")),
                    "publication_geo_source": "archive_v2",
                    "publication_census_join_method": clean(row.get("census_join_method")),
                    "publication_place20_geoid": clean(row.get("place20_geoid")),
                    "publication_census_id_pid6": clean(row.get("census_id_pid6")),
                    "publication_census_id_gidid": clean(row.get("census_id_gidid")),
                    "publication_census_unit_name": clean(row.get("census_unit_name")),
                    "publication_census_unit_type": clean(row.get("census_unit_type")),
                }
                if issue_id and city:
                    archive_issue_index[issue_id] = record
                if slug and city and slug not in archive_slug_index:
                    archive_slug_index[slug] = record

    if publication_locations_jsonl.is_file():
        with publication_locations_jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                slug = clean(row.get("pub_slug"))
                city = clean(row.get("city_name"))
                state = normalize_state(row.get("state_abbr"))
                if not slug or not city:
                    continue
                locations_slug_index[slug] = {
                    "publication_city_name": city,
                    "publication_state_abbr": state,
                    "publication_city_key": clean(row.get("city_key")) or f"{city.lower().replace(' ', '-') }__{state.lower()}".replace(" ", "-"),
                    "publication_cbsa_name": "",
                    "publication_geo_source": "locations_headful",
                    "publication_census_join_method": "locations_headful_slug",
                    "publication_place20_geoid": "",
                    "publication_census_id_pid6": "",
                    "publication_census_id_gidid": "",
                    "publication_census_unit_name": "",
                    "publication_census_unit_type": "",
                }
    return archive_issue_index, archive_slug_index, locations_slug_index


def select_publication_geo(
    *,
    issue_id: str,
    newspaper_slug: str,
    archive_issue_index: dict[str, dict[str, str]],
    archive_slug_index: dict[str, dict[str, str]],
    locations_slug_index: dict[str, dict[str, str]],
) -> dict[str, str]:
    if issue_id in archive_issue_index:
        return archive_issue_index[issue_id]
    if newspaper_slug in archive_slug_index:
        return archive_slug_index[newspaper_slug]
    if newspaper_slug in locations_slug_index:
        return locations_slug_index[newspaper_slug]
    return {
        "publication_city_name": "",
        "publication_state_abbr": "",
        "publication_city_key": "",
        "publication_cbsa_name": "",
        "publication_geo_source": "missing",
        "publication_census_join_method": "",
        "publication_place20_geoid": "",
        "publication_census_id_pid6": "",
        "publication_census_id_gidid": "",
        "publication_census_unit_name": "",
        "publication_census_unit_type": "",
    }


def normalize_ordinance_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.casefold())
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def annotate_reprint_groups(rows: list[dict[str, str]], target_root: Path) -> None:
    indexed_rows: list[dict[str, object]] = []
    by_geo: dict[tuple[str, str], list[dict[str, object]]] = {}

    for row in rows:
        parsed_relpath = clean(row.get("parsed_relpath"))
        parsed_text = ""
        parsed_len = 0
        if parsed_relpath:
            parsed_path = target_root / parsed_relpath
            if parsed_path.is_file():
                parsed_text = parsed_path.read_text(encoding="utf-8", errors="replace")
                parsed_len = len(parsed_text)
        normalized_text = normalize_ordinance_text(parsed_text)[:20000] if parsed_text else ""
        geo_key = (
            normalize_geo_name(row.get("geo_target_city_name")),
            normalize_state(row.get("geo_target_state_abbr")),
        )
        enriched = {
            "row": row,
            "geo_key": geo_key,
            "issue_id": row["issue_id"],
            "issue_date": date.fromisoformat(row["issue_date"]),
            "parsed_relpath": parsed_relpath,
            "parsed_len": parsed_len,
            "normalized_text": normalized_text,
        }
        indexed_rows.append(enriched)
        by_geo.setdefault(geo_key, []).append(enriched)

    parent = {item["issue_id"]: item["issue_id"] for item in indexed_rows}
    similarity_by_issue: dict[tuple[str, str], float] = {}
    detection_method = "same_geo_within_45d_prefix500_ge_0.92_body12000_ge_0.93"

    def find(issue_id: str) -> str:
        while parent[issue_id] != issue_id:
            parent[issue_id] = parent[parent[issue_id]]
            issue_id = parent[issue_id]
        return issue_id

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for geo_key, items in by_geo.items():
        if not geo_key[0]:
            continue
        items.sort(key=lambda item: item["issue_date"])
        for index, left in enumerate(items):
            for right in items[index + 1 :]:
                day_delta = (right["issue_date"] - left["issue_date"]).days
                if day_delta > 45:
                    break
                if not left["normalized_text"] or not right["normalized_text"]:
                    continue
                prefix_ratio = SequenceMatcher(
                    None,
                    left["normalized_text"][:500],
                    right["normalized_text"][:500],
                ).ratio()
                body_ratio = SequenceMatcher(
                    None,
                    left["normalized_text"][:12000],
                    right["normalized_text"][:12000],
                ).ratio()
                if prefix_ratio >= 0.92 and body_ratio >= 0.93:
                    union(left["issue_id"], right["issue_id"])
                    key = tuple(sorted((left["issue_id"], right["issue_id"])))
                    similarity_by_issue[key] = body_ratio

    clusters: dict[str, list[dict[str, object]]] = {}
    for item in indexed_rows:
        clusters.setdefault(find(item["issue_id"]), []).append(item)

    for root_issue_id, cluster in clusters.items():
        cluster.sort(key=lambda item: (item["issue_date"], item["issue_id"]))
        canonical = max(
            cluster,
            key=lambda item: (
                item["parsed_len"],
                -int(item["issue_date"].strftime("%Y%m%d")),
                item["issue_id"],
            ),
        )
        canonical_issue_id = canonical["issue_id"]
        canonical_relpath = canonical["parsed_relpath"]
        cluster_ids = [item["issue_id"] for item in cluster]
        if len(cluster) > 1:
            group_key_material = "|".join(cluster_ids)
            group_id = "reprint_" + hashlib.sha1(group_key_material.encode("utf-8")).hexdigest()[:12]
        else:
            group_id = ""

        for item in cluster:
            row = item["row"]
            if len(cluster) == 1:
                row["reprint_status"] = "unique"
                row["reprint_group_id"] = ""
                row["reprint_group_size"] = "1"
                row["reprint_is_canonical"] = "1"
                row["reprint_canonical_issue_id"] = row["issue_id"]
                row["reprint_canonical_parsed_relpath"] = clean(row.get("parsed_relpath"))
                row["reprint_days_from_canonical"] = "0"
                row["reprint_similarity_to_canonical"] = "1.0000"
                row["reprint_detection_method"] = "none"
                row["reprint_group_issue_ids_json"] = json.dumps([row["issue_id"]], ensure_ascii=True)
                continue

            pair_key = tuple(sorted((item["issue_id"], canonical_issue_id)))
            similarity = 1.0 if item["issue_id"] == canonical_issue_id else similarity_by_issue.get(pair_key, "")
            row["reprint_status"] = "canonical" if item["issue_id"] == canonical_issue_id else "duplicate"
            row["reprint_group_id"] = group_id
            row["reprint_group_size"] = str(len(cluster))
            row["reprint_is_canonical"] = "1" if item["issue_id"] == canonical_issue_id else "0"
            row["reprint_canonical_issue_id"] = canonical_issue_id
            row["reprint_canonical_parsed_relpath"] = canonical_relpath
            row["reprint_days_from_canonical"] = str(
                abs((item["issue_date"] - canonical["issue_date"]).days)
            )
            row["reprint_similarity_to_canonical"] = (
                f"{similarity:.4f}" if similarity != "" else ""
            )
            row["reprint_detection_method"] = detection_method
            row["reprint_group_issue_ids_json"] = json.dumps(cluster_ids, ensure_ascii=True)


def write_readme(target_root: Path) -> None:
    readme_path = target_root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Newspaper Ordinances",
                "",
                "This folder is an export of the full-ordinance bundle into the project-folder layout.",
                "",
                "- `metadata.csv`: bundle metadata with relative paths to parsed ordinances, raw images, transcripts, and reprint-group fields.",
                "- `parsed/<issue_id>.md`: best available clean ordinance text per issue.",
                "- `raw/<issue_id>/*.png`: source page images per issue.",
                "- `transcript/<issue_id>.txt`: source newspaper OCR transcript per issue.",
                "",
                "Reprint handling is metadata-only. Each issue keeps its own files, and `metadata.csv` identifies high-confidence near-time reprints through the `reprint_*` fields.",
                "",
                "Path fields in `metadata.csv` are relative to this folder.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_metadata(target_metadata_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with target_metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_bundle = args.source_bundle.resolve()
    target_root = args.target_root.resolve()

    source_metadata_path = source_bundle / "metadata.csv"
    if not source_metadata_path.exists():
        raise SystemExit(f"Missing source metadata: {source_metadata_path}")

    parsed_dir = target_root / "parsed"
    raw_dir = target_root / "raw"
    transcript_dir = target_root / "transcript"

    ensure_empty_or_overwrite(parsed_dir, args.overwrite)
    ensure_empty_or_overwrite(raw_dir, args.overwrite)
    ensure_empty_or_overwrite(transcript_dir, args.overwrite)

    target_metadata_path = target_root / "metadata.csv"
    if target_metadata_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Target metadata already exists and --overwrite was not set: {target_metadata_path}"
        )

    with source_metadata_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"No rows found in source metadata: {source_metadata_path}")

    archive_issue_index, archive_slug_index, locations_slug_index = load_publication_indexes(
        args.archive_metadata_csv.resolve(),
        args.publication_locations_jsonl.resolve(),
    )
    resolver = build_jurisdiction_resolver(
        archive_metadata_csv=args.archive_metadata_csv.resolve(),
        census_of_gov_xlsx=args.census_of_gov_xlsx.resolve(),
    )

    fieldnames = list(rows[0].keys())
    additional_fields = [
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
        "geo_target_city_name",
        "geo_target_state_abbr",
        "geo_target_level",
        "geo_target_basis",
        "geo_target_state_backfill_used",
        "geo_target_overrode_publication_location",
        "geo_target_match_status",
        "parsed_source",
        "parsed_relpath",
        "raw_dir_relpath",
        "raw_image_count",
        "raw_image_relpaths_json",
        "transcript_relpath",
        "reprint_status",
        "reprint_group_id",
        "reprint_group_size",
        "reprint_is_canonical",
        "reprint_canonical_issue_id",
        "reprint_canonical_parsed_relpath",
        "reprint_days_from_canonical",
        "reprint_similarity_to_canonical",
        "reprint_detection_method",
        "reprint_group_issue_ids_json",
    ]
    for field in additional_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    for field in RESOLVER_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    written_rows: list[dict[str, str]] = []
    for row in rows:
        issue_id = row["issue_id"]
        newspaper_slug = row["newspaper_slug"]
        issue_dir = source_bundle / row["issue_dir_rel"]
        issue_json_path = issue_dir / "issue.json"
        if not issue_json_path.is_file():
            raise SystemExit(f"Missing issue.json for {issue_id}: {issue_json_path}")
        issue_json = json.loads(issue_json_path.read_text(encoding="utf-8"))

        transcript_src = source_bundle / row["issue_transcript_relpath"]
        transcript_dst = transcript_dir / f"{issue_id}.txt"
        if not transcript_src.exists():
            raise SystemExit(f"Missing transcript for {issue_id}: {transcript_src}")
        copy_file(transcript_src, transcript_dst)

        parsed_src = source_bundle / row["ordinance_relpath"] if row["ordinance_relpath"] else None
        raw_ordinance_src = (
            source_bundle / row["ordinance_raw_relpath"]
            if row["ordinance_raw_relpath"]
            else None
        )

        parsed_relpath = ""
        parsed_dst_path = parsed_dir / f"{issue_id}.md"
        parsed_source = ""
        if parsed_src and parsed_src.exists() and parsed_src.read_text(encoding="utf-8").strip():
            copy_file(parsed_src, parsed_dst_path)
            parsed_source = "ordinance_text"
            parsed_relpath = str(parsed_dst_path.relative_to(target_root))
        elif raw_ordinance_src and raw_ordinance_src.exists() and raw_ordinance_src.read_text(encoding="utf-8").strip():
            copy_file(raw_ordinance_src, parsed_dst_path)
            parsed_source = "ordinance_raw"
            parsed_relpath = str(parsed_dst_path.relative_to(target_root))
        else:
            parsed_source = "missing"

        raw_issue_dir = raw_dir / issue_id
        raw_issue_dir.mkdir(parents=True, exist_ok=True)
        image_src_dir = issue_dir / "images"
        if not image_src_dir.exists():
            raise SystemExit(f"Missing images directory for {issue_id}: {image_src_dir}")
        image_relpaths: list[str] = []
        for image_src in sorted(image_src_dir.glob("*.png")):
            image_dst = raw_issue_dir / image_src.name
            copy_file(image_src, image_dst)
            image_relpaths.append(str(image_dst.relative_to(target_root)))

        publication_geo = select_publication_geo(
            issue_id=issue_id,
            newspaper_slug=newspaper_slug,
            archive_issue_index=archive_issue_index,
            archive_slug_index=archive_slug_index,
            locations_slug_index=locations_slug_index,
        )
        resolution = resolve_issue_with_preferred_public_fallback(
            resolver=resolver,
            issue_id=issue_id,
            issue_json=issue_json,
            metadata_row=metadata_namespace(row),
            publication_geo=publication_geo,
        )
        geo_target = derive_geo_target_fields(
            publication_geo=publication_geo,
            resolution=resolution,
        )

        row["issue_dir_rel"] = ""
        row["issue_json_relpath"] = ""
        row["issue_transcript_relpath"] = str(transcript_dst.relative_to(target_root))
        row["ordinance_relpath"] = parsed_relpath
        row["ordinance_raw_relpath"] = ""
        row["ordinance_notes_relpath"] = ""
        row["publication_city_name"] = publication_geo["publication_city_name"]
        row["publication_state_abbr"] = publication_geo["publication_state_abbr"]
        row["publication_city_key"] = publication_geo["publication_city_key"]
        row["publication_cbsa_name"] = publication_geo["publication_cbsa_name"]
        row["publication_geo_source"] = publication_geo["publication_geo_source"]
        row["publication_census_join_method"] = publication_geo["publication_census_join_method"]
        row["publication_place20_geoid"] = publication_geo["publication_place20_geoid"]
        row["publication_census_id_pid6"] = publication_geo["publication_census_id_pid6"]
        row["publication_census_id_gidid"] = publication_geo["publication_census_id_gidid"]
        row["publication_census_unit_name"] = publication_geo["publication_census_unit_name"]
        row["publication_census_unit_type"] = publication_geo["publication_census_unit_type"]
        row.update(stringify_resolution_fields(resolution))
        row["jurisdiction_city_name"] = row.get("jurisdiction_city_name", "")
        row["jurisdiction_state_abbr"] = row.get("jurisdiction_state_abbr", "")
        row["jurisdiction_level"] = row.get("jurisdiction_level", "")
        row["city_key"] = row.get("jurisdiction_key", "")
        row.update(geo_target)
        row["parsed_source"] = parsed_source
        row["parsed_relpath"] = parsed_relpath
        row["raw_dir_relpath"] = str(raw_issue_dir.relative_to(target_root))
        row["raw_image_count"] = str(len(image_relpaths))
        row["raw_image_relpaths_json"] = json.dumps(image_relpaths, ensure_ascii=True)
        row["transcript_relpath"] = str(transcript_dst.relative_to(target_root))
        written_rows.append(row)

    annotate_reprint_groups(written_rows, target_root)
    write_metadata(target_metadata_path, fieldnames, written_rows)
    write_readme(target_root)

    print(
        json.dumps(
            {
                "target_root": str(target_root),
                "metadata_path": str(target_metadata_path),
                "issue_count": len(written_rows),
                "parsed_written": sum(1 for row in written_rows if row["parsed_relpath"]),
                "parsed_from_raw": sum(
                    1 for row in written_rows if row["parsed_source"] == "ordinance_raw"
                ),
                "parsed_missing": sum(
                    1 for row in written_rows if row["parsed_source"] == "missing"
                ),
                "image_count": sum(int(row["raw_image_count"]) for row in written_rows),
                "reprint_groups": len(
                    {
                        row["reprint_group_id"]
                        for row in written_rows
                        if clean(row.get("reprint_group_id"))
                    }
                ),
                "reprint_issues": sum(
                    1 for row in written_rows if row.get("reprint_status") in {"canonical", "duplicate"}
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
