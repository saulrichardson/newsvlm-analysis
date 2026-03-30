#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a single non-full-code source-label remainder pool into a "
            "Desmond-style project folder with metadata.csv at root, clean "
            "ordinances in parsed/, raw page PNGs in raw/<issue_id>/, and OCR "
            "transcripts in transcript/."
        )
    )
    parser.add_argument("--promotion-csv", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument(
        "--second-stage-class",
        default="other_applicable_zoning_law",
        help="Only export rows whose second-stage class matches this value.",
    )
    parser.add_argument(
        "--bundle-title",
        default="Labeled Ordinance Remainder",
        help="README title for the exported corpus.",
    )
    parser.add_argument("--archive-metadata-csv", type=Path, required=True)
    parser.add_argument("--publication-locations-jsonl", type=Path, required=True)
    parser.add_argument("--census-of-gov-xlsx", type=Path, default=DEFAULT_CENSUS_OF_GOV_XLSX)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow exporting into an existing target root.",
    )
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_state(value: object) -> str:
    return clean(value).upper()


def slugify(text: str) -> str:
    value = clean(text).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "unknown"


def make_city_key(city_name: str, state_abbr: str) -> str:
    city = clean(city_name)
    state = normalize_state(state_abbr)
    if not city or not state:
        return ""
    return f"{slugify(city)}__{state.lower()}"


def norm_id(value: object, *, width: int | None = None) -> str:
    text = clean(value)
    text = re.sub(r"\.0+$", "", text)
    if width and text.isdigit():
        text = text.zfill(width)
    return text


def ensure_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    if not path.is_dir():
        raise SystemExit(f"Target path is not a directory: {path}")
    if overwrite:
        return
    if any(path.iterdir()):
        raise SystemExit(f"Target directory is not empty and --overwrite was not set: {path}")


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise SystemExit(f"Missing file to copy: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise SystemExit(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"Expected object JSON in {path}:{line_number}")
            rows.append(payload)
    return rows


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


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

    with archive_metadata_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
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
                "publication_place20_geoid": norm_id(row.get("place20_geoid"), width=7),
                "publication_census_id_pid6": norm_id(row.get("census_id_pid6")),
                "publication_census_id_gidid": norm_id(row.get("census_id_gidid")),
                "publication_census_unit_name": clean(row.get("census_unit_name")),
                "publication_census_unit_type": clean(row.get("census_unit_type")),
            }
            if issue_id and city:
                archive_issue_index[issue_id] = record
            if slug and city and slug not in archive_slug_index:
                archive_slug_index[slug] = record

    with publication_locations_jsonl.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
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
                "publication_city_key": clean(row.get("city_key")) or make_city_key(city, state),
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


def load_archive_geo_lookup(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(
        key=lambda row: (
            0 if clean(row.get("census_id_pid6")) else 1,
            0 if clean(row.get("place20_geoid")) else 1,
            clean(row.get("city_name")).lower(),
        )
    )
    for row in rows:
        city_name = clean(row.get("city_name"))
        state_abbr = normalize_state(row.get("state_abbr"))
        city_key = clean(row.get("city_key")) or make_city_key(city_name, state_abbr)
        if not city_key or city_key in out:
            continue
        out[city_key.lower()] = {
            "city_name": city_name,
            "state_abbr": state_abbr,
            "city_key": city_key.lower(),
            "place20_geoid": norm_id(row.get("place20_geoid"), width=7),
            "county20_fips_primary": norm_id(row.get("county20_fips_primary"), width=5),
            "puma20_geoid_primary": norm_id(row.get("puma20_geoid_primary"), width=7),
            "cbsa23_geoid_primary": norm_id(row.get("cbsa23_geoid_primary"), width=5),
            "cbsa23_name_primary": clean(row.get("cbsa23_name_primary")),
            "census_id_pid6": norm_id(row.get("census_id_pid6")),
            "census_id_gidid": norm_id(row.get("census_id_gidid")),
            "census_unit_name": clean(row.get("census_unit_name")),
            "census_unit_type": clean(row.get("census_unit_type")),
            "census_join_method": clean(row.get("census_join_method")),
        }
    return out


def load_dataset_rows(dataset_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    issues_by_issue_id: dict[str, dict[str, Any]] = {}
    class_by_issue_id: dict[str, dict[str, Any]] = {}

    for row in read_jsonl(dataset_dir / "issues.jsonl"):
        issue_id = clean(row.get("issue_id"))
        if not issue_id:
            raise SystemExit(f"Missing issue_id in {dataset_dir / 'issues.jsonl'}")
        issues_by_issue_id[issue_id] = row

    class_dir = dataset_dir / "classification_records"
    if not class_dir.is_dir():
        raise SystemExit(f"Missing classification_records dir: {class_dir}")
    for path in sorted(class_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        issue_id = clean(obj.get("issue_id"))
        if not issue_id:
            issue_id = clean((obj.get("source_row") or {}).get("issue_id"))
        if not issue_id:
            raise SystemExit(f"Missing issue_id in classification record: {path}")
        class_by_issue_id[issue_id] = obj
    return issues_by_issue_id, class_by_issue_id


def load_transcriptions(results_root: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in sorted(results_root.glob("worker*/transcriptions.csv")):
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                issue_id = clean(row.get("issue_id"))
                if not issue_id:
                    continue
                if issue_id in out:
                    raise SystemExit(f"Duplicate transcription row for issue_id={issue_id} under {results_root}")
                out[issue_id] = row
    return out


def write_metadata(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    target_root: Path,
    *,
    bundle_title: str,
    source_label: str,
    second_stage_class: str,
    exported_count: int,
    skipped_count: int,
) -> None:
    text = (
        f"# {bundle_title}\n\n"
        "This folder contains a filtered non-full-code newspaper ordinance corpus exported "
        "from a single source label and second-stage class.\n\n"
        "Layout:\n"
        "- `metadata.csv`: one row per exported issue\n"
        "- `parsed/<issue_id>.md`: clean ordinance text\n"
        "- `raw/<issue_id>/*.png`: copied source page PNGs\n"
        "- `transcript/<issue_id>.txt`: OCR transcript used for reconstruction\n\n"
        f"Source label: `{source_label}`\n"
        f"Second-stage class filter: `{second_stage_class}`\n"
        f"Exported issues: `{exported_count}`\n"
        f"Matched issues skipped because no clean ordinance text was available: `{skipped_count}`\n"
    )
    (target_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    promotion_csv = args.promotion_csv.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    archive_metadata_csv = args.archive_metadata_csv.expanduser().resolve()
    publication_locations_jsonl = args.publication_locations_jsonl.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    source_label = clean(args.source_label)
    second_stage_class = clean(args.second_stage_class)

    if not promotion_csv.is_file():
        raise SystemExit(f"--promotion-csv not found: {promotion_csv}")
    for path in [dataset_dir, results_root]:
        if not path.exists():
            raise SystemExit(f"Required path not found: {path}")
    if not source_label:
        raise SystemExit("--source-label is required")
    if not second_stage_class:
        raise SystemExit("--second-stage-class must be non-empty")

    parsed_dir = target_root / "parsed"
    raw_dir = target_root / "raw"
    transcript_dir = target_root / "transcript"
    ensure_empty_or_overwrite(parsed_dir, args.overwrite)
    ensure_empty_or_overwrite(raw_dir, args.overwrite)
    ensure_empty_or_overwrite(transcript_dir, args.overwrite)

    target_metadata_path = target_root / "metadata.csv"
    if target_metadata_path.exists() and not args.overwrite:
        raise SystemExit(f"Target metadata already exists and --overwrite was not set: {target_metadata_path}")

    archive_issue_index, archive_slug_index, locations_slug_index = load_publication_indexes(
        archive_metadata_csv, publication_locations_jsonl
    )
    resolver = build_jurisdiction_resolver(
        archive_metadata_csv=archive_metadata_csv,
        census_of_gov_xlsx=args.census_of_gov_xlsx.expanduser().resolve(),
    )

    issue_index, class_index = load_dataset_rows(dataset_dir)
    tx_index = load_transcriptions(results_root)

    selected_rows = [
        row
        for row in load_csv_rows(promotion_csv)
        if clean(row.get("source_label")) == source_label
        and clean(row.get("second_stage_class")) == second_stage_class
    ]
    if not selected_rows:
        raise SystemExit(
            f"No rows found in {promotion_csv} for source_label={source_label!r} "
            f"and second_stage_class={second_stage_class!r}"
        )

    metadata_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for promo in sorted(selected_rows, key=lambda row: clean(row.get("issue_id"))):
        issue_id = clean(promo.get("issue_id"))
        issue_row = issue_index.get(issue_id)
        class_row = class_index.get(issue_id)
        tx_row = tx_index.get(issue_id)
        if issue_row is None or class_row is None:
            raise SystemExit(f"Missing dataset/classification row for issue_id={issue_id}")

        ordinance_text = clean((tx_row or {}).get("ordinance_text"))
        if not ordinance_text:
            skipped_rows.append(
                {
                    "issue_id": issue_id,
                    "source_label": source_label,
                    "second_stage_class": second_stage_class,
                    "skip_reason": "missing_clean_transcription",
                }
            )
            continue

        transcript_src = Path(str(issue_row["issue_transcript_path"]))
        transcript_dst = transcript_dir / f"{issue_id}.txt"
        copy_file(transcript_src, transcript_dst)

        parsed_dst = parsed_dir / f"{issue_id}.md"
        parsed_dst.write_text(ordinance_text + "\n", encoding="utf-8")

        raw_issue_dir = raw_dir / issue_id
        raw_issue_dir.mkdir(parents=True, exist_ok=True)
        image_relpaths: list[str] = []
        pages = issue_row.get("pages") or []
        if not isinstance(pages, list) or not pages:
            raise SystemExit(f"issue_id={issue_id} has no pages payload")
        for page in sorted(pages, key=lambda item: int(item.get("page_num") or 0)):
            page_id = clean(page.get("page_id"))
            page_num = int(page.get("page_num") or 0)
            png_src = Path(str(page.get("png_path") or ""))
            if not png_src.is_file():
                raise SystemExit(f"Missing source PNG for issue_id={issue_id} page_id={page_id}: {png_src}")
            filename = f"{page_num:04d}__{page_id}{png_src.suffix or '.png'}"
            png_dst = raw_issue_dir / filename
            copy_file(png_src, png_dst)
            image_relpaths.append(str(png_dst.relative_to(target_root)))

        source_row = class_row.get("source_row") or {}
        parsed = source_row.get("parsed") or {}
        legal_document = parsed.get("legal_document") or {}
        zoning_presence = parsed.get("zoning_presence") or {}
        ordinance_jurisdiction = parsed.get("ordinance_jurisdiction") or {}
        issue_json = {"classification": parsed}

        newspaper_slug = clean(issue_row.get("newspaper_slug"))
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
            metadata_row=metadata_namespace(
                {
                    "jurisdiction_city_name": clean(ordinance_jurisdiction.get("city")),
                    "jurisdiction_state_abbr": normalize_state(ordinance_jurisdiction.get("state_abbr")),
                    "jurisdiction_level": clean(ordinance_jurisdiction.get("jurisdiction_level")),
                    "jurisdiction_evidence_quote": clean(ordinance_jurisdiction.get("evidence_quote")),
                    "city_key": make_city_key(
                        clean(ordinance_jurisdiction.get("city")),
                        normalize_state(ordinance_jurisdiction.get("state_abbr")),
                    ),
                    "place20_geoid": "",
                    "census_id_pid6": "",
                }
            ),
            publication_geo=publication_geo,
        )
        geo_target = derive_geo_target_fields(
            publication_geo=publication_geo,
            resolution=resolution,
        )

        row = {
                "issue_id": issue_id,
                "issue_date": clean(issue_row.get("issue_date")),
                "newspaper_slug": newspaper_slug,
                "classification_status": clean(source_row.get("status")),
                "classification_model": clean(source_row.get("classifier_model")),
                "source_label": source_label,
                "source_operativity": clean(promo.get("source_operativity")),
                "source_scope": clean(promo.get("source_scope")),
                "page_count": str(issue_row.get("page_count") or len(pages)),
                "document_class": clean(legal_document.get("class")),
                "document_operativity": clean(legal_document.get("operativity")),
                "document_primary_class": clean(zoning_presence.get("primary_class")),
                "document_confidence_0_to_1": clean(legal_document.get("confidence_0_to_1")),
                "document_legal_action": clean(legal_document.get("legal_action")),
                "document_scope": clean(legal_document.get("scope")),
                "document_publication_completeness": clean(legal_document.get("publication_completeness")),
                "document_evidence_quote": clean(legal_document.get("evidence_quote")),
                "zoning_presence_evidence_quote": clean(zoning_presence.get("evidence_quote")),
                "jurisdiction_evidence_quote": clean(ordinance_jurisdiction.get("evidence_quote")),
                "quality_flags_json": json.dumps(parsed.get("quality_flags") or [], ensure_ascii=False),
                "promotion_second_stage_class": second_stage_class,
                "promotion_instrument_family": clean(promo.get("instrument_family")),
                "promotion_operativity": clean(promo.get("second_stage_operativity")),
                "promotion_confidence_0_to_1": clean(promo.get("second_stage_confidence_0_to_1")),
                "promotion_evidence_quote": clean(promo.get("evidence_quote")),
                "promotion_quality_flags_json": clean(promo.get("quality_flags_json")),
                "promotion_rationale": clean(promo.get("rationale")),
                "promotion_response_json_ok": clean(promo.get("response_json_ok")),
                "publication_city_name": publication_geo["publication_city_name"],
                "publication_state_abbr": publication_geo["publication_state_abbr"],
                "publication_city_key": publication_geo["publication_city_key"],
                "publication_cbsa_name": publication_geo["publication_cbsa_name"],
                "publication_geo_source": publication_geo["publication_geo_source"],
                "publication_census_join_method": publication_geo["publication_census_join_method"],
                "publication_place20_geoid": publication_geo["publication_place20_geoid"],
                "publication_census_id_pid6": publication_geo["publication_census_id_pid6"],
                "publication_census_id_gidid": publication_geo["publication_census_id_gidid"],
                "publication_census_unit_name": publication_geo["publication_census_unit_name"],
                "publication_census_unit_type": publication_geo["publication_census_unit_type"],
                "parsed_source": "gemini_flash_ordinance_text",
                "parsed_relpath": str(parsed_dst.relative_to(target_root)),
                "raw_dir_relpath": str(raw_issue_dir.relative_to(target_root)),
                "raw_image_count": str(len(image_relpaths)),
                "raw_image_relpaths_json": json.dumps(image_relpaths, ensure_ascii=True),
                "transcript_relpath": str(transcript_dst.relative_to(target_root)),
                "transcription_contract_ok": clean((tx_row or {}).get("contract_ok")),
                "transcription_ordinance_chars": clean((tx_row or {}).get("ordinance_chars")),
                "transcription_source_line": clean((tx_row or {}).get("source_line")),
        }
        row.update(stringify_resolution_fields(resolution))
        row["city_key"] = row.get("jurisdiction_key", "")
        row.update(geo_target)
        metadata_rows.append(row)

    if not metadata_rows:
        raise SystemExit(
            "No matched issues with clean transcription outputs were exportable"
        )

    fieldnames = list(metadata_rows[0].keys())
    for field in RESOLVER_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    write_metadata(target_metadata_path, fieldnames, metadata_rows)
    write_readme(
        target_root,
        bundle_title=clean(args.bundle_title) or "Labeled Ordinance Remainder",
        source_label=source_label,
        second_stage_class=second_stage_class,
        exported_count=len(metadata_rows),
        skipped_count=len(skipped_rows),
    )

    skipped_path = target_root / "skipped_missing_clean_text.csv"
    if skipped_rows:
        write_metadata(skipped_path, list(skipped_rows[0].keys()), skipped_rows)

    print(
        json.dumps(
            {
                "target_root": str(target_root),
                "metadata_path": str(target_metadata_path),
                "source_label": source_label,
                "second_stage_class": second_stage_class,
                "issue_count": len(metadata_rows),
                "image_count": sum(int(row["raw_image_count"]) for row in metadata_rows),
                "skipped_missing_clean_transcription": len(skipped_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
