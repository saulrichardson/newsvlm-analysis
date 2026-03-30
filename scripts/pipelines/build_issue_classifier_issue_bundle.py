#!/usr/bin/env python3
"""
Build a portable, issue-organized classifier bundle with relative paths.

Output layout:

  <out_dir>/
    README.md
    summary.json
    metadata.csv
    issues/<issue_id>/
      issue.json
      issue_transcript.txt
      pages/<page_num>__<page_id>.txt
      images/<page_num>__<page_id>.png

The bundle is model-output only. It does not include manual review state.
All bundle-local paths in metadata are relative to <out_dir>.

Geography/Census enrichment is sourced from a city-level lookup CSV (by default the
historical v2 metadata file), but the output schema is forward-looking and not tied
to the legacy metadata layout.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_issue_classifier_dataset_manifest as manifest_builder


DEFAULT_GEO_SOURCE_CSV = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/newspapers/archive/v2/metadata.csv"
)
ISSUE_PAGE_DELIMITER = "\n\n----- PAGE BREAK -----\n\n"
SCHEMA_VERSION = "issue_classifier_issue_bundle_v1"
GEO_SOURCE_FIELDS = [
    "city_name",
    "state_abbr",
    "city_key",
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
]
METADATA_FIELDS = [
    "issue_id",
    "issue_date",
    "newspaper_slug",
    "classification_status",
    "classification_model",
    "document_class",
    "document_operativity",
    "document_primary_class",
    "document_confidence_0_to_1",
    "document_legal_action",
    "document_scope",
    "document_publication_completeness",
    "document_evidence_quote",
    "zoning_presence_evidence_quote",
    "jurisdiction_city_name",
    "jurisdiction_state_abbr",
    "jurisdiction_level",
    "jurisdiction_evidence_quote",
    "city_key",
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
    "page_count",
    "quality_flags_json",
    "issue_dir_rel",
    "issue_json_relpath",
    "issue_transcript_relpath",
]


def _norm(value: Any) -> str:
    return str(value or "").strip()



def _norm_empty(value: Any) -> str:
    s = _norm(value)
    return "" if s.lower() in {"nan", "none", "null"} else s



def _norm_numeric_like_id(value: Any, *, width: int | None = None) -> str:
    s = _norm_empty(value)
    if not s:
        return ""
    s = re.sub(r"\.0+$", "", s)
    if width and s.isdigit():
        s = s.zfill(width)
    return s



def _slug(text: str) -> str:
    s = _norm_empty(text).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unknown"



def _city_key(city_name: str, state_abbr: str) -> str:
    city = _norm_empty(city_name)
    state = _norm_empty(state_abbr).lower()
    if not city or not state:
        return ""
    return f"{_slug(city)}__{state}"



def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()



def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
            yield payload



def _safe_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text))
    out = out.strip("_")
    return out or "item"



def _safe_page_text_filename(page_num: int, page_id: str) -> str:
    return f"{int(page_num):04d}__{_safe_name(page_id)}.txt"



def _safe_page_image_filename(page_num: int, page_id: str, source_path: Path) -> str:
    suffix = source_path.suffix or ".png"
    return f"{int(page_num):04d}__{_safe_name(page_id)}{suffix}"



def _selected_rows(
    parsed_jsonl: Path,
    *,
    include_labels: set[str],
    require_status_ok: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_jsonl(parsed_jsonl):
        status = _norm(row.get("status"))
        if require_status_ok and status.lower() != "ok":
            continue
        parsed = row.get("parsed") or {}
        if not isinstance(parsed, dict):
            continue
        legal_document = parsed.get("legal_document") or {}
        if not isinstance(legal_document, dict):
            continue
        label = _norm(legal_document.get("class"))
        if include_labels and label not in include_labels:
            continue
        rows.append(row)
    if not rows:
        raise SystemExit("No rows selected from parsed JSONL")
    return rows



def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise SystemExit(f"Missing file to copy: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)



def _load_geo_lookup(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"--geo-source-csv not found: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"--geo-source-csv is empty: {path}")

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise SystemExit(f"--geo-source-csv has no header: {path}")
        if "city_key" not in fieldnames and not {"city_name", "state_abbr"}.issubset(set(fieldnames)):
            raise SystemExit(
                f"--geo-source-csv must contain city_key or city_name/state_abbr columns: {path}"
            )

        out: dict[str, dict[str, str]] = {}
        for raw in reader:
            row = {k: _norm_empty(v) for k, v in raw.items()}
            city_key = _norm_empty(row.get("city_key")) or _city_key(row.get("city_name", ""), row.get("state_abbr", ""))
            city_key = city_key.lower()
            if not city_key:
                continue
            normalized = {
                "city_name": _norm_empty(row.get("city_name")),
                "state_abbr": _norm_empty(row.get("state_abbr")).upper(),
                "city_key": city_key,
                "place20_geoid": _norm_numeric_like_id(row.get("place20_geoid"), width=7),
                "county20_fips_primary": _norm_numeric_like_id(row.get("county20_fips_primary"), width=5),
                "puma20_geoid_primary": _norm_numeric_like_id(row.get("puma20_geoid_primary"), width=7),
                "cbsa23_geoid_primary": _norm_numeric_like_id(row.get("cbsa23_geoid_primary"), width=5),
                "cbsa23_name_primary": _norm_empty(row.get("cbsa23_name_primary")),
                "census_id_pid6": _norm_numeric_like_id(row.get("census_id_pid6")),
                "census_id_gidid": _norm_numeric_like_id(row.get("census_id_gidid")),
                "census_unit_name": _norm_empty(row.get("census_unit_name")),
                "census_unit_type": _norm_empty(row.get("census_unit_type")),
                "census_join_method": _norm_empty(row.get("census_join_method")),
            }
            existing = out.get(city_key)
            if existing is None:
                out[city_key] = normalized
                continue
            for field in GEO_SOURCE_FIELDS:
                a = _norm_empty(existing.get(field))
                b = _norm_empty(normalized.get(field))
                if a and b and a != b:
                    raise SystemExit(
                        f"Conflicting geo-source values for city_key={city_key} field={field}: {a!r} vs {b!r}"
                    )
                if not a and b:
                    existing[field] = b
    if not out:
        raise SystemExit(f"--geo-source-csv yielded no usable city rows: {path}")
    return out



def _build_issue_json_payload(
    *,
    bundle_built_at: str,
    schema_version: str,
    issue_id: str,
    issue_date: str,
    newspaper_slug: str,
    classifier_model: str,
    classification_status: str,
    parsed: dict[str, Any],
    metadata_row: dict[str, Any],
    issue_transcript_relpath: str,
    pages_payload: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "bundle_built_at": bundle_built_at,
        "issue_id": issue_id,
        "issue_date": issue_date,
        "newspaper_slug": newspaper_slug,
        "classification": {
            "status": classification_status,
            "model": classifier_model,
            "zoning_presence": parsed.get("zoning_presence") or {},
            "legal_document": parsed.get("legal_document") or {},
            "ordinance_jurisdiction": parsed.get("ordinance_jurisdiction") or {},
            "code_components_present": parsed.get("code_components_present") or {},
            "quality_flags": parsed.get("quality_flags") if isinstance(parsed.get("quality_flags"), list) else [],
            "rationale": _norm(parsed.get("rationale")),
        },
        "metadata": metadata_row,
        "issue_transcript_relpath": issue_transcript_relpath,
        "pages": pages_payload,
    }



def _write_readme(out_dir: Path) -> None:
    text = """# Issue Classifier Issue Bundle

This directory is a portable, issue-organized bundle derived from parsed issue-classifier outputs.

## Primary entrypoint

- `metadata.csv`: one row per issue, including classification fields, jurisdiction fields, Census IDs, and relative paths into this bundle.

## Layout

- `issues/<issue_id>/issue.json`: issue-level JSON record with model output and ordered page inventory.
- `issues/<issue_id>/issue_transcript.txt`: full issue transcript built by concatenating page transcripts with `----- PAGE BREAK -----`.
- `issues/<issue_id>/pages/*.txt`: one transcript file per page.
- `issues/<issue_id>/images/*.png`: one copied page image per page.

## Path policy

All `*_relpath` fields in `metadata.csv` and `issue.json` are relative to the bundle root.

## External provenance

Some page records include `*_external_path` fields. Those point back to the original Torch/local artifacts used to produce the bundle and are not required to consume the copied bundle contents.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")



def _parse_labels(raw: str) -> set[str]:
    labels = {_norm(part) for part in raw.split(",") if _norm(part)}
    return labels



def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed-jsonl", type=Path, required=True, help="Parsed classifier outputs JSONL.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Bundle output directory.")
    ap.add_argument("--unique-png-root", type=Path, required=True, help="Canonical unique_png root.")
    ap.add_argument(
        "--geo-source-csv",
        type=Path,
        default=DEFAULT_GEO_SOURCE_CSV,
        help="City-level geography/Census lookup CSV (default: historical v2 metadata.csv).",
    )
    ap.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated legal_document.class labels to include. Default: include all status-ok rows.",
    )
    ap.add_argument(
        "--allow-non-ok-status",
        action="store_true",
        help="Include non-ok classification rows. Default is status=ok only.",
    )
    return ap.parse_args()



def main() -> None:
    args = _parse_args()
    parsed_jsonl = args.parsed_jsonl.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    unique_png_root = args.unique_png_root.expanduser().resolve()
    geo_source_csv = args.geo_source_csv.expanduser().resolve()
    include_labels = _parse_labels(args.labels)
    require_status_ok = not bool(args.allow_non_ok_status)

    if not parsed_jsonl.is_file():
        raise SystemExit(f"--parsed-jsonl not found: {parsed_jsonl}")
    if not unique_png_root.is_dir():
        raise SystemExit(f"--unique-png-root is not a directory: {unique_png_root}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir must be empty or absent: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _selected_rows(parsed_jsonl, include_labels=include_labels, require_status_ok=require_status_ok)
    geo_lookup = _load_geo_lookup(geo_source_csv)

    box_page_ids_by_source, direct_page_refs = manifest_builder._collect_source_refs(rows)
    png_by_page_id = manifest_builder._build_png_index(unique_png_root)

    metadata_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    unmatched_geo_count = 0
    bundle_built_at = dt.datetime.now(dt.timezone.utc).isoformat()

    with tempfile.TemporaryDirectory(prefix="issue_bundle_stage_", dir=str(out_dir.parent)) as tmp_dir_name:
        stage_dir = Path(tmp_dir_name)
        page_plan_by_key = manifest_builder._plan_page_records(rows=rows, out_dir=stage_dir, png_by_page_id=png_by_page_id)
        manifest_builder._enrich_page_plans_with_png_provenance(page_plan_by_key=page_plan_by_key, png_by_page_id=png_by_page_id)
        page_rows_by_key = manifest_builder._materialize_page_texts(
            page_plan_by_key=page_plan_by_key,
            box_page_ids_by_source=box_page_ids_by_source,
            direct_page_refs=direct_page_refs,
        )

        issues_root = out_dir / "issues"
        issues_root.mkdir(parents=True, exist_ok=True)

        for row in rows:
            mapping = row.get("mapping") or {}
            parsed = row.get("parsed") or {}
            legal_document = parsed.get("legal_document") or {}
            zoning_presence = parsed.get("zoning_presence") or {}
            ordinance_jurisdiction = parsed.get("ordinance_jurisdiction") or {}

            issue_id = _norm(row.get("issue_id") or mapping.get("issue_id"))
            issue_date = _norm(row.get("issue_date") or mapping.get("issue_date"))
            newspaper_slug = _norm(row.get("slug") or mapping.get("slug")).lower()
            classifier_model = _norm(row.get("classifier_model"))
            classification_status = _norm(row.get("status"))

            issue_dir_rel = Path("issues") / issue_id
            issue_dir = out_dir / issue_dir_rel
            pages_dir = issue_dir / "pages"
            images_dir = issue_dir / "images"
            issue_dir.mkdir(parents=True, exist_ok=True)
            pages_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)

            pages_payload: list[dict[str, Any]] = []
            page_texts: list[str] = []
            source_pages = mapping.get("source_pages") or []
            if not isinstance(source_pages, list) or not source_pages:
                raise SystemExit(f"issue_id={issue_id} has no source_pages")
            for page in sorted(source_pages, key=lambda item: int(item.get("page_num") or 0)):
                page_id = _norm(page.get("page_id")).lower()
                page_num = int(page.get("page_num") or 0)
                text_source = _norm(page.get("text_source"))
                source_path = _norm(page.get("source_path"))
                page_key = f"{text_source}::{source_path}::{page_id}"
                page_row = page_rows_by_key.get(page_key)
                if page_row is None:
                    raise SystemExit(f"Missing materialized page row for issue_id={issue_id} page_id={page_id}")

                staged_text_path = Path(page_row["materialized_page_text_path"])
                page_text = staged_text_path.read_text(encoding="utf-8", errors="replace")
                page_texts.append(page_text)
                page_text_filename = _safe_page_text_filename(page_num, page_id)
                page_text_relpath = issue_dir_rel / "pages" / page_text_filename
                page_text_dst = out_dir / page_text_relpath
                page_text_dst.write_text(page_text, encoding="utf-8")

                image_src = Path(page_row["png_path"])
                image_filename = _safe_page_image_filename(page_num, page_id, image_src)
                image_relpath = issue_dir_rel / "images" / image_filename
                image_dst = out_dir / image_relpath
                _copy_file(image_src, image_dst)

                source_counts[text_source] += 1
                pages_payload.append(
                    {
                        "page_id": page_id,
                        "page_num": page_num,
                        "text_source": text_source,
                        "page_text_relpath": page_text_relpath.as_posix(),
                        "image_relpath": image_relpath.as_posix(),
                        "page_text_chars": len(page_text),
                        "page_text_sha256": _sha256_text(page_text),
                        "source_text_chars": int(page_row.get("source_text_chars") or 0),
                        "source_text_sha256": _norm(page_row.get("source_text_sha256")),
                        "source_text_origin_external_path": _norm(page_row.get("original_transcript_source_path")),
                        "ocr_input_mode": _norm(page_row.get("ocr_input_mode")),
                        "ocr_input_png_external_path": _norm(page_row.get("ocr_input_png_path")),
                        "ocr_input_provenance_external_path": _norm(page_row.get("ocr_input_provenance_path")),
                        "ocr_input_provenance_type": _norm(page_row.get("ocr_input_provenance_type")),
                        "ocr_layout_external_path": _norm(page_row.get("ocr_layout_path")),
                    }
                )

            issue_transcript = ISSUE_PAGE_DELIMITER.join(page_texts).strip()
            issue_transcript_relpath = issue_dir_rel / "issue_transcript.txt"
            (out_dir / issue_transcript_relpath).write_text(issue_transcript, encoding="utf-8")

            city_name = _norm_empty(ordinance_jurisdiction.get("city"))
            state_abbr = _norm_empty(ordinance_jurisdiction.get("state_abbr")).upper()
            city_key = _city_key(city_name, state_abbr)
            geo = geo_lookup.get(city_key, {}) if city_key else {}
            if city_key and not geo:
                unmatched_geo_count += 1

            metadata_row = {
                "issue_id": issue_id,
                "issue_date": issue_date,
                "newspaper_slug": newspaper_slug,
                "classification_status": classification_status,
                "classification_model": classifier_model,
                "document_class": _norm(legal_document.get("class")),
                "document_operativity": _norm(legal_document.get("operativity")),
                "document_primary_class": _norm(zoning_presence.get("primary_class")),
                "document_confidence_0_to_1": legal_document.get("confidence_0_to_1", ""),
                "document_legal_action": _norm(legal_document.get("legal_action")),
                "document_scope": _norm(legal_document.get("scope")),
                "document_publication_completeness": _norm(legal_document.get("publication_completeness")),
                "document_evidence_quote": _norm(legal_document.get("evidence_quote")),
                "zoning_presence_evidence_quote": _norm(zoning_presence.get("evidence_quote")),
                "jurisdiction_city_name": city_name,
                "jurisdiction_state_abbr": state_abbr,
                "jurisdiction_level": _norm(ordinance_jurisdiction.get("jurisdiction_level")),
                "jurisdiction_evidence_quote": _norm(ordinance_jurisdiction.get("evidence_quote")),
                "city_key": city_key,
                "place20_geoid": _norm_empty(geo.get("place20_geoid")),
                "county20_fips_primary": _norm_empty(geo.get("county20_fips_primary")),
                "puma20_geoid_primary": _norm_empty(geo.get("puma20_geoid_primary")),
                "cbsa23_geoid_primary": _norm_empty(geo.get("cbsa23_geoid_primary")),
                "cbsa23_name_primary": _norm_empty(geo.get("cbsa23_name_primary")),
                "census_id_pid6": _norm_empty(geo.get("census_id_pid6")),
                "census_id_gidid": _norm_empty(geo.get("census_id_gidid")),
                "census_unit_name": _norm_empty(geo.get("census_unit_name")),
                "census_unit_type": _norm_empty(geo.get("census_unit_type")),
                "census_join_method": _norm_empty(geo.get("census_join_method")) if geo else ("missing_jurisdiction" if not city_key else "unmatched_city_key"),
                "page_count": len(pages_payload),
                "quality_flags_json": json.dumps(parsed.get("quality_flags") if isinstance(parsed.get("quality_flags"), list) else [], ensure_ascii=False),
                "issue_dir_rel": issue_dir_rel.as_posix(),
                "issue_json_relpath": (issue_dir_rel / "issue.json").as_posix(),
                "issue_transcript_relpath": issue_transcript_relpath.as_posix(),
            }

            issue_payload = _build_issue_json_payload(
                bundle_built_at=bundle_built_at,
                schema_version=SCHEMA_VERSION,
                issue_id=issue_id,
                issue_date=issue_date,
                newspaper_slug=newspaper_slug,
                classifier_model=classifier_model,
                classification_status=classification_status,
                parsed=parsed,
                metadata_row=metadata_row,
                issue_transcript_relpath=issue_transcript_relpath.as_posix(),
                pages_payload=pages_payload,
            )
            (issue_dir / "issue.json").write_text(json.dumps(issue_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            metadata_rows.append(metadata_row)
            label_counts[metadata_row["document_class"]] += 1

    metadata_rows = sorted(metadata_rows, key=lambda r: (r["issue_date"], r["issue_id"]))
    with (out_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow(row)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "bundle_built_at": bundle_built_at,
        "source_parsed_jsonl": str(parsed_jsonl),
        "geo_source_csv": str(geo_source_csv),
        "issue_count": len(metadata_rows),
        "page_count": int(sum(int(r["page_count"]) for r in metadata_rows)),
        "label_counts": dict(sorted(label_counts.items())),
        "text_source_counts": dict(sorted(source_counts.items())),
        "issues_with_city_key": int(sum(1 for r in metadata_rows if _norm(r["city_key"]))),
        "issues_with_place20_geoid": int(sum(1 for r in metadata_rows if _norm(r["place20_geoid"]))),
        "issues_with_census_pid6": int(sum(1 for r in metadata_rows if _norm(r["census_id_pid6"]))),
        "unmatched_geo_issue_count": unmatched_geo_count,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
