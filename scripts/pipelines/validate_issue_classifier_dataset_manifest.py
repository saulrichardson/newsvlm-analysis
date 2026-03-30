#!/usr/bin/env python3
"""
Validate an issue-classifier dataset manifest.

This checks that the manifest is internally consistent and that the referenced
Torch artifacts actually exist. It also validates OCR image provenance for the
supported text sources in the manifest:

  - paddle_vl15_md
  - gemini_box_results
  - openai_box_results

Outputs a machine-readable JSON report and prints a short summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ISSUE_PAGE_DELIMITER = "\n\n----- PAGE BREAK -----\n\n"


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise SystemExit(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(obj, dict):
                raise SystemExit(f"Expected object JSON in {path}:{line_number}")
            rows.append(obj)
    return rows


def _derive_box_mapping_path(result_jsonl_path: Path) -> Path:
    match = re.search(r"shard(\d+)\.jsonl$", result_jsonl_path.name)
    if not match:
        raise SystemExit(f"Could not derive mapping shard from {result_jsonl_path}")
    shard = match.group(1)
    return result_jsonl_path.parent.parent / f"mapping_shard{shard}.jsonl"


def _derive_paddle_res_json_path(md_path: Path) -> Path:
    return md_path.with_name(f"{md_path.stem}_res.json")


def _load_box_mapping_index(mapping_paths: set[Path]) -> dict[Path, dict[str, dict[str, str]]]:
    out: dict[Path, dict[str, dict[str, str]]] = {}
    for mapping_path in sorted(mapping_paths):
        if not mapping_path.is_file():
            raise SystemExit(f"Missing box mapping file: {mapping_path}")
        per_page: dict[str, dict[str, str]] = {}
        with mapping_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as exc:
                    raise SystemExit(f"Invalid JSON in {mapping_path}:{line_number}: {exc}") from exc
                page_id = _norm(obj.get("page_id")).lower()
                if not page_id:
                    continue
                record = {
                    "png_path": _norm(obj.get("png_path")),
                    "layout_path": _norm(obj.get("layout_path")),
                }
                existing = per_page.get(page_id)
                if existing is not None and existing != record:
                    raise SystemExit(
                        f"Conflicting mapping rows for page_id={page_id} in {mapping_path}: {existing} vs {record}"
                    )
                per_page[page_id] = record
        out[mapping_path] = per_page
    return out


def _validate_page_row(
    row: dict[str, Any],
    *,
    box_mapping_index: dict[Path, dict[str, dict[str, str]]],
) -> list[str]:
    errors: list[str] = []
    page_id = _norm(row.get("page_id")).lower()
    text_source = _norm(row.get("text_source"))
    png_path = Path(_norm(row.get("png_path")))
    materialized_page_text_path = Path(_norm(row.get("materialized_page_text_path")))
    original_transcript_source_path = Path(_norm(row.get("original_transcript_source_path")))
    ocr_input_png_path = Path(_norm(row.get("ocr_input_png_path"))) if _norm(row.get("ocr_input_png_path")) else None
    ocr_input_provenance_path = (
        Path(_norm(row.get("ocr_input_provenance_path"))) if _norm(row.get("ocr_input_provenance_path")) else None
    )
    ocr_input_mode = _norm(row.get("ocr_input_mode"))
    ocr_input_png_path_raw = _norm(row.get("ocr_input_png_path_raw"))
    ocr_input_provenance_type = _norm(row.get("ocr_input_provenance_type"))
    ocr_layout_path = _norm(row.get("ocr_layout_path"))

    if not materialized_page_text_path.is_file():
        errors.append(f"missing materialized_page_text_path: {materialized_page_text_path}")
    if not png_path.is_file():
        errors.append(f"missing png_path: {png_path}")
    if not original_transcript_source_path.is_file():
        errors.append(f"missing original_transcript_source_path: {original_transcript_source_path}")
    if ocr_input_png_path is None or not ocr_input_png_path.is_file():
        errors.append(f"missing ocr_input_png_path: {ocr_input_png_path}")
    if ocr_input_provenance_path is None or not ocr_input_provenance_path.is_file():
        errors.append(f"missing ocr_input_provenance_path: {ocr_input_provenance_path}")

    if text_source == "paddle_vl15_md":
        expected_prov = _derive_paddle_res_json_path(original_transcript_source_path)
        if ocr_input_mode != "full_page_png_direct":
            errors.append(f"unexpected ocr_input_mode for paddle: {ocr_input_mode}")
        if ocr_input_provenance_type != "paddle_res_json":
            errors.append(f"unexpected ocr_input_provenance_type for paddle: {ocr_input_provenance_type}")
        if ocr_input_provenance_path != expected_prov:
            errors.append(f"unexpected paddle provenance path: {ocr_input_provenance_path} != {expected_prov}")
        if expected_prov.is_file():
            res_obj = json.loads(expected_prov.read_text(encoding="utf-8", errors="replace"))
            input_path = _norm(res_obj.get("input_path"))
            if input_path != ocr_input_png_path_raw:
                errors.append(f"paddle raw input_path mismatch: {input_path} != {ocr_input_png_path_raw}")
        if ocr_layout_path:
            errors.append(f"expected blank ocr_layout_path for paddle, found: {ocr_layout_path}")
    elif text_source in {"gemini_box_results", "openai_box_results"}:
        expected_prov = _derive_box_mapping_path(original_transcript_source_path)
        if ocr_input_mode != "page_png_with_box_crops":
            errors.append(f"unexpected ocr_input_mode for box OCR: {ocr_input_mode}")
        if ocr_input_provenance_type != "box_mapping_jsonl":
            errors.append(f"unexpected ocr_input_provenance_type for box OCR: {ocr_input_provenance_type}")
        if ocr_input_provenance_path != expected_prov:
            errors.append(f"unexpected box provenance path: {ocr_input_provenance_path} != {expected_prov}")
        mapping_row = box_mapping_index.get(expected_prov, {}).get(page_id)
        if mapping_row is None:
            errors.append(f"missing page_id={page_id} in mapping file {expected_prov}")
        else:
            if _norm(mapping_row.get("png_path")) != ocr_input_png_path_raw:
                errors.append(
                    f"box raw png_path mismatch for page_id={page_id}: "
                    f"{mapping_row.get('png_path')} != {ocr_input_png_path_raw}"
                )
            if _norm(mapping_row.get("layout_path")) != ocr_layout_path:
                errors.append(
                    f"box layout_path mismatch for page_id={page_id}: "
                    f"{mapping_row.get('layout_path')} != {ocr_layout_path}"
                )
        if not ocr_layout_path:
            errors.append(f"missing ocr_layout_path for box OCR page_id={page_id}")
    else:
        errors.append(f"unsupported text_source: {text_source}")

    if ocr_input_png_path is not None and png_path.name != ocr_input_png_path.name:
        errors.append(f"png basename mismatch: {png_path.name} != {ocr_input_png_path.name}")
    if png_path.stem.lower() != page_id:
        errors.append(f"png_path stem mismatch: {png_path.stem} != {page_id}")
    if materialized_page_text_path.stem.split("__", 1)[-1].lower() != page_id:
        errors.append(f"page_text_path stem mismatch: {materialized_page_text_path} page_id={page_id}")

    return errors


def _manual_spot_check(
    issues: list[dict[str, Any]],
    *,
    sample_per_source: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    taken: Counter[str] = Counter()
    for issue in issues:
        for page in issue.get("pages") or []:
            source = _norm(page.get("text_source"))
            if taken[source] >= sample_per_source:
                continue
            page_text_path = Path(_norm(page.get("materialized_page_text_path")))
            issue_transcript_path = Path(_norm(issue.get("issue_transcript_path")))
            page_text = page_text_path.read_text(encoding="utf-8", errors="replace")
            issue_text = issue_transcript_path.read_text(encoding="utf-8", errors="replace")
            samples.append(
                {
                    "issue_id": _norm(issue.get("issue_id")),
                    "page_id": _norm(page.get("page_id")),
                    "text_source": source,
                    "png_path": _norm(page.get("png_path")),
                    "materialized_page_text_path": _norm(page.get("materialized_page_text_path")),
                    "ocr_input_mode": _norm(page.get("ocr_input_mode")),
                    "ocr_input_provenance_path": _norm(page.get("ocr_input_provenance_path")),
                    "page_text_excerpt": page_text[:800],
                    "issue_contains_page_text": page_text[:200] in issue_text,
                }
            )
            taken[source] += 1
    return samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True, help="Dataset manifest directory.")
    parser.add_argument("--report-json", type=Path, required=True, help="Output JSON validation report.")
    parser.add_argument("--sample-per-source", type=int, default=2, help="Manual sample count per text source.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_dir = args.manifest_dir.expanduser().resolve()
    report_json = args.report_json.expanduser().resolve()

    required_files = [
        manifest_dir / "summary.json",
        manifest_dir / "issues.jsonl",
        manifest_dir / "pages.jsonl",
        manifest_dir / "issues.csv",
        manifest_dir / "pages.csv",
        manifest_dir / "README.md",
        manifest_dir / "prompt_used.txt",
    ]
    required_dirs = [
        manifest_dir / "issue_transcripts",
        manifest_dir / "classification_records",
        manifest_dir / "page_texts",
    ]
    for path in required_files:
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")
    for path in required_dirs:
        if not path.is_dir():
            raise SystemExit(f"Missing required directory: {path}")

    summary = json.loads((manifest_dir / "summary.json").read_text(encoding="utf-8", errors="replace"))
    issues = _read_jsonl(manifest_dir / "issues.jsonl")
    pages = _read_jsonl(manifest_dir / "pages.jsonl")

    box_mapping_paths: set[Path] = set()
    for page in pages:
        text_source = _norm(page.get("text_source"))
        if text_source in {"gemini_box_results", "openai_box_results"}:
            box_mapping_paths.add(Path(_norm(page.get("ocr_input_provenance_path"))))
    box_mapping_index = _load_box_mapping_index(box_mapping_paths)

    issue_errors: list[dict[str, Any]] = []
    page_errors: list[dict[str, Any]] = []

    nested_page_rows = 0
    label_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    text_source_counts: Counter[str] = Counter()
    ocr_input_mode_counts: Counter[str] = Counter()

    seen_issue_ids: set[str] = set()
    seen_page_ids: set[str] = set()
    nested_page_keys: set[tuple[str, str]] = set()

    for issue in issues:
        issue_id = _norm(issue.get("issue_id"))
        if issue_id in seen_issue_ids:
            issue_errors.append({"issue_id": issue_id, "error": "duplicate issue_id"})
        seen_issue_ids.add(issue_id)

        issue_transcript_path = Path(_norm(issue.get("issue_transcript_path")))
        classification_record_path = Path(_norm(issue.get("classification_record_path")))
        pages_nested = issue.get("pages") or []
        if not issue_transcript_path.is_file():
            issue_errors.append({"issue_id": issue_id, "error": f"missing issue_transcript_path: {issue_transcript_path}"})
        if not classification_record_path.is_file():
            issue_errors.append({"issue_id": issue_id, "error": f"missing classification_record_path: {classification_record_path}"})
        if int(issue.get("page_count") or 0) != len(pages_nested):
            issue_errors.append(
                {"issue_id": issue_id, "error": f"page_count mismatch: {issue.get('page_count')} != {len(pages_nested)}"}
            )

        label = _norm((issue.get("predicted") or {}).get("label"))
        status = _norm(issue.get("classification_status"))
        label_counts[label] += 1
        status_counts[status] += 1

        page_texts: list[str] = []
        for page in pages_nested:
            page_id = _norm(page.get("page_id")).lower()
            nested_page_rows += 1
            nested_page_keys.add((issue_id, page_id))
            text_source = _norm(page.get("text_source"))
            text_source_counts[text_source] += 1
            ocr_input_mode_counts[_norm(page.get("ocr_input_mode"))] += 1
            page_text_path = Path(_norm(page.get("materialized_page_text_path")))
            if page_text_path.is_file():
                page_texts.append(page_text_path.read_text(encoding="utf-8", errors="replace"))

        if issue_transcript_path.is_file() and len(page_texts) == len(pages_nested):
            issue_text = issue_transcript_path.read_text(encoding="utf-8", errors="replace")
            expected = ISSUE_PAGE_DELIMITER.join(page_texts).strip()
            if issue_text != expected:
                issue_errors.append({"issue_id": issue_id, "error": "issue transcript does not match joined page texts"})

    for page in pages:
        page_id = _norm(page.get("page_id")).lower()
        if page_id in seen_page_ids:
            page_errors.append({"page_id": page_id, "error": "duplicate page_id in pages.jsonl"})
        seen_page_ids.add(page_id)
        page_issue_id = _norm(page.get("issue_id"))
        if (page_issue_id, page_id) not in nested_page_keys:
            page_errors.append({"page_id": page_id, "issue_id": page_issue_id, "error": "page row missing from nested issue pages"})
        errors = _validate_page_row(page, box_mapping_index=box_mapping_index)
        for error in errors:
            page_errors.append({"page_id": page_id, "issue_id": page_issue_id, "error": error})

    summary_checks = {
        "summary_issue_count_matches": int(summary.get("issue_count") or -1) == len(issues),
        "summary_page_count_matches": int(summary.get("page_count") or -1) == len(pages),
        "summary_label_counts_matches": dict(sorted(summary.get("label_counts", {}).items())) == dict(sorted(label_counts.items())),
        "summary_status_counts_matches": dict(sorted(summary.get("status_counts", {}).items())) == dict(sorted(status_counts.items())),
    }

    issue_transcript_files = sum(1 for _ in (manifest_dir / "issue_transcripts").glob("*.txt"))
    classification_record_files = sum(1 for _ in (manifest_dir / "classification_records").glob("*.json"))
    page_text_files = sum(1 for _ in (manifest_dir / "page_texts").rglob("*.txt"))

    report = {
        "manifest_dir": str(manifest_dir),
        "counts": {
            "issues_jsonl_rows": len(issues),
            "pages_jsonl_rows": len(pages),
            "nested_page_rows": nested_page_rows,
            "issue_transcript_files": issue_transcript_files,
            "classification_record_files": classification_record_files,
            "page_text_files": page_text_files,
        },
        "summary_checks": summary_checks,
        "text_source_counts": dict(sorted(text_source_counts.items())),
        "ocr_input_mode_counts": dict(sorted(ocr_input_mode_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "issue_errors_count": len(issue_errors),
        "page_errors_count": len(page_errors),
        "issue_errors_sample": issue_errors[:50],
        "page_errors_sample": page_errors[:50],
        "manual_samples": _manual_spot_check(issues, sample_per_source=args.sample_per_source),
        "ok": not issue_errors and not page_errors and all(summary_checks.values()),
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"validated manifest ok={report['ok']} issues={len(issues)} pages={len(pages)} "
        f"issue_errors={len(issue_errors)} page_errors={len(page_errors)} report={report_json}"
    )


if __name__ == "__main__":
    main()
