#!/usr/bin/env python3
"""
Build an issue-level dataset manifest for classifier outputs.

This materializes a self-describing bundle that is easy to consume on Torch:

  <out_dir>/
    README.md
    summary.json
    prompt_used.txt
    issues.jsonl
    pages.jsonl
    issue_transcripts/<issue_id>.txt
    classification_records/<issue_id>.json
    page_texts/<issue_id>/<page_num>__<page_id>.txt

`issues.jsonl` is the primary dataset. Each row includes:
  - issue identifiers
  - classifier result summary
  - absolute Torch paths to the materialized issue transcript
  - absolute Torch paths to the per-issue classification JSON
  - nested page metadata with:
      * original transcript source path
      * materialized per-page transcript path
      * canonical PNG path

This script requires parsed classifier outputs where each row contains
`mapping.source_pages`. It intentionally fails for `issue_txt_dir` mode because
that mode does not preserve page-level provenance.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import export_issue_zoning_issue_classifier_batch_requests as exporter


ISSUE_PAGE_DELIMITER = "\n\n----- PAGE BREAK -----\n\n"


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _sha256(text: str) -> str:
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


def _parse_csv_values(raw: str) -> set[str]:
    return {_norm(part).lower() for part in str(raw or "").split(",") if _norm(part)}


def _selected_rows(
    path: Path,
    *,
    zoning_presence_primary_classes: set[str],
    legal_document_classes: set[str],
    exclude_operativities: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        parsed = row.get("parsed")
        if not isinstance(parsed, dict):
            continue
        zoning_presence = parsed.get("zoning_presence")
        if not isinstance(zoning_presence, dict):
            continue
        legal_document = parsed.get("legal_document")
        if not isinstance(legal_document, dict):
            continue
        if zoning_presence_primary_classes:
            primary_class = _norm(zoning_presence.get("primary_class")).lower()
            if primary_class not in zoning_presence_primary_classes:
                continue
        if legal_document_classes:
            legal_class = _norm(legal_document.get("class")).lower()
            if legal_class not in legal_document_classes:
                continue
        if exclude_operativities:
            operativity = _norm(legal_document.get("operativity")).lower()
            if operativity in exclude_operativities:
                continue
        rows.append(row)
    return rows


def _safe_issue_filename(issue_id: str) -> str:
    return exporter._safe_issue_filename(issue_id)


def _safe_page_filename(page_num: int, page_id: str) -> str:
    return f"{int(page_num):04d}__{_safe_issue_filename(page_id)}.txt"


def _load_direct_page_text(path: Path, *, source: str) -> str:
    if source == "paddle_vl15_md":
        return path.read_text(encoding="utf-8", errors="replace")
    if source == "vlm_page":
        return exporter._extract_clean_text_from_vlm_page(path)
    raise SystemExit(f"Unsupported direct source type: {source}")


def _build_dataset_issue_transcript(
    pages: list[exporter.PageText],
    *,
    max_issue_chars: int,
    issue_id: str,
) -> str:
    text = ISSUE_PAGE_DELIMITER.join(page.text for page in pages).strip()
    return exporter._bounded_text_or_fail(text, max_issue_chars, label=f"issue_id={issue_id} issue_transcript")


def _collect_source_refs(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, str]]]:
    box_page_ids_by_source: dict[str, dict[str, set[str]]] = {
        "gemini_box_results": defaultdict(set),
        "openai_box_results": defaultdict(set),
    }
    direct_page_refs: dict[str, dict[str, str]] = {}

    for row in rows:
        mapping = row.get("mapping") or {}
        issue_id = _norm(row.get("issue_id") or mapping.get("issue_id"))
        source_mode = _norm(mapping.get("source_mode"))
        if source_mode == "issue_txt_dir":
            raise SystemExit(
                f"issue_id={issue_id} came from issue_txt_dir mode; page-level provenance is unavailable"
            )
        source_pages = mapping.get("source_pages")
        if not isinstance(source_pages, list) or not source_pages:
            raise SystemExit(f"issue_id={issue_id} has no source_pages")

        for page in source_pages:
            if not isinstance(page, dict):
                raise SystemExit(f"issue_id={issue_id} has malformed source_pages entry: {page!r}")
            page_id = _norm(page.get("page_id")).lower()
            text_source = _norm(page.get("text_source"))
            source_path = _norm(page.get("source_path"))
            if not page_id or not text_source or not source_path:
                raise SystemExit(f"issue_id={issue_id} has incomplete page provenance: {page!r}")
            source_key = f"{text_source}::{source_path}::{page_id}"
            if text_source in {"paddle_vl15_md", "vlm_page"}:
                direct_page_refs[source_key] = {
                    "page_id": page_id,
                    "text_source": text_source,
                    "source_path": source_path,
                }
            elif text_source in {"gemini_box_results", "openai_box_results"}:
                box_page_ids_by_source[text_source][source_path].add(page_id)
            else:
                raise SystemExit(f"Unsupported text_source={text_source} for issue_id={issue_id}")

    return box_page_ids_by_source, direct_page_refs


def _load_page_text_lookup(
    *,
    box_page_ids_by_source: dict[str, dict[str, set[str]]],
    direct_page_refs: dict[str, dict[str, str]],
) -> dict[str, str]:
    page_text_by_key: dict[str, str] = {}

    for source_key, payload in sorted(direct_page_refs.items()):
        source = payload["text_source"]
        source_path = Path(payload["source_path"]).expanduser()
        if not source_path.is_file():
            raise SystemExit(f"Missing direct source file: {source_path}")
        text = _load_direct_page_text(source_path, source=source).strip()
        if not text:
            raise SystemExit(f"Empty text loaded from {source_path} ({source})")
        page_text_by_key[source_key] = text

    loader_by_source = {
        "gemini_box_results": exporter._load_gemini_box_results_jsonl,
        "openai_box_results": exporter._load_openai_box_results_jsonl,
    }
    for source_name, page_ids_by_path in box_page_ids_by_source.items():
        loader = loader_by_source[source_name]
        for source_path, page_ids in sorted(page_ids_by_path.items()):
            path = Path(source_path).expanduser()
            if not path.is_file():
                raise SystemExit(f"Missing {source_name} shard file: {path}")
            loaded = loader([path], wanted_page_ids=set(page_ids))
            for page_id in sorted(page_ids):
                page_key = f"{source_name}::{source_path}::{page_id}"
                page_text = loaded.get(page_id)
                if page_text is None or not page_text.text.strip():
                    raise SystemExit(f"Missing page_id={page_id} in {source_name} shard {path}")
                page_text_by_key[page_key] = page_text.text

    return page_text_by_key


def _build_png_index(unique_png_root: Path) -> dict[str, str]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    for dirpath, _dirnames, filenames in os.walk(unique_png_root):
        for name in filenames:
            if not name.endswith(".png"):
                continue
            page_id = name[: -len(".png")].strip().lower()
            if not page_id:
                continue
            candidates[page_id].append(Path(dirpath) / name)

    out: dict[str, str] = {}
    duplicate_stems = 0
    duplicate_ignored = 0
    duplicate_conflicts = 0
    for page_id, paths in sorted(candidates.items()):
        if len(paths) > 1:
            duplicate_stems += 1
            duplicate_ignored += len(paths) - 1
            sizes = {p.stat().st_size for p in paths}
            if len(sizes) > 1:
                duplicate_conflicts += 1
        best = sorted(paths, key=lambda p: (-p.stat().st_size, str(p)))[0]
        out[page_id] = str(best)
    if not out:
        raise SystemExit(f"No PNG files found under --unique-png-root={unique_png_root}")
    if duplicate_stems:
        print(
            f"warn: unique_png_root duplicate stems detected: stems={duplicate_stems} ignored_files={duplicate_ignored} "
            f"conflicting_sizes={duplicate_conflicts}; selected largest file per page_id",
            file=sys.stderr,
        )
    return out


def _resolve_png_path_from_raw(raw_path: str, *, page_id: str, png_by_page_id: dict[str, str]) -> str:
    raw = _norm(raw_path)
    if raw:
        raw_file = Path(raw)
        if raw_file.is_file():
            return str(raw_file)
        canonical = png_by_page_id.get(page_id, "")
        if canonical and Path(canonical).name == raw_file.name:
            return canonical
    canonical = png_by_page_id.get(page_id, "")
    if canonical:
        return canonical
    raise SystemExit(f"Could not resolve page PNG for page_id={page_id} raw_path={raw_path!r}")


def _derive_paddle_res_json_path(md_path: Path) -> Path:
    return md_path.with_name(f"{md_path.stem}_res.json")


def _derive_box_mapping_path(*, text_source: str, result_jsonl_path: Path) -> Path:
    name = result_jsonl_path.name
    match = re.search(r"shard(\d+)\.jsonl$", name)
    if not match:
        raise SystemExit(f"Could not derive mapping shard from {result_jsonl_path}")
    shard = match.group(1)
    return result_jsonl_path.parent.parent / f"mapping_shard{shard}.jsonl"


def _load_box_mapping_rows(mapping_path: Path, *, wanted_page_ids: set[str]) -> dict[str, dict[str, str]]:
    if not mapping_path.is_file():
        raise SystemExit(f"Missing box mapping file: {mapping_path}")
    out: dict[str, dict[str, str]] = {}
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
            if page_id not in wanted_page_ids:
                continue
            raw_png_path = _norm(obj.get("png_path"))
            layout_path = _norm(obj.get("layout_path"))
            existing = out.get(page_id)
            current = {
                "ocr_input_png_path_raw": raw_png_path,
                "ocr_layout_path": layout_path,
            }
            if existing is not None and existing != current:
                raise SystemExit(
                    f"Conflicting mapping rows for page_id={page_id} in {mapping_path}: {existing} vs {current}"
                )
            out[page_id] = current
    missing = sorted(wanted_page_ids - set(out))
    if missing:
        raise SystemExit(f"Missing page_ids in {mapping_path}: {missing[:5]}")
    return out


def _enrich_page_plans_with_png_provenance(
    *,
    page_plan_by_key: dict[str, dict[str, Any]],
    png_by_page_id: dict[str, str],
) -> None:
    box_wanted_by_mapping: dict[Path, set[str]] = defaultdict(set)
    box_keys_by_mapping: dict[tuple[Path, str], list[str]] = defaultdict(list)

    for page_key, page_plan in page_plan_by_key.items():
        text_source = _norm(page_plan.get("text_source"))
        page_id = _norm(page_plan.get("page_id")).lower()
        source_path = Path(_norm(page_plan.get("original_transcript_source_path"))).expanduser()

        if text_source == "paddle_vl15_md":
            res_json_path = _derive_paddle_res_json_path(source_path)
            if not res_json_path.is_file():
                raise SystemExit(f"Missing Paddle res JSON for page_id={page_id}: {res_json_path}")
            try:
                res_obj = json.loads(res_json_path.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                raise SystemExit(f"Invalid JSON in Paddle res file {res_json_path}: {exc}") from exc
            raw_png_path = _norm(res_obj.get("input_path"))
            if not raw_png_path:
                raise SystemExit(f"Missing input_path in Paddle res file {res_json_path}")
            page_plan["ocr_input_mode"] = "full_page_png_direct"
            page_plan["ocr_input_png_path_raw"] = raw_png_path
            page_plan["ocr_input_png_path"] = _resolve_png_path_from_raw(
                raw_png_path,
                page_id=page_id,
                png_by_page_id=png_by_page_id,
            )
            page_plan["ocr_input_provenance_path"] = str(res_json_path)
            page_plan["ocr_input_provenance_type"] = "paddle_res_json"
            page_plan["ocr_layout_path"] = ""
        elif text_source in {"gemini_box_results", "openai_box_results"}:
            mapping_path = _derive_box_mapping_path(text_source=text_source, result_jsonl_path=source_path)
            box_wanted_by_mapping[mapping_path].add(page_id)
            box_keys_by_mapping[(mapping_path, page_id)].append(page_key)
        elif text_source == "vlm_page":
            page_plan["ocr_input_mode"] = "full_page_png_direct"
            page_plan["ocr_input_png_path_raw"] = png_by_page_id[page_id]
            page_plan["ocr_input_png_path"] = png_by_page_id[page_id]
            page_plan["ocr_input_provenance_path"] = ""
            page_plan["ocr_input_provenance_type"] = "canonical_png_fallback"
            page_plan["ocr_layout_path"] = ""
        else:
            raise SystemExit(f"Unsupported text_source for PNG provenance: {text_source}")

    for mapping_path, wanted_page_ids in sorted(box_wanted_by_mapping.items()):
        loaded = _load_box_mapping_rows(mapping_path, wanted_page_ids=wanted_page_ids)
        for page_id, values in loaded.items():
            for page_key in box_keys_by_mapping[(mapping_path, page_id)]:
                page_plan = page_plan_by_key[page_key]
                page_plan["ocr_input_mode"] = "page_png_with_box_crops"
                page_plan["ocr_input_png_path_raw"] = values["ocr_input_png_path_raw"]
                page_plan["ocr_input_png_path"] = _resolve_png_path_from_raw(
                    values["ocr_input_png_path_raw"],
                    page_id=page_id,
                    png_by_page_id=png_by_page_id,
                )
                page_plan["ocr_input_provenance_path"] = str(mapping_path)
                page_plan["ocr_input_provenance_type"] = "box_mapping_jsonl"
                page_plan["ocr_layout_path"] = values["ocr_layout_path"]


def _plan_page_records(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    png_by_page_id: dict[str, str],
 ) -> dict[str, dict[str, Any]]:
    page_text_root = out_dir / "page_texts"
    page_text_root.mkdir(parents=True, exist_ok=True)

    page_plan_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        mapping = row.get("mapping") or {}
        issue_id = _norm(row.get("issue_id") or mapping.get("issue_id"))
        issue_date = _norm(row.get("issue_date") or mapping.get("issue_date"))
        slug = _norm(row.get("slug") or mapping.get("slug")).lower()
        source_pages = mapping.get("source_pages") or []
        page_text_dir = page_text_root / _safe_issue_filename(issue_id)
        page_text_dir.mkdir(parents=True, exist_ok=True)
        for page in sorted(source_pages, key=lambda item: int(item.get("page_num") or 0)):
            page_id = _norm(page.get("page_id")).lower()
            page_num = int(page.get("page_num") or 0)
            text_source = _norm(page.get("text_source"))
            source_path = _norm(page.get("source_path"))
            page_key = f"{text_source}::{source_path}::{page_id}"
            materialized_page_path = page_text_dir / _safe_page_filename(page_num, page_id)
            png_path = png_by_page_id.get(page_id)
            if not png_path:
                raise SystemExit(f"Missing canonical PNG path for issue_id={issue_id} page_id={page_id}")

            page_plan = {
                "issue_id": issue_id,
                "issue_date": issue_date,
                "newspaper_slug": slug,
                "page_id": page_id,
                "page_num": page_num,
                "text_source": text_source,
                "original_transcript_source_path": source_path,
                "materialized_page_text_path": str(materialized_page_path),
                "materialized_page_text_sha256": "",
                "materialized_page_text_chars": 0,
                "source_text_sha256": _norm(page.get("text_sha256")),
                "source_text_chars": int(page.get("text_chars") or 0),
                "png_path": png_path,
                "ocr_input_mode": "",
                "ocr_input_png_path_raw": "",
                "ocr_input_png_path": "",
                "ocr_input_provenance_path": "",
                "ocr_input_provenance_type": "",
                "ocr_layout_path": "",
            }
            existing = page_plan_by_key.get(page_key)
            if existing is not None and existing != page_plan:
                raise SystemExit(f"Conflicting page plan for {page_key}")
            page_plan_by_key[page_key] = page_plan

    return page_plan_by_key

def _materialize_page_texts(
    *,
    page_plan_by_key: dict[str, dict[str, Any]],
    box_page_ids_by_source: dict[str, dict[str, set[str]]],
    direct_page_refs: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    page_rows_by_key: dict[str, dict[str, Any]] = {}

    for source_key, payload in sorted(direct_page_refs.items()):
        source = payload["text_source"]
        source_path = Path(payload["source_path"]).expanduser()
        if not source_path.is_file():
            raise SystemExit(f"Missing direct source file: {source_path}")
        text = _load_direct_page_text(source_path, source=source).strip()
        if not text:
            raise SystemExit(f"Empty text loaded from {source_path} ({source})")
        page_plan = dict(page_plan_by_key[source_key])
        page_plan["materialized_page_text_sha256"] = _sha256(text)
        page_plan["materialized_page_text_chars"] = len(text)
        if not int(page_plan.get("source_text_chars") or 0):
            page_plan["source_text_chars"] = len(text)
        materialized_path = Path(page_plan["materialized_page_text_path"])
        materialized_path.parent.mkdir(parents=True, exist_ok=True)
        materialized_path.write_text(text, encoding="utf-8")
        page_rows_by_key[source_key] = page_plan

    loader_by_source = {
        "gemini_box_results": exporter._load_gemini_box_results_jsonl,
        "openai_box_results": exporter._load_openai_box_results_jsonl,
    }
    for source_name, page_ids_by_path in box_page_ids_by_source.items():
        loader = loader_by_source[source_name]
        for source_path, page_ids in sorted(page_ids_by_path.items()):
            path = Path(source_path).expanduser()
            if not path.is_file():
                raise SystemExit(f"Missing {source_name} shard file: {path}")
            loaded = loader([path], wanted_page_ids=set(page_ids))
            for page_id in sorted(page_ids):
                page_key = f"{source_name}::{source_path}::{page_id}"
                page_text = loaded.get(page_id)
                if page_text is None or not page_text.text.strip():
                    raise SystemExit(f"Missing page_id={page_id} in {source_name} shard {path}")
                page_plan = dict(page_plan_by_key[page_key])
                page_plan["materialized_page_text_sha256"] = _sha256(page_text.text)
                page_plan["materialized_page_text_chars"] = len(page_text.text)
                if not int(page_plan.get("source_text_chars") or 0):
                    page_plan["source_text_chars"] = len(page_text.text)
                materialized_path = Path(page_plan["materialized_page_text_path"])
                materialized_path.parent.mkdir(parents=True, exist_ok=True)
                materialized_path.write_text(page_text.text, encoding="utf-8")
                page_rows_by_key[page_key] = page_plan

    if len(page_rows_by_key) != len(page_plan_by_key):
        missing = sorted(set(page_plan_by_key) - set(page_rows_by_key))
        raise SystemExit(
            f"Failed to materialize all pages: have={len(page_rows_by_key)} expected={len(page_plan_by_key)} missing_example={missing[:3]}"
        )

    return page_rows_by_key


def _materialize_issue_dataset(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    page_rows_by_key: dict[str, dict[str, Any]],
    input_parsed_jsonl: Path,
    copied_prompt_path: Path,
    review_packet_dir: Path | None,
) -> dict[str, Any]:
    issue_transcripts_dir = out_dir / "issue_transcripts"
    classification_dir = out_dir / "classification_records"
    issue_transcripts_dir.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)

    label_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    issue_count = 0
    page_count = 0

    issues_jsonl_path = out_dir / "issues.jsonl"
    pages_jsonl_path = out_dir / "pages.jsonl"
    issues_csv_path = out_dir / "issues.csv"
    pages_csv_path = out_dir / "pages.csv"

    with (
        issues_jsonl_path.open("w", encoding="utf-8") as issues_jsonl_handle,
        pages_jsonl_path.open("w", encoding="utf-8") as pages_jsonl_handle,
        issues_csv_path.open("w", encoding="utf-8", newline="") as issues_csv_handle,
        pages_csv_path.open("w", encoding="utf-8", newline="") as pages_csv_handle,
    ):
        issues_csv_writer = csv.DictWriter(issues_csv_handle, fieldnames=ISSUE_CSV_FIELDS)
        pages_csv_writer = csv.DictWriter(pages_csv_handle, fieldnames=PAGE_CSV_FIELDS)
        issues_csv_writer.writeheader()
        pages_csv_writer.writeheader()

        for row in rows:
            mapping = row.get("mapping") or {}
            parsed = row.get("parsed") or {}
            legal_document = parsed.get("legal_document") or {}
            zoning_presence = parsed.get("zoning_presence") or {}
            issue_id = _norm(row.get("issue_id") or mapping.get("issue_id"))
            issue_date = _norm(row.get("issue_date") or mapping.get("issue_date"))
            slug = _norm(row.get("slug") or mapping.get("slug")).lower()
            predicted_label = _norm(legal_document.get("class"))
            classification_status = _norm(row.get("status"))
            label_counts[predicted_label] += 1
            status_counts[classification_status] += 1

            source_pages = mapping.get("source_pages") or []
            issue_pages_materialized: list[dict[str, Any]] = []
            page_payloads_for_transcript: list[exporter.PageText] = []
            for page in sorted(source_pages, key=lambda item: int(item.get("page_num") or 0)):
                page_id = _norm(page.get("page_id")).lower()
                page_num = int(page.get("page_num") or 0)
                text_source = _norm(page.get("text_source"))
                source_path = _norm(page.get("source_path"))
                page_key = f"{text_source}::{source_path}::{page_id}"
                page_record = page_rows_by_key.get(page_key)
                if page_record is None:
                    raise SystemExit(f"Missing materialized page record for issue_id={issue_id} page_id={page_id}")

                issue_pages_materialized.append(page_record)
                pages_jsonl_handle.write(json.dumps(page_record, ensure_ascii=False) + "\n")
                pages_csv_writer.writerow(_page_csv_row(page_record))
                page_count += 1

                materialized_path = Path(page_record["materialized_page_text_path"])
                page_text = materialized_path.read_text(encoding="utf-8", errors="replace")
                page_payloads_for_transcript.append(
                    exporter.PageText(
                        page_id=page_id,
                        issue_id=issue_id,
                        issue_date=issue_date,
                        newspaper_slug=slug,
                        page_num=page_num,
                        text=page_text,
                        text_sha256=_norm(page_record.get("materialized_page_text_sha256")),
                        text_chars=int(page_record.get("materialized_page_text_chars") or len(page_text)),
                        source=text_source,
                        source_path=source_path,
                    )
                )

            issue_transcript = _build_dataset_issue_transcript(
                page_payloads_for_transcript,
                max_issue_chars=10**9,
                issue_id=issue_id,
            )
            issue_transcript_path = issue_transcripts_dir / f"{_safe_issue_filename(issue_id)}.txt"
            issue_transcript_path.write_text(issue_transcript, encoding="utf-8")

            classification_record_path = classification_dir / f"{_safe_issue_filename(issue_id)}.json"
            classification_payload = {
                "issue_id": issue_id,
                "issue_date": issue_date,
                "newspaper_slug": slug,
                "classifier_model": _norm(row.get("classifier_model")),
                "input_parsed_jsonl": str(input_parsed_jsonl),
                "classification_prompt_path": str(copied_prompt_path),
                "source_row": row,
            }
            classification_record_path.write_text(
                json.dumps(classification_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            review_packet_record_path = ""
            if review_packet_dir is not None:
                candidate = review_packet_dir / predicted_label / f"{_safe_issue_filename(issue_id)}.json"
                if candidate.is_file():
                    review_packet_record_path = str(candidate.resolve())

            issue_row = {
                "issue_id": issue_id,
                "issue_date": issue_date,
                "newspaper_slug": slug,
                "classifier_model": _norm(row.get("classifier_model")),
                "classification_status": classification_status,
                "classification_error": row.get("error"),
                "classification_prompt_path": str(copied_prompt_path),
                "classification_record_path": str(classification_record_path),
                "review_packet_record_path": review_packet_record_path,
                "issue_transcript_path": str(issue_transcript_path),
                "issue_transcript_chars": len(issue_transcript),
                "issue_transcript_sha256": _sha256(issue_transcript),
                "page_count": len(issue_pages_materialized),
                "predicted": {
                    "primary_class": _norm(zoning_presence.get("primary_class")),
                    "label": predicted_label,
                    "operativity": _norm(legal_document.get("operativity")),
                    "legal_action": _norm(legal_document.get("legal_action")),
                    "scope": _norm(legal_document.get("scope")),
                    "publication_completeness": _norm(legal_document.get("publication_completeness")),
                    "confidence_0_to_1": legal_document.get("confidence_0_to_1"),
                },
                "evidence": {
                    "zoning_presence_quote": _norm(zoning_presence.get("evidence_quote")),
                    "legal_document_quote": _norm(legal_document.get("evidence_quote")),
                    "rationale": _norm(parsed.get("rationale")),
                    "quality_flags": parsed.get("quality_flags") if isinstance(parsed.get("quality_flags"), list) else [],
                },
                "pages": issue_pages_materialized,
            }
            issues_jsonl_handle.write(json.dumps(issue_row, ensure_ascii=False) + "\n")
            issues_csv_writer.writerow(_issue_csv_row(issue_row))
            issue_count += 1

    return {
        "issue_count": issue_count,
        "page_count": page_count,
        "label_counts": label_counts,
        "status_counts": status_counts,
    }


PAGE_CSV_FIELDS = [
    "issue_id",
    "issue_date",
    "newspaper_slug",
    "page_id",
    "page_num",
    "text_source",
    "original_transcript_source_path",
    "materialized_page_text_path",
    "materialized_page_text_sha256",
    "materialized_page_text_chars",
    "source_text_sha256",
    "source_text_chars",
    "png_path",
    "ocr_input_mode",
    "ocr_input_png_path_raw",
    "ocr_input_png_path",
    "ocr_input_provenance_path",
    "ocr_input_provenance_type",
    "ocr_layout_path",
]


ISSUE_CSV_FIELDS = [
    "issue_id",
    "issue_date",
    "newspaper_slug",
    "classifier_model",
    "classification_status",
    "classification_error_json",
    "classification_prompt_path",
    "classification_record_path",
    "review_packet_record_path",
    "issue_transcript_path",
    "issue_transcript_chars",
    "issue_transcript_sha256",
    "page_count",
    "predicted_primary_class",
    "predicted_label",
    "predicted_operativity",
    "predicted_legal_action",
    "predicted_scope",
    "predicted_publication_completeness",
    "predicted_confidence_0_to_1",
    "zoning_presence_quote",
    "legal_document_quote",
    "rationale",
    "quality_flags_json",
]


def _page_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row.get(name, "") for name in PAGE_CSV_FIELDS}


def _issue_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    predicted = row.get("predicted") or {}
    evidence = row.get("evidence") or {}
    return {
        "issue_id": row.get("issue_id", ""),
        "issue_date": row.get("issue_date", ""),
        "newspaper_slug": row.get("newspaper_slug", ""),
        "classifier_model": row.get("classifier_model", ""),
        "classification_status": row.get("classification_status", ""),
        "classification_error_json": json.dumps(row.get("classification_error"), ensure_ascii=False),
        "classification_prompt_path": row.get("classification_prompt_path", ""),
        "classification_record_path": row.get("classification_record_path", ""),
        "review_packet_record_path": row.get("review_packet_record_path", ""),
        "issue_transcript_path": row.get("issue_transcript_path", ""),
        "issue_transcript_chars": row.get("issue_transcript_chars", ""),
        "issue_transcript_sha256": row.get("issue_transcript_sha256", ""),
        "page_count": row.get("page_count", ""),
        "predicted_primary_class": predicted.get("primary_class", ""),
        "predicted_label": predicted.get("label", ""),
        "predicted_operativity": predicted.get("operativity", ""),
        "predicted_legal_action": predicted.get("legal_action", ""),
        "predicted_scope": predicted.get("scope", ""),
        "predicted_publication_completeness": predicted.get("publication_completeness", ""),
        "predicted_confidence_0_to_1": predicted.get("confidence_0_to_1", ""),
        "zoning_presence_quote": evidence.get("zoning_presence_quote", ""),
        "legal_document_quote": evidence.get("legal_document_quote", ""),
        "rationale": evidence.get("rationale", ""),
        "quality_flags_json": json.dumps(evidence.get("quality_flags", []), ensure_ascii=False),
    }


def _write_readme(
    *,
    out_dir: Path,
    input_parsed_jsonl: Path,
    unique_png_root: Path,
    zoning_presence_primary_classes: set[str],
    legal_document_classes: set[str],
    exclude_operativities: set[str],
    issue_count: int,
    page_count: int,
    label_counts: Counter[str],
) -> None:
    readme = out_dir / "README.md"
    lines = [
        "# Issue Classifier Dataset Manifest",
        "",
        "This directory is a Torch-side issue-level dataset bundle derived from parsed classifier outputs.",
        "",
        "## Primary files",
        "",
        "- `issues.jsonl`: one row per issue, including classification summary, issue transcript path, and nested page metadata.",
        "- `issues.csv`: flat issue-level view for quick spreadsheet/pandas use.",
        "- `pages.jsonl`: one row per page, flattened from `issues.jsonl`.",
        "- `pages.csv`: CSV view of `pages.jsonl` for quick spreadsheet/pandas use.",
        "- `summary.json`: run metadata and counts.",
        "- `prompt_used.txt`: exact classifier prompt copied into this bundle.",
        "- `issue_transcripts/`: materialized issue transcripts, built by concatenating page transcripts with a simple page-break delimiter and no per-page metadata wrapper.",
        "- `page_texts/`: materialized per-page transcript text files.",
        "- `classification_records/`: one JSON file per issue with the original parsed classifier row embedded.",
        "",
        "## Source inputs",
        "",
        f"- parsed classifier rows: `{input_parsed_jsonl}`",
        f"- canonical page PNG root: `{unique_png_root}`",
        f"- zoning_presence.primary_class filter: `{sorted(zoning_presence_primary_classes) if zoning_presence_primary_classes else []}`",
        f"- legal_document.class filter: `{sorted(legal_document_classes) if legal_document_classes else []}`",
        f"- excluded operativity values: `{sorted(exclude_operativities) if exclude_operativities else []}`",
        "",
        "## Counts",
        "",
        f"- issues: {issue_count}",
        f"- pages: {page_count}",
        f"- labels: {dict(sorted(label_counts.items()))}",
        "",
        "## Notes",
        "",
        "- All paths written in `issues.jsonl` and `pages.jsonl` are absolute Torch paths.",
        "- `issues.*` expose `classification_status` plus serialized `classification_error_json` for non-ok rows.",
        "- `pages[].original_transcript_source_path` points to the original OCR/VLM artifact.",
        "- `pages[].materialized_page_text_path` points to a normalized plain-text file created by this script.",
        "- `pages[].png_path` points to the resolved Torch page PNG used by this manifest.",
        "- `pages[].ocr_input_png_path_raw` records the exact page-image path string found in OCR provenance artifacts when available.",
        "- `pages[].ocr_input_png_path` resolves that OCR image path onto Torch when possible.",
        "- `pages[].ocr_input_provenance_path` points to the provenance artifact (`*_res.json` for Paddle, `mapping_shard*.jsonl` for box OCR).",
        "- `pages[].ocr_input_mode` distinguishes full-page OCR from box-crop OCR.",
        "",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-jsonl", type=Path, required=True, help="Parsed classifier outputs JSONL.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--unique-png-root",
        type=Path,
        required=True,
        help="Canonical unique_png directory used to resolve page image paths.",
    )
    parser.add_argument(
        "--classification-prompt-path",
        type=Path,
        required=True,
        help="Prompt file to copy into the dataset bundle as prompt_used.txt.",
    )
    parser.add_argument(
        "--review-packet-dir",
        type=Path,
        default=None,
        help="Optional review packet root; if provided, matching per-issue review JSON paths are recorded.",
    )
    parser.add_argument(
        "--zoning-presence-primary-classes",
        default="",
        help=(
            "Optional comma-separated filter on parsed.zoning_presence.primary_class. "
            "Default keeps all rows, preserving existing behavior."
        ),
    )
    parser.add_argument(
        "--legal-document-classes",
        default="",
        help=(
            "Optional comma-separated filter on parsed.legal_document.class. "
            "Default keeps all classes."
        ),
    )
    parser.add_argument(
        "--exclude-operativities",
        default="",
        help=(
            "Optional comma-separated exclusion filter on parsed.legal_document.operativity. "
            "Example: proposed"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    parsed_jsonl = args.parsed_jsonl.expanduser().resolve()
    unique_png_root = args.unique_png_root.expanduser().resolve()
    prompt_path = args.classification_prompt_path.expanduser().resolve()
    review_packet_dir = args.review_packet_dir.expanduser().resolve() if args.review_packet_dir else None
    out_dir = args.out_dir.expanduser().resolve()
    zoning_presence_primary_classes = _parse_csv_values(args.zoning_presence_primary_classes)
    legal_document_classes = _parse_csv_values(args.legal_document_classes)
    exclude_operativities = _parse_csv_values(args.exclude_operativities)

    if not parsed_jsonl.is_file():
        raise SystemExit(f"--parsed-jsonl not found: {parsed_jsonl}")
    if not unique_png_root.is_dir():
        raise SystemExit(f"--unique-png-root is not a directory: {unique_png_root}")
    if not prompt_path.is_file():
        raise SystemExit(f"--classification-prompt-path not found: {prompt_path}")
    if review_packet_dir is not None and not review_packet_dir.is_dir():
        raise SystemExit(f"--review-packet-dir is not a directory: {review_packet_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    copied_prompt_path = out_dir / "prompt_used.txt"
    copied_prompt_path.write_text(prompt_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    rows = _selected_rows(
        parsed_jsonl,
        zoning_presence_primary_classes=zoning_presence_primary_classes,
        legal_document_classes=legal_document_classes,
        exclude_operativities=exclude_operativities,
    )
    if not rows:
        raise SystemExit(f"No rows found in {parsed_jsonl}")

    png_by_page_id = _build_png_index(unique_png_root)
    box_page_ids_by_source, direct_page_refs = _collect_source_refs(rows)
    page_plan_by_key = _plan_page_records(
        rows=rows,
        out_dir=out_dir,
        png_by_page_id=png_by_page_id,
    )
    _enrich_page_plans_with_png_provenance(
        page_plan_by_key=page_plan_by_key,
        png_by_page_id=png_by_page_id,
    )
    page_rows_by_key = _materialize_page_texts(
        page_plan_by_key=page_plan_by_key,
        box_page_ids_by_source=box_page_ids_by_source,
        direct_page_refs=direct_page_refs,
    )

    manifest_summary = _materialize_issue_dataset(
        rows=rows,
        out_dir=out_dir,
        page_rows_by_key=page_rows_by_key,
        input_parsed_jsonl=parsed_jsonl,
        copied_prompt_path=copied_prompt_path,
        review_packet_dir=review_packet_dir,
    )

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_parsed_jsonl": str(parsed_jsonl),
        "unique_png_root": str(unique_png_root),
        "classification_prompt_path": str(copied_prompt_path),
        "review_packet_dir": str(review_packet_dir) if review_packet_dir is not None else "",
        "zoning_presence_primary_classes_filter": sorted(zoning_presence_primary_classes),
        "legal_document_classes_filter": sorted(legal_document_classes),
        "exclude_operativities_filter": sorted(exclude_operativities),
        "issue_count": manifest_summary["issue_count"],
        "page_count": manifest_summary["page_count"],
        "label_counts": dict(sorted(manifest_summary["label_counts"].items())),
        "status_counts": dict(sorted(manifest_summary["status_counts"].items())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(
        out_dir=out_dir,
        input_parsed_jsonl=parsed_jsonl,
        unique_png_root=unique_png_root,
        zoning_presence_primary_classes=zoning_presence_primary_classes,
        legal_document_classes=legal_document_classes,
        exclude_operativities=exclude_operativities,
        issue_count=manifest_summary["issue_count"],
        page_count=manifest_summary["page_count"],
        label_counts=manifest_summary["label_counts"],
    )

    print(
        f"built dataset manifest issues={manifest_summary['issue_count']} pages={manifest_summary['page_count']} "
        f"labels={dict(sorted(manifest_summary['label_counts'].items()))} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
