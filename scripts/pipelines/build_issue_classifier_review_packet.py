#!/usr/bin/env python3
"""
Build a browser-review packet for issue-level zoning-classifier outputs.

The packet is optimized for local manual review of likely printed-law positives.
Input is the parsed JSONL produced by:

  scripts/pipelines/rehydrate_issue_zoning_issue_classifier_results.py

Output packet layout:

  <out_dir>/
    metadata.json
    index.jsonl
    review_sheet.csv
    review_events.jsonl
    review_snapshot.csv
    <predicted_label>/<issue_id>.json

Each per-issue JSON record contains the model output, source provenance, the full
issue transcript used for classification, and per-page source text where available.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import export_issue_zoning_issue_classifier_batch_requests as exporter


TARGET_LABELS = [
    "code_publication_full_issue",
    "code_publication_excerpt_or_installment",
    "amendment_substantial_text",
    "amendment_targeted_text",
    "zoning_ordinance_limited_scope",
    "map_rezoning_order",
    "variance_special_use_order",
]

LABEL_PRIORITY = {
    "code_publication_full_issue": 0,
    "code_publication_excerpt_or_installment": 1,
    "amendment_substantial_text": 2,
    "amendment_targeted_text": 3,
    "zoning_ordinance_limited_scope": 4,
    "map_rezoning_order": 5,
    "variance_special_use_order": 6,
    "procedural_notice_only": 7,
    "non_zoning_ordinance": 8,
    "none": 9,
    "uncertain": 10,
    "unlabeled": 11,
}

ALL_LEGAL_DOCUMENT_LABELS = [
    "none",
    "code_publication_full_issue",
    "code_publication_excerpt_or_installment",
    "amendment_substantial_text",
    "amendment_targeted_text",
    "zoning_ordinance_limited_scope",
    "map_rezoning_order",
    "variance_special_use_order",
    "procedural_notice_only",
    "non_zoning_ordinance",
    "uncertain",
]

SIGNAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "has_be_it_ordained": re.compile(r"\bbe it ordain(?:ed|s)?\b|\bdoes ordain\b", flags=re.I),
    "has_section_clause": re.compile(r"\b(?:section|sec\.?|§)\s*\d+[A-Za-z0-9.-]*\b", flags=re.I),
    "has_amend_formula": re.compile(r"\bis hereby amended\b|\bamended to read as follows\b|\bamend(?:ing)? section\b", flags=re.I),
    "has_effective": re.compile(r"\bshall take effect\b|\beffective date\b|\bshall become effective\b", flags=re.I),
    "has_public_hearing": re.compile(r"\bpublic hearing\b|\bhearing will be held\b", flags=re.I),
    "has_minutes": re.compile(r"\bminutes\b|\bmotion carried\b|\bapproved the minutes\b", flags=re.I),
    "has_notice_copy_available": re.compile(r"\bcopy (?:may|can) be examined\b|\bavailable for public inspection\b", flags=re.I),
    "has_variance_terms": re.compile(r"\bvariance\b|\bspecial exception\b|\bconditional use\b|\bspecial use\b", flags=re.I),
    "has_rezone_terms": re.compile(r"\brezone\b|\brezoning\b|\bzoning map\b|\bfrom [A-Z0-9-]+ to [A-Z0-9-]+\b", flags=re.I),
    "has_building_housing": re.compile(r"\bbuilding\b|\bhousing\b|\bsetback\b|\blot\b|\bheight\b|\bdensity\b", flags=re.I),
}


def _eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


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


def _parse_labels(raw: str) -> list[str]:
    labels = [_norm(part) for part in raw.split(",")]
    labels = [label for label in labels if label]
    if not labels:
        return []
    unknown = sorted(set(labels) - set(ALL_LEGAL_DOCUMENT_LABELS))
    if unknown:
        raise SystemExit(f"Unknown labels: {', '.join(unknown)}")
    return labels


def _should_include_row(
    row: dict[str, Any],
    *,
    include_labels: set[str],
    include_all: bool,
    require_status_ok: bool,
    only_zoning_legal_text: bool,
    exclude_proposed: bool,
) -> bool:
    if require_status_ok and _norm(row.get("status")).lower() != "ok":
        return False
    parsed = row.get("parsed") or {}
    if not isinstance(parsed, dict):
        return False
    legal_document = parsed.get("legal_document") or {}
    zoning_presence = parsed.get("zoning_presence") or {}
    if not isinstance(legal_document, dict):
        return False
    if not isinstance(zoning_presence, dict):
        return False
    predicted_label = _norm(legal_document.get("class"))
    predicted_operativity = _norm(legal_document.get("operativity")).lower()
    primary_class = _norm(zoning_presence.get("primary_class")).lower()
    if only_zoning_legal_text and primary_class != "zoning_legal_text":
        return False
    if exclude_proposed and predicted_operativity == "proposed":
        return False
    if include_all or not include_labels:
        return True
    return predicted_label in include_labels


def _safe_text_preview(text: str, *, limit: int = 220) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _detect_signals(text: str) -> dict[str, int]:
    return {signal_name: int(bool(pattern.search(text))) for signal_name, pattern in SIGNAL_PATTERNS.items()}


def _page_count_from_transcript(text: str) -> int:
    page_count = text.count("=== PAGE START ===")
    return page_count if page_count > 0 else int(bool(text.strip()))


def _load_direct_page_text(path: Path, *, source: str) -> str:
    if source == "paddle_vl15_md":
        return path.read_text(encoding="utf-8", errors="replace")
    if source == "vlm_page":
        return exporter._extract_clean_text_from_vlm_page(path)
    raise SystemExit(f"Unsupported direct page source: {source}")


def _collect_candidate_rows(
    parsed_jsonl: Path,
    *,
    include_labels: set[str],
    include_all: bool,
    require_status_ok: bool,
    only_zoning_legal_text: bool,
    exclude_proposed: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, set[str]]], dict[str, dict[str, str]]]:
    selected_rows: list[dict[str, Any]] = []
    box_page_ids_by_source: dict[str, dict[str, set[str]]] = {
        "gemini_box_results": defaultdict(set),
        "openai_box_results": defaultdict(set),
    }
    direct_page_refs: dict[str, dict[str, str]] = {}

    for row in _read_jsonl(parsed_jsonl):
        if not _should_include_row(
            row,
            include_labels=include_labels,
            include_all=include_all,
            require_status_ok=require_status_ok,
            only_zoning_legal_text=only_zoning_legal_text,
            exclude_proposed=exclude_proposed,
        ):
            continue
        mapping = row.get("mapping") or {}
        if not isinstance(mapping, dict):
            raise SystemExit(f"Row missing mapping for issue_id={row.get('issue_id')}")
        issue_id = _norm(row.get("issue_id") or mapping.get("issue_id"))
        if not issue_id:
            raise SystemExit("Encountered parsed row without issue_id")
        if mapping.get("source_mode") == "issue_txt_dir":
            selected_rows.append(row)
            continue
        source_pages = mapping.get("source_pages")
        if not isinstance(source_pages, list) or not source_pages:
            raise SystemExit(f"issue_id={issue_id} has no source_pages and is not issue_txt_dir mode")
        for page in source_pages:
            if not isinstance(page, dict):
                raise SystemExit(f"issue_id={issue_id} has invalid source_pages entry")
            page_id = _norm(page.get("page_id")).lower()
            text_source = _norm(page.get("text_source"))
            source_path = _norm(page.get("source_path"))
            if not page_id or not text_source or not source_path:
                raise SystemExit(f"issue_id={issue_id} has incomplete page provenance: {page}")
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
        selected_rows.append(row)

    return selected_rows, box_page_ids_by_source, direct_page_refs


def _rebuild_source_refs_for_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, str]]]:
    box_page_ids_by_source: dict[str, dict[str, set[str]]] = {
        "gemini_box_results": defaultdict(set),
        "openai_box_results": defaultdict(set),
    }
    direct_page_refs: dict[str, dict[str, str]] = {}
    for row in rows:
        mapping = row.get("mapping") or {}
        if mapping.get("source_mode") == "issue_txt_dir":
            continue
        for page in mapping.get("source_pages") or []:
            page_id = _norm(page.get("page_id")).lower()
            text_source = _norm(page.get("text_source"))
            source_path = _norm(page.get("source_path"))
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
                raise SystemExit(f"Unsupported text_source={text_source} for issue_id={row.get('issue_id')}")
    return box_page_ids_by_source, direct_page_refs


def _load_page_text_lookup(
    *,
    box_page_ids_by_source: dict[str, dict[str, set[str]]],
    direct_page_refs: dict[str, dict[str, str]],
    on_missing_source: str,
) -> dict[str, str]:
    page_text_by_key: dict[str, str] = {}
    missing_keys: list[str] = []

    for source_key, payload in sorted(direct_page_refs.items()):
        source = payload["text_source"]
        source_path = Path(payload["source_path"]).expanduser()
        if not source_path.is_file():
            if on_missing_source == "skip_issue":
                missing_keys.append(source_key)
                continue
            raise SystemExit(f"Missing direct source file: {source_path}")
        text = _load_direct_page_text(source_path, source=source).strip()
        if not text:
            if on_missing_source == "skip_issue":
                missing_keys.append(source_key)
                continue
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
                if on_missing_source == "skip_issue":
                    for page_id in page_ids:
                        missing_keys.append(f"{source_name}::{source_path}::{page_id}")
                    continue
                raise SystemExit(f"Missing {source_name} shard file: {path}")
            loaded = loader([path], wanted_page_ids=set(page_ids))
            for page_id in page_ids:
                page_key = f"{source_name}::{source_path}::{page_id}"
                page_text = loaded.get(page_id)
                if page_text is None or not page_text.text.strip():
                    if on_missing_source == "skip_issue":
                        missing_keys.append(page_key)
                        continue
                    raise SystemExit(f"Missing page_id={page_id} in {source_name} shard {path}")
                page_text_by_key[page_key] = page_text.text

    if missing_keys:
        _eprint(f"warn: missing page text for {len(missing_keys)} page references; affected issues will be skipped")
    return page_text_by_key


def _build_issue_record(
    row: dict[str, Any],
    *,
    page_text_by_key: dict[str, str],
    packet_built_at: str,
) -> dict[str, Any]:
    parsed = row.get("parsed") or {}
    mapping = row.get("mapping") or {}
    legal_document = parsed.get("legal_document") or {}
    zoning_presence = parsed.get("zoning_presence") or {}
    issue_id = _norm(row.get("issue_id") or mapping.get("issue_id"))
    predicted_label = _norm(legal_document.get("class")) or "unlabeled"
    predicted_operativity = _norm(legal_document.get("operativity"))
    quality_flags = parsed.get("quality_flags") or []
    if not isinstance(quality_flags, list):
        quality_flags = []

    source_pages_full: list[dict[str, Any]] = []
    issue_transcript = ""
    source_mode = _norm(mapping.get("source_mode"))
    if source_mode == "issue_txt_dir":
        transcript_path = Path(_norm(mapping.get("source_path"))).expanduser()
        if not transcript_path.is_file():
            raise SystemExit(f"Missing issue transcript file for issue_id={issue_id}: {transcript_path}")
        issue_transcript = transcript_path.read_text(encoding="utf-8", errors="replace")
    else:
        source_pages = mapping.get("source_pages")
        if not isinstance(source_pages, list) or not source_pages:
            raise SystemExit(f"issue_id={issue_id} has no source_pages")
        page_texts: list[exporter.PageText] = []
        for page in sorted(source_pages, key=lambda page_row: int(page_row.get("page_num") or 0)):
            page_id = _norm(page.get("page_id")).lower()
            page_num = int(page.get("page_num") or 0)
            text_source = _norm(page.get("text_source"))
            source_path = _norm(page.get("source_path"))
            lookup_key = f"{text_source}::{source_path}::{page_id}"
            page_text = page_text_by_key.get(lookup_key)
            if page_text is None:
                raise KeyError(lookup_key)
            source_pages_full.append(
                {
                    "page_id": page_id,
                    "page_num": page_num,
                    "text_source": text_source,
                    "source_path": source_path,
                    "text": page_text,
                }
            )
            page_texts.append(
                exporter.PageText(
                    page_id=page_id,
                    issue_id=issue_id,
                    issue_date=_norm(mapping.get("issue_date")),
                    newspaper_slug=_norm(mapping.get("slug")).lower(),
                    page_num=page_num,
                    text=page_text,
                    text_sha256=exporter._sha256(page_text),
                    text_chars=len(page_text),
                    source=text_source,
                    source_path=source_path,
                )
            )
        issue_transcript = exporter._build_issue_transcript(page_texts, max_issue_chars=0, issue_id=issue_id)

    issue_chars = len(issue_transcript)
    page_count = len(source_pages_full) if source_pages_full else _page_count_from_transcript(issue_transcript)
    signals = _detect_signals(issue_transcript)
    preview = _safe_text_preview(issue_transcript)

    response_meta = {
        field_name: row.get(field_name)
        for field_name in [
            "status",
            "classifier_model",
            "output_text_sha256",
            "response_id",
            "response_created_at",
            "response_completed_at",
            "total_tokens",
            "rehydrated_at",
        ]
        if row.get(field_name) not in {None, ""}
    }

    return {
        "schema_version": "issue_classifier_review_packet_v1",
        "packet_built_at": packet_built_at,
        "issue_id": issue_id,
        "issue_date": _norm(row.get("issue_date") or mapping.get("issue_date")),
        "slug": _norm(row.get("slug") or mapping.get("slug")).lower(),
        "predicted": {
            "label": predicted_label,
            "operativity": predicted_operativity,
            "confidence_0_to_1": legal_document.get("confidence_0_to_1"),
            "legal_action": _norm(legal_document.get("legal_action")),
            "publication_completeness": _norm(legal_document.get("publication_completeness")),
            "scope": _norm(legal_document.get("scope")),
            "primary_class": _norm(zoning_presence.get("primary_class")),
            "zoning_has_reference": bool(zoning_presence.get("has_any_zoning_reference")),
            "quality_flags": list(quality_flags),
            "zoning_evidence_quote": _norm(zoning_presence.get("evidence_quote")),
            "legal_evidence_quote": _norm(legal_document.get("evidence_quote")),
            "rationale": _norm(parsed.get("rationale")),
        },
        "response_meta": response_meta,
        "model_output": parsed,
        "mapping": mapping,
        "source_pages_full": source_pages_full,
        "issue_transcript": issue_transcript,
        "issue_chars": issue_chars,
        "page_count": page_count,
        "preview": preview,
        "signals": signals,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: row.get(field_name, "") for field_name in fieldnames})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a browser review packet for issue-classifier outputs.")
    parser.add_argument("--parsed-jsonl", required=True, help="Parsed outputs JSONL from rehydrate_issue_zoning_issue_classifier_results.py")
    parser.add_argument("--out-dir", required=True, help="Directory to write the review packet into.")
    parser.add_argument(
        "--labels",
        default="",
        help=(
            "Optional comma-separated predicted legal_document.class labels to include. "
            "If omitted, all classes that pass the printed-law gating filters are included."
        ),
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include every parsed issue instead of filtering by --labels.",
    )
    parser.add_argument(
        "--allow-non-ok-status",
        action="store_true",
        help="Include rows whose parsed status is not ok.",
    )
    parser.add_argument(
        "--include-non-zoning-legal-text",
        action="store_true",
        help="Do not restrict the packet to rows with zoning_presence.primary_class=zoning_legal_text.",
    )
    parser.add_argument(
        "--include-proposed",
        action="store_true",
        help="Do not exclude rows whose predicted operativity is proposed.",
    )
    parser.add_argument(
        "--on-missing-source",
        choices=["error", "skip_issue"],
        default="error",
        help="How to handle missing source-text files needed to reconstruct the reviewed issue transcript.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional hard cap on the number of included issues, applied after filtering.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    parsed_jsonl = Path(args.parsed_jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not parsed_jsonl.is_file():
        raise SystemExit(f"--parsed-jsonl not found: {parsed_jsonl}")
    include_labels = set(_parse_labels(args.labels))
    packet_built_at = _now_iso()
    only_zoning_legal_text = not args.include_non_zoning_legal_text
    exclude_proposed = not args.include_proposed

    candidate_rows, box_page_ids_by_source, direct_page_refs = _collect_candidate_rows(
        parsed_jsonl,
        include_labels=include_labels,
        include_all=args.include_all,
        require_status_ok=not args.allow_non_ok_status,
        only_zoning_legal_text=only_zoning_legal_text,
        exclude_proposed=exclude_proposed,
    )
    if args.limit > 0:
        candidate_rows = candidate_rows[: args.limit]
        box_page_ids_by_source, direct_page_refs = _rebuild_source_refs_for_rows(candidate_rows)

    page_text_by_key = _load_page_text_lookup(
        box_page_ids_by_source=box_page_ids_by_source,
        direct_page_refs=direct_page_refs,
        on_missing_source=args.on_missing_source,
    )

    issue_records: list[dict[str, Any]] = []
    skipped_issue_ids: list[str] = []
    for row in candidate_rows:
        try:
            record = _build_issue_record(row, page_text_by_key=page_text_by_key, packet_built_at=packet_built_at)
        except KeyError as exc:
            if args.on_missing_source == "skip_issue":
                skipped_issue_ids.append(_norm(row.get("issue_id")))
                _eprint(f"warn: skipping issue_id={row.get('issue_id')} because source text was missing ({exc})")
                continue
            raise SystemExit(f"Missing page text while building issue_id={row.get('issue_id')}: {exc}") from exc
        issue_records.append(record)

    if not issue_records:
        raise SystemExit("No review records were built")

    issue_records.sort(
        key=lambda record: (
            LABEL_PRIORITY.get(record["predicted"]["label"], 999),
            -float(record["predicted"]["confidence_0_to_1"] or 0),
            record["issue_id"],
        )
    )

    label_counts = Counter(record["predicted"]["label"] for record in issue_records)
    index_rows: list[dict[str, Any]] = []
    review_sheet_rows: list[dict[str, Any]] = []
    for record in issue_records:
        predicted_label = record["predicted"]["label"]
        file_name = exporter._safe_issue_filename(record["issue_id"]) + ".json"
        item_path = (out_dir / predicted_label / file_name).resolve()
        relative_item_path = str(item_path.relative_to(out_dir))
        _write_json(item_path, record)
        signals = record["signals"]
        index_row = {
            "issue_id": record["issue_id"],
            "issue_date": record["issue_date"],
            "slug": record["slug"],
            "predicted_label": predicted_label,
            "predicted_operativity": record["predicted"]["operativity"],
            "predicted_primary_class": record["predicted"]["primary_class"],
            "confidence_0_to_1": record["predicted"]["confidence_0_to_1"],
            "quality_flags": record["predicted"]["quality_flags"],
            "page_count": record["page_count"],
            "issue_chars": record["issue_chars"],
            "preview": record["preview"],
            "relative_path": relative_item_path,
            "path": str(item_path),
        }
        index_rows.append(index_row)
        review_sheet_rows.append(
            {
                "issue_id": record["issue_id"],
                "label": predicted_label,
                "operativity": record["predicted"]["operativity"],
                "confidence": record["predicted"]["confidence_0_to_1"],
                "flags": "|".join(record["predicted"]["quality_flags"]),
                "pages": record["page_count"],
                "chars": record["issue_chars"],
                "has_be_it_ordained": signals["has_be_it_ordained"],
                "has_section_clause": signals["has_section_clause"],
                "has_amend_formula": signals["has_amend_formula"],
                "has_effective": signals["has_effective"],
                "has_public_hearing": signals["has_public_hearing"],
                "has_minutes": signals["has_minutes"],
                "has_notice_copy_available": signals["has_notice_copy_available"],
                "has_variance_terms": signals["has_variance_terms"],
                "has_rezone_terms": signals["has_rezone_terms"],
                "has_building_housing": signals["has_building_housing"],
                "preview": record["preview"],
                "relative_path": relative_item_path,
                "path": str(item_path),
            }
        )

    metadata = {
        "schema_version": "issue_classifier_review_packet_v1",
        "packet_built_at": packet_built_at,
        "parsed_jsonl": str(parsed_jsonl),
        "labels_filter": sorted(include_labels),
        "include_all": bool(args.include_all),
        "require_status_ok": not args.allow_non_ok_status,
        "only_zoning_legal_text": only_zoning_legal_text,
        "exclude_proposed": exclude_proposed,
        "on_missing_source": args.on_missing_source,
        "issue_count": len(issue_records),
        "skipped_issue_count": len(skipped_issue_ids),
        "skipped_issue_ids": skipped_issue_ids,
        "label_counts": dict(sorted(label_counts.items())),
    }

    _write_json(out_dir / "metadata.json", metadata)
    _write_jsonl(out_dir / "index.jsonl", index_rows)
    _write_csv(
        out_dir / "review_sheet.csv",
        review_sheet_rows,
        fieldnames=[
            "issue_id",
            "label",
            "operativity",
            "confidence",
            "flags",
            "pages",
            "chars",
            "has_be_it_ordained",
            "has_section_clause",
            "has_amend_formula",
            "has_effective",
            "has_public_hearing",
            "has_minutes",
            "has_notice_copy_available",
            "has_variance_terms",
            "has_rezone_terms",
            "has_building_housing",
            "preview",
            "path",
        ],
    )
    review_events_path = out_dir / "review_events.jsonl"
    review_events_path.parent.mkdir(parents=True, exist_ok=True)
    review_events_path.touch(exist_ok=True)
    _write_csv(
        out_dir / "review_snapshot.csv",
        [],
        fieldnames=[
            "issue_id",
            "predicted_label",
            "predicted_operativity",
            "manual_label",
            "manual_operativity",
            "review_decision",
            "strict_operative_target",
            "exact_label_match",
            "exact_operativity_match",
            "note",
            "reviewed_at",
            "path",
        ],
    )

    _eprint(
        "built review packet "
        f"issues={len(issue_records)} labels={dict(sorted(label_counts.items()))} "
        f"out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
