#!/usr/bin/env python3
"""
Materialize Gemini Batch promotion results for hidden full-ordinance detection.

Input:
  - one or more *_prepared run roots produced by
    prepare_full_ordinance_promotion_gateway_run.py
  - completed Gemini batch outputs downloaded under:
      workers/workerXXX/gemini_batch_submission/completed_outputs/*_output.jsonl

This script joins the batch outputs back to the source dataset metadata, normalizes
the run-local request key, and writes:
  - promotion_results.jsonl
  - promotion_results.csv
  - promotion_summary.json

It can also write a merged result set across multiple prepared roots.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize full-ordinance promotion batch outputs.")
    ap.add_argument(
        "--prepared-dir",
        action="append",
        required=True,
        help="Prepared promotion run root (repeatable).",
    )
    ap.add_argument(
        "--merged-output-dir",
        default="",
        help="Optional directory for merged results across all prepared dirs.",
    )
    return ap.parse_args()


def _json_lines(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _load_issue_index(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    issues_path = dataset_dir / "issues.jsonl"
    rows = _json_lines(issues_path)
    by_request_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        issue_transcript_path = Path(str(row["issue_transcript_path"]))
        request_id = issue_transcript_path.stem
        if request_id in by_request_id:
            raise SystemExit(f"Duplicate request_id derived from issue_transcript_path stem: {request_id}")
        by_request_id[request_id] = row
    return rows, by_request_id


def _extract_response_text(obj: dict[str, Any]) -> str:
    response = obj.get("response") or {}
    candidates = response.get("candidates") or []
    texts: list[str] = []
    for cand in candidates:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    if not texts:
        return ""
    return "\n".join(texts).strip()


def _parse_decision_text(text: str) -> dict[str, Any] | None:
    if not text.strip():
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        decoder = json.JSONDecoder()
        obj, idx = decoder.raw_decode(text)
        remainder = text[idx:].strip()
        if not remainder or set(remainder) <= {"}"}:
            return obj
    except Exception:
        pass
    return None


def _extract_file_name_from_submission_record(record_path: Path) -> str:
    records = _json_lines(record_path)
    if not records:
        raise SystemExit(f"No submission records in {record_path}")
    if len(records) != 1:
        raise SystemExit(f"Expected one record in {record_path}, found {len(records)}")
    batch_name = str(records[0].get("batch_name") or "").strip()
    if not batch_name:
        raise SystemExit(f"Missing batch_name in {record_path}")
    return batch_name.replace("/", "__") + "_output.jsonl"


def _materialize_prepared_dir(prepared_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not prepared_dir.is_dir():
        raise SystemExit(f"--prepared-dir is not a directory: {prepared_dir}")
    if prepared_dir.name.endswith("_prepared"):
        dataset_dir = prepared_dir.with_name(prepared_dir.name[:-9] + "_dataset")
    else:
        raise SystemExit(f"Cannot infer dataset dir from prepared dir name: {prepared_dir.name}")
    if not dataset_dir.is_dir():
        raise SystemExit(f"Inferred dataset dir does not exist: {dataset_dir}")

    issue_rows, issue_index = _load_issue_index(dataset_dir)
    result_rows: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()

    worker_dirs = sorted((prepared_dir / "workers").glob("worker*"))
    if not worker_dirs:
        raise SystemExit(f"No worker directories found under {prepared_dir / 'workers'}")

    for worker_dir in worker_dirs:
        record_path = worker_dir / "requests" / "submitted_gemini_batches.jsonl"
        output_name = _extract_file_name_from_submission_record(record_path)
        output_path = worker_dir / "gemini_batch_submission" / "completed_outputs" / output_name
        if not output_path.is_file():
            raise SystemExit(f"Missing completed output file: {output_path}")
        for obj in _json_lines(output_path):
            request_id = str(obj.get("key") or "").strip()
            if not request_id:
                raise SystemExit(f"Missing key in {output_path}")
            if request_id in seen_request_ids:
                raise SystemExit(f"Duplicate request_id across outputs: {request_id}")
            seen_request_ids.add(request_id)

            source_row = issue_index.get(request_id)
            if source_row is None:
                raise SystemExit(f"Output key {request_id} not found in dataset issue index for {dataset_dir}")

            response_text = _extract_response_text(obj)
            parsed = _parse_decision_text(response_text)
            parsed_ok = parsed is not None
            second_stage = (parsed or {}).get("second_stage") or {}
            signals = (parsed or {}).get("full_ordinance_signals") or {}
            quality_flags = (parsed or {}).get("quality_flags") or []

            predicted = source_row.get("predicted") or {}
            evidence = source_row.get("evidence") or {}
            result_rows.append(
                {
                    "request_id": request_id,
                    "issue_id": str(source_row.get("issue_id") or ""),
                    "issue_date": str(source_row.get("issue_date") or ""),
                    "newspaper_slug": str(source_row.get("newspaper_slug") or ""),
                    "source_label": str(predicted.get("label") or ""),
                    "source_operativity": str(predicted.get("operativity") or ""),
                    "source_scope": str(predicted.get("scope") or ""),
                    "page_count": int(source_row.get("page_count") or 0),
                    "classification_rationale": str(evidence.get("rationale") or ""),
                    "response_text": response_text,
                    "response_json_ok": parsed_ok,
                    "second_stage_class": str(second_stage.get("class") or ""),
                    "instrument_family": str(second_stage.get("instrument_family") or ""),
                    "second_stage_operativity": str(second_stage.get("operativity") or ""),
                    "second_stage_confidence_0_to_1": second_stage.get("confidence_0_to_1"),
                    "evidence_quote": str(second_stage.get("evidence_quote") or ""),
                    "official_publication_framing": bool(signals.get("official_publication_framing")),
                    "broad_code_structure": bool(signals.get("broad_code_structure")),
                    "district_framework_present": bool(signals.get("district_framework_present")),
                    "multiple_substantive_components_present": bool(
                        signals.get("multiple_substantive_components_present")
                    ),
                    "quality_flags_json": json.dumps(quality_flags, ensure_ascii=False),
                    "rationale": str((parsed or {}).get("rationale") or ""),
                    "promoted_to_full_ordinance": str(second_stage.get("class") or "") == "full_ordinance_text",
                    "issue_transcript_path": str(source_row.get("issue_transcript_path") or ""),
                }
            )

    expected = len(issue_rows)
    if len(result_rows) != expected:
        raise SystemExit(
            f"Result count mismatch for {prepared_dir}: expected {expected}, found {len(result_rows)}"
        )

    out_jsonl = prepared_dir / "promotion_results.jsonl"
    out_csv = prepared_dir / "promotion_results.csv"
    out_summary = prepared_dir / "promotion_summary.json"

    result_rows_sorted = sorted(result_rows, key=lambda row: row["issue_id"])
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in result_rows_sorted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = list(result_rows_sorted[0].keys()) if result_rows_sorted else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows_sorted)

    summary = {
        "prepared_dir": str(prepared_dir),
        "dataset_dir": str(dataset_dir),
        "issues_total": expected,
        "response_json_ok": sum(1 for row in result_rows_sorted if row["response_json_ok"]),
        "promoted_to_full_ordinance": sum(1 for row in result_rows_sorted if row["promoted_to_full_ordinance"]),
        "by_second_stage_class": {},
        "by_instrument_family": {},
    }
    for row in result_rows_sorted:
        cls = row["second_stage_class"] or "blank"
        fam = row["instrument_family"] or "blank"
        summary["by_second_stage_class"][cls] = summary["by_second_stage_class"].get(cls, 0) + 1
        summary["by_instrument_family"][fam] = summary["by_instrument_family"].get(fam, 0) + 1
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result_rows_sorted, summary


def main() -> None:
    args = _parse_args()
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for raw_dir in args.prepared_dir:
        prepared_dir = Path(raw_dir).expanduser().resolve()
        rows, summary = _materialize_prepared_dir(prepared_dir)
        all_rows.extend(rows)
        summaries.append(summary)

    if args.merged_output_dir:
        merged_dir = Path(args.merged_output_dir).expanduser().resolve()
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_jsonl = merged_dir / "promotion_results_merged.jsonl"
        merged_csv = merged_dir / "promotion_results_merged.csv"
        merged_summary = merged_dir / "promotion_summary_merged.json"

        rows_sorted = sorted(all_rows, key=lambda row: (row["source_label"], row["issue_id"]))
        with merged_jsonl.open("w", encoding="utf-8") as f:
            for row in rows_sorted:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fieldnames = list(rows_sorted[0].keys()) if rows_sorted else []
        with merged_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)

        merged = {
            "prepared_dirs": [str(Path(p).expanduser().resolve()) for p in args.prepared_dir],
            "issues_total": len(rows_sorted),
            "promoted_to_full_ordinance": sum(1 for row in rows_sorted if row["promoted_to_full_ordinance"]),
            "by_source_label": {},
            "by_second_stage_class": {},
            "prepared_summaries": summaries,
        }
        for row in rows_sorted:
            src = row["source_label"] or "blank"
            cls = row["second_stage_class"] or "blank"
            merged["by_source_label"][src] = merged["by_source_label"].get(src, 0) + 1
            merged["by_second_stage_class"][cls] = merged["by_second_stage_class"].get(cls, 0) + 1
        merged_summary.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "prepared_dirs": [str(Path(p).expanduser().resolve()) for p in args.prepared_dir],
                "issues_total": len(all_rows),
                "promoted_to_full_ordinance": sum(1 for row in all_rows if row["promoted_to_full_ordinance"]),
                "merged_output_dir": str(Path(args.merged_output_dir).expanduser().resolve())
                if args.merged_output_dir
                else "",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
