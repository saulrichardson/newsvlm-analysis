#!/usr/bin/env python3
"""
Rehydrate page-level v5 classifications from OpenAI Batch issue-level outputs.

This script reads:
  - mapping_shard*.jsonl (expected pages per custom_id)
  - *_output.jsonl (OpenAI batch completed outputs)

And writes:
  - page_classification_outputs.jsonl
  - page_classification_outputs.csv
  - issue_parse_audit.csv
  - classification_tally_summary.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fullnewspaper_v3_common import extract_openai_output_text, norm_str, parse_json_from_text


ALLOWED_YNU = {"yes", "no", "uncertain"}
ALLOWED_ENACT = {"enacted", "proposed", "unknown"}
ALLOWED_ZONING_DOMAIN = {"zoning", "other_law", "none", "uncertain"}
ALLOWED_DOCUMENT_FORM = {"ordinance", "amendment_or_rezoning", "legal_notice", "uncertain", "n/a"}
ALLOWED_SCOPE = {"comprehensive_citywide_code", "noncomprehensive_zoning_ordinance", "unknown", "n/a"}
ALLOWED_EVIDENCE_STRENGTH = {"strong", "moderate", "weak", "none"}
ALLOWED_FRAGMENT = {"yes", "no", "uncertain", "n/a"}
ALLOWED_PAGE_CLASS = {
    "zoning_ordinance_comprehensive",
    "zoning_ordinance_noncomprehensive",
    "zoning_amendment_or_rezoning",
    "zoning_legal_notice",
    "building_code_or_other_law",
    "zoning_narrative_nonverbatim",
    "non_zoning",
    "uncertain",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rehydrate page-level v5 labels from OpenAI batch outputs.")
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing requests/mapping_shard*.jsonl and openai_batch_submission/completed_outputs/*.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-dir>/rehydrated_v5_page_labels).",
    )
    ap.add_argument(
        "--mapping-glob",
        default="requests/mapping_shard*.jsonl",
        help="Glob (relative to run-dir) for mapping shards.",
    )
    ap.add_argument(
        "--completed-glob",
        default="openai_batch_submission/completed_outputs/*_output.jsonl",
        help="Glob (relative to run-dir) for completed batch outputs.",
    )
    ap.add_argument(
        "--prefer-latest-duplicate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When duplicate custom_id responses exist, keep the last seen row.",
    )
    return ap.parse_args()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame.from_records(rows).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _coerce_page_obj(obj: dict[str, Any], page_meta: dict[str, Any]) -> dict[str, Any]:
    is_verbatim = norm_str(obj.get("is_verbatim_legal_text")).lower()
    if is_verbatim not in ALLOWED_YNU:
        is_verbatim = "uncertain"

    zoning_presence = norm_str(obj.get("zoning_legal_text_presence")).lower()
    if zoning_presence not in ALLOWED_YNU:
        zoning_presence = "uncertain"

    zoning_domain = norm_str(obj.get("zoning_legal_domain")).lower()
    if zoning_domain not in ALLOWED_ZONING_DOMAIN:
        zoning_domain = "uncertain"

    zoning_form = norm_str(obj.get("zoning_legal_document_form")).lower()
    if zoning_form not in ALLOWED_DOCUMENT_FORM:
        zoning_form = "uncertain"

    ordinance_scope = norm_str(obj.get("ordinance_scope_signal")).lower()
    if ordinance_scope not in ALLOWED_SCOPE:
        ordinance_scope = "unknown"

    enactment_strength = norm_str(obj.get("enactment_evidence_strength")).lower()
    if enactment_strength not in ALLOWED_EVIDENCE_STRENGTH:
        enactment_strength = "none"

    is_fragment = norm_str(obj.get("is_fragment")).lower()
    if is_fragment not in ALLOWED_FRAGMENT:
        is_fragment = "uncertain"

    page_class = norm_str(obj.get("page_class")).lower()
    if page_class not in ALLOWED_PAGE_CLASS:
        page_class = "uncertain"

    contains = norm_str(obj.get("contains_verbatim_zoning_law_language")).lower()
    if contains not in ALLOWED_YNU:
        contains = zoning_presence

    enact = norm_str(obj.get("enactment_status_signal")).lower()
    if enact not in ALLOWED_ENACT:
        enact = "unknown"

    conf = obj.get("confidence_0_to_1")
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    conf_f = max(0.0, min(1.0, conf_f))

    evidence_quotes: list[str] = []
    raw_quotes = obj.get("evidence_quotes")
    if isinstance(raw_quotes, list):
        evidence_quotes = [str(x).strip() for x in raw_quotes if str(x).strip()][:3]

    return {
        "page_id": norm_str(obj.get("page_id")) or norm_str(page_meta.get("page_id")),
        "issue_id": norm_str(page_meta.get("issue_id")),
        "issue_date": norm_str(page_meta.get("issue_date")),
        "newspaper_slug": norm_str(page_meta.get("newspaper_slug")),
        "page_num": int(page_meta.get("page_num") or 0),
        "contains_verbatim_zoning_law_language": contains,
        "is_verbatim_legal_text": is_verbatim,
        "zoning_legal_text_presence": zoning_presence,
        "zoning_legal_domain": zoning_domain,
        "zoning_legal_document_form": zoning_form,
        "ordinance_scope_signal": ordinance_scope,
        "enactment_evidence_strength": enactment_strength,
        "is_fragment": is_fragment,
        "page_class": page_class,
        "label_schema": "v5_page_class",
        "enactment_status_signal": enact,
        "confidence_0_to_1": conf_f,
        "evidence_quotes": evidence_quotes,
        "notes": norm_str(obj.get("notes")),
    }


def _extract_issue_page_outputs(parsed: dict[str, Any], expected_page_ids: set[str]) -> dict[str, dict[str, Any]]:
    page_lists: list[Any] = []
    for key in ("page_outputs", "pages", "results", "classifications", "page_results"):
        val = parsed.get(key)
        if isinstance(val, list):
            page_lists.append(val)
    if not page_lists and parsed.get("page_id"):
        page_lists = [[parsed]]

    out: dict[str, dict[str, Any]] = {}
    for lst in page_lists:
        for item in lst:
            if not isinstance(item, dict):
                continue
            pid = norm_str(item.get("page_id")).lower()
            if not pid:
                continue
            if expected_page_ids and pid not in expected_page_ids:
                continue
            out[pid] = item
    return out


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if norm_str(args.output_dir) else run_dir / "rehydrated_v5_page_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_files = sorted(run_dir.glob(args.mapping_glob))
    output_files = sorted(run_dir.glob(args.completed_glob))
    if not mapping_files:
        raise SystemExit(f"No mapping files matched: {args.mapping_glob}")
    if not output_files:
        raise SystemExit(f"No completed output files matched: {args.completed_glob}")

    # custom_id -> ordered page metas for issue
    mapping_by_cid: dict[str, list[dict[str, Any]]] = {}
    for mp in mapping_files:
        for row in _iter_jsonl(mp):
            cid = norm_str(row.get("custom_id"))
            if not cid:
                continue
            mapping_by_cid.setdefault(cid, []).append(
                {
                    "custom_id": cid,
                    "issue_id": norm_str(row.get("issue_id")),
                    "issue_date": norm_str(row.get("issue_date")),
                    "newspaper_slug": norm_str(row.get("newspaper_slug")),
                    "page_id": norm_str(row.get("page_id")).lower(),
                    "page_num": int(row.get("page_num") or 0),
                    "text_chars": int(row.get("text_chars") or 0),
                    "text_sha256": norm_str(row.get("text_sha256")),
                    "vlm_path": norm_str(row.get("vlm_path")),
                }
            )
    for cid, rows in list(mapping_by_cid.items()):
        dedup: dict[str, dict[str, Any]] = {}
        for r in sorted(rows, key=lambda x: (int(x.get("page_num") or 0), norm_str(x.get("page_id")))):
            dedup[norm_str(r.get("page_id"))] = r
        mapping_by_cid[cid] = list(dedup.values())

    # custom_id -> raw output object (dedup by custom_id)
    result_by_cid: dict[str, dict[str, Any]] = {}
    duplicate_result_rows = 0
    for op in output_files:
        for row in _iter_jsonl(op):
            cid = norm_str(row.get("custom_id"))
            if not cid:
                continue
            if cid in result_by_cid:
                duplicate_result_rows += 1
                if not bool(args.prefer_latest_duplicate):
                    continue
            result_by_cid[cid] = row

    parse_audit_rows: list[dict[str, Any]] = []
    out_rows: list[dict[str, Any]] = []
    issue_parse_ok = 0
    issue_parse_fail = 0
    issue_page_count_mismatch = 0

    for cid in sorted(mapping_by_cid.keys()):
        pages = mapping_by_cid.get(cid, [])
        expected_page_ids = {norm_str(r.get("page_id")).lower() for r in pages if norm_str(r.get("page_id"))}

        had_response = 0
        parse_ok = 0
        output_text = ""
        parsed: dict[str, Any] | None = None
        parsed_pages: dict[str, dict[str, Any]] = {}

        if cid in result_by_cid:
            had_response = 1
            body = ((result_by_cid[cid].get("response") or {}).get("body")) or {}
            if isinstance(body, dict):
                output_text = extract_openai_output_text(body)
                parsed = parse_json_from_text(output_text)

        if isinstance(parsed, dict):
            parsed_pages = _extract_issue_page_outputs(parsed, expected_page_ids)
            if parsed_pages:
                parse_ok = 1

        if parse_ok:
            issue_parse_ok += 1
        else:
            issue_parse_fail += 1

        if parse_ok and len(parsed_pages) != len(expected_page_ids):
            issue_page_count_mismatch += 1

        parse_audit_rows.append(
            {
                "custom_id": cid,
                "issue_id": norm_str(pages[0].get("issue_id")) if pages else "",
                "newspaper_slug": norm_str(pages[0].get("newspaper_slug")) if pages else "",
                "expected_page_count": int(len(expected_page_ids)),
                "parsed_page_count": int(len(parsed_pages)),
                "had_response": int(had_response),
                "parse_ok": int(parse_ok),
                "page_count_mismatch": int(parse_ok and (len(parsed_pages) != len(expected_page_ids))),
                "output_text_chars": int(len(output_text)),
            }
        )

        for page_meta in pages:
            pid = norm_str(page_meta.get("page_id")).lower()
            page_obj = parsed_pages.get(pid) or {}
            row = _coerce_page_obj(page_obj, page_meta)
            row["custom_id"] = cid
            row["parse_ok"] = int(1 if page_obj else 0)
            row["had_response"] = int(had_response)
            row["text_chars"] = int(page_meta.get("text_chars") or 0)
            row["text_sha256"] = norm_str(page_meta.get("text_sha256"))
            row["vlm_path"] = norm_str(page_meta.get("vlm_path"))
            out_rows.append(row)

    out_rows = sorted(
        out_rows,
        key=lambda x: (
            norm_str(x.get("newspaper_slug")),
            norm_str(x.get("issue_date")),
            int(x.get("page_num") or 0),
            norm_str(x.get("page_id")),
        ),
    )

    _write_jsonl(out_dir / "page_classification_outputs.jsonl", out_rows)
    _write_csv(out_dir / "page_classification_outputs.csv", out_rows)
    _write_csv(out_dir / "issue_parse_audit.csv", parse_audit_rows)

    # Strict V5 tallies.
    df = pd.DataFrame.from_records(out_rows)
    page_class_counts = (
        df["page_class"].value_counts(dropna=False).to_dict()
        if (not df.empty and "page_class" in df.columns)
        else {}
    )
    issue_primary_v5_counts: dict[str, int] = {}
    if not df.empty and "issue_id" in df.columns and "page_class" in df.columns:
        pri = {lab: i for i, lab in enumerate(sorted(ALLOWED_PAGE_CLASS))}
        tmp = (
            df.groupby(["issue_id", "page_class"], dropna=False)["page_id"]
            .count()
            .reset_index(name="n")
            .sort_values(
                ["issue_id", "n", "page_class"],
                ascending=[True, False, True],
                kind="stable",
            )
        )
        dominant = tmp.groupby("issue_id", as_index=False).first()
        issue_primary_v5_counts = (
            dominant["page_class"].value_counts(dropna=False).to_dict()
        )

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "mapping_files": [str(p) for p in mapping_files],
        "completed_output_files": [str(p) for p in output_files],
        "requests_total": int(len(mapping_by_cid)),
        "responses_total": int(len(result_by_cid)),
        "duplicate_result_rows_removed": int(duplicate_result_rows),
        "issues_total": int(len(mapping_by_cid)),
        "issues_parse_ok": int(issue_parse_ok),
        "issues_parse_fail": int(issue_parse_fail),
        "issues_page_count_mismatch": int(issue_page_count_mismatch),
        "pages_total_from_outputs": int(len(out_rows)),
        "page_class_counts": {str(k): int(v) for k, v in page_class_counts.items()},
        "issue_primary_page_class_counts": {
            str(k): int(v) for k, v in issue_primary_v5_counts.items()
        },
    }
    (out_dir / "classification_tally_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"done_rehydrate_issue_batch_page_classification_v5 "
        f"issues={summary['issues_total']} pages={summary['pages_total_from_outputs']} out={out_dir}"
    )


if __name__ == "__main__":
    main()
