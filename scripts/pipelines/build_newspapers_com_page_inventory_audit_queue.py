#!/usr/bin/env python3
"""Build a prioritized issue-by-issue queue for Newspapers.com page inventory checks.

This joins the live full-ordinance metadata with:
- the full-ordinance completeness eval
- the missing-page audit
- the current Newspapers.com paper-match audit

The output is a queue designed for low-volume, manual or semi-manual browsing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SEVERITY_ORDER = {
    "definitely_incomplete": 0,
    "likely_incomplete": 1,
    "uncertain": 2,
    "complete_or_nearly_complete": 3,
}

AUDIT_ORDER = {
    "high_confidence_missing_other_pages": 0,
    "incomplete_text_cause_unclear": 1,
    "no_direct_gap_clue": 2,
}

MATCH_ORDER = {
    "matched": 0,
    "ambiguous": 1,
    "matched_no_browse_base": 2,
    "not_found": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path(
            "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/newspaper_ordinances/metadata.csv"
        ),
    )
    parser.add_argument(
        "--completeness-csv",
        type=Path,
        default=Path(
            "artifacts/scratch/full_ordinance_completeness_eval_20260318_full_gpt54_v10/all_predictions.csv"
        ),
    )
    parser.add_argument(
        "--missing-page-audit-csv",
        type=Path,
        default=Path("artifacts/reports/full_ordinance_missing_page_audit_20260313/audit.csv"),
    )
    parser.add_argument(
        "--availability-csv",
        type=Path,
        default=Path("artifacts/reports/newspapers_com_availability_full_20260318/newspapers_com_issue_availability.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/newspapers_com_page_inventory_audit_20260318"),
    )
    return parser.parse_args()


def read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row[key]: row for row in rows}


def parse_raw_page_numbers(raw_dir: Path) -> list[int]:
    page_numbers: list[int] = []
    for png in sorted(raw_dir.glob("*.png")):
        match = re.search(r"-p-(\d+)\.png$", png.name)
        if match:
            page_numbers.append(int(match.group(1)))
    return page_numbers


def format_page_numbers(page_numbers: list[int]) -> str:
    return ",".join(str(page) for page in page_numbers)


def recommended_next_action(match_status: str, availability_reason: str) -> str:
    if match_status == "matched":
        return "browse_exact_issue_and_record_visible_pages"
    if match_status == "ambiguous":
        return "resolve_paper_match_before_issue_browse"
    if match_status == "matched_no_browse_base":
        return "recover_browse_base_from_matched_paper_page"
    if availability_reason == "outside_paper_year_range":
        return "recheck_paper_match_or_confirm_year_gap"
    return "improve_paper_match_search"


def priority_group(
    match_status: str,
    completeness_label: str,
    audit_category: str,
) -> str:
    if match_status == "matched" and audit_category == "high_confidence_missing_other_pages":
        return "P0_matched_high_confidence_missing"
    if match_status == "matched" and completeness_label in {
        "definitely_incomplete",
        "likely_incomplete",
        "uncertain",
    }:
        return "P1_matched_model_incomplete_or_uncertain"
    if match_status == "matched":
        return "P2_matched_model_complete"
    if match_status in {"ambiguous", "matched_no_browse_base"}:
        return "P3_fix_match_then_browse"
    return "P4_unmatched_titles"


def build_output_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    metadata_by_issue = read_csv_by_key(args.metadata_csv, "issue_id")
    completeness_by_issue = read_csv_by_key(args.completeness_csv, "issue_id")
    audit_by_issue = read_csv_by_key(args.missing_page_audit_csv, "issue_id")
    availability_by_issue = read_csv_by_key(args.availability_csv, "issue_id")

    metadata_root = args.metadata_csv.parent
    rows: list[dict[str, str]] = []
    for issue_id, meta in metadata_by_issue.items():
        completeness = completeness_by_issue[issue_id]
        audit = audit_by_issue[issue_id]
        availability = availability_by_issue[issue_id]

        raw_dir = metadata_root / meta["raw_dir_relpath"]
        raw_page_numbers = parse_raw_page_numbers(raw_dir)

        estimated_missing_share = completeness["estimated_missing_share_0_to_1"]
        try:
            estimated_missing_share_float = float(estimated_missing_share)
        except ValueError:
            estimated_missing_share_float = -1.0

        row = {
            "issue_id": issue_id,
            "issue_date": meta["issue_date"],
            "newspaper_slug": meta["newspaper_slug"],
            "newspaper_display_name": availability["newspaper_display_name"],
            "publication_city_name": meta["newspaperarchive_publication_city_name"],
            "publication_state_abbr": meta["newspaperarchive_publication_state_abbr"],
            "full_ordinance_origin": meta["full_ordinance_origin"],
            "parsed_relpath": meta["parsed_relpath"],
            "raw_dir_relpath": meta["raw_dir_relpath"],
            "attached_raw_page_count": str(len(raw_page_numbers)),
            "attached_raw_page_numbers_csv": format_page_numbers(raw_page_numbers),
            "attached_raw_filenames_json": json.dumps(sorted(p.name for p in raw_dir.glob("*.png"))),
            "heuristic_audit_category": audit["audit_category"],
            "model_completeness_label": completeness["completeness_label"],
            "estimated_missing_share_0_to_1": estimated_missing_share,
            "missing_share_band": completeness["missing_share_band"],
            "model_confidence_0_to_1": completeness["confidence_0_to_1"],
            "newspapers_com_match_status": availability["match_status"],
            "newspapers_com_availability_reason": availability["availability_reason"],
            "newspapers_com_matched_paper_title": availability["matched_paper_title"],
            "newspapers_com_matched_paper_location": availability["matched_paper_location"],
            "newspapers_com_matched_paper_date_range": availability["matched_paper_date_range"],
            "newspapers_com_matched_paper_url": availability["matched_paper_url"],
            "newspapers_com_matched_paper_browse_base": availability["matched_paper_browse_base"],
            "newspapers_com_exact_issue_url": availability["exact_issue_url"],
            "newspapers_com_previous_exact_issue_check_status": availability["exact_issue_check_status"],
            "newspapers_com_previous_exact_issue_page_title": availability["exact_issue_page_title"],
            "recommended_next_action": recommended_next_action(
                availability["match_status"], availability["availability_reason"]
            ),
            "priority_group": priority_group(
                availability["match_status"],
                completeness["completeness_label"],
                audit["audit_category"],
            ),
            "_sort_match": str(MATCH_ORDER.get(availability["match_status"], 9)),
            "_sort_audit": str(AUDIT_ORDER.get(audit["audit_category"], 9)),
            "_sort_severity": str(SEVERITY_ORDER.get(completeness["completeness_label"], 9)),
            "_sort_missing_share": f"{-estimated_missing_share_float:.6f}",
            "_sort_raw_count": f"{len(raw_page_numbers):04d}",
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["priority_group"],
            int(row["_sort_match"]),
            int(row["_sort_audit"]),
            int(row["_sort_severity"]),
            float(row["_sort_missing_share"]),
            int(row["_sort_raw_count"]),
            row["issue_id"],
        )
    )

    for idx, row in enumerate(rows, start=1):
        row["priority_rank"] = str(idx)

    return rows


def write_csv(path: Path, rows: list[dict[str, str]], drop_internal_sort_fields: bool) -> None:
    if not rows:
        raise ValueError("No rows to write")
    if drop_internal_sort_fields:
        cleaned_rows = []
        for row in rows:
            cleaned_rows.append({k: v for k, v in row.items() if not k.startswith("_")})
        rows = cleaned_rows
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_results_template(path: Path, queue_rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "priority_rank",
        "issue_id",
        "issue_date",
        "newspaper_display_name",
        "attached_raw_page_numbers_csv",
        "check_status",
        "checked_at",
        "newspapers_com_issue_page_confirmed",
        "newspapers_com_visible_page_numbers_csv",
        "newspapers_com_visible_page_count",
        "has_additional_pages_beyond_attached_raw",
        "additional_page_numbers_csv",
        "notes",
    ]
    rows = [
        {
            "priority_rank": row["priority_rank"],
            "issue_id": row["issue_id"],
            "issue_date": row["issue_date"],
            "newspaper_display_name": row["newspaper_display_name"],
            "attached_raw_page_numbers_csv": row["attached_raw_page_numbers_csv"],
            "check_status": "pending",
            "checked_at": "",
            "newspapers_com_issue_page_confirmed": "",
            "newspapers_com_visible_page_numbers_csv": "",
            "newspapers_com_visible_page_count": "",
            "has_additional_pages_beyond_attached_raw": "",
            "additional_page_numbers_csv": "",
            "notes": "",
        }
        for row in queue_rows
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    summary = {
        "issue_rows": len(rows),
        "priority_group_counts": Counter(row["priority_group"] for row in rows),
        "match_status_counts": Counter(row["newspapers_com_match_status"] for row in rows),
        "completeness_label_counts": Counter(row["model_completeness_label"] for row in rows),
        "heuristic_audit_category_counts": Counter(row["heuristic_audit_category"] for row in rows),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    rows = build_output_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    queue_csv = args.output_dir / "issue_page_inventory_queue.csv"
    results_template_csv = args.output_dir / "issue_page_inventory_results_template.csv"
    summary_json = args.output_dir / "summary.json"

    write_csv(queue_csv, rows, drop_internal_sort_fields=True)
    write_results_template(results_template_csv, rows)
    write_summary(summary_json, rows)

    print(f"Wrote {queue_csv}")
    print(f"Wrote {results_template_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
