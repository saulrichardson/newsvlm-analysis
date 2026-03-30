#!/usr/bin/env python3
"""Build a concrete page-recovery plan for confirmed Newspapers.com issues.

The goal is operational, not speculative: for each issue whose exact issue date
has been confirmed on Newspapers.com, describe the pages already attached in
the current corpus and the next pages to fetch.

Current planning rules:

1. Core fetch pages:
   - every missing page between the minimum and maximum attached page number
   - the immediate page before the first attached page, when it is > 1
   - the immediate page after the last attached page

2. Optional ring-2 fetch pages:
   - two pages before the first attached page, when it is > 2
   - two pages after the last attached page

This reflects the user request to plan around pages in-between and pages before
and after the currently attached set, while keeping one clean first-pass target
set.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_CONFIRMED_CSV = Path(
    "artifacts/reports/newspapers_com_availability_full_20260319_breadth_batch1/"
    "confirmed_exact_issue_dates_combined_after_breadth_batch1.csv"
)
DEFAULT_METADATA_CSV = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/"
    "raw_data/newspaper_ordinances/metadata.csv"
)
DEFAULT_AUDIT_CSV = Path("artifacts/reports/full_ordinance_missing_page_audit_20260313/audit.csv")
DEFAULT_COMPLETENESS_CSV = Path(
    "artifacts/scratch/full_ordinance_completeness_eval_20260318_full_gpt54_v10/"
    "all_predictions.csv"
)
DEFAULT_RAW_ROOT = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/"
    "raw_data/newspaper_ordinances/raw"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/newspapers_com_page_recovery_plan_20260319")

PAGE_NUM_RE = re.compile(r"-p-(\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmed-csv", type=Path, default=DEFAULT_CONFIRMED_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--completeness-csv", type=Path, default=DEFAULT_COMPLETENESS_CSV)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = [row for row in rows if not row.get(key)]
    if missing:
        raise ValueError(f"{path} has rows missing required key '{key}'")
    return {row[key]: row for row in rows}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def parse_attached_pages(issue_dir: Path) -> list[int]:
    if not issue_dir.exists():
        raise FileNotFoundError(f"Missing raw issue directory: {issue_dir}")

    pages: list[int] = []
    for image_path in sorted(issue_dir.glob("*.png")):
        match = PAGE_NUM_RE.search(image_path.name)
        if not match:
            raise ValueError(f"Could not parse page number from {image_path}")
        pages.append(int(match.group(1)))
    if not pages:
        raise ValueError(f"No PNGs found in {issue_dir}")
    return pages


def serialize_pages(pages: list[int]) -> str:
    return ",".join(str(page) for page in pages)


def compute_priority(audit_category: str, completeness_label: str, interior_missing_pages: list[int]) -> tuple[str, str]:
    if audit_category == "high_confidence_missing_other_pages":
        return ("P0", "Direct missing-page clue already present in parsed ordinance.")
    if audit_category == "incomplete_text_cause_unclear" or completeness_label in {"definitely_incomplete", "likely_incomplete"}:
        return ("P1", "Current ordinance is already flagged incomplete or likely incomplete.")
    if completeness_label == "uncertain" or interior_missing_pages:
        return ("P2", "Current ordinance is uncertain or has missing pages inside the attached span.")
    return ("P3", "No direct gap clue; fetch neighbors as a lower-priority completeness check.")


def build_plan_rows(
    confirmed_rows: list[dict[str, str]],
    metadata_by_issue: dict[str, dict[str, str]],
    audit_by_issue: dict[str, dict[str, str]],
    completeness_by_issue: dict[str, dict[str, str]],
    raw_root: Path,
) -> list[dict[str, str]]:
    plan_rows: list[dict[str, str]] = []

    for confirmed in confirmed_rows:
        issue_id = confirmed["issue_id"]
        metadata = metadata_by_issue.get(issue_id)
        audit = audit_by_issue.get(issue_id)
        completeness = completeness_by_issue.get(issue_id)
        if metadata is None:
            raise KeyError(f"{issue_id} missing from metadata")
        if audit is None:
            raise KeyError(f"{issue_id} missing from audit CSV")
        if completeness is None:
            raise KeyError(f"{issue_id} missing from completeness CSV")

        attached_pages = parse_attached_pages(raw_root / issue_id)
        first_page = min(attached_pages)
        last_page = max(attached_pages)

        interior_missing_pages = [
            page
            for page in range(first_page, last_page + 1)
            if page not in attached_pages
        ]
        immediate_boundary_pages = []
        if first_page > 1:
            immediate_boundary_pages.append(first_page - 1)
        immediate_boundary_pages.append(last_page + 1)

        core_fetch_pages = sorted(set(interior_missing_pages + immediate_boundary_pages))

        ring2_boundary_pages = []
        if first_page > 2:
            ring2_boundary_pages.append(first_page - 2)
        ring2_boundary_pages.append(last_page + 2)
        optional_ring2_pages = sorted(set(page for page in ring2_boundary_pages if page not in core_fetch_pages))

        priority_tier, priority_reason = compute_priority(
            audit["audit_category"],
            completeness["completeness_label"],
            interior_missing_pages,
        )

        plan_rows.append(
            {
                "issue_id": issue_id,
                "issue_date": confirmed["issue_date"],
                "newspaper_display_name": metadata["newspaper_slug"].replace("-", " ").title()
                if not confirmed.get("newspaper_display_name")
                else confirmed["newspaper_display_name"],
                "matched_paper_url": confirmed.get("matched_paper_url", ""),
                "exact_issue_url": confirmed.get("exact_issue_url", ""),
                "full_ordinance_origin": metadata["full_ordinance_origin"],
                "audit_category": audit["audit_category"],
                "completeness_label": completeness["completeness_label"],
                "estimated_missing_share_0_to_1": completeness["estimated_missing_share_0_to_1"],
                "raw_image_count_current": str(len(attached_pages)),
                "attached_pages": serialize_pages(attached_pages),
                "attached_first_page": str(first_page),
                "attached_last_page": str(last_page),
                "inbetween_missing_pages": serialize_pages(interior_missing_pages),
                "inbetween_missing_page_count": str(len(interior_missing_pages)),
                "immediate_boundary_pages": serialize_pages(immediate_boundary_pages),
                "core_fetch_pages": serialize_pages(core_fetch_pages),
                "core_fetch_page_count": str(len(core_fetch_pages)),
                "optional_ring2_pages": serialize_pages(optional_ring2_pages),
                "optional_ring2_page_count": str(len(optional_ring2_pages)),
                "priority_tier": priority_tier,
                "priority_reason": priority_reason,
                "parsed_relpath": metadata["parsed_relpath"],
                "raw_dir_relpath": metadata["raw_dir_relpath"],
                "transcript_relpath": metadata["transcript_relpath"],
            }
        )

    plan_rows.sort(
        key=lambda row: (
            row["priority_tier"],
            -int(row["core_fetch_page_count"]),
            row["issue_id"],
        )
    )
    return plan_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(plan_rows: list[dict[str, str]]) -> dict[str, object]:
    priority_issue_counts = Counter(row["priority_tier"] for row in plan_rows)
    priority_core_page_counts: Counter[str] = Counter()

    total_core_fetch_pages = 0
    total_optional_ring2_pages = 0
    issues_with_interior_gaps = 0
    single_page_issues = 0

    for row in plan_rows:
        priority_core_page_counts[row["priority_tier"]] += int(row["core_fetch_page_count"])
        total_core_fetch_pages += int(row["core_fetch_page_count"])
        total_optional_ring2_pages += int(row["optional_ring2_page_count"])
        if int(row["inbetween_missing_page_count"]) > 0:
            issues_with_interior_gaps += 1
        if int(row["raw_image_count_current"]) == 1:
            single_page_issues += 1

    top_core_fetch_issues = [
        {
            "issue_id": row["issue_id"],
            "core_fetch_page_count": int(row["core_fetch_page_count"]),
            "core_fetch_pages": row["core_fetch_pages"],
            "attached_pages": row["attached_pages"],
            "priority_tier": row["priority_tier"],
        }
        for row in sorted(
            plan_rows,
            key=lambda row: (-int(row["core_fetch_page_count"]), row["issue_id"]),
        )[:15]
    ]

    return {
        "confirmed_issue_count": len(plan_rows),
        "total_core_fetch_pages": total_core_fetch_pages,
        "total_optional_ring2_pages": total_optional_ring2_pages,
        "issues_with_interior_gaps": issues_with_interior_gaps,
        "single_page_issues": single_page_issues,
        "priority_issue_counts": dict(priority_issue_counts),
        "priority_core_page_counts": dict(priority_core_page_counts),
        "top_core_fetch_issues": top_core_fetch_issues,
    }


def write_markdown_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Confirmed Issue Page Recovery Plan",
        "",
        f"- Confirmed exact issues on Newspapers.com: {summary['confirmed_issue_count']}",
        f"- Core fetch pages total: {summary['total_core_fetch_pages']}",
        f"- Optional ring-2 pages total: {summary['total_optional_ring2_pages']}",
        f"- Issues with in-between gaps: {summary['issues_with_interior_gaps']}",
        f"- Issues with only one current raw page: {summary['single_page_issues']}",
        "",
        "## Priority split",
    ]

    priority_issue_counts = summary["priority_issue_counts"]
    priority_core_page_counts = summary["priority_core_page_counts"]
    for tier in ["P0", "P1", "P2", "P3"]:
        lines.append(
            f"- {tier}: {priority_issue_counts.get(tier, 0)} issues, "
            f"{priority_core_page_counts.get(tier, 0)} core fetch pages"
        )

    lines.extend(
        [
            "",
            "## Recovery rule",
            "- Core fetch: all missing pages between current attached pages, plus one page before the first attached page and one page after the last attached page.",
            "- Optional ring-2: two pages before the first attached page and two pages after the last attached page.",
            "",
            "## Largest core fetch sets",
        ]
    )

    for row in summary["top_core_fetch_issues"]:
        lines.append(
            f"- {row['issue_id']}: {row['core_fetch_page_count']} core pages "
            f"from attached {row['attached_pages']} -> fetch {row['core_fetch_pages']} "
            f"({row['priority_tier']})"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    confirmed_rows = read_csv_rows(args.confirmed_csv)
    metadata_by_issue = read_csv_by_key(args.metadata_csv, "issue_id")
    audit_by_issue = read_csv_by_key(args.audit_csv, "issue_id")
    completeness_by_issue = read_csv_by_key(args.completeness_csv, "issue_id")

    plan_rows = build_plan_rows(
        confirmed_rows=confirmed_rows,
        metadata_by_issue=metadata_by_issue,
        audit_by_issue=audit_by_issue,
        completeness_by_issue=completeness_by_issue,
        raw_root=args.raw_root,
    )
    if not plan_rows:
        raise ValueError("No plan rows were built")

    write_csv(args.output_dir / "confirmed_issue_page_recovery_plan.csv", plan_rows)
    summary = build_summary(plan_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown_report(args.output_dir / "report.md", summary)

    print(json.dumps(summary, indent=2))
    print(args.output_dir / "confirmed_issue_page_recovery_plan.csv")


if __name__ == "__main__":
    main()
