#!/usr/bin/env python3
"""Build a simple rule-only page fetch plan for confirmed Newspapers.com issues.

This intentionally ignores missingness clues, model outputs, and prior audits.
It only uses:
  - the confirmed exact-issue set
  - the raw page numbers already attached for each issue

Rules:
1. Core rule:
   - fetch every page missing between the minimum and maximum attached page
   - fetch the immediate page before the first attached page, when > 1
   - fetch the immediate page after the last attached page

2. Extended rule:
   - add two pages before the first attached page, when > 2
   - add two pages after the last attached page

The core rule is the intended first-pass download target.
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
DEFAULT_RAW_ROOT = Path(
    "/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/"
    "raw_data/newspaper_ordinances/raw"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/newspapers_com_rule_only_fetch_plan_20260319")

PAGE_NUM_RE = re.compile(r"-p-(\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmed-csv", type=Path, default=DEFAULT_CONFIRMED_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    return {row[key]: row for row in rows}


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


def build_issue_plan_row(
    confirmed_row: dict[str, str],
    metadata_row: dict[str, str],
    raw_root: Path,
) -> dict[str, str]:
    issue_id = confirmed_row["issue_id"]
    attached_pages = parse_attached_pages(raw_root / issue_id)
    first_page = min(attached_pages)
    last_page = max(attached_pages)

    inbetween_missing_pages = [
        page
        for page in range(first_page, last_page + 1)
        if page not in attached_pages
    ]

    immediate_before_after = []
    if first_page > 1:
        immediate_before_after.append(first_page - 1)
    immediate_before_after.append(last_page + 1)

    core_fetch_pages = sorted(set(inbetween_missing_pages + immediate_before_after))

    ring2_pages = []
    if first_page > 2:
        ring2_pages.append(first_page - 2)
    ring2_pages.append(last_page + 2)

    extended_fetch_pages = sorted(set(core_fetch_pages + ring2_pages))

    return {
        "issue_id": issue_id,
        "issue_date": confirmed_row["issue_date"],
        "newspaper_display_name": confirmed_row.get("newspaper_display_name", ""),
        "matched_paper_url": confirmed_row.get("matched_paper_url", ""),
        "exact_issue_url": confirmed_row.get("exact_issue_url", ""),
        "raw_dir_relpath": metadata_row["raw_dir_relpath"],
        "parsed_relpath": metadata_row["parsed_relpath"],
        "transcript_relpath": metadata_row["transcript_relpath"],
        "attached_pages": serialize_pages(attached_pages),
        "attached_page_count": str(len(attached_pages)),
        "attached_first_page": str(first_page),
        "attached_last_page": str(last_page),
        "inbetween_missing_pages": serialize_pages(inbetween_missing_pages),
        "inbetween_missing_page_count": str(len(inbetween_missing_pages)),
        "core_fetch_pages": serialize_pages(core_fetch_pages),
        "core_fetch_page_count": str(len(core_fetch_pages)),
        "extended_fetch_pages": serialize_pages(extended_fetch_pages),
        "extended_fetch_page_count": str(len(extended_fetch_pages)),
        "rule_name": "fill_between_plus_one_before_after",
        "rule_notes": (
            "Core pages include all missing pages between current attachments plus "
            "one page before the first attachment and one page after the last."
        ),
    }


def build_page_manifest(issue_rows: list[dict[str, str]], page_field: str) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    for issue_row in issue_rows:
        pages = [
            int(page)
            for page in issue_row[page_field].split(",")
            if page.strip()
        ]
        for page_num in pages:
            manifest_rows.append(
                {
                    "issue_id": issue_row["issue_id"],
                    "issue_date": issue_row["issue_date"],
                    "newspaper_display_name": issue_row["newspaper_display_name"],
                    "matched_paper_url": issue_row["matched_paper_url"],
                    "exact_issue_url": issue_row["exact_issue_url"],
                    "page_num": str(page_num),
                }
            )
    manifest_rows.sort(key=lambda row: (row["issue_id"], int(row["page_num"])))
    return manifest_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(issue_rows: list[dict[str, str]], core_manifest_rows: list[dict[str, str]], extended_manifest_rows: list[dict[str, str]]) -> dict[str, object]:
    single_page_issue_count = sum(
        1 for row in issue_rows if int(row["attached_page_count"]) == 1
    )
    interior_gap_issue_count = sum(
        1 for row in issue_rows if int(row["inbetween_missing_page_count"]) > 0
    )
    attached_page_count_distribution = Counter(
        row["attached_page_count"] for row in issue_rows
    )
    largest_core_fetch_issues = [
        {
            "issue_id": row["issue_id"],
            "attached_pages": row["attached_pages"],
            "core_fetch_pages": row["core_fetch_pages"],
            "core_fetch_page_count": int(row["core_fetch_page_count"]),
        }
        for row in sorted(
            issue_rows,
            key=lambda row: (-int(row["core_fetch_page_count"]), row["issue_id"]),
        )[:20]
    ]

    return {
        "confirmed_issue_count": len(issue_rows),
        "core_fetch_page_total": len(core_manifest_rows),
        "extended_fetch_page_total": len(extended_manifest_rows),
        "single_page_issue_count": single_page_issue_count,
        "interior_gap_issue_count": interior_gap_issue_count,
        "attached_page_count_distribution": dict(attached_page_count_distribution),
        "largest_core_fetch_issues": largest_core_fetch_issues,
    }


def write_markdown_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Rule-Only Confirmed Issue Fetch Plan",
        "",
        "- Rule set ignores clue-based prioritization and only uses currently attached page numbers.",
        f"- Confirmed exact issues: {summary['confirmed_issue_count']}",
        f"- Core fetch pages: {summary['core_fetch_page_total']}",
        f"- Extended fetch pages: {summary['extended_fetch_page_total']}",
        f"- Single-page issues: {summary['single_page_issue_count']}",
        f"- Issues with in-between gaps: {summary['interior_gap_issue_count']}",
        "",
        "## Rule",
        "- Core: all missing pages between the first and last attached page, plus one page before and one page after.",
        "- Extended: core plus two pages before and two pages after.",
        "",
        "## Largest core fetch sets",
    ]

    for row in summary["largest_core_fetch_issues"]:
        lines.append(
            f"- {row['issue_id']}: attached {row['attached_pages']} -> core fetch {row['core_fetch_pages']} "
            f"({row['core_fetch_page_count']} pages)"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    confirmed_rows = read_csv_rows(args.confirmed_csv)
    metadata_by_issue = read_csv_by_key(args.metadata_csv, "issue_id")

    issue_rows = [
        build_issue_plan_row(
            confirmed_row=confirmed_row,
            metadata_row=metadata_by_issue[confirmed_row["issue_id"]],
            raw_root=args.raw_root,
        )
        for confirmed_row in confirmed_rows
    ]
    issue_rows.sort(
        key=lambda row: (-int(row["core_fetch_page_count"]), row["issue_id"])
    )

    core_manifest_rows = build_page_manifest(issue_rows, "core_fetch_pages")
    extended_manifest_rows = build_page_manifest(issue_rows, "extended_fetch_pages")

    write_csv(args.output_dir / "confirmed_issue_rule_only_fetch_plan.csv", issue_rows)
    write_csv(args.output_dir / "confirmed_issue_rule_only_core_page_manifest.csv", core_manifest_rows)
    write_csv(
        args.output_dir / "confirmed_issue_rule_only_extended_page_manifest.csv",
        extended_manifest_rows,
    )

    summary = build_summary(issue_rows, core_manifest_rows, extended_manifest_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown_report(args.output_dir / "report.md", summary)

    print(json.dumps(summary, indent=2))
    print(args.output_dir / "confirmed_issue_rule_only_fetch_plan.csv")


if __name__ == "__main__":
    main()
