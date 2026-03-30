#!/usr/bin/env python3
"""Conservative retry pass for Newspapers.com rows blocked by Cloudflare.

This script re-checks previously blocked exact-date rows in a single headed
browser session with explicit pacing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright


OUTPUT_COLUMNS = [
    "issue_id",
    "issue_date",
    "newspaper_slug",
    "newspaper_display_name",
    "matched_paper_title",
    "matched_paper_url",
    "matched_paper_browse_base",
    "retry_exact_issue_url",
    "retry_final_url",
    "retry_exact_issue_page_title",
    "retry_status",
    "retry_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-availability-csv", required=True, type=Path)
    parser.add_argument("--storage-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--existing-retry-csv",
        type=Path,
        help="Optional existing retry CSV to resume from; completed issue_ids will be skipped.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--paper-wait-ms", type=int, default=5000)
    parser.add_argument("--browse-wait-ms", type=int, default=4000)
    parser.add_argument("--exact-wait-ms", type=int, default=7000)
    parser.add_argument("--blocked-retry-wait-ms", type=int, default=10000)
    parser.add_argument("--between-issues-wait-ms", type=int, default=6000)
    return parser.parse_args()


def load_blocked_rows(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="")))
    blocked = []
    for row in rows:
        title = row.get("exact_issue_page_title", "")
        if any(marker in title for marker in ["Cloudflare", "Access denied", "Just a moment", "Attention Required"]):
            blocked.append(row)
    return blocked


def sleep_ms(page, center_ms: int, jitter_ms: int = 1200) -> None:
    low = max(0, center_ms - jitter_ms)
    high = center_ms + jitter_ms
    page.wait_for_timeout(random.randint(low, high))


def classify_title(title: str, final_url: str, issue_date: str) -> tuple[str, str]:
    issue_dt = datetime.strptime(issue_date, "%Y-%m-%d")
    exact_date_fragment = f"{issue_dt.strftime('%b')} {issue_dt.day}, {issue_dt.year}"
    exact_path_suffix = issue_dt.strftime("/%Y/%m/%d/")

    if any(marker in title for marker in ["Cloudflare", "Access denied", "Just a moment", "Attention Required"]):
        return "blocked_cloudflare", "cloudflare_blocked"
    if exact_date_fragment in title and final_url.endswith(exact_path_suffix):
        return "available", "exact_issue_page_found"
    if not final_url.endswith(exact_path_suffix):
        return "unavailable", "redirected_away_from_exact_issue_path"
    return "unresolved_non_date_page", "exact_issue_page_not_confirmed"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "retried_rows": len(rows),
                "retry_status_counts": Counter(row["retry_status"] for row in rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    blocked_rows = load_blocked_rows(args.issue_availability_csv)
    blocked_rows = blocked_rows[args.offset :]
    if args.limit is not None:
        blocked_rows = blocked_rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    completed_issue_ids: set[str] = set()
    retry_csv = args.output_dir / "newspapers_com_blocked_retry_results.csv"
    summary_json = args.output_dir / "summary.json"

    if args.existing_retry_csv is not None:
        existing_rows = list(csv.DictReader(args.existing_retry_csv.open(newline="")))
        output_rows.extend(existing_rows)
        completed_issue_ids.update(row["issue_id"] for row in existing_rows)

    blocked_rows = [row for row in blocked_rows if row["issue_id"] not in completed_issue_ids]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False, slow_mo=200)
        context = browser.new_context(storage_state=str(args.storage_state))
        page = context.new_page()

        for idx, row in enumerate(blocked_rows, start=1):
            issue_url = row["exact_issue_url"]
            print(f"[{idx}/{len(blocked_rows)}] {row['issue_id']}", flush=True)

            page.goto(row["matched_paper_url"], wait_until="domcontentloaded")
            sleep_ms(page, args.paper_wait_ms)
            page.goto(row["matched_paper_browse_base"], wait_until="domcontentloaded")
            sleep_ms(page, args.browse_wait_ms)
            page.goto(issue_url, wait_until="domcontentloaded")
            sleep_ms(page, args.exact_wait_ms)

            title = page.title()
            final_url = page.url
            status, reason = classify_title(title, final_url, row["issue_date"])

            if status == "blocked_cloudflare":
                sleep_ms(page, args.blocked_retry_wait_ms)
                page.goto(issue_url, wait_until="domcontentloaded")
                sleep_ms(page, args.exact_wait_ms)
                title = page.title()
                final_url = page.url
                status, reason = classify_title(title, final_url, row["issue_date"])

            output_rows.append(
                {
                    "issue_id": row["issue_id"],
                    "issue_date": row["issue_date"],
                    "newspaper_slug": row["newspaper_slug"],
                    "newspaper_display_name": row["newspaper_display_name"],
                    "matched_paper_title": row["matched_paper_title"],
                    "matched_paper_url": row["matched_paper_url"],
                    "matched_paper_browse_base": row["matched_paper_browse_base"],
                    "retry_exact_issue_url": issue_url,
                    "retry_final_url": final_url,
                    "retry_exact_issue_page_title": title,
                    "retry_status": status,
                    "retry_reason": reason,
                }
            )

            write_csv(retry_csv, output_rows)
            write_summary(summary_json, output_rows)

            sleep_ms(page, args.between_issues_wait_ms)

        context.close()
        browser.close()

    write_csv(retry_csv, output_rows)
    write_summary(summary_json, output_rows)
    print(f"Wrote {retry_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
