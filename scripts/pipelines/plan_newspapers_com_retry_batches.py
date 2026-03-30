#!/usr/bin/env python3
"""Create a resumable batch manifest for Newspapers.com blocked retries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-availability-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.issue_availability_csv.open(newline="")))
    blocked = []
    for row in rows:
        title = row.get("exact_issue_page_title", "")
        if any(marker in title for marker in ["Cloudflare", "Access denied", "Just a moment", "Attention Required"]):
            blocked.append(row)

    blocked = sorted(blocked, key=lambda row: (row["newspaper_slug"], row["issue_date"], row["issue_id"]))
    output_rows = []
    for idx, row in enumerate(blocked):
        batch_number = idx // args.batch_size + 1
        batch_offset = idx
        output_rows.append(
            {
                "batch_number": batch_number,
                "batch_index_within_batch": idx % args.batch_size,
                "global_blocked_index": idx,
                "issue_id": row["issue_id"],
                "issue_date": row["issue_date"],
                "newspaper_slug": row["newspaper_slug"],
                "newspaper_display_name": row["newspaper_display_name"],
                "matched_paper_title": row["matched_paper_title"],
                "matched_paper_url": row["matched_paper_url"],
                "matched_paper_browse_base": row["matched_paper_browse_base"],
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
