#!/usr/bin/env python3
"""Merge slow retry results back into the main Newspapers.com audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-issue-csv", required=True, type=Path)
    parser.add_argument("--retry-csv", action="append", dest="retry_csvs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_rows = read_csv(args.base_issue_csv)
    if not base_rows:
        raise ValueError(f"No rows in {args.base_issue_csv}")

    retry_by_issue: dict[str, dict[str, str]] = {}
    for retry_csv in args.retry_csvs:
        for row in read_csv(retry_csv):
            issue_id = row["issue_id"]
            if issue_id in retry_by_issue:
                raise ValueError(f"Duplicate retry row for issue_id {issue_id}")
            retry_by_issue[issue_id] = row

    merged_rows = []
    merged_retry_count = 0
    for row in base_rows:
        retry = retry_by_issue.get(row["issue_id"])
        if retry is None:
            merged_rows.append(row)
            continue

        merged_retry_count += 1
        row = dict(row)
        row["exact_issue_url"] = retry["retry_exact_issue_url"]
        row["exact_issue_page_title"] = retry["retry_exact_issue_page_title"]
        row["exact_issue_check_status"] = retry["retry_status"]
        row["availability_reason"] = retry["retry_reason"]
        merged_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = args.output_dir / "newspapers_com_issue_availability_merged.csv"
    summary_json = args.output_dir / "summary.json"

    write_csv(merged_csv, list(base_rows[0].keys()), merged_rows)

    summary = {
        "base_issue_rows": len(base_rows),
        "retry_rows_merged": merged_retry_count,
        "exact_issue_check_status_counts": Counter(
            row["exact_issue_check_status"] for row in merged_rows
        ),
        "availability_reason_counts": Counter(row["availability_reason"] for row in merged_rows),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote {merged_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
