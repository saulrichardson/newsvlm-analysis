#!/usr/bin/env python3
"""Merge chunked Newspapers.com audit outputs into single deliverables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ISSUE_CSV_NAME = "newspapers_com_issue_availability.csv"
GROUPED_CSV_NAME = "newspapers_com_newspaper_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", dest="input_dirs", required=True, type=Path)
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
    all_issue_rows: list[dict[str, str]] = []
    all_grouped_rows: list[dict[str, str]] = []

    for input_dir in args.input_dirs:
        issue_path = input_dir / ISSUE_CSV_NAME
        grouped_path = input_dir / GROUPED_CSV_NAME
        if not issue_path.exists():
            raise FileNotFoundError(issue_path)
        if not grouped_path.exists():
            raise FileNotFoundError(grouped_path)
        all_issue_rows.extend(read_csv(issue_path))
        all_grouped_rows.extend(read_csv(grouped_path))

    issue_by_id = {}
    for row in all_issue_rows:
        issue_id = row["issue_id"]
        if issue_id in issue_by_id:
            raise ValueError(f"Duplicate issue_id across chunks: {issue_id}")
        issue_by_id[issue_id] = row

    newspaper_by_slug = {}
    for row in all_grouped_rows:
        slug = row["newspaper_slug"]
        if slug in newspaper_by_slug:
            raise ValueError(f"Duplicate newspaper_slug across chunks: {slug}")
        newspaper_by_slug[slug] = row

    merged_issue_rows = sorted(
        issue_by_id.values(),
        key=lambda item: (item["newspaper_slug"], item["issue_date"], item["issue_id"]),
    )
    merged_grouped_rows = sorted(
        newspaper_by_slug.values(),
        key=lambda item: item["newspaper_slug"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    issue_out = args.output_dir / ISSUE_CSV_NAME
    grouped_out = args.output_dir / GROUPED_CSV_NAME
    summary_out = args.output_dir / "summary.json"

    write_csv(issue_out, list(merged_issue_rows[0].keys()), merged_issue_rows)
    write_csv(grouped_out, list(merged_grouped_rows[0].keys()), merged_grouped_rows)

    summary = {
        "input_chunk_dirs": [str(path) for path in args.input_dirs],
        "merged_issue_rows": len(merged_issue_rows),
        "merged_newspaper_rows": len(merged_grouped_rows),
        "issue_status_counts": Counter(
            row["exact_issue_check_status"] for row in merged_issue_rows
        ),
        "match_status_counts": Counter(row["match_status"] for row in merged_issue_rows),
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote {issue_out}")
    print(f"Wrote {grouped_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
