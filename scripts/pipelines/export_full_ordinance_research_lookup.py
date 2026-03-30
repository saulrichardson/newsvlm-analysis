#!/usr/bin/env python3
"""Export research-agent handoff files for the live full-ordinance corpus.

This script reads the public full-ordinance metadata CSV and writes:

1. An issue-level CSV with one row per required issue date.
2. A grouped-by-newspaper CSV with all required dates for each title.
3. A plain-text checklist grouped by newspaper for quick handoff.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ISSUE_COLUMNS = [
    "issue_id",
    "issue_date",
    "newspaper_slug",
    "newspaper_display_name",
    "newspaperarchive_publication_city_name",
    "newspaperarchive_publication_state_abbr",
    "extracted_jurisdiction_city_name",
    "extracted_jurisdiction_state_abbr",
    "page_count",
    "raw_image_count",
    "full_ordinance_origin",
]

GROUPED_COLUMNS = [
    "newspaper_slug",
    "newspaper_display_name",
    "newspaperarchive_publication_city_name",
    "newspaperarchive_publication_state_abbr",
    "issue_count",
    "earliest_issue_date",
    "latest_issue_date",
    "needed_issue_dates",
    "needed_issue_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def slug_to_display_name(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("-"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {
        "issue_id",
        "issue_date",
        "newspaper_slug",
        "newspaperarchive_publication_city_name",
        "newspaperarchive_publication_state_abbr",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def build_issue_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    issue_rows = []
    for row in sorted(rows, key=lambda item: (item["newspaper_slug"], item["issue_date"], item["issue_id"])):
        issue_rows.append(
            {
                "issue_id": row["issue_id"],
                "issue_date": row["issue_date"],
                "newspaper_slug": row["newspaper_slug"],
                "newspaper_display_name": slug_to_display_name(row["newspaper_slug"]),
                "newspaperarchive_publication_city_name": row.get(
                    "newspaperarchive_publication_city_name", ""
                ),
                "newspaperarchive_publication_state_abbr": row.get(
                    "newspaperarchive_publication_state_abbr", ""
                ),
                "extracted_jurisdiction_city_name": row.get("extracted_jurisdiction_city_name", ""),
                "extracted_jurisdiction_state_abbr": row.get("extracted_jurisdiction_state_abbr", ""),
                "page_count": row.get("page_count", ""),
                "raw_image_count": row.get("raw_image_count", ""),
                "full_ordinance_origin": row.get("full_ordinance_origin", ""),
            }
        )
    return issue_rows


def build_grouped_rows(issue_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in issue_rows:
        grouped[row["newspaper_slug"]].append(row)

    output_rows = []
    for slug in sorted(grouped):
        entries = sorted(grouped[slug], key=lambda item: (item["issue_date"], item["issue_id"]))
        dates = [entry["issue_date"] for entry in entries]
        issue_ids = [entry["issue_id"] for entry in entries]
        first = entries[0]
        output_rows.append(
            {
                "newspaper_slug": slug,
                "newspaper_display_name": first["newspaper_display_name"],
                "newspaperarchive_publication_city_name": first[
                    "newspaperarchive_publication_city_name"
                ],
                "newspaperarchive_publication_state_abbr": first[
                    "newspaperarchive_publication_state_abbr"
                ],
                "issue_count": str(len(entries)),
                "earliest_issue_date": dates[0],
                "latest_issue_date": dates[-1],
                "needed_issue_dates": "; ".join(dates),
                "needed_issue_ids": "; ".join(issue_ids),
            }
        )
    return output_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text_checklist(path: Path, grouped_rows: list[dict[str, str]]) -> None:
    lines = [
        "Full Ordinance Newspaper Lookup Checklist",
        "",
        "One block per newspaper slug with all issue dates needed from the current full-ordinance corpus.",
        "",
    ]
    for row in grouped_rows:
        city = row["newspaperarchive_publication_city_name"]
        state = row["newspaperarchive_publication_state_abbr"]
        lines.append(f"{row['newspaper_display_name']} [{row['newspaper_slug']}]")
        lines.append(f"Publication location: {city}, {state}")
        lines.append(f"Issue count: {row['issue_count']}")
        lines.append(f"Date range: {row['earliest_issue_date']} to {row['latest_issue_date']}")
        lines.append(f"Needed issue dates: {row['needed_issue_dates']}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    rows = read_rows(args.metadata_csv)
    issue_rows = build_issue_rows(rows)
    grouped_rows = build_grouped_rows(issue_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    issue_csv = args.output_dir / "full_ordinance_issue_lookup.csv"
    grouped_csv = args.output_dir / "full_ordinance_newspaper_lookup_grouped.csv"
    checklist_txt = args.output_dir / "full_ordinance_newspaper_lookup_checklist.txt"

    write_csv(issue_csv, ISSUE_COLUMNS, issue_rows)
    write_csv(grouped_csv, GROUPED_COLUMNS, grouped_rows)
    write_text_checklist(checklist_txt, grouped_rows)

    print(f"Wrote {len(issue_rows)} issue rows to {issue_csv}")
    print(f"Wrote {len(grouped_rows)} grouped newspaper rows to {grouped_csv}")
    print(f"Wrote checklist to {checklist_txt}")


if __name__ == "__main__":
    main()
