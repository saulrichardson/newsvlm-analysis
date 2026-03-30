#!/usr/bin/env python3
"""Merge Newspapers.com family-discovery batches into a cumulative result set.

This script rebuilds the current authoritative family-discovery status by:

1. reading selected family-discovery `issue_results.csv` files,
2. keeping the latest row per `issue_id`,
3. combining newly confirmed `available` rows with the pre-family strict
   confirmation file.

It is intentionally conservative:
- it excludes the superseded original `batch1` run,
- it does not reinterpret statuses,
- it preserves the exact issue URLs produced by the batch scripts.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path


BASE_DIR = Path("/Users/saulrichardson/projects/newspapers/newspaper-analysis")
REPORT_DIR = BASE_DIR / "reports" / "newspapers_com_family_discovery_cumulative_20260321"
PRIOR_CONFIRMED_PATH = (
    BASE_DIR
    / "reports"
    / "newspapers_com_availability_full_20260319_breadth_batch1"
    / "confirmed_exact_issue_dates_combined_after_breadth_batch1.csv"
)

INPUT_GLOBS = [
    "artifacts/scratch/newspapers_family_discovery_batch*_20260319/issue_results.csv",
    "artifacts/scratch/newspapers_family_discovery_batch*_20260321/issue_results.csv",
]

EXCLUDED_PATHS = {
    "artifacts/scratch/newspapers_family_discovery_batch1_20260319/issue_results.csv",
}


def load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pattern in INPUT_GLOBS:
        for path_str in sorted(glob.glob(str(BASE_DIR / pattern))):
            rel = str(Path(path_str).relative_to(BASE_DIR))
            if rel in EXCLUDED_PATHS:
                continue
            with open(path_str, newline="") as handle:
                batch_rows = list(csv.DictReader(handle))
            for row in batch_rows:
                enriched = dict(row)
                enriched["source_csv"] = rel
                rows.append(enriched)
    return rows


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest_by_issue: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda r: (r["source_csv"], r["issue_id"])):
        latest_by_issue[row["issue_id"]] = row
    return sorted(
        latest_by_issue.values(),
        key=lambda r: (r["newspaper_display_name"].lower(), r["issue_date"], r["issue_id"]),
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_combined_confirmed(normalized_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    with PRIOR_CONFIRMED_PATH.open(newline="") as handle:
        prior_rows = list(csv.DictReader(handle))

    available_rows = [row for row in normalized_rows if row["exact_issue_status"] == "available"]

    all_keys = set()
    for row in prior_rows + available_rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    combined: list[dict[str, str]] = []
    seen_issue_ids: set[str] = set()
    for row in prior_rows + available_rows:
        issue_id = row.get("issue_id", "")
        if issue_id in seen_issue_ids:
            continue
        seen_issue_ids.add(issue_id)
        combined.append({key: row.get(key, "") for key in fieldnames})
    return sorted(
        combined,
        key=lambda r: (
            r.get("newspaper_display_name", "").lower(),
            r.get("issue_date", ""),
            r.get("issue_id", ""),
        ),
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows()
    normalized_rows = normalize_rows(raw_rows)

    merged_results_path = REPORT_DIR / "family_discovery_issue_results_merged.csv"
    write_csv(merged_results_path, normalized_rows)

    status_counts: dict[str, int] = {}
    for row in normalized_rows:
        status = row["exact_issue_status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "merged_issue_rows": len(normalized_rows),
        "status_counts": status_counts,
    }
    summary_path = REPORT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    combined_confirmed = build_combined_confirmed(normalized_rows)
    combined_confirmed_path = REPORT_DIR / "confirmed_exact_issue_dates_combined_after_family_discovery.csv"
    write_csv(combined_confirmed_path, combined_confirmed)

    print(f"merged_results_path={merged_results_path}")
    print(f"summary_path={summary_path}")
    print(f"combined_confirmed_path={combined_confirmed_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"combined_confirmed_count={len(combined_confirmed)}")


if __name__ == "__main__":
    main()
