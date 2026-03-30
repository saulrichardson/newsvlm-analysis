#!/usr/bin/env python3
"""Merge and normalize Safari phase-1 exact-issue results.

This script rebuilds the authoritative result set for the matched-paper
Newspapers.com phase-1 audit by:

1. reading all Safari batch/chunk result CSVs,
2. reclassifying them with the current exact-issue classifier,
3. keeping the latest row per issue_id,
4. combining confirmed `available` rows with the prior strict-confirmation file.

It is intentionally conservative and excludes known provisional outputs.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

BASE_DIR = Path("/Users/saulrichardson/projects/newspapers/newspaper-analysis")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.safari_confirm_exact_issue_dates import classify_issue
REPORT_DIR = BASE_DIR / "reports" / "newspapers_com_availability_full_20260319_matched_unconfirmed"
PRIOR_CONFIRMED_PATH = (
    BASE_DIR
    / "reports"
    / "newspapers_com_availability_full_20260318"
    / "confirmed_exact_issue_dates_from_slow_retries.csv"
)

INPUT_GLOBS = [
    "artifacts/scratch/safari_phase1_batch*_20260319/safari_exact_issue_results.csv",
    "artifacts/scratch/safari_phase1_chunk12_offset33_20260319/safari_exact_issue_results.csv",
    "artifacts/scratch/safari_phase1_chunk13_offset48_20260319/safari_exact_issue_results.csv",
    "artifacts/scratch/safari_phase1_recheck_stale_rows_20260319/results/safari_exact_issue_results.csv",
    "artifacts/scratch/safari_phase1_chunk14b_offset63_20260319/safari_exact_issue_results.csv",
    "artifacts/scratch/safari_phase1_chunk15_offset73_20260319/safari_exact_issue_results.csv",
    "artifacts/scratch/safari_phase1_chunk16_offset93_20260319/safari_exact_issue_results.csv",
    "artifacts/scratch/safari_phase1_chunk17_offset113_20260319/safari_exact_issue_results.csv",
    "artifacts/scratch/safari_phase1_chunk18_offset133_20260319/safari_exact_issue_results.csv",
    "artifacts/scratch/safari_phase1_chunk19_offset153_20260319/safari_exact_issue_results.csv",
    "artifacts/scratch/safari_phase1_chunk20_offset173_20260319/safari_exact_issue_results.csv",
]

EXCLUDED_PATHS = {
    "artifacts/scratch/safari_phase1_chunk14_offset63_20260319/safari_exact_issue_results.csv",
}


def load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pattern in INPUT_GLOBS:
        for path_str in sorted(glob.glob(str(BASE_DIR / pattern))):
            rel = str(Path(path_str).relative_to(BASE_DIR))
            if rel in EXCLUDED_PATHS:
                continue
            with open(path_str, newline="") as handle:
                rows.extend(list(csv.DictReader(handle)))
    return rows


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest_by_issue: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda r: (r.get("checked_at_utc", ""), r["issue_id"])):
        status, reason = classify_issue(
            issue_date=row["issue_date"],
            final_url=row["final_safari_url"],
            final_title=row["final_safari_title"],
            exact_issue_url=row["exact_issue_url"],
            matched_paper_url=row["matched_paper_url"],
        )
        normalized = dict(row)
        normalized["check_status"] = status
        normalized["check_reason"] = reason
        latest_by_issue[row["issue_id"]] = normalized
    return sorted(latest_by_issue.values(), key=lambda r: (int(r["priority_rank"]), r["issue_id"]))


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

    available_rows = [row for row in normalized_rows if row["check_status"] == "available"]
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
    return sorted(combined, key=lambda r: (r.get("newspaper_slug", ""), r.get("issue_date", ""), r.get("issue_id", "")))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows()
    normalized_rows = normalize_rows(raw_rows)

    merged_results_path = REPORT_DIR / "safari_phase1_matched_unconfirmed_182_merged.csv"
    write_csv(merged_results_path, normalized_rows)

    summary = {
        "checked_issue_rows": len(normalized_rows),
        "status_counts": {},
    }
    for row in normalized_rows:
        summary["status_counts"][row["check_status"]] = summary["status_counts"].get(row["check_status"], 0) + 1

    summary_path = REPORT_DIR / "safari_phase1_matched_unconfirmed_182_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    combined_confirmed = build_combined_confirmed(normalized_rows)
    combined_confirmed_path = REPORT_DIR / "confirmed_exact_issue_dates_combined_with_safari_phase1_full_182.csv"
    write_csv(combined_confirmed_path, combined_confirmed)

    print(f"merged_results_path={merged_results_path}")
    print(f"summary_path={summary_path}")
    print(f"combined_confirmed_path={combined_confirmed_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"combined_confirmed_count={len(combined_confirmed)}")


if __name__ == "__main__":
    main()
