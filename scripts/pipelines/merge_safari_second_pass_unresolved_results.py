#!/usr/bin/env python3
"""Merge second-pass unresolved resolution into the phase-1 matched-paper audit."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path("/Users/saulrichardson/projects/newspapers/newspaper-analysis")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


PHASE1_MERGED = (
    BASE_DIR
    / "reports"
    / "newspapers_com_availability_full_20260319_matched_unconfirmed"
    / "safari_phase1_matched_unconfirmed_182_merged.csv"
)
SECOND_PASS = (
    BASE_DIR
    / "artifacts/scratch"
    / "safari_unresolved_second_pass_full_20260319"
    / "resolved_unresolved_results.csv"
)
PRIOR_CONFIRMED = (
    BASE_DIR
    / "reports"
    / "newspapers_com_availability_full_20260318"
    / "confirmed_exact_issue_dates_from_slow_retries.csv"
)
REPORT_DIR = BASE_DIR / "reports" / "newspapers_com_availability_full_20260319_extras"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    phase1_rows = read_csv(PHASE1_MERGED)
    second_pass_rows = read_csv(SECOND_PASS)

    second_by_issue = {row["issue_id"]: row for row in second_pass_rows}

    merged_rows: list[dict[str, str]] = []
    for row in phase1_rows:
        issue_id = row["issue_id"]
        if issue_id in second_by_issue:
            resolved = second_by_issue[issue_id]
            # One known cross-paper anomaly from the second pass should remain unresolved.
            if issue_id == "ludington-daily-news__1959-09-04":
                row["check_status"] = "unresolved_non_date_page"
                row["check_reason"] = "second_pass_cross_paper_anomaly_left_unresolved"
            else:
                row["check_status"] = resolved["resolved_check_status"]
                row["check_reason"] = resolved["resolved_check_reason"]
        merged_rows.append(row)

    merged_out = REPORT_DIR / "safari_matched_unconfirmed_182_plus_unresolved_second_pass.csv"
    write_csv(merged_out, merged_rows)

    summary = {"checked_issue_rows": len(merged_rows), "status_counts": {}}
    for row in merged_rows:
        summary["status_counts"][row["check_status"]] = summary["status_counts"].get(row["check_status"], 0) + 1
    summary_out = REPORT_DIR / "safari_matched_unconfirmed_182_plus_unresolved_second_pass_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    prior_rows = read_csv(PRIOR_CONFIRMED)
    all_keys = set()
    available_rows = [row for row in merged_rows if row["check_status"] == "available"]
    for row in prior_rows + available_rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)
    combined: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in prior_rows + available_rows:
        issue_id = row.get("issue_id", "")
        if issue_id in seen:
            continue
        seen.add(issue_id)
        combined.append({key: row.get(key, "") for key in fieldnames})
    combined.sort(key=lambda r: (r.get("newspaper_slug", ""), r.get("issue_date", ""), r.get("issue_id", "")))
    combined_out = REPORT_DIR / "confirmed_exact_issue_dates_combined_after_unresolved_second_pass.csv"
    write_csv(combined_out, combined)

    print(f"merged_out={merged_out}")
    print(f"summary_out={summary_out}")
    print(f"combined_out={combined_out}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"combined_confirmed_count={len(combined)}")


if __name__ == "__main__":
    main()
