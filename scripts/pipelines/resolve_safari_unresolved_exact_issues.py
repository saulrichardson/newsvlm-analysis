#!/usr/bin/env python3
"""Second-pass Safari resolution for unresolved exact-issue rows.

This script revisits rows previously classified as `unresolved_non_date_page`
and uses Safari's rendered document text as an additional evidence source.

Resolution rules are intentionally conservative:

1. Family-level unavailability:
   If a representative page for a matched paper exposes a year selector and a
   row's target year is absent from that selector, the row is marked
   `unavailable`.

2. Row-level availability:
   If the rendered document text explicitly contains the target issue date
   (for example in a date selector or "Search within ..." control), the row is
   marked `available`.

3. Row-level unavailability:
   If a row-specific page exposes a year selector and the target year is absent,
   the row is marked `unavailable`.

4. Otherwise the row remains `unresolved_non_date_page`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


BLOCK_MARKERS = ("Cloudflare", "Access denied", "Just a moment", "Attention Required")

OUTPUT_COLUMNS = [
    "issue_id",
    "issue_date",
    "newspaper_display_name",
    "matched_paper_url",
    "exact_issue_url",
    "final_safari_url",
    "final_safari_title",
    "resolved_check_status",
    "resolved_check_reason",
    "year_selector_year_count",
    "year_selector_min_year",
    "year_selector_max_year",
    "text_snippet",
    "checked_at_utc",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "artifacts/reports/newspapers_com_availability_full_20260319_matched_unconfirmed/safari_phase1_matched_unconfirmed_182_merged.csv"
        ),
    )
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--page-settle-seconds", type=float, default=8.0)
    ap.add_argument("--between-rows-seconds", type=float, default=15.0)
    ap.add_argument(
        "--stop-on-blocked",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return ap.parse_args()


def run_osascript(script: str, timeout_seconds: float = 30.0) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_seconds,
    )
    return result.stdout.strip()


def ensure_safari_window() -> None:
    run_osascript(
        """
tell application "Safari"
    activate
    if (count of windows) = 0 then
        make new document
    end if
end tell
"""
    )


def set_front_tab_url(url: str) -> None:
    escaped = url.replace('"', '\\"')
    run_osascript(
        f'''
tell application "Safari"
    activate
    set URL of current tab of front window to "{escaped}"
end tell
'''
    )


def get_front_tab_url() -> str:
    return run_osascript('tell application "Safari" to get URL of current tab of front window')


def get_front_tab_title() -> str:
    return run_osascript('tell application "Safari" to get name of current tab of front window')


def get_front_document_text() -> str:
    return run_osascript('tell application "Safari" to get text of front document', timeout_seconds=45.0)


def load_unresolved_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    unresolved = [row for row in rows if row["check_status"] == "unresolved_non_date_page"]
    if not unresolved:
        raise SystemExit("No unresolved rows found")
    return unresolved


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    summary = {
        "checked_rows": len(rows),
        "status_counts": Counter(row["resolved_check_status"] for row in rows),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def extract_years_from_text(text: str) -> list[int]:
    years = sorted({int(match) for match in re.findall(r"(?<!\d)(18\d{2}|19\d{2}|20\d{2})(?!\d)", text)})
    return years


def exact_date_strings(issue_date: str) -> set[str]:
    dt = datetime.strptime(issue_date, "%Y-%m-%d")
    month_short = dt.strftime("%b")
    month_long = dt.strftime("%B")
    day = dt.day
    weekday_short = dt.strftime("%a")
    weekday_long = dt.strftime("%A")
    return {
        f"{month_short} {day}, {dt.year}, {weekday_short}",
        f"{month_long} {day}, {dt.year}, {weekday_long}",
        f"Search within {month_short} {day}, {dt.year}, {weekday_short}",
        f"Search within {month_long} {day}, {dt.year}, {weekday_long}",
    }


def classify_from_text(issue_date: str, title: str, text: str) -> tuple[str, str, list[int]]:
    if any(marker in title for marker in BLOCK_MARKERS):
        return "blocked_cloudflare", "cloudflare_or_access_challenge", []

    years = extract_years_from_text(text)
    target_year = int(issue_date[:4])

    for candidate in exact_date_strings(issue_date):
        if candidate in text:
            return "available", "exact_issue_date_present_in_rendered_text", years

    if years and target_year not in years:
        return "unavailable", "target_year_absent_from_year_selector", years

    return "unresolved_non_date_page", "no_exact_date_text_confirmation", years


def main() -> None:
    args = parse_args()
    unresolved_rows = load_unresolved_rows(args.input_csv)
    unresolved_rows = unresolved_rows[args.offset :]
    if args.limit is not None:
        unresolved_rows = unresolved_rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "resolved_unresolved_results.csv"
    summary_json = args.output_dir / "summary.json"
    family_probe_csv = args.output_dir / "family_probes.csv"

    ensure_safari_window()
    output_rows: list[dict[str, str]] = []
    family_probe_rows: list[dict[str, str]] = []

    # Family probe to cheaply rule out rows whose target years are not offered.
    rows_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in unresolved_rows:
        rows_by_paper[row["matched_paper_url"]].append(row)

    family_year_coverage: dict[str, set[int]] = {}
    for paper_url, rows in rows_by_paper.items():
        representative_url = rows[0]["exact_issue_url"]
        representative_issue_id = rows[0]["issue_id"]
        print(f"[family {len(family_probe_rows)+1}/{len(rows_by_paper)}] {rows[0]['newspaper_display_name']} :: {representative_issue_id}", flush=True)
        set_front_tab_url(representative_url)
        time.sleep(args.page_settle_seconds)
        title = get_front_tab_title()
        text = get_front_document_text()
        status, _, years = classify_from_text(rows[0]["issue_date"], title, text)
        # Family coverage is useful even if this representative row itself is unresolved.
        family_year_coverage[paper_url] = set(years)
        family_probe_rows.append(
            {
                "newspaper_display_name": rows[0]["newspaper_display_name"],
                "representative_issue_id": representative_issue_id,
                "matched_paper_url": paper_url,
                "representative_exact_issue_url": representative_url,
                "representative_final_title": title,
                "representative_status": status,
                "year_selector_year_count": len(years),
                "year_selector_min_year": min(years) if years else "",
                "year_selector_max_year": max(years) if years else "",
            }
        )
        with family_probe_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(family_probe_rows[0].keys()))
            writer.writeheader()
            writer.writerows(family_probe_rows)
        if status == "blocked_cloudflare" and args.stop_on_blocked:
            # write empty progress and stop hard
            write_csv(results_csv, output_rows)
            write_summary(summary_json, output_rows)
            raise SystemExit("Encountered Cloudflare/access challenge during family probe")
        time.sleep(args.between_rows_seconds)

    for index, row in enumerate(unresolved_rows, start=1):
        issue_id = row["issue_id"]
        target_year = int(row["issue_date"][:4])
        paper_url = row["matched_paper_url"]
        years = sorted(family_year_coverage.get(paper_url, set()))

        print(f"[{index}/{len(unresolved_rows)}] {issue_id}", flush=True)

        if years and target_year not in years:
            resolved_status = "unavailable"
            resolved_reason = "target_year_absent_from_family_year_selector"
            final_url = row["exact_issue_url"]
            final_title = row["final_safari_title"]
            text = ""
            print(f"    status={resolved_status}", flush=True)
            print(f"    reason={resolved_reason}", flush=True)
        else:
            set_front_tab_url(row["exact_issue_url"])
            time.sleep(args.page_settle_seconds)
            final_url = get_front_tab_url()
            final_title = get_front_tab_title()
            text = get_front_document_text()
            resolved_status, resolved_reason, years = classify_from_text(
                issue_date=row["issue_date"],
                title=final_title,
                text=text,
            )
            family_year_coverage[paper_url] = set(years) or family_year_coverage.get(paper_url, set())
            print(f"    status={resolved_status}", flush=True)
            print(f"    title={final_title}", flush=True)
            if resolved_status == "blocked_cloudflare" and args.stop_on_blocked:
                print("Encountered Cloudflare/access challenge; stopping immediately.", flush=True)
                break
            time.sleep(args.between_rows_seconds)

        output_rows.append(
            {
                "issue_id": issue_id,
                "issue_date": row["issue_date"],
                "newspaper_display_name": row["newspaper_display_name"],
                "matched_paper_url": row["matched_paper_url"],
                "exact_issue_url": row["exact_issue_url"],
                "final_safari_url": final_url,
                "final_safari_title": final_title,
                "resolved_check_status": resolved_status,
                "resolved_check_reason": resolved_reason,
                "year_selector_year_count": str(len(years)),
                "year_selector_min_year": str(min(years)) if years else "",
                "year_selector_max_year": str(max(years)) if years else "",
                "text_snippet": " ".join(text.split())[:500],
                "checked_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        write_csv(results_csv, output_rows)
        write_summary(summary_json, output_rows)
        sys.stdout.flush()

    write_csv(results_csv, output_rows)
    write_summary(summary_json, output_rows)
    print(f"Wrote {results_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
