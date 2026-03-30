#!/usr/bin/env python3
"""Conservative Safari-managed exact-issue confirmation for matched rows.

This phase-1 workflow avoids Playwright/Chromium and instead drives the user's
real Safari session via AppleScript. It only answers one question:

Does Newspapers.com have the exact issue date we need?

Classification is intentionally strict:
- available: tab title explicitly contains the exact month/day/year and the URL
  remains on the exact /YYYY/MM/DD/ path
- blocked_cloudflare: title indicates a challenge/access block
- unavailable: Safari redirected away from the exact /YYYY/MM/DD/ path
- unresolved_non_date_page: stayed on the exact path, but title did not confirm
  the exact issue date

The script persists results after every checked issue and can resume from an
existing CSV. It is designed for tiny batches only.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


BLOCK_MARKERS = ("Cloudflare", "Access denied", "Just a moment", "Attention Required")

OUTPUT_COLUMNS = [
    "batch_number",
    "priority_rank",
    "issue_id",
    "issue_date",
    "newspaper_display_name",
    "matched_paper_url",
    "exact_issue_url",
    "final_safari_url",
    "final_safari_title",
    "check_status",
    "check_reason",
    "checked_at_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-csv",
        type=Path,
        default=Path(
            "artifacts/reports/newspapers_com_availability_full_20260319_matched_unconfirmed/matched_unconfirmed_exact_issue_queue.csv"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--existing-results-csv", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-number", type=int)
    parser.add_argument("--paper-settle-seconds", type=float, default=8.0)
    parser.add_argument("--issue-timeout-seconds", type=float, default=22.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--between-issues-seconds", type=float, default=15.0)
    parser.add_argument(
        "--stop-on-blocked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the run immediately if Safari lands on a Cloudflare/access page.",
    )
    return parser.parse_args()


def run_osascript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def ensure_safari_window() -> None:
    script = """
tell application "Safari"
    activate
    if (count of windows) = 0 then
        make new document
    else
        make new document
    end if
end tell
"""
    run_osascript(script)


def set_front_tab_url(url: str) -> None:
    escaped = url.replace('"', '\\"')
    script = f'''
tell application "Safari"
    activate
    set URL of current tab of front window to "{escaped}"
end tell
'''
    run_osascript(script)


def get_front_tab_url() -> str:
    return run_osascript('tell application "Safari" to get URL of current tab of front window')


def get_front_tab_title() -> str:
    return run_osascript('tell application "Safari" to get name of current tab of front window')


def wait_for_page_change(
    baseline_url: str,
    baseline_title: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[str, str]:
    deadline = time.time() + timeout_seconds
    last_url = baseline_url
    last_title = baseline_title

    while True:
        last_url = get_front_tab_url()
        last_title = get_front_tab_title()
        if last_url != baseline_url or last_title != baseline_title:
            return last_url, last_title
        if time.time() >= deadline:
            return last_url, last_title
        time.sleep(poll_seconds)


def wait_for_url_equals(expected_url: str, timeout_seconds: float, poll_seconds: float) -> tuple[str, str]:
    deadline = time.time() + timeout_seconds
    last_url = ""
    last_title = ""

    while True:
        last_url = get_front_tab_url()
        last_title = get_front_tab_title()
        if last_url == expected_url:
            return last_url, last_title
        if time.time() >= deadline:
            return last_url, last_title
        time.sleep(poll_seconds)


def read_queue_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {
        "issue_id",
        "issue_date",
        "newspaper_display_name",
        "newspapers_com_matched_paper_url",
        "newspapers_com_exact_issue_url",
        "newspapers_com_matched_paper_browse_base",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def build_exact_issue_url(row: dict[str, str]) -> str:
    explicit = row["newspapers_com_exact_issue_url"].strip()
    if explicit:
        return explicit
    browse_base = row["newspapers_com_matched_paper_browse_base"].strip()
    if not browse_base:
        raise ValueError(f"{row['issue_id']} is missing both exact issue URL and browse base")
    if not browse_base.endswith("/"):
        browse_base += "/"
    issue_dt = datetime.strptime(row["issue_date"], "%Y-%m-%d")
    return f"{browse_base}{issue_dt:%Y/%m/%d/}"


def load_existing_results(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "checked_rows": len(rows),
                "status_counts": Counter(row["check_status"] for row in rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


def normalize_url(url: str) -> str:
    return url.split("#", 1)[0].split("?", 1)[0]


def classify_issue(
    issue_date: str,
    final_url: str,
    final_title: str,
    exact_issue_url: str,
    matched_paper_url: str,
) -> tuple[str, str]:
    issue_dt = datetime.strptime(issue_date, "%Y-%m-%d")
    exact_fragment = f"{issue_dt.strftime('%b')} {issue_dt.day}, {issue_dt.year}"
    exact_suffix = issue_dt.strftime("/%Y/%m/%d/")
    normalized_final_url = normalize_url(final_url)
    exact_issue_base = exact_issue_url.removesuffix(exact_suffix)

    if any(marker in final_title for marker in BLOCK_MARKERS):
        return "blocked_cloudflare", "cloudflare_or_access_challenge"
    if exact_fragment in final_title and normalized_final_url == exact_issue_url:
        return "available", "exact_issue_page_found"
    if normalized_final_url.endswith(exact_suffix) and not normalized_final_url.startswith(exact_issue_base):
        return "unresolved_non_date_page", "cross_paper_date_page"
    if normalized_final_url != exact_issue_url and (
        normalized_final_url == matched_paper_url or normalized_final_url.startswith(exact_issue_base)
    ):
        return "unavailable", "redirected_away_from_exact_issue_path"
    if normalized_final_url != exact_issue_url:
        return "unresolved_non_date_page", "cross_paper_or_stale_page_state"
    return "unresolved_non_date_page", "exact_issue_not_confirmed_in_title"


def wait_for_issue_state(
    issue_date: str,
    exact_issue_url: str,
    matched_paper_url: str,
    baseline_url: str,
    baseline_title: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[str, str, str, str]:
    deadline = time.time() + timeout_seconds
    last_url = ""
    last_title = ""

    while True:
        last_url = get_front_tab_url()
        last_title = get_front_tab_title()
        if last_url == baseline_url and last_title == baseline_title and time.time() < deadline:
            time.sleep(poll_seconds)
            continue
        status, reason = classify_issue(issue_date, last_url, last_title, exact_issue_url, matched_paper_url)
        if status in {"available", "blocked_cloudflare", "unavailable"}:
            return last_url, last_title, status, reason
        if time.time() >= deadline:
            return last_url, last_title, status, reason
        time.sleep(poll_seconds)


def main() -> None:
    args = parse_args()
    rows = read_queue_rows(args.queue_csv)

    if args.batch_number is not None:
        rows = [row for row in rows if int(row["batch_number"]) == args.batch_number]

    rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "safari_exact_issue_results.csv"
    summary_json = args.output_dir / "summary.json"

    output_rows = load_existing_results(args.existing_results_csv)
    completed_ids = {row["issue_id"] for row in output_rows}
    rows = [row for row in rows if row["issue_id"] not in completed_ids]

    ensure_safari_window()
    set_front_tab_url("https://www.newspapers.com/")
    time.sleep(max(4.0, args.paper_settle_seconds))

    for index, row in enumerate(rows, start=1):
        issue_id = row["issue_id"]
        matched_paper_url = row["newspapers_com_matched_paper_url"]
        exact_issue_url = build_exact_issue_url(row)

        print(f"[{index}/{len(rows)}] {issue_id}", flush=True)
        set_front_tab_url("about:blank")
        wait_for_url_equals(
            expected_url="about:blank",
            timeout_seconds=max(4.0, args.poll_seconds * 3),
            poll_seconds=args.poll_seconds,
        )
        set_front_tab_url(matched_paper_url)
        paper_url, paper_title = wait_for_page_change(
            baseline_url="about:blank",
            baseline_title=get_front_tab_title(),
            timeout_seconds=max(args.paper_settle_seconds, 10.0),
            poll_seconds=args.poll_seconds,
        )
        time.sleep(args.paper_settle_seconds)
        paper_url = get_front_tab_url()
        paper_title = get_front_tab_title()
        set_front_tab_url(exact_issue_url)

        final_url, final_title, status, reason = wait_for_issue_state(
            issue_date=row["issue_date"],
            exact_issue_url=exact_issue_url,
            matched_paper_url=matched_paper_url,
            baseline_url=paper_url,
            baseline_title=paper_title,
            timeout_seconds=args.issue_timeout_seconds,
            poll_seconds=args.poll_seconds,
        )

        output_rows.append(
            {
                "batch_number": row["batch_number"],
                "priority_rank": row["priority_rank"],
                "issue_id": issue_id,
                "issue_date": row["issue_date"],
                "newspaper_display_name": row["newspaper_display_name"],
                "matched_paper_url": matched_paper_url,
                "exact_issue_url": exact_issue_url,
                "final_safari_url": final_url,
                "final_safari_title": final_title,
                "check_status": status,
                "check_reason": reason,
                "checked_at_utc": datetime.now(UTC).isoformat(),
            }
        )

        write_csv(results_csv, output_rows)
        write_summary(summary_json, output_rows)

        print(f"    status={status}", flush=True)
        print(f"    title={final_title}", flush=True)

        if status == "blocked_cloudflare" and args.stop_on_blocked:
            print("Encountered Cloudflare/access challenge; stopping immediately.", flush=True)
            break

        time.sleep(args.between_issues_seconds)

    write_csv(results_csv, output_rows)
    write_summary(summary_json, output_rows)
    print(f"Wrote {results_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
