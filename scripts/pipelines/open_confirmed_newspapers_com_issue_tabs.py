#!/usr/bin/env python3
"""Open one browser tab per strictly confirmed Newspapers.com exact issue."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirmed-issues-csv",
        type=Path,
        default=Path(
            "artifacts/reports/newspapers_com_availability_full_20260318/confirmed_exact_issue_dates_from_slow_retries.csv"
        ),
    )
    parser.add_argument(
        "--storage-state",
        type=Path,
        default=Path("artifacts/scratch/newspapers_com_storage_state_20260318.json"),
    )
    parser.add_argument("--between-tabs-seconds", type=float, default=8.0)
    parser.add_argument("--hold-open-seconds", type=int, default=3600)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.confirmed_issues_csv)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False, slow_mo=250)
        context = browser.new_context(storage_state=str(args.storage_state))

        home = context.new_page()
        home.goto("https://www.newspapers.com/", wait_until="domcontentloaded")
        home.wait_for_timeout(7000)
        print("Opened Newspapers.com home tab", flush=True)

        for index, row in enumerate(rows, start=1):
            page = context.new_page()
            issue_url = row["exact_issue_url"]
            print(
                f"[{index}/{len(rows)}] opening {row['issue_id']} -> {issue_url}",
                flush=True,
            )
            page.goto(issue_url, wait_until="domcontentloaded")
            page.wait_for_timeout(5000)
            print(f"    title={page.title()}", flush=True)
            time.sleep(args.between_tabs_seconds)

        print(
            f"All tabs opened. Holding browser open for {args.hold_open_seconds} seconds.",
            flush=True,
        )
        time.sleep(args.hold_open_seconds)

        context.close()
        browser.close()


if __name__ == "__main__":
    main()
