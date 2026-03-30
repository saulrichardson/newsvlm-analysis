#!/usr/bin/env python3
"""Download Newspapers.com page JPEGs from a matched page manifest.

This script uses the same real-Chrome token extraction flow proven by
`probe_newspapers_com_image_download.py`, but wraps it in a resumable,
low-rate batch loop suitable for calibration runs.

Operational constraints:

1. A real Google Chrome session must already be running on a CDP port.
2. That Chrome profile must already be signed into a paid Newspapers.com account.
3. The script intentionally stops on rate limiting or Cloudflare rather than
   trying to "push through" challenges.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse

import probe_newspapers_com_image_download as probe


RESULT_FIELDNAMES = [
    "run_index",
    "issue_id",
    "issue_date",
    "page_num",
    "preferred_image_id",
    "preferred_image_page_url",
    "status",
    "output_path",
    "byte_count",
    "content_type",
    "page_title",
    "error_type",
    "error_message",
]


@dataclass(frozen=True)
class ManifestRow:
    issue_id: str
    issue_date: str
    page_num: str
    preferred_image_id: str
    preferred_image_page_url: str
    raw_row: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Newspapers.com page JPEGs from a preferred image-page manifest "
            "using a real logged-in Chrome session."
        )
    )
    parser.add_argument(
        "--manifest-csv",
        required=True,
        help="CSV with preferred_image_page_url and metadata columns.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where images and run metadata should be written.",
    )
    parser.add_argument(
        "--chrome-debug-base",
        default="http://127.0.0.1:9223",
        help="Base URL for the real Chrome CDP instance.",
    )
    parser.add_argument(
        "--chrome-app-name",
        default="Google Chrome",
        help="macOS app name used by AppleScript for front-tab navigation.",
    )
    parser.add_argument(
        "--page-load-seconds",
        type=float,
        default=6.0,
        help="Seconds to wait after navigating each image page.",
    )
    parser.add_argument(
        "--sleep-between-pages",
        type=float,
        default=45.0,
        help="Cooldown between successful downloads.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of new pages to attempt this run.",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Optional offset into the remaining undownloaded rows.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "issue_id",
            "issue_date",
            "page_num",
            "preferred_image_id",
            "preferred_image_page_url",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = []
        for row in reader:
            rows.append(
                ManifestRow(
                    issue_id=row["issue_id"].strip(),
                    issue_date=row["issue_date"].strip(),
                    page_num=row["page_num"].strip(),
                    preferred_image_id=row["preferred_image_id"].strip(),
                    preferred_image_page_url=row["preferred_image_page_url"].strip(),
                    raw_row={k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()},
                )
            )
        return rows


def load_completed_keys(results_csv: Path) -> set[tuple[str, str]]:
    if not results_csv.exists():
        return set()
    with results_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        completed: set[tuple[str, str]] = set()
        for row in reader:
            if row.get("status") == "downloaded":
                completed.add((row["issue_id"], row["page_num"]))
        return completed


def append_result(results_csv: Path, row: dict[str, str]) -> None:
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def error_type_and_message(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, HTTPError):
        body = b""
        try:
            body = exc.read()
        except Exception:
            body = b""
        body_text = body.decode("utf-8", errors="replace")
        if exc.code == 429:
            return "rate_limited", body_text or str(exc)
        return f"http_{exc.code}", body_text or str(exc)
    return exc.__class__.__name__, str(exc)


def build_output_path(base_dir: Path, row: ManifestRow) -> Path:
    issue_dir = base_dir / row.issue_id
    issue_dir.mkdir(parents=True, exist_ok=True)
    return issue_dir / f"{row.page_num.zfill(4)}__{row.preferred_image_id}.jpg"


def probe_live_page(
    row: ManifestRow,
    *,
    chrome_app_name: str,
    chrome_debug_base: str,
    page_load_seconds: float,
) -> dict[str, Any]:
    probe.navigate_front_tab(chrome_app_name, row.preferred_image_page_url)
    time.sleep(page_load_seconds)
    ws_url = probe.find_page_ws_url(chrome_debug_base, row.preferred_image_page_url)
    live_probe = asyncio.run(probe.cdp_evaluate_json(ws_url, probe.page_probe_expression()))
    title = str(live_probe.get("title", ""))
    body = str(live_probe.get("bodySnippet", ""))
    if "Access denied" in title or "Cloudflare" in title:
        raise RuntimeError(f"Cloudflare challenge page detected: {title}")
    if "Sign in to Newspapers.com" in title or "Sign in to Newspapers.com" in body:
        raise RuntimeError("Chrome session is no longer signed into Newspapers.com")
    if "/image/" not in str(live_probe.get("url", "")):
        raise RuntimeError(
            f"Unexpected page after navigation for {row.preferred_image_page_url}: {title}"
        )
    return live_probe


def main() -> int:
    args = parse_args()
    manifest_rows = read_manifest(Path(args.manifest_csv))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "results.csv"
    summary_json = output_dir / "summary.json"
    completed = load_completed_keys(results_csv)

    remaining = [
        row for row in manifest_rows if (row.issue_id, row.page_num) not in completed
    ]
    if args.start_offset:
        remaining = remaining[args.start_offset :]
    if args.limit is not None:
        remaining = remaining[: args.limit]

    summary: dict[str, Any] = {
        "manifest_csv": args.manifest_csv,
        "output_dir": str(output_dir),
        "total_manifest_rows": len(manifest_rows),
        "already_downloaded_rows": len(completed),
        "run_candidate_rows": len(remaining),
        "downloaded_this_run": 0,
        "stopped_reason": "",
        "last_issue_id": "",
        "last_page_num": "",
        "sleep_between_pages": args.sleep_between_pages,
        "page_load_seconds": args.page_load_seconds,
    }
    write_summary(summary_json, summary)

    for run_index, row in enumerate(remaining, start=1):
        output_path = build_output_path(output_dir, row)
        try:
            live_probe = probe_live_page(
                row,
                chrome_app_name=args.chrome_app_name,
                chrome_debug_base=args.chrome_debug_base,
                page_load_seconds=args.page_load_seconds,
            )
            full_image_url = probe.build_full_image_url(live_probe)
            download_meta = probe.download_binary(full_image_url, output_path)
            result = {
                "run_index": str(run_index),
                "issue_id": row.issue_id,
                "issue_date": row.issue_date,
                "page_num": row.page_num,
                "preferred_image_id": row.preferred_image_id,
                "preferred_image_page_url": row.preferred_image_page_url,
                "status": "downloaded",
                "output_path": str(output_path),
                "byte_count": str(download_meta["byte_count"]),
                "content_type": str(download_meta["content_type"]),
                "page_title": str(live_probe.get("title", "")),
                "error_type": "",
                "error_message": "",
            }
            append_result(results_csv, result)
            summary["downloaded_this_run"] += 1
            summary["last_issue_id"] = row.issue_id
            summary["last_page_num"] = row.page_num
            write_summary(summary_json, summary)
            print(
                f"Downloaded {row.issue_id} page {row.page_num} -> {output_path}",
                file=sys.stderr,
            )
            time.sleep(args.sleep_between_pages)
        except Exception as exc:
            err_type, err_msg = error_type_and_message(exc)
            result = {
                "run_index": str(run_index),
                "issue_id": row.issue_id,
                "issue_date": row.issue_date,
                "page_num": row.page_num,
                "preferred_image_id": row.preferred_image_id,
                "preferred_image_page_url": row.preferred_image_page_url,
                "status": "failed",
                "output_path": str(output_path),
                "byte_count": "",
                "content_type": "",
                "page_title": "",
                "error_type": err_type,
                "error_message": err_msg,
            }
            append_result(results_csv, result)
            summary["stopped_reason"] = err_type
            summary["last_issue_id"] = row.issue_id
            summary["last_page_num"] = row.page_num
            write_summary(summary_json, summary)
            raise

    summary["stopped_reason"] = "completed_run"
    write_summary(summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
