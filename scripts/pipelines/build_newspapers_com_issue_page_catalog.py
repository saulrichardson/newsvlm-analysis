#!/usr/bin/env python3
"""Enumerate Newspapers.com issue pages from confirmed exact issue URLs.

This script treats the browse hierarchy as the source of truth for which page
numbers exist inside each confirmed issue. It can also join that live issue
inventory against a page-fetch manifest, such as the rule-only 2-before/2-after
plan, to produce concrete `/image/<id>/` targets for recovery work.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.7680.80 Safari/537.36"
)


@dataclass(frozen=True)
class ConfirmedIssue:
    issue_id: str
    issue_date: str
    newspaper_display_name: str
    matched_paper_url: str
    exact_issue_url: str


@dataclass(frozen=True)
class BrowseBranch:
    node_type: str
    name: str
    display_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate Newspapers.com page inventories from confirmed exact issue "
            "URLs and optionally join them to a target page manifest."
        )
    )
    parser.add_argument(
        "--confirmed-csv",
        required=True,
        help="CSV of confirmed exact issue URLs.",
    )
    parser.add_argument(
        "--target-page-manifest",
        help=(
            "Optional CSV with issue_id and page_num columns. If provided, the "
            "script produces a concrete image-page manifest for those targets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output CSV/JSON artifacts should be written.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between issue API calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum retries for transient rate limiting.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=15.0,
        help="Base backoff in seconds for 429 retries.",
    )
    return parser.parse_args()


def read_confirmed_issues(path: Path) -> list[ConfirmedIssue]:
    issues_by_id: dict[str, ConfirmedIssue] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "issue_id",
            "issue_date",
            "newspaper_display_name",
            "matched_paper_url",
            "exact_issue_url",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            issue_id = row["issue_id"].strip()
            if not issue_id:
                raise ValueError(f"{path} contains a blank issue_id row")
            issues_by_id[issue_id] = ConfirmedIssue(
                issue_id=issue_id,
                issue_date=row["issue_date"].strip(),
                newspaper_display_name=row["newspaper_display_name"].strip(),
                matched_paper_url=row["matched_paper_url"].strip(),
                exact_issue_url=row["exact_issue_url"].strip(),
            )
    return sorted(issues_by_id.values(), key=lambda issue: (issue.issue_date, issue.issue_id))


def read_target_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"issue_id", "page_num"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = []
        for row in reader:
            issue_id = row["issue_id"].strip()
            page_num = row["page_num"].strip()
            if not issue_id or not page_num:
                raise ValueError(f"{path} contains blank issue_id/page_num values")
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
        return rows


def exact_issue_url_to_api_url(exact_issue_url: str) -> str:
    parsed = urlparse(exact_issue_url)
    if parsed.scheme != "https":
        raise ValueError(f"Expected https exact issue URL, got: {exact_issue_url}")
    if parsed.netloc != "www.newspapers.com":
        raise ValueError(f"Expected www.newspapers.com exact issue URL, got: {exact_issue_url}")
    path = parsed.path.rstrip("/")
    prefix = "/browse/"
    if not path.startswith(prefix):
        raise ValueError(f"Expected /browse/ path, got: {exact_issue_url}")
    suffix = path[len(prefix) :]
    if not suffix:
        raise ValueError(f"Could not derive browse suffix from: {exact_issue_url}")
    return f"https://www.newspapers.com/api/browse/1/{suffix}"


def fetch_json(url: str, *, max_retries: int, retry_backoff_seconds: float) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except HTTPError as exc:
            if exc.code != 429 or attempt >= max_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            fallback_wait_seconds = retry_backoff_seconds * (attempt + 1)
            if retry_after:
                try:
                    wait_seconds = max(float(retry_after), fallback_wait_seconds)
                except ValueError:
                    wait_seconds = fallback_wait_seconds
            else:
                wait_seconds = fallback_wait_seconds
            print(
                f"429 from {url}; retrying in {wait_seconds:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Unreachable retry loop for {url}")


def safe_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def enumerate_issue_pages(
    issue: ConfirmedIssue,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    api_url = exact_issue_url_to_api_url(issue.exact_issue_url)
    payload = fetch_json(
        api_url,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    node = payload.get("node") or {}
    if node.get("type") != "DATE":
        raise ValueError(
            f"Expected DATE browse node for {issue.issue_id}, got {node.get('type')!r}"
        )

    branch_counts: defaultdict[str, int] = defaultdict(int)
    page_rows: list[dict[str, str]] = []
    numeric_page_numbers: list[int] = []

    def walk_children(
        current_api_url: str,
        current_payload: dict,
        branches: tuple[BrowseBranch, ...],
    ) -> None:
        children = current_payload.get("children") or []
        for position, child in enumerate(children, start=1):
            child_name = str(child.get("name", "")).strip()
            child_display_name = str(child.get("displayName", "")).strip()
            if not child_name or not child_display_name:
                raise ValueError(
                    f"Missing browse child name/displayName for {issue.issue_id} at "
                    f"{current_api_url}"
                )
            if child_name.isdigit():
                page_num_int = safe_int(child_display_name)
                if page_num_int is not None:
                    numeric_page_numbers.append(page_num_int)
                edition_path = "/".join(branch.name for branch in branches)
                edition_display_name = " / ".join(
                    branch.display_name or branch.name for branch in branches
                )
                branch_key = edition_display_name or "(root)"
                branch_counts[branch_key] += 1
                page_rows.append(
                    {
                        "issue_id": issue.issue_id,
                        "issue_date": issue.issue_date,
                        "newspaper_display_name": issue.newspaper_display_name,
                        "matched_paper_url": issue.matched_paper_url,
                        "exact_issue_url": issue.exact_issue_url,
                        "browse_api_url": current_api_url,
                        "page_position": str(position),
                        "page_num": child_display_name,
                        "page_num_int": "" if page_num_int is None else str(page_num_int),
                        "image_id": child_name,
                        "image_page_url": f"https://www.newspapers.com/image/{child_name}/",
                        "edition_path": edition_path,
                        "edition_display_name": edition_display_name,
                        "branch_depth": str(len(branches)),
                    }
                )
                continue

            child_api_url = f"{current_api_url.rstrip('/')}/{child_name}"
            child_payload = fetch_json(
                child_api_url,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            child_node = child_payload.get("node") or {}
            child_type = str(child_node.get("type", "")).strip()
            if not child_type:
                raise ValueError(
                    f"Missing child node type for {issue.issue_id} at {child_api_url}"
                )
            walk_children(
                child_api_url,
                child_payload,
                branches
                + (
                    BrowseBranch(
                        node_type=child_type,
                        name=child_name,
                        display_name=child_display_name,
                    ),
                ),
            )

    walk_children(api_url, payload, tuple())

    issue_summary = {
        "issue_id": issue.issue_id,
        "issue_date": issue.issue_date,
        "newspaper_display_name": issue.newspaper_display_name,
        "matched_paper_url": issue.matched_paper_url,
        "exact_issue_url": issue.exact_issue_url,
        "browse_api_url": api_url,
        "browse_node_type": str(node.get("type", "")),
        "browse_child_count": str(len(page_rows)),
        "numeric_page_count": str(len(numeric_page_numbers)),
        "first_numeric_page": "" if not numeric_page_numbers else str(min(numeric_page_numbers)),
        "last_numeric_page": "" if not numeric_page_numbers else str(max(numeric_page_numbers)),
        "edition_branch_count": str(len(branch_counts)),
        "edition_branch_breakdown": "; ".join(
            f"{branch}={count}" for branch, count in sorted(branch_counts.items())
        ),
    }
    return issue_summary, page_rows


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV without a schema: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_rows(path: Path, rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_target_join_rows(
    target_rows: list[dict[str, str]],
    page_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    pages_by_issue_and_num: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in page_rows:
        pages_by_issue_and_num[(row["issue_id"], row["page_num"])].append(row)

    joined_rows: list[dict[str, str]] = []
    matched_count = 0
    unmatched_count = 0
    issues_with_unmatched: set[str] = set()
    issues_with_full_match: dict[str, bool] = defaultdict(lambda: True)
    issue_target_counts: defaultdict[str, int] = defaultdict(int)
    issue_match_counts: defaultdict[str, int] = defaultdict(int)
    ambiguous_match_count = 0

    for target in target_rows:
        issue_id = target["issue_id"]
        page_num = target["page_num"]
        issue_target_counts[issue_id] += 1
        candidates = sorted(
            pages_by_issue_and_num.get((issue_id, page_num), []),
            key=lambda row: (
                0 if row["edition_path"] == "" else 1,
                0 if row["edition_path"] == "main-edition" else 1,
                row["edition_display_name"],
                int(row["page_position"]),
            ),
        )
        matched = bool(candidates)
        if matched:
            matched_count += 1
            issue_match_counts[issue_id] += 1
            if len(candidates) > 1:
                ambiguous_match_count += 1
        else:
            unmatched_count += 1
            issues_with_unmatched.add(issue_id)
            issues_with_full_match[issue_id] = False
        preferred = candidates[0] if candidates else None
        joined_rows.append(
            {
                **target,
                "matched_issue_page": "true" if matched else "false",
                "candidate_count": "0" if not matched else str(len(candidates)),
                "candidate_image_ids": "" if not matched else ";".join(row["image_id"] for row in candidates),
                "candidate_image_page_urls": (
                    "" if not matched else ";".join(row["image_page_url"] for row in candidates)
                ),
                "candidate_edition_paths": (
                    "" if not matched else ";".join(row["edition_path"] for row in candidates)
                ),
                "preferred_image_id": "" if not matched else preferred["image_id"],
                "preferred_image_page_url": "" if not matched else preferred["image_page_url"],
                "preferred_browse_api_url": "" if not matched else preferred["browse_api_url"],
                "preferred_edition_path": "" if not matched else preferred["edition_path"],
                "preferred_edition_display_name": (
                    "" if not matched else preferred["edition_display_name"]
                ),
            }
        )

    fully_matched_issue_count = 0
    for issue_id, total_targets in issue_target_counts.items():
        if issue_match_counts[issue_id] == total_targets:
            fully_matched_issue_count += 1

    summary = {
        "target_page_rows": len(target_rows),
        "matched_target_page_rows": matched_count,
        "unmatched_target_page_rows": unmatched_count,
        "ambiguous_target_page_rows": ambiguous_match_count,
        "issues_with_any_target_pages": len(issue_target_counts),
        "issues_with_all_target_pages_matched": fully_matched_issue_count,
        "issues_with_any_unmatched_target_pages": len(issues_with_unmatched),
    }
    return joined_rows, summary


def main() -> int:
    args = parse_args()
    confirmed_csv = Path(args.confirmed_csv)
    output_dir = Path(args.output_dir)
    target_manifest = Path(args.target_page_manifest) if args.target_page_manifest else None

    issues = read_confirmed_issues(confirmed_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    issue_summary_path = output_dir / "issue_page_inventory_summary.csv"
    page_inventory_path = output_dir / "issue_page_inventory.csv"
    processed_issue_ids: set[str] = set()
    if issue_summary_path.exists():
        processed_issue_ids = {
            row["issue_id"] for row in read_csv_rows(issue_summary_path) if row.get("issue_id")
        }

    newly_enumerated_count = 0
    for idx, issue in enumerate(issues, start=1):
        if issue.issue_id in processed_issue_ids:
            continue
        issue_summary, issue_pages = enumerate_issue_pages(
            issue,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
        append_csv_rows(issue_summary_path, [issue_summary])
        append_csv_rows(page_inventory_path, issue_pages)
        processed_issue_ids.add(issue.issue_id)
        newly_enumerated_count += 1
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)
        if idx % 25 == 0:
            print(
                f"Checked {idx}/{len(issues)} issues; "
                f"enumerated {len(processed_issue_ids)} total",
                file=sys.stderr,
            )

    issue_summary_rows = read_csv_rows(issue_summary_path)
    page_rows = read_csv_rows(page_inventory_path)

    summary: dict[str, object] = {
        "confirmed_issue_count": len(issues),
        "issue_page_inventory_rows": len(page_rows),
        "issues_enumerated_successfully": len(issue_summary_rows),
        "issues_enumerated_in_this_run": newly_enumerated_count,
    }

    if target_manifest:
        target_rows = read_target_manifest(target_manifest)
        joined_rows, joined_summary = build_target_join_rows(target_rows, page_rows)
        write_csv(output_dir / "target_page_image_manifest.csv", joined_rows)
        preferred_only_rows = [
            row for row in joined_rows if row["matched_issue_page"] == "true"
        ]
        if preferred_only_rows:
            write_csv(output_dir / "target_page_image_manifest_preferred_only.csv", preferred_only_rows)
        summary["target_page_join"] = joined_summary

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
