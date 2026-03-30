#!/usr/bin/env python3
"""Audit Newspapers.com availability for the full-ordinance issue list.

This script expects:
1. An issue-level CSV with one row per needed issue date.
2. A Playwright storage-state JSON captured from an authenticated session.

It produces:
1. An issue-level availability CSV.
2. A grouped summary CSV by newspaper.
3. A JSON summary with high-level counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


STATE_ABBR_TO_NAME = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "GU": "Guam",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "VI": "U.S. Virgin Islands",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


ISSUE_OUTPUT_COLUMNS = [
    "issue_id",
    "issue_date",
    "newspaper_slug",
    "newspaper_display_name",
    "newspaperarchive_publication_city_name",
    "newspaperarchive_publication_state_abbr",
    "search_query",
    "search_status",
    "match_status",
    "matched_paper_title",
    "matched_paper_location",
    "matched_paper_date_range",
    "matched_paper_url",
    "matched_paper_browse_base",
    "exact_issue_check_status",
    "exact_issue_url",
    "exact_issue_page_title",
    "availability_reason",
]

GROUPED_OUTPUT_COLUMNS = [
    "newspaper_slug",
    "newspaper_display_name",
    "newspaperarchive_publication_city_name",
    "newspaperarchive_publication_state_abbr",
    "issue_count",
    "available_count",
    "unavailable_count",
    "not_found_count",
    "ambiguous_count",
    "search_error_count",
    "matched_paper_title",
    "matched_paper_location",
    "matched_paper_date_range",
    "matched_paper_url",
]


@dataclass
class Candidate:
    title: str
    location: str
    date_range: str
    page_count: str
    paper_url: str
    paper_slug: str
    paper_id: str
    score: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-csv", required=True, type=Path)
    parser.add_argument("--storage-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit-newspapers", type=int)
    parser.add_argument("--offset-newspapers", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_issue_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {
        "issue_id",
        "issue_date",
        "newspaper_slug",
        "newspaper_display_name",
        "newspaperarchive_publication_city_name",
        "newspaperarchive_publication_state_abbr",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def group_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["newspaper_slug"]].append(row)

    output = []
    for slug in sorted(grouped):
        entries = sorted(grouped[slug], key=lambda item: (item["issue_date"], item["issue_id"]))
        first = entries[0]
        output.append(
            {
                "newspaper_slug": slug,
                "newspaper_display_name": first["newspaper_display_name"],
                "publication_city": first["newspaperarchive_publication_city_name"],
                "publication_state_abbr": first["newspaperarchive_publication_state_abbr"],
                "issue_rows": entries,
            }
        )
    return output


def parse_year_range(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d{4})\D+(\d{4})", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def extract_results(page) -> list[dict[str, str]]:
    script = """
() => {
  const cards = [];
  const anchors = document.querySelectorAll('main a[href^="/paper/"]');
  for (const anchor of anchors) {
    const heading = anchor.querySelector('h2');
    const items = [...anchor.querySelectorAll('li')].map(li => li.innerText.trim()).filter(Boolean);
    if (!heading || items.length < 2) continue;
    const href = anchor.href;
    if (!href) continue;
    cards.push({
      title: heading.innerText.trim(),
      location: items[0] || '',
      date_range: items[1] || '',
      page_count: items[2] || '',
      paper_url: href,
    });
  }
  const unique = [];
  const seen = new Set();
  for (const row of cards) {
    if (seen.has(row.paper_url)) continue;
    seen.add(row.paper_url);
    unique.push(row);
  }
  return unique;
}
"""
    return page.evaluate(script)


def score_candidate(
    *,
    candidate_title: str,
    candidate_slug: str,
    candidate_location: str,
    wanted_title: str,
    wanted_slug: str,
    publication_city: str,
    publication_state_abbr: str,
) -> int:
    score = 0
    title_norm = normalize(candidate_title)
    wanted_norm = normalize(wanted_title)
    if title_norm == wanted_norm:
        score += 40
    elif wanted_norm in title_norm or title_norm in wanted_norm:
        score += 20

    if candidate_slug == wanted_slug:
        score += 100

    location_norm = normalize(candidate_location)
    city_norm = normalize(publication_city)
    if city_norm and city_norm in location_norm:
        score += 20

    state_name = STATE_ABBR_TO_NAME.get(publication_state_abbr, "")
    state_norm = normalize(state_name)
    if state_norm and state_norm in location_norm:
        score += 10

    return score


def build_candidates(
    result_rows: list[dict[str, str]],
    wanted_slug: str,
    wanted_title: str,
    publication_city: str,
    publication_state_abbr: str,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for row in result_rows:
        match = re.search(r"/paper/([^/]+)/(\d+)/", row["paper_url"])
        if not match:
            continue
        paper_slug = match.group(1)
        paper_id = match.group(2)
        score = score_candidate(
            candidate_title=row["title"],
            candidate_slug=paper_slug,
            candidate_location=row["location"],
            wanted_title=wanted_title,
            wanted_slug=wanted_slug,
            publication_city=publication_city,
            publication_state_abbr=publication_state_abbr,
        )
        candidates.append(
            Candidate(
                title=row["title"],
                location=row["location"],
                date_range=row["date_range"],
                page_count=row["page_count"],
                paper_url=row["paper_url"],
                paper_slug=paper_slug,
                paper_id=paper_id,
                score=score,
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def get_browse_base(page, paper_url: str, paper_slug: str, paper_id: str) -> str | None:
    page.goto(paper_url, wait_until="domcontentloaded")
    page.wait_for_timeout(500)
    href = page.evaluate(
        """([paperSlug, paperId]) => {
            const target = `${paperSlug}_${paperId}`;
            const anchors = [...document.querySelectorAll('a[href*="/browse/"]')];
            const match = anchors.find(a => (a.href || '').includes(target));
            return match ? match.href : null;
        }""",
        [paper_slug, paper_id],
    )
    if href and not href.endswith("/"):
        href += "/"
    return href


def exact_issue_available(page, browse_base: str, issue_date: str) -> tuple[str, str, str]:
    dt = datetime.strptime(issue_date, "%Y-%m-%d")
    date_url = f"{browse_base}{dt:%Y/%m/%d/}"
    page.goto(date_url, wait_until="domcontentloaded")
    page.wait_for_timeout(400)
    title = page.title()
    if any(marker in title for marker in ["Cloudflare", "Access denied", "Just a moment", "Attention Required"]):
        return "blocked_cloudflare", date_url, title
    available = str(dt.year) in title
    if available:
        return "available", date_url, title
    return "unavailable", date_url, title


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    issue_rows = load_issue_rows(args.issue_csv)
    grouped = group_rows(issue_rows)
    grouped = grouped[args.offset_newspapers :]
    if args.limit_newspapers is not None:
        grouped = grouped[: args.limit_newspapers]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    issue_output_rows: list[dict[str, Any]] = []
    grouped_output_rows: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=args.headless)
        context = browser.new_context(storage_state=str(args.storage_state))
        page = context.new_page()

        for index, newspaper in enumerate(grouped, start=1):
            search_query = newspaper["newspaper_display_name"]
            search_url = f"https://www.newspapers.com/papers/?titleKeyword={quote(search_query)}"
            group_rows_for_paper = newspaper["issue_rows"]
            matched_candidate: Candidate | None = None
            browse_base: str | None = None
            search_status = "ok"
            match_status = "not_found"
            print(
                f"[{index}/{len(grouped)}] {newspaper['newspaper_slug']} -> search '{search_query}'",
                flush=True,
            )

            try:
                page.goto(search_url, wait_until="domcontentloaded")
                page.wait_for_timeout(900)
                result_rows = extract_results(page)
                candidates = build_candidates(
                    result_rows=result_rows,
                    wanted_slug=newspaper["newspaper_slug"],
                    wanted_title=newspaper["newspaper_display_name"],
                    publication_city=newspaper["publication_city"],
                    publication_state_abbr=newspaper["publication_state_abbr"],
                )
            except PlaywrightTimeoutError:
                search_status = "timeout"
                candidates = []
            except Exception as exc:  # fail loud per row without aborting whole batch
                search_status = f"error:{type(exc).__name__}"
                candidates = []

            if candidates:
                top = candidates[0]
                if len(candidates) > 1 and top.score == candidates[1].score:
                    match_status = "ambiguous"
                elif top.score > 0:
                    match_status = "matched"
                    matched_candidate = top
                else:
                    match_status = "not_found"

            if matched_candidate is not None:
                try:
                    browse_base = get_browse_base(
                        page=page,
                        paper_url=matched_candidate.paper_url,
                        paper_slug=matched_candidate.paper_slug,
                        paper_id=matched_candidate.paper_id,
                    )
                    if not browse_base:
                        match_status = "matched_no_browse_base"
                except Exception as exc:
                    match_status = f"matched_browse_error:{type(exc).__name__}"

            available_count = 0
            unavailable_count = 0
            not_found_count = 0
            ambiguous_count = 0
            search_error_count = 0

            year_start, year_end = (
                parse_year_range(matched_candidate.date_range)
                if matched_candidate is not None
                else (None, None)
            )

            for issue_row in group_rows_for_paper:
                output = {
                    "issue_id": issue_row["issue_id"],
                    "issue_date": issue_row["issue_date"],
                    "newspaper_slug": issue_row["newspaper_slug"],
                    "newspaper_display_name": issue_row["newspaper_display_name"],
                    "newspaperarchive_publication_city_name": issue_row[
                        "newspaperarchive_publication_city_name"
                    ],
                    "newspaperarchive_publication_state_abbr": issue_row[
                        "newspaperarchive_publication_state_abbr"
                    ],
                    "search_query": search_query,
                    "search_status": search_status,
                    "match_status": match_status,
                    "matched_paper_title": matched_candidate.title if matched_candidate else "",
                    "matched_paper_location": matched_candidate.location if matched_candidate else "",
                    "matched_paper_date_range": matched_candidate.date_range if matched_candidate else "",
                    "matched_paper_url": matched_candidate.paper_url if matched_candidate else "",
                    "matched_paper_browse_base": browse_base or "",
                    "exact_issue_check_status": "",
                    "exact_issue_url": "",
                    "exact_issue_page_title": "",
                    "availability_reason": "",
                }

                if search_status != "ok":
                    output["exact_issue_check_status"] = "not_checked"
                    output["availability_reason"] = "search_error"
                    search_error_count += 1
                elif match_status == "ambiguous":
                    output["exact_issue_check_status"] = "not_checked"
                    output["availability_reason"] = "ambiguous_paper_match"
                    ambiguous_count += 1
                elif matched_candidate is None or not browse_base:
                    output["exact_issue_check_status"] = "not_checked"
                    output["availability_reason"] = "paper_not_found"
                    not_found_count += 1
                else:
                    issue_year = int(issue_row["issue_date"][:4])
                    if year_start is not None and year_end is not None and not (year_start <= issue_year <= year_end):
                        output["exact_issue_check_status"] = "not_checked"
                        output["availability_reason"] = "outside_paper_year_range"
                        unavailable_count += 1
                    else:
                        try:
                            status, exact_url, page_title = exact_issue_available(
                                page=page,
                                browse_base=browse_base,
                                issue_date=issue_row["issue_date"],
                            )
                            output["exact_issue_check_status"] = status
                            output["exact_issue_url"] = exact_url
                            output["exact_issue_page_title"] = page_title
                            if status == "available":
                                output["availability_reason"] = "exact_issue_page_found"
                                available_count += 1
                            elif status == "blocked_cloudflare":
                                output["availability_reason"] = "cloudflare_blocked"
                                search_error_count += 1
                            else:
                                output["availability_reason"] = "exact_issue_page_missing"
                                unavailable_count += 1
                        except Exception as exc:
                            output["exact_issue_check_status"] = "error"
                            output["availability_reason"] = f"exact_issue_error:{type(exc).__name__}"
                            search_error_count += 1

                issue_output_rows.append(output)

            grouped_output_rows.append(
                {
                    "newspaper_slug": newspaper["newspaper_slug"],
                    "newspaper_display_name": newspaper["newspaper_display_name"],
                    "newspaperarchive_publication_city_name": newspaper["publication_city"],
                    "newspaperarchive_publication_state_abbr": newspaper["publication_state_abbr"],
                    "issue_count": len(group_rows_for_paper),
                    "available_count": available_count,
                    "unavailable_count": unavailable_count,
                    "not_found_count": not_found_count,
                    "ambiguous_count": ambiguous_count,
                    "search_error_count": search_error_count,
                    "matched_paper_title": matched_candidate.title if matched_candidate else "",
                    "matched_paper_location": matched_candidate.location if matched_candidate else "",
                    "matched_paper_date_range": matched_candidate.date_range if matched_candidate else "",
                    "matched_paper_url": matched_candidate.paper_url if matched_candidate else "",
                }
            )

        context.close()
        browser.close()

    issue_csv = args.output_dir / "newspapers_com_issue_availability.csv"
    grouped_csv = args.output_dir / "newspapers_com_newspaper_summary.csv"
    summary_json = args.output_dir / "summary.json"

    write_csv(issue_csv, ISSUE_OUTPUT_COLUMNS, issue_output_rows)
    write_csv(grouped_csv, GROUPED_OUTPUT_COLUMNS, grouped_output_rows)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_issue_rows": len(issue_rows),
        "processed_newspapers": len(grouped),
        "issue_status_counts": {
            key: sum(1 for row in issue_output_rows if row["exact_issue_check_status"] == key)
            for key in sorted({row["exact_issue_check_status"] for row in issue_output_rows})
        },
        "match_status_counts": {
            key: sum(1 for row in issue_output_rows if row["match_status"] == key)
            for key in sorted({row["match_status"] for row in issue_output_rows})
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {issue_csv}")
    print(f"Wrote {grouped_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
