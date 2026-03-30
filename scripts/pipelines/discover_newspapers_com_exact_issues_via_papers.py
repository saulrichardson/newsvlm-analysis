#!/usr/bin/env python3
"""Discover Newspapers.com exact issue availability via the live `/papers/` page.

Workflow:

1. Use a real signed-in Chrome session to query the Newspapers.com `/papers/`
   search surface for a newspaper title.
2. Choose a candidate paper result using visible title/location text.
3. Open the paper page and extract the canonical browse-base link.
4. Confirm exact issue dates for that newspaper family via the browse API.

This is designed to make progress on the remaining unresolved families without
using Playwright Chromium or bursty page traffic.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

import websockets


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.7680.80 Safari/537.36"
)


@dataclass(frozen=True)
class IssueRow:
    issue_id: str
    issue_date: str
    newspaper_display_name: str
    city: str
    state: str
    search_query: str
    raw_row: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a real Newspapers.com browser session to resolve paper families via "
            "/papers/ and then confirm exact issue dates via the browse API."
        )
    )
    parser.add_argument(
        "--base-csv",
        default="artifacts/reports/newspapers_com_availability_full_20260318/newspapers_com_issue_availability.csv",
        help="Base issue-level Newspapers.com audit CSV.",
    )
    parser.add_argument(
        "--confirmed-csv",
        default="artifacts/reports/newspapers_com_availability_full_20260319_breadth_batch1/confirmed_exact_issue_dates_combined_after_breadth_batch1.csv",
        help="CSV of issue_ids already confirmed available.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where discovery artifacts should be written.",
    )
    parser.add_argument(
        "--chrome-debug-base",
        default="http://127.0.0.1:9223",
        help="Base URL for the real Chrome CDP instance.",
    )
    parser.add_argument(
        "--family-limit",
        type=int,
        default=5,
        help="Maximum number of newspaper families to process this run.",
    )
    parser.add_argument(
        "--family-offset",
        type=int,
        default=0,
        help="Offset into the ranked family list.",
    )
    parser.add_argument(
        "--page-load-seconds",
        type=float,
        default=4.0,
        help="Fixed wait after each real-browser navigation.",
    )
    parser.add_argument(
        "--sleep-between-families",
        type=float,
        default=8.0,
        help="Cooldown between family probes.",
    )
    parser.add_argument(
        "--max-api-retries",
        type=int,
        default=4,
        help="Maximum retries for browse API 429s.",
    )
    parser.add_argument(
        "--api-backoff-seconds",
        type=float,
        default=20.0,
        help="Base backoff seconds for browse API 429s.",
    )
    return parser.parse_args()


def chrome_json(debug_base: str, path: str) -> Any:
    with urlopen(f"{debug_base.rstrip('/')}{path}", timeout=30) as response:
        return json.load(response)


def list_page_tabs(debug_base: str) -> list[dict[str, Any]]:
    return [
        page
        for page in chrome_json(debug_base, "/json/list")
        if page.get("type") == "page"
    ]


def find_or_open_papers_tab(debug_base: str) -> dict[str, Any]:
    pages = list_page_tabs(debug_base)
    for page in pages:
        if page.get("url", "").startswith("https://www.newspapers.com/papers/"):
            return page
    # Fall back to any Newspapers.com page and navigate it into /papers/.
    for page in pages:
        if "newspapers.com" in page.get("url", ""):
            navigate_page_ws(page["webSocketDebuggerUrl"], "https://www.newspapers.com/papers/")
            time.sleep(4)
            return page
    raise RuntimeError("Could not find any open Newspapers.com page tab to reuse")


async def cdp_evaluate_json(ws_url: str, expression: str) -> Any:
    async with websockets.connect(ws_url, max_size=2**27) as ws:
        await ws.send(
            json.dumps(
                {
                    "id": 1,
                    "method": "Runtime.evaluate",
                    "params": {"expression": expression, "returnByValue": True},
                }
            )
        )
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") != 1:
                continue
            result = msg.get("result", {}).get("result", {})
            return json.loads(result.get("value", "null"))


async def cdp_navigate(ws_url: str, target_url: str) -> None:
    async with websockets.connect(ws_url, max_size=2**27) as ws:
        await ws.send(
            json.dumps(
                {
                    "id": 1,
                    "method": "Page.navigate",
                    "params": {"url": target_url},
                }
            )
        )
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == 1:
                return


def navigate_page_ws(ws_url: str, target_url: str) -> None:
    asyncio.run(cdp_navigate(ws_url, target_url))


def read_issue_rows(base_csv: Path, confirmed_csv: Path) -> list[IssueRow]:
    confirmed_ids = {row["issue_id"] for row in csv.DictReader(confirmed_csv.open())}
    rows = []
    for row in csv.DictReader(base_csv.open()):
        if row["issue_id"] in confirmed_ids:
            continue
        rows.append(
            IssueRow(
                issue_id=row["issue_id"].strip(),
                issue_date=row["issue_date"].strip(),
                newspaper_display_name=row["newspaper_display_name"].strip(),
                city=row["newspaperarchive_publication_city_name"].strip(),
                state=row["newspaperarchive_publication_state_abbr"].strip(),
                search_query=row["search_query"].strip(),
                raw_row={k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()},
            )
        )
    return rows


def rank_families(rows: list[IssueRow]) -> list[tuple[str, list[IssueRow]]]:
    grouped: dict[str, list[IssueRow]] = defaultdict(list)
    for row in rows:
        grouped[row.newspaper_display_name].append(row)
    families = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0].lower()),
    )
    return families


def papers_search_expression() -> str:
    return r"""JSON.stringify((() => {
  const body = document.body?.innerText || '';
  const showingMatch = body.match(/Showing\s+(\d+)\s+papers/i);
  const cards = Array.from(document.querySelectorAll('a[href*="/paper/"]')).map((a) => ({
    text: a.innerText.trim().replace(/\s+/g, ' '),
    href: a.href,
  }));
  const unique = [];
  const seen = new Set();
  for (const card of cards) {
    if (!card.href || seen.has(card.href)) continue;
    seen.add(card.href);
    unique.push(card);
  }
  return {
    url: location.href,
    title: document.title,
    bodySnippet: body.slice(0, 2500),
    showing: showingMatch ? Number(showingMatch[1]) : null,
    cards: unique.slice(0, 12),
  };
})())"""


def paper_page_expression() -> str:
    return r"""JSON.stringify((() => {
  const links = Array.from(document.querySelectorAll('a')).map((a) => ({
    text: a.innerText.trim().replace(/\s+/g, ' '),
    href: a.href,
  })).filter((item) => item.href);
  const browseLinks = links.filter((item) =>
    item.href.startsWith('https://www.newspapers.com/browse/')
  );
  return {
    url: location.href,
    title: document.title,
    bodySnippet: (document.body?.innerText || '').slice(0, 2500),
    browseLinks: browseLinks.slice(0, 20),
  };
})())"""


def normalize_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2}


def parse_year_range(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d{4})\s*[–-]\s*(\d{4})", text)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    if end < start:
        return None
    return start, end


def score_paper_card(card: dict[str, str], family_rows: list[IssueRow]) -> int:
    score = 0
    text = card["text"].lower()
    sample_row = family_rows[0]
    family_years = [int(row.issue_date[:4]) for row in family_rows]
    title_text = sample_row.newspaper_display_name.lower()
    title_tokens = normalize_tokens(sample_row.newspaper_display_name)
    query_tokens = normalize_tokens(sample_row.search_query)

    if title_text in text:
        score += 12
    for token in sorted(title_tokens | query_tokens):
        if token in text:
            score += 1
    if sample_row.city and sample_row.city.lower() in text:
        score += 5
    if sample_row.state and sample_row.state.lower() in text:
        score += 2
    # Reward alias text that mentions "also known as" rather than penalizing it.
    if "also known as" in text:
        score += 1

    year_range = parse_year_range(card["text"])
    if year_range is not None:
        start, end = year_range
        covered_years = sum(start <= year <= end for year in family_years)
        if covered_years == 0:
            score -= 100
        else:
            score += covered_years * 3
            if covered_years == len(family_years):
                score += 8
    return score


def choose_card(
    cards: list[dict[str, str]],
    family_rows: list[IssueRow],
) -> tuple[str, dict[str, str] | None]:
    if not cards:
        return "no_results", None
    scored = [
        (score_paper_card(card, family_rows), idx, card)
        for idx, card in enumerate(cards)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, _, best_card = scored[0]
    if best_score <= 0:
        return "unscored_results", None
    if len(scored) > 1 and scored[1][0] == best_score:
        return "ambiguous_best_score", None
    return "selected", best_card


def choose_browse_base(browse_links: list[dict[str, str]], paper_url: str) -> str | None:
    paper_id = urlparse(paper_url).path.rstrip("/").split("/")[-1]
    candidates = []
    for link in browse_links:
        href = link["href"]
        if f"_{paper_id}/" in href:
            candidates.append(href)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0].rstrip("/") + "/"


def exact_issue_api_url(browse_base: str, issue_date: str) -> str:
    yyyy, mm, dd = issue_date.split("-")
    parsed = urlparse(browse_base)
    path = parsed.path.rstrip("/")
    if not path.startswith("/browse/"):
        raise ValueError(f"Expected browse path, got: {browse_base}")
    suffix = path[len("/browse/") :]
    return f"https://www.newspapers.com/api/browse/1/{suffix}/{yyyy}/{mm}/{dd}"


def fetch_json(url: str, *, max_retries: int, backoff_seconds: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code == 404:
                raise
            if exc.code != 429 or attempt >= max_retries:
                raise
            wait = backoff_seconds * (attempt + 1)
            time.sleep(wait)
    raise RuntimeError(f"Unreachable retry loop for {url}")


def confirm_exact_issue(
    browse_base: str,
    issue_date: str,
    *,
    max_retries: int,
    backoff_seconds: float,
) -> tuple[str, str, str]:
    api_url = exact_issue_api_url(browse_base, issue_date)
    try:
        payload = fetch_json(api_url, max_retries=max_retries, backoff_seconds=backoff_seconds)
    except HTTPError as exc:
        if exc.code == 404:
            return "unavailable", api_url.replace("/api/browse/1/", "/browse/"), "404"
        raise
    node = payload.get("node") or {}
    if node.get("type") == "DATE":
        exact_issue_url = api_url.replace("/api/browse/1/", "/browse/")
        return "available", exact_issue_url, str(len(payload.get("children") or []))
    return "unexpected_node", api_url.replace("/api/browse/1/", "/browse/"), str(node.get("type"))


def append_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_issue_rows(Path(args.base_csv), Path(args.confirmed_csv))
    families = rank_families(rows)
    families = families[args.family_offset : args.family_offset + args.family_limit]
    if not families:
        raise RuntimeError("No newspaper families selected for this run")

    papers_tab = find_or_open_papers_tab(args.chrome_debug_base)
    ws_url = papers_tab["webSocketDebuggerUrl"]

    family_csv = output_dir / "family_results.csv"
    issue_csv = output_dir / "issue_results.csv"
    summary_json = output_dir / "summary.json"

    summary: dict[str, Any] = {
        "family_limit": args.family_limit,
        "family_offset": args.family_offset,
        "families_selected": len(families),
        "families_processed": 0,
        "issues_marked_available": 0,
        "issues_marked_unavailable": 0,
        "issues_unresolved": 0,
        "current_family": "",
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    family_fieldnames = [
        "newspaper_display_name",
        "family_issue_count",
        "query_string",
        "papers_search_url",
        "papers_result_status",
        "papers_showing_count",
        "selected_paper_url",
        "selected_paper_text",
        "browse_base",
    ]
    issue_fieldnames = [
        "issue_id",
        "issue_date",
        "newspaper_display_name",
        "query_string",
        "selected_paper_url",
        "browse_base",
        "exact_issue_status",
        "exact_issue_url",
        "exact_issue_detail",
    ]

    for family_name, family_rows in families:
        sample = family_rows[0]
        summary["current_family"] = family_name
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

        query_string = family_name
        papers_search_url = (
            "https://www.newspapers.com/papers/?titleKeyword=" + quote_plus(query_string)
        )
        navigate_page_ws(ws_url, papers_search_url)
        time.sleep(args.page_load_seconds)
        search_state = asyncio.run(cdp_evaluate_json(ws_url, papers_search_expression()))
        title = str(search_state.get("title", ""))
        if "Access denied" in title or "Cloudflare" in title:
            raise RuntimeError(f"Cloudflare challenge on /papers/ search page: {title}")

        cards = search_state.get("cards") or []
        result_status, selected_card = choose_card(cards, family_rows)
        selected_paper_url = "" if selected_card is None else selected_card["href"]
        selected_paper_text = "" if selected_card is None else selected_card["text"]
        browse_base = ""

        if selected_card is not None:
            navigate_page_ws(ws_url, selected_paper_url)
            time.sleep(args.page_load_seconds)
            paper_state = asyncio.run(cdp_evaluate_json(ws_url, paper_page_expression()))
            paper_title = str(paper_state.get("title", ""))
            if "Access denied" in paper_title or "Cloudflare" in paper_title:
                raise RuntimeError(f"Cloudflare challenge on paper page: {paper_title}")
            browse_base = choose_browse_base(paper_state.get("browseLinks") or [], selected_paper_url) or ""
            if not browse_base:
                result_status = "selected_without_browse_base"

        append_csv(
            family_csv,
            family_fieldnames,
            [
                {
                    "newspaper_display_name": family_name,
                    "family_issue_count": str(len(family_rows)),
                    "query_string": query_string,
                    "papers_search_url": papers_search_url,
                    "papers_result_status": result_status,
                    "papers_showing_count": (
                        "" if search_state.get("showing") is None else str(search_state["showing"])
                    ),
                    "selected_paper_url": selected_paper_url,
                    "selected_paper_text": selected_paper_text,
                    "browse_base": browse_base,
                }
            ],
        )

        issue_rows_out: list[dict[str, str]] = []
        if browse_base:
            selected_year_range = parse_year_range(selected_paper_text)
            for issue in family_rows:
                issue_year = int(issue.issue_date[:4])
                if selected_year_range is not None:
                    start, end = selected_year_range
                    if not (start <= issue_year <= end):
                        exact_status, exact_issue_url, detail = (
                            "out_of_range_for_selected_paper",
                            "",
                            f"{start}-{end}",
                        )
                    else:
                        exact_status, exact_issue_url, detail = confirm_exact_issue(
                            browse_base,
                            issue.issue_date,
                            max_retries=args.max_api_retries,
                            backoff_seconds=args.api_backoff_seconds,
                        )
                else:
                    exact_status, exact_issue_url, detail = confirm_exact_issue(
                        browse_base,
                        issue.issue_date,
                        max_retries=args.max_api_retries,
                        backoff_seconds=args.api_backoff_seconds,
                    )
                issue_rows_out.append(
                    {
                        "issue_id": issue.issue_id,
                        "issue_date": issue.issue_date,
                        "newspaper_display_name": issue.newspaper_display_name,
                        "query_string": query_string,
                        "selected_paper_url": selected_paper_url,
                        "browse_base": browse_base,
                        "exact_issue_status": exact_status,
                        "exact_issue_url": exact_issue_url,
                        "exact_issue_detail": detail,
                    }
                )
                if exact_status == "available":
                    summary["issues_marked_available"] += 1
                elif exact_status == "unavailable":
                    summary["issues_marked_unavailable"] += 1
                else:
                    summary["issues_unresolved"] += 1
        else:
            for issue in family_rows:
                issue_rows_out.append(
                    {
                        "issue_id": issue.issue_id,
                        "issue_date": issue.issue_date,
                        "newspaper_display_name": issue.newspaper_display_name,
                        "query_string": query_string,
                        "selected_paper_url": selected_paper_url,
                        "browse_base": browse_base,
                        "exact_issue_status": "unresolved_family_match",
                        "exact_issue_url": "",
                        "exact_issue_detail": result_status,
                    }
                )
                summary["issues_unresolved"] += 1

        append_csv(issue_csv, issue_fieldnames, issue_rows_out)
        summary["families_processed"] += 1
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        time.sleep(args.sleep_between_families)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
