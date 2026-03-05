#!/usr/bin/env python3
"""
LLM-only page-level semantic tagging for five-newspaper v3 pipeline.

This stage tags pages as:
  - verbatim_legal_candidate
  - zoning_narrative_only
  - non_zoning
  - uncertain

Page tags are audit metadata and are not hard filters for anchor cleaning.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from fullnewspaper_v3_common import (
    eprint,
    enum,
    extract_openai_output_text,
    find_newspaper_city_map,
    iter_jsonl,
    load_issue_rows_for_city,
    norm_str,
    parse_json_from_text,
    read_error_jsonl,
    read_result_jsonl,
    run_gateway_requests,
    safe_float,
    write_csv,
    write_jsonl,
    write_openai_requests,
)


ALLOWED_PAGE_TAGS = {
    "verbatim_legal_candidate",
    "zoning_narrative_only",
    "non_zoning",
    "uncertain",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM-only page-level semantic tagging.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Run directory containing panels/<city_key>/issue_texts.jsonl.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-dir>/fullnewspaper_v3/page_tagging).",
    )
    ap.add_argument(
        "--newspapers",
        default="cazenovia-republican,albion-evening-recorder,bountiful-davis-county-clipper,abilene-reporter-news,appleton-post-crescent",
        help="Comma-separated newspaper slugs.",
    )
    ap.add_argument("--model", default="openai:gpt-5")
    ap.add_argument(
        "--max-page-chars",
        type=int,
        default=0,
        help="Hard cap for per-page text included in prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument(
        "--gateway-runner",
        default=str(Path(__file__).resolve().parents[1] / "scripts" / "run_openai_requests_via_gateway.py"),
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default=str(Path(__file__).resolve().parents[1] / "agent-gateway" / "src"),
    )
    ap.add_argument("--gov-env-path", default=".env")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _bounded_text_or_fail(text: str, max_chars: int, label: str) -> str:
    s = str(text or "")
    if max_chars > 0 and len(s) > max_chars:
        raise ValueError(
            f"{label} length {len(s)} exceeds hard cap {max_chars}. "
            "Clipping/condensing is disabled; increase cap or reduce input scope."
        )
    return s


def _split_issue_into_pages(issue_text: str, page_ids: list[str]) -> list[dict[str, Any]]:
    """
    Deterministic page segmentation for merged issue text.

    We partition contiguous character ranges into len(page_ids) buckets.
    This is index bookkeeping only; semantic labeling is still LLM-only.
    """
    text = str(issue_text or "")
    n = max(1, len(page_ids))
    if not page_ids:
        page_ids = [f"page_{i+1}" for i in range(n)]

    total = len(text)
    if total == 0:
        return [
            {
                "page_id": page_ids[i] if i < len(page_ids) else f"page_{i+1}",
                "start_char": 0,
                "end_char": 0,
                "text": "",
            }
            for i in range(n)
        ]

    cuts: list[tuple[int, int]] = []
    start = 0
    for i in range(n):
        if i == n - 1:
            end = total
        else:
            target = int(round((i + 1) * total / n))
            end = target
        cuts.append((start, end))
        start = end

    out: list[dict[str, Any]] = []
    for i, (s, e) in enumerate(cuts):
        pid = page_ids[i] if i < len(page_ids) else f"page_{i+1}"
        out.append({"page_id": pid, "start_char": int(s), "end_char": int(e), "text": text[s:e]})
    return out


def _segments_from_page_rows(issue: dict[str, Any], page_ids: list[str]) -> list[dict[str, Any]] | None:
    """
    Preferred segmentation when issue_texts.jsonl contains true per-page text rows.

    Expected shape:
      issue["page_rows"] = [{"page_id": ..., "text": ...}, ...]
    """
    rows = issue.get("page_rows")
    if not isinstance(rows, list) or not rows:
        return None
    by_pid: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        pid = norm_str(r.get("page_id"))
        if not pid:
            continue
        by_pid[pid] = r
    if not by_pid:
        return None

    segs: list[dict[str, Any]] = []
    cursor = 0
    for pid in page_ids:
        txt = norm_str((by_pid.get(pid) or {}).get("text"))
        start = cursor
        end = cursor + len(txt)
        segs.append({"page_id": pid, "start_char": int(start), "end_char": int(end), "text": txt})
        cursor = end
    return segs


def _build_prompt(issue: dict[str, Any], page_segments: list[dict[str, Any]], max_page_chars: int) -> str:
    schema = {
        "issue_id": norm_str(issue.get("issue_id")),
        "page_tags": [
            {
                "page_id": "string",
                "tag": "verbatim_legal_candidate",
                "confidence_0_to_1": 0.9,
                "evidence": "short quoted phrase from the page text",
            }
        ],
        "issue_notes": "",
    }

    lines: list[str] = []
    lines.append("You are classifying newspaper pages for zoning-law extraction audit metadata.")
    lines.append("Return ONLY JSON. No markdown.")
    lines.append("")
    lines.append("Tag each page with exactly one label:")
    lines.append("- verbatim_legal_candidate: page contains verbatim zoning ordinance/amendment/legal-notice language.")
    lines.append("- zoning_narrative_only: page discusses zoning but does not print operative legal text.")
    lines.append("- non_zoning: no zoning-relevant content.")
    lines.append("- uncertain: ambiguous.")
    lines.append("")
    lines.append("Important constraints:")
    lines.append("- Do NOT force pages into legal_candidate if they only summarize or report on zoning.")
    lines.append("- Distinguish legal text vs narrative.")
    lines.append("- Use page_ids exactly as provided.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Issue metadata: issue_id={norm_str(issue.get('issue_id'))} "
        f"date={norm_str(issue.get('issue_date'))} "
        f"city={norm_str(issue.get('city_name'))},{norm_str(issue.get('state_abbr')).upper()} "
        f"newspaper={norm_str(issue.get('pub_slug'))}"
    )
    lines.append("")
    lines.append("Pages:")
    for p in page_segments:
        lines.append("")
        lines.append(
            f"[PAGE] page_id={norm_str(p.get('page_id'))} "
            f"char_range=({int(p.get('start_char') or 0)}-{int(p.get('end_char') or 0)})"
        )
        lines.append(
            _bounded_text_or_fail(
                norm_str(p.get("text")),
                max_page_chars,
                label=f"issue_id={norm_str(issue.get('issue_id'))} page_id={norm_str(p.get('page_id'))} page_text",
            )
        )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _coerce_parse(parsed_obj: dict[str, Any], issue: dict[str, Any], page_ids: list[str]) -> dict[str, Any]:
    obj = dict(parsed_obj or {})
    raw = obj.get("page_tags")
    if not isinstance(raw, list):
        raw = []

    by_pid: dict[str, dict[str, Any]] = {}
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        pid = norm_str(rec.get("page_id"))
        if not pid or pid not in page_ids:
            continue
        by_pid[pid] = {
            "page_id": pid,
            "tag": enum(rec.get("tag"), ALLOWED_PAGE_TAGS, "uncertain"),
            "confidence_0_to_1": safe_float(rec.get("confidence_0_to_1"), default=0.0),
            "evidence": norm_str(rec.get("evidence")),
        }

    out_rows: list[dict[str, Any]] = []
    for pid in page_ids:
        if pid in by_pid:
            out_rows.append(by_pid[pid])
        else:
            out_rows.append(
                {
                    "page_id": pid,
                    "tag": "uncertain",
                    "confidence_0_to_1": 0.0,
                    "evidence": "",
                }
            )

    return {
        "issue_id": norm_str(obj.get("issue_id")) or norm_str(issue.get("issue_id")),
        "page_tags": out_rows,
        "issue_notes": norm_str(obj.get("issue_notes")),
    }


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if norm_str(args.output_dir)
        else run_dir / "fullnewspaper_v3" / "page_tagging"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    newspapers = [norm_str(x).lower() for x in norm_str(args.newspapers).split(",") if norm_str(x)]
    city_map = find_newspaper_city_map(run_dir, newspapers)
    missing = [s for s in newspapers if s not in city_map]
    if missing:
        raise SystemExit(f"Could not map newspaper slug(s) to city panel: {missing}")

    all_issues: list[dict[str, Any]] = []
    for slug in newspapers:
        city_key = city_map[slug]
        rows = load_issue_rows_for_city(run_dir, city_key)
        for r in rows:
            if norm_str(r.get("pub_slug")).lower() == slug:
                all_issues.append(r)
    all_issues = sorted(all_issues, key=lambda r: (norm_str(r.get("pub_slug")), norm_str(r.get("issue_date")), norm_str(r.get("issue_id"))))
    if not all_issues:
        raise SystemExit("No issues loaded for requested newspapers.")

    req_rows: list[tuple[str, str]] = []
    issue_meta: dict[str, dict[str, Any]] = {}
    for issue in all_issues:
        issue_id = norm_str(issue.get("issue_id"))
        slug = norm_str(issue.get("pub_slug")).lower()
        cid = f"page_tag_v3::{slug}::{issue_id}"
        page_ids_raw = issue.get("page_ids")
        page_ids = [norm_str(x) for x in page_ids_raw] if isinstance(page_ids_raw, list) else []
        page_ids = [x for x in page_ids if x]
        if not page_ids:
            n = max(1, int(issue.get("page_count") or 1))
            page_ids = [f"{issue_id}__page_{i+1}" for i in range(n)]
        segments = _segments_from_page_rows(issue, page_ids)
        if not isinstance(segments, list) or not segments:
            segments = _split_issue_into_pages(norm_str(issue.get("text")), page_ids)
        prompt = _build_prompt(issue, segments, max_page_chars=int(args.max_page_chars))
        req_rows.append((cid, prompt))
        issue_meta[cid] = {"issue": issue, "page_ids": page_ids, "segments": segments}

    req_dir = out_dir / "requests"
    res_dir = out_dir / "results"
    write_openai_requests(req_dir / "openai_requests_shard000.jsonl", req_rows, model=str(args.model))
    run_gateway_requests(
        request_dir=req_dir,
        output_dir=res_dir,
        runner_path=Path(args.gateway_runner).expanduser().resolve(),
        model=str(args.model),
        gateway_pythonpath=Path(args.gateway_pythonpath).expanduser().resolve() if norm_str(args.gateway_pythonpath) else None,
        timeout_s=float(args.timeout),
        concurrency=int(args.concurrency),
        gov_env_path=Path(args.gov_env_path).expanduser().resolve(),
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )

    result_map = read_result_jsonl(res_dir)
    error_map = read_error_jsonl(res_dir)

    issue_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []

    for cid, _prompt in req_rows:
        meta = issue_meta[cid]
        issue = meta["issue"]
        page_ids = meta["page_ids"]
        segments = meta["segments"]

        output_text = ""
        had_error = 0
        if cid in result_map:
            body = (((result_map[cid].get("response") or {}).get("body")) or {})
            output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in error_map:
            had_error = 1
            body = (((error_map[cid].get("response") or {}).get("body")) or {})
            output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))

        parsed = parse_json_from_text(output_text)
        parse_ok = int(isinstance(parsed, dict) and bool(parsed))
        coerced = _coerce_parse(parsed if isinstance(parsed, dict) else {}, issue, page_ids)

        issue_row = {
            "newspaper_slug": norm_str(issue.get("pub_slug")).lower(),
            "city_key": norm_str(issue.get("city_key")),
            "issue_id": norm_str(issue.get("issue_id")),
            "issue_date": norm_str(issue.get("issue_date")),
            "page_count": int(len(page_ids)),
            "page_tags": coerced.get("page_tags") or [],
            "page_segments": segments,
            "issue_notes": norm_str(coerced.get("issue_notes")),
            "parse_ok": parse_ok,
            "had_error": had_error,
        }
        issue_rows.append(issue_row)

        for pt in issue_row["page_tags"]:
            flat_rows.append(
                {
                    "newspaper_slug": issue_row["newspaper_slug"],
                    "city_key": issue_row["city_key"],
                    "issue_id": issue_row["issue_id"],
                    "issue_date": issue_row["issue_date"],
                    "page_id": norm_str(pt.get("page_id")),
                    "tag": norm_str(pt.get("tag")),
                    "confidence_0_to_1": safe_float(pt.get("confidence_0_to_1"), default=0.0),
                    "evidence": norm_str(pt.get("evidence")),
                    "parse_ok": parse_ok,
                }
            )

    issue_df = pd.DataFrame.from_records(issue_rows)
    flat_df = pd.DataFrame.from_records(flat_rows)
    issue_df = issue_df.sort_values(["newspaper_slug", "issue_date", "issue_id"]).reset_index(drop=True)
    flat_df = flat_df.sort_values(["newspaper_slug", "issue_date", "issue_id", "page_id"]).reset_index(drop=True)

    write_jsonl(out_dir / "page_tags.jsonl", issue_rows)
    write_csv(out_dir / "page_tags_flat.csv", flat_df)

    summary = (
        flat_df.groupby(["newspaper_slug", "tag"], dropna=False).size().rename("page_count").reset_index()
        if not flat_df.empty
        else pd.DataFrame(columns=["newspaper_slug", "tag", "page_count"])
    )
    write_csv(out_dir / "page_tag_summary.csv", summary)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "newspapers": newspapers,
        "city_map": city_map,
        "issue_count": int(len(issue_rows)),
        "page_count": int(len(flat_rows)),
        "parse_ok_share": float(issue_df["parse_ok"].mean()) if not issue_df.empty else math.nan,
        "model": str(args.model),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eprint(f"Done. page_tagging output: {out_dir}")


if __name__ == "__main__":
    main()
