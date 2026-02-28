#!/usr/bin/env python3
"""
LLM-only anchor-level zoning legal text cleaning for v3 pipeline.

Variants:
  - single_pass
  - two_pass (selection + recall)
  - three_pass (selection + recall + adjudication)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from fullnewspaper_v3_common import (
    eprint,
    enum,
    extract_openai_output_text,
    iter_jsonl,
    norm_str,
    parse_json_from_text,
    read_error_jsonl,
    read_result_jsonl,
    run_gateway_requests,
    safe_float,
    safe_int,
    write_csv,
    write_jsonl,
    write_openai_requests,
)


ALLOWED_ANCHOR_ROLE = {
    "verbatim_legal_clause",
    "legal_notice_language",
    "zoning_related_narrative",
    "general_news_or_opinion",
    "uncertain",
}
ALLOWED_VARIANTS = {"single_pass", "two_pass", "three_pass"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM-only anchor-level zoning cleaning.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Run directory containing panels/<city_key>/issue_texts.jsonl.",
    )
    ap.add_argument(
        "--page-tags-jsonl",
        default="",
        help="Path to page_tags.jsonl from llm_page_tag_v3.py (default: <run-dir>/fullnewspaper_v3/page_tagging/page_tags.jsonl).",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-dir>/fullnewspaper_v3/cleaning).",
    )
    ap.add_argument(
        "--newspapers",
        default="cazenovia-republican,albion-evening-recorder,bountiful-davis-county-clipper,abilene-reporter-news,appleton-post-crescent",
        help="Comma-separated newspaper slugs.",
    )
    ap.add_argument("--variant", default="three_pass", choices=sorted(ALLOWED_VARIANTS))
    ap.add_argument(
        "--strict-narrative-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, uses stronger anti-narrative guardrails in recall/adjudication prompts.",
    )
    ap.add_argument("--model", default="openai:gpt-5")
    ap.add_argument(
        "--max-anchor-text-chars",
        type=int,
        default=0,
        help="Hard cap for per-anchor text included in prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument(
        "--max-anchor-block-chars",
        type=int,
        default=520,
        help="Target character budget per anchor block before splitting into the next anchor.",
    )
    ap.add_argument(
        "--max-lines-per-anchor-block",
        type=int,
        default=6,
        help="Maximum number of source lines grouped into one anchor block.",
    )
    ap.add_argument(
        "--max-anchors-per-issue",
        type=int,
        default=0,
        help="Hard cap for anchors per issue. 0 means no cap. If positive and exceeded, the run fails (no truncation).",
    )
    ap.add_argument("--limit-issues", type=int, default=0, help="Optional cap for calibration runs. 0 = all issues.")
    ap.add_argument(
        "--issue-ids-file",
        default="",
        help="Optional newline-delimited issue_id filter. When provided, only those issue_ids are processed.",
    )
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument(
        "--gateway-runner",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/experimental/scripts/run_openai_requests_via_gateway.py",
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/agent-gateway/src",
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


def _iter_line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    s = str(text or "")
    for line in s.splitlines(keepends=True):
        end = start + len(line)
        spans.append((start, end, line.rstrip("\n")))
        start = end
    if not spans and s:
        spans.append((0, len(s), s))
    return spans


def _split_long_line(line: str, start_char: int, end_char: int) -> list[tuple[int, int, str]]:
    clean = str(line or "")
    if len(clean) <= 650:
        return [(start_char, end_char, clean)]
    parts = re.split(r"(?<=[\.\?\!\;\:])\s+", clean)
    out: list[tuple[int, int, str]] = []
    cursor = start_char
    for p in parts:
        seg = str(p or "").strip()
        if not seg:
            continue
        pos = clean.find(seg, max(0, cursor - start_char))
        if pos == -1:
            pos = 0
        s_abs = start_char + pos
        e_abs = min(end_char, s_abs + len(seg))
        out.append((s_abs, e_abs, seg))
        cursor = e_abs
    return out or [(start_char, end_char, clean)]


def _build_anchor_blocks(
    spans: list[tuple[int, int, str]],
    *,
    max_block_chars: int,
    max_lines_per_block: int,
) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    cur_lines: list[str] = []
    cur_chars = 0
    cur_line_count = 0

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_lines, cur_chars, cur_line_count
        if cur_start is None or cur_end is None or not cur_lines:
            cur_start = None
            cur_end = None
            cur_lines = []
            cur_chars = 0
            cur_line_count = 0
            return
        txt = "\n".join([norm_str(x) for x in cur_lines if norm_str(x)]).strip()
        if txt:
            blocks.append((int(cur_start), int(cur_end), txt))
        cur_start = None
        cur_end = None
        cur_lines = []
        cur_chars = 0
        cur_line_count = 0

    for s, e, line in spans:
        txt = norm_str(line)
        if not txt:
            flush()
            continue
        pieces = _split_long_line(txt, s, e)
        for ps, pe, ptxt in pieces:
            seg = norm_str(ptxt)
            if not seg:
                continue
            seg_len = len(seg)
            if cur_start is None:
                cur_start = int(ps)
                cur_end = int(pe)
                cur_lines = [seg]
                cur_chars = seg_len
                cur_line_count = 1
                continue
            would_chars = cur_chars + 1 + seg_len
            if would_chars <= max(120, int(max_block_chars)) and cur_line_count < max(1, int(max_lines_per_block)):
                cur_end = int(pe)
                cur_lines.append(seg)
                cur_chars = would_chars
                cur_line_count += 1
            else:
                flush()
                cur_start = int(ps)
                cur_end = int(pe)
                cur_lines = [seg]
                cur_chars = seg_len
                cur_line_count = 1
    flush()
    return blocks


def _build_anchors(
    issue_text: str,
    page_segments: list[dict[str, Any]],
    issue_id: str,
    max_anchors: int,
    *,
    max_block_chars: int,
    max_lines_per_block: int,
) -> list[dict[str, Any]]:
    spans = _iter_line_spans(issue_text)
    anchors: list[dict[str, Any]] = []

    def page_for_mid(mid: int) -> str:
        for seg in page_segments:
            s = int(seg.get("start_char") or 0)
            e = int(seg.get("end_char") or 0)
            if s <= mid < max(e, s + 1):
                return norm_str(seg.get("page_id"))
        if page_segments:
            return norm_str(page_segments[-1].get("page_id"))
        return ""

    blocks = _build_anchor_blocks(
        spans,
        max_block_chars=max_block_chars,
        max_lines_per_block=max_lines_per_block,
    )
    idx = 0
    for ps, pe, ptxt in blocks:
        t = norm_str(ptxt)
        if not t:
            continue
        idx += 1
        mid = (ps + pe) // 2
        anchors.append(
            {
                "anchor_id": f"A{idx:05d}",
                "start_char": int(ps),
                "end_char": int(pe),
                "page_id": page_for_mid(mid),
                "text": t,
            }
        )
    if max_anchors > 0 and len(anchors) > max_anchors:
        raise ValueError(
            f"issue_id={issue_id} produced {len(anchors)} anchors, exceeding hard cap {max_anchors}. "
            "Anchor truncation is disabled; increase --max-anchors-per-issue or set 0 for no cap."
        )
    return anchors


def _build_pass1_prompt(
    issue: dict[str, Any],
    anchors: list[dict[str, Any]],
    max_anchor_chars: int,
    *,
    strict_narrative_mode: bool = False,
) -> str:
    schema = {
        "issue_id": norm_str(issue.get("issue_id")),
        "kept_anchors": [
            {
                "anchor_id": "A00001",
                "anchor_role": "verbatim_legal_clause",
                "confidence_0_to_1": 0.9,
                "reason_short": "",
            }
        ],
        "issue_notes": "",
    }
    lines: list[str] = []
    lines.append("You are extracting ONLY verbatim zoning law language from newspaper text.")
    lines.append("Return ONLY JSON. No markdown.")
    lines.append("")
    lines.append("Select ONLY anchors that should be kept for the legal-text corpus.")
    lines.append("Do NOT output dropped anchors.")
    lines.append("Allowed anchor_role for kept anchors:")
    lines.append("- verbatim_legal_clause")
    lines.append("- legal_notice_language")
    lines.append("")
    lines.append("Critical guardrail:")
    lines.append("- Do NOT treat zoning-related narrative, summaries, commentary, or reporting as legal text.")
    lines.append("- Keep anchors only when they are clearly verbatim legal clauses or legal-notice language.")
    lines.append("- If none qualify, return kept_anchors as an empty list.")
    if bool(strict_narrative_mode):
        lines.append("- Precision mode: if an anchor is ambiguous between narrative and law, do NOT keep it.")
        lines.append("- Mentions of hearings/proposals without operative legal clauses should be dropped.")
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
    lines.append("Anchors:")
    for a in anchors:
        lines.append("")
        lines.append(f"[ANCHOR] id={a['anchor_id']} page_id={norm_str(a.get('page_id'))}")
        lines.append(
            _bounded_text_or_fail(
                norm_str(a.get("text")),
                max_anchor_chars,
                label=f"issue_id={norm_str(issue.get('issue_id'))} anchor_id={norm_str(a.get('anchor_id'))} anchor_text",
            )
        )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _build_pass2_prompt(
    issue: dict[str, Any],
    dropped_anchors: list[dict[str, Any]],
    max_anchor_chars: int,
    *,
    strict_narrative_mode: bool = False,
) -> str:
    schema = {
        "issue_id": norm_str(issue.get("issue_id")),
        "rescue_anchor_ids": ["A00042"],
        "issue_notes": "",
    }
    lines: list[str] = []
    lines.append("You are auditing previously dropped anchors for missed verbatim zoning legal text.")
    lines.append("Return ONLY JSON. No markdown.")
    lines.append("")
    lines.append("Select rescue_anchor_ids ONLY when the anchor is genuinely verbatim legal clause text or formal legal notice language.")
    lines.append("Do NOT rescue zoning-related narrative.")
    if bool(strict_narrative_mode):
        lines.append("Precision mode: if uncertain, do not rescue.")
        lines.append("Do not rescue policy summaries or commentary that does not quote operative legal text.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(f"Issue metadata: issue_id={norm_str(issue.get('issue_id'))}")
    lines.append("")
    lines.append("Dropped anchors to review:")
    for a in dropped_anchors:
        lines.append("")
        lines.append(f"[ANCHOR] id={a['anchor_id']} page_id={norm_str(a.get('page_id'))}")
        lines.append(
            _bounded_text_or_fail(
                norm_str(a.get("text")),
                max_anchor_chars,
                label=f"issue_id={norm_str(issue.get('issue_id'))} anchor_id={norm_str(a.get('anchor_id'))} anchor_text",
            )
        )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _build_pass3_prompt(
    issue: dict[str, Any],
    candidate_anchors: list[dict[str, Any]],
    max_anchor_chars: int,
    *,
    strict_narrative_mode: bool = False,
) -> str:
    schema = {
        "issue_id": norm_str(issue.get("issue_id")),
        "final_keep_anchor_ids": ["A00001"],
        "issue_notes": "",
    }
    lines: list[str] = []
    lines.append("Adjudicate final keep list for zoning law corpus.")
    lines.append("Return ONLY JSON. No markdown.")
    lines.append("")
    lines.append("Keep only anchors that are verbatim legal clauses or formal legal notices.")
    lines.append("Drop narrative, summaries, and commentary even if zoning-related.")
    if bool(strict_narrative_mode):
        lines.append("Precision mode: default to drop when legal-operativity is unclear.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(f"Issue metadata: issue_id={norm_str(issue.get('issue_id'))}")
    lines.append("")
    lines.append("Candidate anchors:")
    for a in candidate_anchors:
        lines.append("")
        lines.append(f"[ANCHOR] id={a['anchor_id']} page_id={norm_str(a.get('page_id'))}")
        lines.append(
            _bounded_text_or_fail(
                norm_str(a.get("text")),
                max_anchor_chars,
                label=f"issue_id={norm_str(issue.get('issue_id'))} anchor_id={norm_str(a.get('anchor_id'))} anchor_text",
            )
        )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _coerce_pass1(parsed_obj: dict[str, Any], issue_id: str, anchor_ids: set[str]) -> dict[str, Any]:
    obj = dict(parsed_obj or {})
    rows = obj.get("kept_anchors") if isinstance(obj.get("kept_anchors"), list) else []
    if not rows and isinstance(obj.get("anchors"), list):
        # Backward compatibility with older prompt shape.
        rows = obj.get("anchors")  # type: ignore[assignment]
    out_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        aid = norm_str(r.get("anchor_id"))
        if not aid or aid not in anchor_ids or aid in seen:
            continue
        seen.add(aid)
        role = enum(r.get("anchor_role"), {"verbatim_legal_clause", "legal_notice_language"}, "")
        if role not in {"verbatim_legal_clause", "legal_notice_language"}:
            continue
        out_rows.append(
            {
                "anchor_id": aid,
                "anchor_role": role,
                "keep_for_law_corpus": 1,
                "confidence_0_to_1": safe_float(r.get("confidence_0_to_1"), default=0.0),
                "reason_short": norm_str(r.get("reason_short")),
            }
        )
    return {
        "issue_id": norm_str(obj.get("issue_id")) or issue_id,
        "anchors": out_rows,
        "issue_notes": norm_str(obj.get("issue_notes")),
    }


def _coerce_pass2(parsed_obj: dict[str, Any], issue_id: str, allowed_ids: set[str]) -> dict[str, Any]:
    obj = dict(parsed_obj or {})
    ids_raw = obj.get("rescue_anchor_ids")
    rescue_ids: list[str] = []
    if isinstance(ids_raw, list):
        for x in ids_raw:
            aid = norm_str(x)
            if aid and aid in allowed_ids and aid not in rescue_ids:
                rescue_ids.append(aid)
    return {
        "issue_id": norm_str(obj.get("issue_id")) or issue_id,
        "rescue_anchor_ids": rescue_ids,
        "issue_notes": norm_str(obj.get("issue_notes")),
    }


def _coerce_pass3(parsed_obj: dict[str, Any], issue_id: str, allowed_ids: set[str]) -> dict[str, Any]:
    obj = dict(parsed_obj or {})
    ids_raw = obj.get("final_keep_anchor_ids")
    keep_ids: list[str] = []
    if isinstance(ids_raw, list):
        for x in ids_raw:
            aid = norm_str(x)
            if aid and aid in allowed_ids and aid not in keep_ids:
                keep_ids.append(aid)
    return {
        "issue_id": norm_str(obj.get("issue_id")) or issue_id,
        "final_keep_anchor_ids": keep_ids,
        "issue_notes": norm_str(obj.get("issue_notes")),
    }


def _load_issue_pool(run_dir: Path, newspapers: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    panel_dir = run_dir / "panels"
    for city_dir in sorted(panel_dir.iterdir()):
        p = city_dir / "issue_texts.jsonl"
        if not city_dir.is_dir() or not p.is_file():
            continue
        for r in iter_jsonl(p):
            slug = norm_str(r.get("pub_slug")).lower()
            if slug in newspapers:
                out.append(r)
    out = sorted(out, key=lambda r: (norm_str(r.get("pub_slug")), norm_str(r.get("issue_date")), norm_str(r.get("issue_id"))))
    return out


def _issue_metrics_row(issue: dict[str, Any], anchors: list[dict[str, Any]]) -> dict[str, Any]:
    kept = [a for a in anchors if int(a.get("keep_for_law_corpus") or 0) == 1]
    total_chars = len(norm_str(issue.get("text")))
    kept_chars = sum(len(norm_str(a.get("text"))) for a in kept)
    return {
        "newspaper_slug": norm_str(issue.get("pub_slug")).lower(),
        "city_key": norm_str(issue.get("city_key")),
        "issue_id": norm_str(issue.get("issue_id")),
        "issue_date": norm_str(issue.get("issue_date")),
        "anchor_count": int(len(anchors)),
        "kept_anchor_count": int(len(kept)),
        "input_chars": int(total_chars),
        "kept_chars": int(kept_chars),
        "keep_ratio": (float(kept_chars) / float(total_chars)) if total_chars > 0 else math.nan,
    }


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    newspapers = [norm_str(x).lower() for x in norm_str(args.newspapers).split(",") if norm_str(x)]
    if not newspapers:
        raise SystemExit("No newspapers provided.")

    page_tags_path = (
        Path(args.page_tags_jsonl).expanduser().resolve()
        if norm_str(args.page_tags_jsonl)
        else run_dir / "fullnewspaper_v3" / "page_tagging" / "page_tags.jsonl"
    )
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if norm_str(args.output_dir)
        else run_dir / "fullnewspaper_v3" / "cleaning"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    page_tag_rows = iter_jsonl(page_tags_path)
    page_tag_map: dict[str, dict[str, Any]] = {}
    for r in page_tag_rows:
        iid = norm_str(r.get("issue_id"))
        if iid:
            page_tag_map[iid] = r

    issues = _load_issue_pool(run_dir, newspapers)
    issue_id_filter_path = Path(args.issue_ids_file).expanduser().resolve() if norm_str(args.issue_ids_file) else None
    issue_id_filter: set[str] = set()
    if issue_id_filter_path is not None:
        if not issue_id_filter_path.is_file():
            raise SystemExit(f"issue ids file not found: {issue_id_filter_path}")
        for line in issue_id_filter_path.read_text(encoding="utf-8").splitlines():
            iid = norm_str(line)
            if iid:
                issue_id_filter.add(iid)
        issues = [r for r in issues if norm_str(r.get("issue_id")) in issue_id_filter]
    if int(args.limit_issues) > 0:
        issues = issues[: int(args.limit_issues)]
    if not issues:
        raise SystemExit("No issues loaded for newspapers.")

    # Build anchors and pass1 requests.
    pass1_rows: list[tuple[str, str]] = []
    meta_by_issue: dict[str, dict[str, Any]] = {}
    for issue in issues:
        iid = norm_str(issue.get("issue_id"))
        slug = norm_str(issue.get("pub_slug")).lower()
        page_row = page_tag_map.get(iid) or {}
        segs = page_row.get("page_segments")
        if not isinstance(segs, list):
            segs = []
        anchors = _build_anchors(
            norm_str(issue.get("text")),
            segs,
            issue_id=iid,
            max_anchors=int(args.max_anchors_per_issue),
            max_block_chars=int(args.max_anchor_block_chars),
            max_lines_per_block=int(args.max_lines_per_anchor_block),
        )
        prompt = _build_pass1_prompt(
            issue,
            anchors,
            max_anchor_chars=int(args.max_anchor_text_chars),
            strict_narrative_mode=bool(args.strict_narrative_mode),
        )
        cid = f"anchor_clean_v3_pass1::{slug}::{iid}"
        pass1_rows.append((cid, prompt))
        meta_by_issue[iid] = {"issue": issue, "anchors": anchors, "slug": slug}

    pass1_dir = out_dir / "pass1"
    write_openai_requests(pass1_dir / "requests" / "openai_requests_shard000.jsonl", pass1_rows, model=str(args.model))
    run_gateway_requests(
        request_dir=pass1_dir / "requests",
        output_dir=pass1_dir / "results",
        runner_path=Path(args.gateway_runner).expanduser().resolve(),
        model=str(args.model),
        gateway_pythonpath=Path(args.gateway_pythonpath).expanduser().resolve() if norm_str(args.gateway_pythonpath) else None,
        timeout_s=float(args.timeout),
        concurrency=int(args.concurrency),
        gov_env_path=Path(args.gov_env_path).expanduser().resolve(),
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )
    p1_res = read_result_jsonl(pass1_dir / "results")
    p1_err = read_error_jsonl(pass1_dir / "results")

    pass1_parsed: dict[str, dict[str, Any]] = {}
    for cid, _prompt in pass1_rows:
        issue_id = cid.split("::", 2)[2]
        meta = meta_by_issue[issue_id]
        anchor_ids = {norm_str(a.get("anchor_id")) for a in meta["anchors"]}
        output_text = ""
        if cid in p1_res:
            body = (((p1_res[cid].get("response") or {}).get("body")) or {})
            output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in p1_err:
            body = (((p1_err[cid].get("response") or {}).get("body")) or {})
            output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
        parsed = parse_json_from_text(output_text)
        parse_ok = int(isinstance(parsed, dict) and bool(parsed))
        coerced = _coerce_pass1(parsed if isinstance(parsed, dict) else {}, issue_id, anchor_ids)
        coerced["parse_ok"] = parse_ok
        pass1_parsed[issue_id] = coerced

    # Pass2 recall for two_pass/three_pass.
    pass2_parsed: dict[str, dict[str, Any]] = {}
    pass2_rows: list[tuple[str, str]] = []
    if args.variant in {"two_pass", "three_pass"}:
        for issue_id, p1 in pass1_parsed.items():
            meta = meta_by_issue[issue_id]
            issue = meta["issue"]
            dec_by_id = {norm_str(r.get("anchor_id")): r for r in (p1.get("anchors") or []) if isinstance(r, dict)}
            dropped = [
                a
                for a in meta["anchors"]
                if int((dec_by_id.get(norm_str(a.get("anchor_id"))) or {}).get("keep_for_law_corpus", 0)) != 1
            ]
            if not dropped:
                continue
            cid = f"anchor_clean_v3_pass2::{meta['slug']}::{issue_id}"
            pass2_rows.append(
                (
                    cid,
                    _build_pass2_prompt(
                        issue,
                        dropped,
                        max_anchor_chars=int(args.max_anchor_text_chars),
                        strict_narrative_mode=bool(args.strict_narrative_mode),
                    ),
                )
            )

        pass2_dir = out_dir / "pass2"
        if pass2_rows:
            write_openai_requests(pass2_dir / "requests" / "openai_requests_shard000.jsonl", pass2_rows, model=str(args.model))
            run_gateway_requests(
                request_dir=pass2_dir / "requests",
                output_dir=pass2_dir / "results",
                runner_path=Path(args.gateway_runner).expanduser().resolve(),
                model=str(args.model),
                gateway_pythonpath=Path(args.gateway_pythonpath).expanduser().resolve() if norm_str(args.gateway_pythonpath) else None,
                timeout_s=float(args.timeout),
                concurrency=int(args.concurrency),
                gov_env_path=Path(args.gov_env_path).expanduser().resolve(),
                skip_existing=bool(args.skip_existing),
                dry_run=bool(args.dry_run),
            )
            p2_res = read_result_jsonl(pass2_dir / "results")
            p2_err = read_error_jsonl(pass2_dir / "results")
            for cid, _prompt in pass2_rows:
                issue_id = cid.split("::", 2)[2]
                output_text = ""
                if cid in p2_res:
                    body = (((p2_res[cid].get("response") or {}).get("body")) or {})
                    output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
                elif cid in p2_err:
                    body = (((p2_err[cid].get("response") or {}).get("body")) or {})
                    output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
                parsed = parse_json_from_text(output_text)
                parse_ok = int(isinstance(parsed, dict) and bool(parsed))
                dropped_ids = {
                    norm_str(a.get("anchor_id"))
                    for a in meta_by_issue[issue_id]["anchors"]
                    if norm_str(a.get("anchor_id"))
                }
                coerced = _coerce_pass2(parsed if isinstance(parsed, dict) else {}, issue_id, dropped_ids)
                coerced["parse_ok"] = parse_ok
                pass2_parsed[issue_id] = coerced

    # Pass3 adjudication for three_pass.
    pass3_parsed: dict[str, dict[str, Any]] = {}
    pass3_rows: list[tuple[str, str]] = []
    if args.variant == "three_pass":
        for issue_id, p1 in pass1_parsed.items():
            meta = meta_by_issue[issue_id]
            issue = meta["issue"]
            by_id = {norm_str(a.get("anchor_id")): a for a in meta["anchors"]}
            keep_ids = [
                norm_str(r.get("anchor_id"))
                for r in (p1.get("anchors") or [])
                if isinstance(r, dict) and int(r.get("keep_for_law_corpus") or 0) == 1
            ]
            rescue_ids = (pass2_parsed.get(issue_id) or {}).get("rescue_anchor_ids") or []
            cand_ids = sorted({x for x in keep_ids + rescue_ids if x in by_id})
            if not cand_ids:
                continue
            cand_anchors = [by_id[x] for x in cand_ids if x in by_id]
            cid = f"anchor_clean_v3_pass3::{meta['slug']}::{issue_id}"
            pass3_rows.append(
                (
                    cid,
                    _build_pass3_prompt(
                        issue,
                        cand_anchors,
                        max_anchor_chars=int(args.max_anchor_text_chars),
                        strict_narrative_mode=bool(args.strict_narrative_mode),
                    ),
                )
            )

        pass3_dir = out_dir / "pass3"
        if pass3_rows:
            write_openai_requests(pass3_dir / "requests" / "openai_requests_shard000.jsonl", pass3_rows, model=str(args.model))
            run_gateway_requests(
                request_dir=pass3_dir / "requests",
                output_dir=pass3_dir / "results",
                runner_path=Path(args.gateway_runner).expanduser().resolve(),
                model=str(args.model),
                gateway_pythonpath=Path(args.gateway_pythonpath).expanduser().resolve() if norm_str(args.gateway_pythonpath) else None,
                timeout_s=float(args.timeout),
                concurrency=int(args.concurrency),
                gov_env_path=Path(args.gov_env_path).expanduser().resolve(),
                skip_existing=bool(args.skip_existing),
                dry_run=bool(args.dry_run),
            )
            p3_res = read_result_jsonl(pass3_dir / "results")
            p3_err = read_error_jsonl(pass3_dir / "results")
            for cid, _prompt in pass3_rows:
                issue_id = cid.split("::", 2)[2]
                output_text = ""
                if cid in p3_res:
                    body = (((p3_res[cid].get("response") or {}).get("body")) or {})
                    output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
                elif cid in p3_err:
                    body = (((p3_err[cid].get("response") or {}).get("body")) or {})
                    output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
                parsed = parse_json_from_text(output_text)
                parse_ok = int(isinstance(parsed, dict) and bool(parsed))
                # Allowed set is pass1 keep + pass2 rescue.
                p1 = pass1_parsed.get(issue_id) or {}
                keep_ids = {
                    norm_str(r.get("anchor_id"))
                    for r in (p1.get("anchors") or [])
                    if isinstance(r, dict) and int(r.get("keep_for_law_corpus") or 0) == 1
                }
                rescue_ids = set((pass2_parsed.get(issue_id) or {}).get("rescue_anchor_ids") or [])
                allowed = {x for x in (keep_ids | rescue_ids) if x}
                coerced = _coerce_pass3(parsed if isinstance(parsed, dict) else {}, issue_id, allowed)
                coerced["parse_ok"] = parse_ok
                pass3_parsed[issue_id] = coerced

    # Finalize issue-level anchor decisions.
    issue_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for issue in issues:
        iid = norm_str(issue.get("issue_id"))
        meta = meta_by_issue[iid]
        anchors = meta["anchors"]
        by_id = {norm_str(a.get("anchor_id")): dict(a) for a in anchors}
        p1 = pass1_parsed.get(iid) or {"anchors": [], "parse_ok": 0, "issue_notes": ""}
        dec1 = {norm_str(r.get("anchor_id")): r for r in (p1.get("anchors") or []) if isinstance(r, dict)}
        rescue_ids = set((pass2_parsed.get(iid) or {}).get("rescue_anchor_ids") or [])
        final_keep_ids_pass3 = set((pass3_parsed.get(iid) or {}).get("final_keep_anchor_ids") or [])

        final_keep_ids: set[str] = set()
        if args.variant == "single_pass":
            final_keep_ids = set(dec1.keys())
        elif args.variant == "two_pass":
            final_keep_ids = set(dec1.keys()) | rescue_ids
        else:
            # three_pass final decision comes from adjudication.
            final_keep_ids = final_keep_ids_pass3

        final_anchors: list[dict[str, Any]] = []
        for a in anchors:
            aid = norm_str(a.get("anchor_id"))
            d1 = dec1.get(aid) or {}
            role = enum(d1.get("anchor_role"), ALLOWED_ANCHOR_ROLE, "uncertain")
            keep = int(aid in final_keep_ids)
            # Keep only legal roles in final output.
            if keep == 1 and role not in {"verbatim_legal_clause", "legal_notice_language"}:
                keep = 0
            final_anchors.append(
                {
                    "anchor_id": aid,
                    "start_char": int(a.get("start_char") or 0),
                    "end_char": int(a.get("end_char") or 0),
                    "page_id": norm_str(a.get("page_id")),
                    "text": norm_str(a.get("text")),
                    "anchor_role": role,
                    "keep_for_law_corpus": keep,
                    "confidence_0_to_1": safe_float(d1.get("confidence_0_to_1"), default=0.0),
                    "reason_short": norm_str(d1.get("reason_short")),
                }
            )

        kept_sorted = [a for a in final_anchors if int(a.get("keep_for_law_corpus") or 0) == 1]
        kept_sorted = sorted(kept_sorted, key=lambda x: (int(x.get("start_char") or 0), norm_str(x.get("anchor_id"))))
        clean_text = "\n".join(norm_str(a.get("text")) for a in kept_sorted if norm_str(a.get("text"))).strip()

        notes = []
        if norm_str(p1.get("issue_notes")):
            notes.append(norm_str(p1.get("issue_notes")))
        if norm_str((pass2_parsed.get(iid) or {}).get("issue_notes")):
            notes.append(norm_str((pass2_parsed.get(iid) or {}).get("issue_notes")))
        if norm_str((pass3_parsed.get(iid) or {}).get("issue_notes")):
            notes.append(norm_str((pass3_parsed.get(iid) or {}).get("issue_notes")))

        issue_row = {
            "variant": str(args.variant),
            "newspaper_slug": norm_str(issue.get("pub_slug")).lower(),
            "city_key": norm_str(issue.get("city_key")),
            "issue_id": iid,
            "issue_date": norm_str(issue.get("issue_date")),
            "classification_label_prior": norm_str(issue.get("classification_label")),
            "anchor_count": int(len(final_anchors)),
            "kept_anchor_count": int(len(kept_sorted)),
            "anchors": final_anchors,
            "kept_anchor_ids": [norm_str(a.get("anchor_id")) for a in kept_sorted],
            "clean_zoning_text": clean_text,
            "issue_notes": " | ".join([x for x in notes if x]),
            "pass1_parse_ok": int(p1.get("parse_ok") or 0),
            "pass2_parse_ok": int((pass2_parsed.get(iid) or {}).get("parse_ok") or 0),
            "pass3_parse_ok": int((pass3_parsed.get(iid) or {}).get("parse_ok") or 0),
        }
        issue_rows.append(issue_row)

        clean_rows.append(
            {
                "newspaper_slug": issue_row["newspaper_slug"],
                "city_key": issue_row["city_key"],
                "issue_id": issue_row["issue_id"],
                "issue_date": issue_row["issue_date"],
                "clean_zoning_text": clean_text,
                "clean_chars": int(len(clean_text)),
                "kept_anchor_count": int(issue_row["kept_anchor_count"]),
                "anchor_count": int(issue_row["anchor_count"]),
                "variant": str(args.variant),
            }
        )
        metric_rows.append(_issue_metrics_row(issue, final_anchors))

    issue_rows = sorted(issue_rows, key=lambda r: (norm_str(r.get("newspaper_slug")), norm_str(r.get("issue_date")), norm_str(r.get("issue_id"))))
    clean_df = pd.DataFrame.from_records(clean_rows)
    metric_df = pd.DataFrame.from_records(metric_rows)

    write_jsonl(out_dir / "anchor_decisions.jsonl", issue_rows)
    write_jsonl(out_dir / "clean_zoning_text.jsonl", clean_rows)
    write_csv(out_dir / "clean_zoning_text_flat.csv", clean_df)
    write_csv(out_dir / "clean_issue_metrics.csv", metric_df)

    summary = (
        metric_df.groupby("newspaper_slug", dropna=False)
        .agg(
            issue_count=("issue_id", "nunique"),
            mean_keep_ratio=("keep_ratio", "mean"),
            mean_anchor_count=("anchor_count", "mean"),
            mean_kept_anchor_count=("kept_anchor_count", "mean"),
        )
        .reset_index()
        if not metric_df.empty
        else pd.DataFrame(columns=["newspaper_slug", "issue_count", "mean_keep_ratio", "mean_anchor_count", "mean_kept_anchor_count"])
    )
    write_csv(out_dir / "clean_summary_by_newspaper.csv", summary)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "page_tags_jsonl": str(page_tags_path),
        "variant": str(args.variant),
        "strict_narrative_mode": bool(args.strict_narrative_mode),
        "model": str(args.model),
        "issue_count": int(len(issue_rows)),
        "newspapers": newspapers,
        "issue_ids_file": str(issue_id_filter_path) if issue_id_filter_path is not None else "",
        "issue_id_filter_count": int(len(issue_id_filter)),
        "pass1_parse_share": float(pd.Series([int((pass1_parsed.get(norm_str(r.get("issue_id"))) or {}).get("parse_ok") or 0) for r in issues]).mean()) if issues else math.nan,
        "pass2_parse_share": float(pd.Series([int((pass2_parsed.get(norm_str(r.get("issue_id"))) or {}).get("parse_ok") or 0) for r in issues]).mean()) if issues else math.nan,
        "pass3_parse_share": float(pd.Series([int((pass3_parsed.get(norm_str(r.get("issue_id"))) or {}).get("parse_ok") or 0) for r in issues]).mean()) if issues else math.nan,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eprint(f"Done. cleaning output: {out_dir}")


if __name__ == "__main__":
    main()
