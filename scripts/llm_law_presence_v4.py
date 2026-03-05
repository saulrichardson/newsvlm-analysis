#!/usr/bin/env python3
"""
LLM issue-level law-presence gate for fullnewspaper v4 pilot.

Purpose:
- Determine whether an issue contains verbatim zoning-law language.
- Provide evidence anchor ids for downstream ontology and audit traceability.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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
    write_csv,
    write_jsonl,
    write_openai_requests,
)


ALLOWED_PRESENCE = {"yes", "no", "uncertain"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM law-presence gate v4.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Run directory containing panels/<city_key>/issue_texts.jsonl.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-dir>/fullnewspaper_v4/law_presence).",
    )
    ap.add_argument(
        "--newspapers",
        default="",
        help="Comma-separated newspaper slugs.",
    )
    ap.add_argument("--model", default="openai:gpt-5")
    ap.add_argument(
        "--max-anchors-per-issue",
        type=int,
        default=0,
        help="Hard cap for number of anchors per issue. 0 means no cap. If positive and exceeded, the run fails (no truncation).",
    )
    ap.add_argument(
        "--max-anchor-chars",
        type=int,
        default=0,
        help="Hard cap for per-anchor text included in prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument("--max-block-chars", type=int, default=520)
    ap.add_argument("--max-lines-per-block", type=int, default=6)
    ap.add_argument("--limit-issues", type=int, default=0)
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


def _load_issue_pool(run_dir: Path, newspapers: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    panel_root = run_dir / "panels"
    for city_dir in sorted(panel_root.iterdir()):
        p = city_dir / "issue_texts.jsonl"
        if not city_dir.is_dir() or not p.is_file():
            continue
        for r in iter_jsonl(p):
            slug = norm_str(r.get("pub_slug")).lower()
            if slug in newspapers:
                out.append(r)
    out = sorted(out, key=lambda r: (norm_str(r.get("pub_slug")), norm_str(r.get("issue_date")), norm_str(r.get("issue_id"))))
    return out


def _bounded_text_or_fail(text: str, max_chars: int, label: str) -> str:
    s = norm_str(text)
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
    txt = norm_str(line)
    if len(txt) <= 650:
        return [(start_char, end_char, txt)]
    parts = [p.strip() for p in txt.replace(";", ". ").split(". ") if p.strip()]
    out: list[tuple[int, int, str]] = []
    cur = start_char
    for p in parts:
        end = min(end_char, cur + len(p))
        out.append((cur, end, p))
        cur = min(end_char, end + 1)
    return out or [(start_char, end_char, txt)]


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
    cur_n = 0

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_lines, cur_chars, cur_n
        if cur_start is None or cur_end is None or not cur_lines:
            cur_start, cur_end, cur_lines, cur_chars, cur_n = None, None, [], 0, 0
            return
        txt = "\n".join([norm_str(x) for x in cur_lines if norm_str(x)]).strip()
        if txt:
            blocks.append((int(cur_start), int(cur_end), txt))
        cur_start, cur_end, cur_lines, cur_chars, cur_n = None, None, [], 0, 0

    for s, e, line in spans:
        t = norm_str(line)
        if not t:
            flush()
            continue
        pieces = _split_long_line(t, s, e)
        for ps, pe, seg in pieces:
            seg = norm_str(seg)
            if not seg:
                continue
            if cur_start is None:
                cur_start, cur_end, cur_lines, cur_chars, cur_n = int(ps), int(pe), [seg], len(seg), 1
                continue
            next_chars = cur_chars + 1 + len(seg)
            if next_chars <= max(120, int(max_block_chars)) and cur_n < max(1, int(max_lines_per_block)):
                cur_end = int(pe)
                cur_lines.append(seg)
                cur_chars = next_chars
                cur_n += 1
            else:
                flush()
                cur_start, cur_end, cur_lines, cur_chars, cur_n = int(ps), int(pe), [seg], len(seg), 1
    flush()
    return blocks


def _build_anchors(
    issue_text: str,
    page_segments: list[dict[str, Any]],
    *,
    issue_id: str,
    max_anchors: int,
    max_block_chars: int,
    max_lines_per_block: int,
) -> list[dict[str, Any]]:
    spans = _iter_line_spans(issue_text)

    def page_for_mid(mid: int) -> str:
        for seg in page_segments:
            s = int(seg.get("start_char") or 0)
            e = int(seg.get("end_char") or 0)
            if s <= mid < max(e, s + 1):
                return norm_str(seg.get("page_id"))
        if page_segments:
            return norm_str(page_segments[-1].get("page_id"))
        return ""

    anchors: list[dict[str, Any]] = []
    for i, (s, e, txt) in enumerate(
        _build_anchor_blocks(
            spans,
            max_block_chars=max_block_chars,
            max_lines_per_block=max_lines_per_block,
        ),
        start=1,
    ):
        mid = (s + e) // 2
        anchors.append(
            {
                "anchor_id": f"A{i:05d}",
                "start_char": int(s),
                "end_char": int(e),
                "page_id": page_for_mid(mid),
                "text": norm_str(txt),
            }
        )
    if max_anchors > 0 and len(anchors) > max_anchors:
        raise ValueError(
            f"issue_id={issue_id} produced {len(anchors)} anchors, exceeding hard cap {max_anchors}. "
            "Anchor truncation is disabled; increase --max-anchors-per-issue or set 0 for no cap."
        )
    return anchors


def _build_prompt(issue: dict[str, Any], anchors: list[dict[str, Any]], max_anchor_chars: int) -> str:
    schema = {
        "issue_id": norm_str(issue.get("issue_id")),
        "law_presence": "yes",
        "law_presence_confidence_0_to_1": 0.8,
        "law_presence_evidence_anchor_ids": ["A00001", "A00002"],
        "notes": "",
    }
    lines: list[str] = []
    lines.append("Determine whether this issue contains verbatim zoning-law language.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Decision rule:")
    lines.append("- yes: contains verbatim legal ordinance/amendment/notice language related to zoning.")
    lines.append("- no: no verbatim zoning legal language.")
    lines.append("- uncertain: insufficient evidence.")
    lines.append("- Narrative about zoning policy is NOT verbatim legal language.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Issue metadata: issue_id={norm_str(issue.get('issue_id'))} "
        f"date={norm_str(issue.get('issue_date'))} city={norm_str(issue.get('city_name'))},{norm_str(issue.get('state_abbr')).upper()} "
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


def _coerce(parsed_obj: dict[str, Any], issue_id: str, allowed_anchor_ids: set[str]) -> dict[str, Any]:
    obj = dict(parsed_obj or {})
    raw_ids = obj.get("law_presence_evidence_anchor_ids")
    ids: list[str] = []
    if isinstance(raw_ids, list):
        for x in raw_ids:
            aid = norm_str(x)
            if aid and aid in allowed_anchor_ids and aid not in ids:
                ids.append(aid)
    return {
        "issue_id": norm_str(obj.get("issue_id")) or issue_id,
        "law_presence": enum(obj.get("law_presence"), ALLOWED_PRESENCE, "uncertain"),
        "law_presence_confidence_0_to_1": max(0.0, min(1.0, safe_float(obj.get("law_presence_confidence_0_to_1"), default=0.0))),
        "law_presence_evidence_anchor_ids": ids,
        "notes": norm_str(obj.get("notes")),
    }


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    newspapers = [norm_str(x).lower() for x in norm_str(args.newspapers).split(",") if norm_str(x)]
    if not newspapers:
        raise SystemExit("No newspapers provided.")

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if norm_str(args.output_dir)
        else run_dir / "fullnewspaper_v4" / "law_presence"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    issues = _load_issue_pool(run_dir, newspapers)
    if int(args.limit_issues) > 0:
        issues = issues[: int(args.limit_issues)]
    if not issues:
        raise SystemExit("No issues loaded.")

    req_rows: list[tuple[str, str]] = []
    meta: dict[str, dict[str, Any]] = {}
    for issue in issues:
        iid = norm_str(issue.get("issue_id"))
        slug = norm_str(issue.get("pub_slug")).lower()
        anchors = _build_anchors(
            norm_str(issue.get("text")),
            issue.get("pages") if isinstance(issue.get("pages"), list) else [],
            issue_id=iid,
            max_anchors=int(args.max_anchors_per_issue),
            max_block_chars=int(args.max_block_chars),
            max_lines_per_block=int(args.max_lines_per_block),
        )
        cid = f"law_presence_v4::{slug}::{iid}"
        req_rows.append((cid, _build_prompt(issue, anchors, max_anchor_chars=int(args.max_anchor_chars))))
        meta[cid] = {"issue": issue, "anchors": anchors}

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

    out_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for cid, _ in req_rows:
        issue = meta[cid]["issue"]
        anchors = meta[cid]["anchors"]
        iid = norm_str(issue.get("issue_id"))
        allowed_ids = {norm_str(a.get("anchor_id")) for a in anchors if norm_str(a.get("anchor_id"))}
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
        r = _coerce(parsed if isinstance(parsed, dict) else {}, iid, allowed_ids)
        row = {
            "newspaper_slug": norm_str(issue.get("pub_slug")).lower(),
            "city_key": norm_str(issue.get("city_key")),
            "issue_id": iid,
            "issue_date": norm_str(issue.get("issue_date")),
            "law_presence": norm_str(r.get("law_presence")),
            "law_presence_confidence_0_to_1": safe_float(r.get("law_presence_confidence_0_to_1"), default=0.0),
            "law_presence_evidence_anchor_ids": r.get("law_presence_evidence_anchor_ids") if isinstance(r.get("law_presence_evidence_anchor_ids"), list) else [],
            "notes": norm_str(r.get("notes")),
            "anchor_count": int(len(anchors)),
            "parse_ok": parse_ok,
            "had_error": had_error,
        }
        out_rows.append(row)
        for aid in row["law_presence_evidence_anchor_ids"]:
            flat_rows.append(
                {
                    "newspaper_slug": row["newspaper_slug"],
                    "city_key": row["city_key"],
                    "issue_id": row["issue_id"],
                    "issue_date": row["issue_date"],
                    "anchor_id": norm_str(aid),
                }
            )

    out_rows = sorted(out_rows, key=lambda r: (norm_str(r.get("newspaper_slug")), norm_str(r.get("issue_date")), norm_str(r.get("issue_id"))))
    write_jsonl(out_dir / "law_presence_v4.jsonl", out_rows)
    write_csv(out_dir / "law_presence_v4.csv", pd.DataFrame.from_records(out_rows))
    write_csv(out_dir / "law_presence_evidence_anchors_v4.csv", pd.DataFrame.from_records(flat_rows))

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "newspapers": newspapers,
        "model": str(args.model),
        "issue_count": int(len(out_rows)),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eprint(f"Done. law presence output: {out_dir}")


if __name__ == "__main__":
    main()
