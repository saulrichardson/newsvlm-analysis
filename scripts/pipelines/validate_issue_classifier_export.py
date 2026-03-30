#!/usr/bin/env python3
"""
Validate exported ISSUE-level classifier batch request shards.

This is a "pre-flight" sanity check before submitting large Batch jobs. It verifies:
  - mapping_shardNNN.jsonl line counts match provider request line counts
  - IDs match: mapping.custom_id == openai.custom_id == gemini.key
  - (optional) exported prompts include *all pages* for an issue:
      - count of "=== PAGE START ===" equals len(mapping.source_pages)
      - each source page_id appears in the prompt as "page_id=<...>"

It can also write a durable issue "whitelist" of what will be submitted.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parse_shard_index(filename: str, *, prefix: str, suffix: str) -> int:
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(f"Unexpected filename: {filename}")
    token = filename[len(prefix) : -len(suffix)]
    if not token.isdigit():
        raise ValueError(f"Unexpected shard token in filename: {filename}")
    return int(token)


def _line_count(path: Path) -> int:
    out = subprocess.check_output(["wc", "-l", str(path)], text=True).strip()
    return int(out.split()[0])


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _extract_openai_prompt_text(req_obj: dict[str, Any]) -> str:
    body = req_obj.get("body")
    if not isinstance(body, dict):
        return ""
    inp = body.get("input")
    if not isinstance(inp, list) or not inp:
        return ""
    msg0 = inp[0]
    if not isinstance(msg0, dict):
        return ""
    content = msg0.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    for c in content:
        if isinstance(c, dict) and c.get("type") == "input_text":
            t = c.get("text")
            if isinstance(t, str):
                return t
    return ""


def _extract_gemini_prompt_text(req_obj: dict[str, Any]) -> str:
    req = req_obj.get("request")
    if not isinstance(req, dict):
        return ""
    contents = req.get("contents")
    if not isinstance(contents, list):
        return ""
    # Look for the first part with a "text" field.
    for c in contents:
        if not isinstance(c, dict):
            continue
        parts = c.get("parts")
        if not isinstance(parts, list):
            continue
        for p in parts:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                return str(p.get("text") or "")
    return ""


@dataclass(frozen=True)
class _MappingRow:
    shard: int
    issue_id: str
    custom_id: str
    pages: list[str]


def _load_mapping(mapping_path: Path) -> list[_MappingRow]:
    shard = _parse_shard_index(mapping_path.name, prefix="mapping_shard", suffix=".jsonl")
    rows: list[_MappingRow] = []
    for obj in _iter_jsonl(mapping_path):
        cid = obj.get("custom_id") or obj.get("id") or obj.get("issue_id")
        if not isinstance(cid, str) or not cid.strip():
            raise SystemExit(f"Bad mapping row in {mapping_path}: missing custom_id")
        issue_id = obj.get("issue_id") if isinstance(obj.get("issue_id"), str) else cid
        pages: list[str] = []
        sp = obj.get("source_pages")
        if isinstance(sp, list):
            for p in sp:
                if isinstance(p, dict) and isinstance(p.get("page_id"), str):
                    pages.append(p["page_id"])
        rows.append(_MappingRow(shard=shard, issue_id=str(issue_id), custom_id=cid.strip(), pages=pages))
    return rows


def _validate_prompt_pages(*, prompt: str, expected_pages: list[str], issue_id: str, provider: str) -> str | None:
    if not prompt.strip():
        return f"{provider}: empty prompt for issue_id={issue_id}"
    if f"ISSUE_ID: {issue_id}" not in prompt:
        return f"{provider}: missing ISSUE_ID marker for issue_id={issue_id}"
    # Check count of PAGE START markers.
    expected_n = len(expected_pages)
    got_n = len(re.findall(r"^=== PAGE START ===$", prompt, flags=re.M))
    if expected_n and got_n != expected_n:
        return f"{provider}: page marker count mismatch issue_id={issue_id} expected={expected_n} got={got_n}"
    # Check each page id appears.
    for pid in expected_pages:
        if f"page_id={pid}" not in prompt:
            return f"{provider}: missing page_id={pid} in prompt issue_id={issue_id}"
    return None


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate exported issue classifier requests before submission.")
    ap.add_argument("--request-dir", required=True, help="Directory containing requests/*.jsonl shard files.")
    ap.add_argument(
        "--provider",
        choices=["openai", "gemini", "both"],
        default="both",
        help="Which provider request files to validate (must exist).",
    )
    ap.add_argument(
        "--check-pages",
        action="store_true",
        help="Do deep validation that prompts include all pages for each checked issue (expensive).",
    )
    ap.add_argument(
        "--max-check-issues",
        type=int,
        default=25,
        help="Limit deep page checks to first N issues (0 = check all).",
    )
    ap.add_argument(
        "--write-whitelist",
        action="store_true",
        help="Write requests/issue_whitelist.txt with one issue_id per line (sorted).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser().resolve()
    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")

    want_openai = str(args.provider) in {"openai", "both"}
    want_gemini = str(args.provider) in {"gemini", "both"}

    mapping_paths = sorted(request_dir.glob("mapping_shard*.jsonl"))
    if not mapping_paths:
        raise SystemExit(f"No mapping_shard*.jsonl found in {request_dir}")

    # Load mapping rows across shards.
    mapping_rows: list[_MappingRow] = []
    mapping_by_shard: dict[int, list[_MappingRow]] = {}
    for mp in mapping_paths:
        rows = _load_mapping(mp)
        mapping_rows.extend(rows)
        mapping_by_shard[_parse_shard_index(mp.name, prefix="mapping_shard", suffix=".jsonl")] = rows

    if args.write_whitelist:
        wl_path = request_dir / "issue_whitelist.txt"
        wl_path.write_text("\n".join(sorted({r.issue_id for r in mapping_rows})) + "\n", encoding="utf-8")
        _eprint(f"wrote {wl_path} issues={len({r.issue_id for r in mapping_rows})}")

    # Per-shard count checks.
    problems: list[str] = []
    for shard, rows in sorted(mapping_by_shard.items()):
        mapping_count = len(rows)
        if want_openai:
            op = request_dir / f"openai_requests_shard{shard:03d}.jsonl"
            if not op.is_file():
                problems.append(f"missing openai shard file: {op}")
            else:
                openai_count = _line_count(op)
                if openai_count != mapping_count:
                    problems.append(f"count mismatch shard{shard:03d}: mapping={mapping_count} openai={openai_count}")
        if want_gemini:
            gp = request_dir / f"gemini_requests_shard{shard:03d}.jsonl"
            if not gp.is_file():
                problems.append(f"missing gemini shard file: {gp}")
            else:
                gemini_count = _line_count(gp)
                if gemini_count != mapping_count:
                    problems.append(f"count mismatch shard{shard:03d}: mapping={mapping_count} gemini={gemini_count}")

    # ID set checks + optional deep prompt checks.
    checked_issues = 0
    max_check = int(args.max_check_issues)
    for shard, rows in sorted(mapping_by_shard.items()):
        expected = [r.custom_id for r in rows]
        expected_set = set(expected)

        if want_openai:
            op = request_dir / f"openai_requests_shard{shard:03d}.jsonl"
            if op.is_file():
                # For large runs, prompts are huge (they contain the full issue transcript).
                # Never load all prompts into memory. Stream requests and only validate a
                # small prefix of issues (controlled by --max-check-issues).
                check_map: dict[str, tuple[str, list[str]]] = {}
                if args.check_pages:
                    if max_check == 0:
                        check_rows = rows
                    else:
                        remaining = max(0, max_check - checked_issues)
                        check_rows = rows[:remaining]
                    check_map = {r.custom_id: (r.issue_id, r.pages) for r in check_rows}

                got_set: set[str] = set()
                for obj in _iter_jsonl(op):
                    cid = obj.get("custom_id")
                    if not isinstance(cid, str) or not cid.strip():
                        continue
                    cid = cid.strip()
                    got_set.add(cid)
                    if args.check_pages and check_map and cid in check_map:
                        issue_id, pages = check_map[cid]
                        prompt = _extract_openai_prompt_text(obj)
                        err = _validate_prompt_pages(prompt=prompt, expected_pages=pages, issue_id=issue_id, provider="openai")
                        if err:
                            problems.append(err)
                        checked_issues += 1
                        del check_map[cid]
                if got_set != expected_set:
                    problems.append(f"custom_id mismatch shard{shard:03d}: mapping_only={len(expected_set-got_set)} openai_only={len(got_set-expected_set)}")

        if want_gemini:
            gp = request_dir / f"gemini_requests_shard{shard:03d}.jsonl"
            if gp.is_file():
                check_map: dict[str, tuple[str, list[str]]] = {}
                if args.check_pages:
                    if max_check == 0:
                        check_rows = rows
                    else:
                        remaining = max(0, max_check - checked_issues)
                        check_rows = rows[:remaining]
                    check_map = {r.custom_id: (r.issue_id, r.pages) for r in check_rows}

                got_set: set[str] = set()
                for obj in _iter_jsonl(gp):
                    key = obj.get("key")
                    if not isinstance(key, str) or not key.strip():
                        continue
                    key = key.strip()
                    got_set.add(key)
                    if args.check_pages and check_map and key in check_map:
                        issue_id, pages = check_map[key]
                        prompt = _extract_gemini_prompt_text(obj)
                        err = _validate_prompt_pages(prompt=prompt, expected_pages=pages, issue_id=issue_id, provider="gemini")
                        if err:
                            problems.append(err)
                        checked_issues += 1
                        del check_map[key]
                if got_set != expected_set:
                    problems.append(f"key mismatch shard{shard:03d}: mapping_only={len(expected_set-got_set)} gemini_only={len(got_set-expected_set)}")

        if max_check and checked_issues >= max_check:
            break

    if problems:
        _eprint("validation_failed=true")
        for p in problems[:50]:
            _eprint("problem: " + p)
        if len(problems) > 50:
            _eprint(f"... plus {len(problems)-50} more problems")
        raise SystemExit(f"Validation failed with {len(problems)} problems.")

    _eprint(
        json.dumps(
            {
                "validation_ok": True,
                "request_dir": str(request_dir),
                "issues_total": len({r.issue_id for r in mapping_rows}),
                "checked_issues_deep": checked_issues if args.check_pages else 0,
                "providers": str(args.provider),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
