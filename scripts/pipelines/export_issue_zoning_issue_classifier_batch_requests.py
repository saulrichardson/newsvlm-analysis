#!/usr/bin/env python3
"""
Export batch request shards for *ISSUE-level* classification.

This is the repeatable packaging step that sits between:
  - transcripts (Paddle VL1.5 Markdown and/or legacy VLM transcripts)
and:
  - execution (OpenAI Batch API and/or Gemini Batch API; or a live provider via vendor/agent-gateway).

Key properties (project policy):
  - No clipping/truncation of page or issue text. If caps are exceeded, the run FAILS.
  - Issue-level prompts include ALL pages for the issue (when --unique-png-root is provided).
  - We write provenance + mapping artifacts so results can be rehydrated/audited later.

Inputs (choose one mode):
  Mode A: Issue transcript files (smoke tests / ad-hoc)
    --issue-txt-dir DIR
      DIR contains <issue_id>.txt files where filename stem is the issue_id.

  Mode B: Page transcripts (full production)
    --unique-png-root DIR
      DIR contains unique_png/<page_id>.png; used as the canonical page list for completeness.
    --paddle-vl15-jsonl PATH(.gz)  (optional, repeatable)
      JSONL or JSONL.GZ from newspaper-parsing/scripts/export_vl15_markdown_jsonl.py
    --vlm-page-roots DIR1,DIR2... (optional)
      Directories containing per-page <page_id>.vlm.json files (box transcripts).
    --openai-box-results-jsonl PATH (optional, repeatable)
      OpenAI batch/gateway results JSONL(s) where each custom_id is <page_id>:<box_id> and
      output_text parses to {"status":"ok|unreadable|error", "transcript":"..."}.
    --gemini-box-results-jsonl PATH (optional, repeatable)
      Gemini batch results JSONL(s) where each key is <page_id>:<box_id> and
      response text parses to {"status":"ok|unreadable|error", "transcript":"..."} (sometimes fenced).

Outputs (under --output-dir):
  - inputs/ (repro metadata)
  - requests/openai_requests_shardNNN.jsonl
  - requests/gemini_requests_shardNNN.jsonl (optional)
  - requests/mapping_shardNNN.jsonl  (issue->page inventory + hashes + source provenance)
  - prompt_used.txt
  - skipped_issues.csv / missing_pages.csv (if applicable)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import gzip
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Tuple


MONTH_TO_NUM = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


MissingIssuePolicy = Literal["require", "skip", "ignore"]


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _open_text_or_gzip(path: Path) -> Iterable[str]:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


def _repair_truncated_json(candidate: str) -> str:
    """
    Best-effort fix for truncated JSON: closes open quotes/brackets/braces.

    This is intentionally conservative; it's used only as a fallback when models
    return cut-off JSON or wrap it with extra text.
    """
    s = _norm(candidate)
    if not s:
        return s
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    fixed = s
    if in_str:
        fixed += '"'
    if stack:
        fixed += "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    return fixed


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Parse a JSON object out of model text. Handles common wrappers:
      - markdown fences ```json ... ```
      - leading/trailing commentary
      - truncated JSON (best-effort repair)

    Returns the *largest* parsed JSON dict candidate, or None.
    """
    s = _norm(text)
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)

    candidates: list[dict[str, Any]] = []

    def _add(obj: Any) -> None:
        if isinstance(obj, dict):
            candidates.append(obj)

    try:
        _add(json.loads(s))
    except Exception:
        pass

    first = s.find("{")
    if first != -1:
        candidate = s[first:]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        repaired = _repair_truncated_json(candidate)
        try:
            _add(json.loads(repaired))
        except Exception:
            pass

    dec = json.JSONDecoder()
    start = first
    while start != -1:
        try:
            obj, _ = dec.raw_decode(s[start:])
            _add(obj)
        except Exception:
            pass
        start = s.find("{", start + 1)

    if not candidates:
        return None
    # Prefer the largest dict by serialized length (usually the "real" payload).
    return sorted(candidates, key=lambda x: len(json.dumps(x, ensure_ascii=False)), reverse=True)[0]


def _parse_page_id(page_id: str) -> dict[str, Any] | None:
    """
    Parse:
      <newspaper_slug>-<mon>-<dd>-<yyyy>-p-<num>
    Example:
      abilene-reporter-news-nov-28-1945-p-22
    """
    s = _norm(page_id).lower()
    m = re.match(r"^(?P<slug>.+)-(?P<mon>[a-z]{3})-(?P<day>\d{2})-(?P<year>\d{4})-p-(?P<page>\d+)$", s)
    if not m:
        return None
    mon = _norm(m.group("mon"))
    mm = MONTH_TO_NUM.get(mon)
    if not mm:
        return None
    issue_date = f"{m.group('year')}-{mm}-{m.group('day')}"
    slug = _norm(m.group("slug"))
    return {
        "page_id": s,
        "slug": slug,
        "issue_date": issue_date,
        "issue_id": f"{slug}__{issue_date}",
        "page_num": int(m.group("page")),
    }


def _safe_issue_filename(issue_id: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(issue_id))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "issue"


def _bounded_text_or_fail(text: str, max_chars: int, label: str) -> str:
    s = str(text or "")
    if max_chars > 0 and len(s) > max_chars:
        raise ValueError(
            f"{label} length {len(s)} exceeds hard cap {max_chars}. "
            "Clipping/condensing is disabled; increase cap or reduce input scope."
        )
    return s


def _box_sort_key(box: dict[str, Any]) -> tuple[int, str]:
    bid = _norm(box.get("id"))
    m = re.search(r"(\d+)$", bid)
    if m:
        return (int(m.group(1)), bid)
    return (10**9, bid)


def _extract_clean_text_from_vlm_page(path: Path) -> str:
    """Read a per-page *.vlm.json file and concatenate usable box transcripts."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    boxes = obj.get("boxes")
    if not isinstance(boxes, list):
        return ""
    parts: list[str] = []
    for box in sorted([x for x in boxes if isinstance(x, dict)], key=_box_sort_key):
        status = _norm(box.get("status")).lower()
        txt = _norm(box.get("transcript"))
        if not txt:
            continue
        if status in {"unreadable", "error"}:
            continue
        parts.append(txt)
    text = "\n\n".join(parts).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _extract_openai_output_text(body: dict[str, Any]) -> str:
    out = body.get("output")
    if not isinstance(out, list):
        return ""
    parts: list[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "output_text":
                parts.append(str(c.get("text") or ""))
    return "".join(parts)


@dataclass(frozen=True)
class PageText:
    page_id: str
    issue_id: str
    issue_date: str
    newspaper_slug: str
    page_num: int
    text: str
    text_sha256: str
    text_chars: int
    source: str
    source_path: str


def _iter_unique_png_pages(unique_png_root: Path) -> Iterator[tuple[str, Path]]:
    if not unique_png_root.is_dir():
        raise SystemExit(f"--unique-png-root is not a directory: {unique_png_root}")
    # Torch's `unique_png` tree can include nested subdirectories (e.g. backfills like
    # `unique_png/data/downloads/...`). We treat the *filename stem* as the canonical
    # page_id and dedupe if the same stem appears in multiple directories.
    #
    # NOTE: The exporter only uses page_id for grouping + completeness; it does not
    # read image bytes. So for duplicates, "first path wins" is fine.
    seen: set[str] = set()
    dup_count = 0
    for dirpath, _dirnames, filenames in os.walk(unique_png_root):
        for name in filenames:
            if not name.endswith(".png"):
                continue
            page_id = name[: -len(".png")].strip().lower()
            if not page_id:
                continue
            if page_id in seen:
                dup_count += 1
                continue
            seen.add(page_id)
            yield page_id, Path(dirpath) / name
    if dup_count:
        _eprint(f"warn: unique_png_root duplicate stems ignored: {dup_count}")


def _load_paddle_vl15_jsonl(paths: list[Path]) -> dict[str, PageText]:
    """
    Load Paddle VL1.5 Markdown export JSONL (optionally .gz) into page_id -> PageText.

    Expected keys (from newspaper-parsing/scripts/export_vl15_markdown_jsonl.py):
      slug (page_id), md_path, status, text_md
    """
    out: dict[str, PageText] = {}
    for p in paths:
        if not p.is_file():
            _eprint(f"warn: paddle jsonl missing: {p}")
            continue
        for raw in _open_text_or_gzip(p):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            page_id = _norm(obj.get("slug") or obj.get("page_id")).lower()
            if not page_id:
                continue
            parsed = _parse_page_id(page_id)
            if not parsed:
                continue
            text = _norm(obj.get("text_md"))
            # Treat missing markdown as empty (we'll fill from other sources).
            sha = _sha256(text) if text else ""
            out[page_id] = PageText(
                page_id=page_id,
                issue_id=_norm(parsed.get("issue_id")),
                issue_date=_norm(parsed.get("issue_date")),
                newspaper_slug=_norm(parsed.get("slug")).lower(),
                page_num=int(parsed.get("page_num") or 0),
                text=text,
                text_sha256=sha,
                text_chars=len(text),
                source="paddle_vl15_md",
                source_path=_norm(obj.get("md_path") or ""),
            )
    return out


def _iter_paddle_run_dirs(run_root: Path, run_prefix: str) -> list[Path]:
    return sorted([p for p in run_root.glob(run_prefix + "*") if p.is_dir()])


def _iter_layout_tsv_slugs(layout_tsv: Path) -> Iterator[str]:
    """
    Yield page_id slugs from a Paddle layout TSV (first column is image path).

    This mirrors newspaper-parsing/scripts/export_vl15_markdown_jsonl.py, but we only
    need slugs here (we read Markdown from per-page .md files lazily).
    """
    with layout_tsv.open("r", encoding="utf-8", newline="") as f:
        rr = csv.reader(f, delimiter="\t")
        next(rr, None)  # header
        for row in rr:
            if not row:
                continue
            img = _norm(row[0])
            if not img:
                continue
            slug = Path(os.path.expandvars(img)).stem.strip().lower()
            if slug:
                yield slug


def _build_paddle_md_ok_index(
    *,
    run_root: Path,
    run_prefix: str,
    layout_variant: str,
    vl15_variant: str,
) -> dict[str, Path]:
    """
    Build a page_id -> md_path index for pages with non-empty Paddle VL1.5 Markdown.

    This avoids loading all page text into memory: we only keep paths and read the
    Markdown at issue-export time.
    """
    if not run_root.is_dir():
        return {}
    runs = _iter_paddle_run_dirs(run_root, run_prefix)
    if not runs:
        return {}

    ok: dict[str, Path] = {}
    missing_layout_tsv = 0
    total_pages = 0
    for run_dir in runs:
        layout_tsv = run_dir / "reports" / f"paddle_layout_{layout_variant}.tsv"
        if not layout_tsv.exists() or layout_tsv.stat().st_size == 0:
            missing_layout_tsv += 1
            continue
        pvl_root = run_dir / "outputs" / "sources" / "paddle_vl15" / vl15_variant
        for slug in _iter_layout_tsv_slugs(layout_tsv):
            total_pages += 1
            if slug in ok:
                continue
            md_path = pvl_root / slug / "raw" / f"{slug}.md"
            try:
                if md_path.exists() and md_path.stat().st_size > 0:
                    ok[slug] = md_path
            except Exception:
                continue

    _eprint(
        f"paddle_md_ok_index runs={len(runs)} total_pages={total_pages} ok_md={len(ok)} missing_layout_tsv={missing_layout_tsv}"
    )
    return ok


def _load_vlm_page_roots(
    roots: list[Path], *, source_label: str, wanted_page_ids: set[str] | None = None
) -> dict[str, PageText]:
    """Load per-page <page_id>.vlm.json from one or more roots (recursive)."""
    out: dict[str, PageText] = {}
    for root in roots:
        if not root.is_dir():
            _eprint(f"warn: vlm root missing: {root}")
            continue
        for path in root.rglob("*.vlm.json"):
            page_id = path.name[: -len(".vlm.json")].strip().lower()
            if wanted_page_ids is not None and page_id not in wanted_page_ids:
                continue
            parsed = _parse_page_id(page_id)
            if not parsed:
                continue
            text = _extract_clean_text_from_vlm_page(path)
            if not text.strip():
                continue
            out[page_id] = PageText(
                page_id=page_id,
                issue_id=_norm(parsed.get("issue_id")),
                issue_date=_norm(parsed.get("issue_date")),
                newspaper_slug=_norm(parsed.get("slug")).lower(),
                page_num=int(parsed.get("page_num") or 0),
                text=text,
                text_sha256=_sha256(text),
                text_chars=len(text),
                source=source_label,
                source_path=str(path),
            )
    return out


def _load_openai_box_results_jsonl(paths: list[Path], *, wanted_page_ids: set[str] | None = None) -> dict[str, PageText]:
    """
    Load OpenAI per-box results JSONL and aggregate to page_id -> PageText.

    Each result line expected to have:
      - custom_id: <page_id>:<box_id>
      - response.body.output[].content[].text = JSON string containing:
          {"status": "ok|unreadable|...", "transcript": "..."}

    We keep only status==ok transcripts.
    """
    # We process one JSONL at a time to avoid holding the entire box universe in memory.
    out: dict[str, PageText] = {}
    for p in paths:
        if not p.is_file():
            _eprint(f"warn: openai box results missing: {p}")
            continue
        # page_id -> box_id -> (box_sort_key, transcript)
        #
        # Dedupe by (page_id, box_id) because shard JSONLs can contain retries/duplicates.
        # If duplicates exist, keep the longest transcript (often the least truncated).
        by_page: dict[str, dict[str, tuple[tuple[int, str], str]]] = {}
        for raw in _open_text_or_gzip(p):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = _norm(obj.get("custom_id"))
            if not cid:
                continue
            if ":" in cid:
                page_id, box_id = cid.rsplit(":", 1)
            else:
                page_id, box_id = cid, ""
            page_id = page_id.strip().lower()
            if not page_id:
                continue
            if wanted_page_ids is not None and page_id not in wanted_page_ids:
                continue
            body = ((obj.get("response") or {}).get("body") or {})
            if not isinstance(body, dict):
                continue
            out_text = _extract_openai_output_text(body)
            if not out_text.strip():
                continue
            parsed = _parse_json_from_text(out_text)
            if parsed is None:
                continue
            if not isinstance(parsed, dict):
                continue
            if _norm(parsed.get("status")).lower() != "ok":
                continue
            tx = _norm(parsed.get("transcript"))
            if not tx:
                continue
            box_id_norm = _norm(box_id)
            sort_key = _box_sort_key({"id": box_id_norm})
            existing = by_page.setdefault(page_id, {}).get(box_id_norm)
            if existing is None or len(tx) > len(existing[1]):
                by_page[page_id][box_id_norm] = (sort_key, tx)
        for page_id, items in by_page.items():
            parsed = _parse_page_id(page_id)
            if not parsed:
                continue
            items_sorted = sorted(items.values(), key=lambda x: x[0])
            text = "\n\n".join(t for _, t in items_sorted).strip()
            if not text:
                continue
            pt = PageText(
                page_id=page_id,
                issue_id=_norm(parsed.get("issue_id")),
                issue_date=_norm(parsed.get("issue_date")),
                newspaper_slug=_norm(parsed.get("slug")).lower(),
                page_num=int(parsed.get("page_num") or 0),
                text=text,
                text_sha256=_sha256(text),
                text_chars=len(text),
                source="openai_box_results",
                source_path=str(p),
            )
            existing_pt = out.get(page_id)
            if existing_pt is None or pt.text_chars > existing_pt.text_chars:
                out[page_id] = pt
    return out


def _extract_gemini_model_text(result_obj: dict[str, Any]) -> str:
    resp = result_obj.get("response") or {}
    try:
        return str(resp["candidates"][0]["content"]["parts"][0]["text"] or "")
    except Exception:
        return ""


def _load_gemini_box_results_jsonl(paths: list[Path], *, wanted_page_ids: set[str] | None = None) -> dict[str, PageText]:
    """
    Load Gemini per-box batch output JSONL and aggregate to page_id -> PageText.

    Expected shape (see old-ocr/scripts/rehydrate_gemini_batch_results.py):
      - key: "<page_id>:<box_id>"
      - response.candidates[0].content.parts[0].text is JSON (sometimes fenced) with:
          {"status": "ok|unreadable|...", "transcript": "..."}
    """
    # Process one JSONL at a time to keep memory bounded.
    out: dict[str, PageText] = {}
    for p in paths:
        if not p.is_file():
            _eprint(f"warn: gemini box results missing: {p}")
            continue
        # page_id -> box_id -> (box_sort_key, transcript)
        # Dedupe by (page_id, box_id) for retry/append safety; keep longest transcript.
        by_page: dict[str, dict[str, tuple[tuple[int, str], str]]] = {}
        for raw in _open_text_or_gzip(p):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = _norm(obj.get("key"))
            if not key:
                continue
            if ":" in key:
                page_id, box_id = key.rsplit(":", 1)
            else:
                page_id, box_id = key, ""
            page_id = page_id.strip().lower()
            if not page_id:
                continue
            if wanted_page_ids is not None and page_id not in wanted_page_ids:
                continue
            model_text = _extract_gemini_model_text(obj)
            if not model_text.strip():
                continue
            parsed = _parse_json_from_text(model_text)
            if parsed is None:
                continue
            if _norm(parsed.get("status")).lower() != "ok":
                continue
            tx = _norm(parsed.get("transcript"))
            if not tx:
                continue
            box_id_norm = _norm(box_id)
            sort_key = _box_sort_key({"id": box_id_norm})
            existing = by_page.setdefault(page_id, {}).get(box_id_norm)
            if existing is None or len(tx) > len(existing[1]):
                by_page[page_id][box_id_norm] = (sort_key, tx)
        for page_id, items in by_page.items():
            parsed_pid = _parse_page_id(page_id)
            if not parsed_pid:
                continue
            items_sorted = sorted(items.values(), key=lambda x: x[0])
            text = "\n\n".join(t for _, t in items_sorted).strip()
            if not text:
                continue
            pt = PageText(
                page_id=page_id,
                issue_id=_norm(parsed_pid.get("issue_id")),
                issue_date=_norm(parsed_pid.get("issue_date")),
                newspaper_slug=_norm(parsed_pid.get("slug")).lower(),
                page_num=int(parsed_pid.get("page_num") or 0),
                text=text,
                text_sha256=_sha256(text),
                text_chars=len(text),
                source="gemini_box_results",
                source_path=str(p),
            )
            existing_pt = out.get(page_id)
            if existing_pt is None or pt.text_chars > existing_pt.text_chars:
                out[page_id] = pt
    return out


def _iter_issue_txt_dir(dir_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (issue_id, transcript_text) from <issue_id>.txt files in dir_path."""
    if not dir_path.is_dir():
        raise SystemExit(f"--issue-txt-dir is not a directory: {dir_path}")
    for p in sorted(dir_path.glob("*.txt")):
        issue_id = p.stem.strip()
        if not issue_id:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        yield issue_id, text


def _parse_issues_spec(raw: str) -> set[str]:
    spec = _norm(raw)
    if not spec:
        return set()
    if spec.startswith("@"):
        p = Path(spec[1:]).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Issues list file not found: {p}")
        vals: list[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = _norm(line)
            if not s or s.startswith("#"):
                continue
            vals.append(s)
        return set(vals)
    return {x.strip() for x in spec.split(",") if x.strip()}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _build_issue_transcript(pages: list[PageText], *, max_issue_chars: int, issue_id: str) -> str:
    blocks: list[str] = []
    for p in pages:
        blocks.append(
            "\n".join(
                [
                    "=== PAGE START ===",
                    f"page_id={p.page_id}",
                    f"page_num={p.page_num}",
                    f"text_source={p.source}",
                    "page_text:",
                    p.text,
                    "=== PAGE END ===",
                ]
            )
        )
    s = "\n\n".join(blocks).strip()
    return _bounded_text_or_fail(s, max_issue_chars, label=f"issue_id={issue_id} issue_transcript")


def _make_openai_request(
    custom_id: str,
    prompt: str,
    *,
    model: str,
    reasoning_effort: str = "",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "stream": False,
    }
    effort = _norm(reasoning_effort).lower()
    if effort:
        body["reasoning"] = {"effort": effort}
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def _make_gemini_request(
    key: str,
    prompt: str,
    *,
    response_mime_type: str = "",
    thinking_budget: int | None = None,
    thinking_level: str = "",
) -> dict[str, Any]:
    # Google GenAI Batch format (see old-ocr/scripts/submit_batch_shards.py):
    #   {"key": "...", "request": {"contents": [{"role": "user", "parts": [{"text": "..."}]}]}}
    req: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    generation_config: dict[str, Any] = {}
    mime = _norm(response_mime_type)
    if mime:
        generation_config["response_mime_type"] = mime
    thinking_cfg: dict[str, Any] = {}
    level = _norm(thinking_level).lower()
    if level:
        thinking_cfg["thinking_level"] = level
    if thinking_budget is not None:
        thinking_cfg["thinking_budget"] = int(thinking_budget)
    if thinking_cfg:
        generation_config["thinking_config"] = thinking_cfg
    if generation_config:
        req["generation_config"] = generation_config
    return {
        "key": key,
        "request": req,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export issue-level zoning classifier request shards.")
    ap.add_argument("--output-dir", required=True, help="Directory to write run artifacts into.")

    ap.add_argument(
        "--prompt-path",
        required=True,
        help="Issue classifier prompt text file (static instructions + schema). Issue transcript is appended after it.",
    )
    ap.add_argument(
        "--provider",
        choices=["openai", "gemini", "both"],
        default="openai",
        help="Which provider request JSONLs to emit under requests/.",
    )
    ap.add_argument("--openai-model", default="gpt-5-mini", help="OpenAI model name to embed in OpenAI request bodies.")
    ap.add_argument(
        "--openai-reasoning-effort",
        default="medium",
        help="OpenAI Responses API reasoning.effort to embed in request bodies (empty disables).",
    )
    ap.add_argument(
        "--gemini-response-mime-type",
        default="application/json",
        help="Gemini request config.response_mime_type to embed in batch requests (empty disables).",
    )
    ap.add_argument(
        "--gemini-thinking-budget",
        type=int,
        default=None,
        help="Gemini 2.5 thinking_budget to embed in request config.thinking_config (e.g. 32768).",
    )
    ap.add_argument(
        "--gemini-thinking-level",
        default="",
        help="Gemini 3 thinking_level to embed in request config.thinking_config (e.g. high).",
    )

    # Mode A: issue transcripts
    ap.add_argument(
        "--issue-txt-dir",
        default="",
        help="Directory containing <issue_id>.txt issue transcripts. If set, page-aggregation inputs are ignored.",
    )

    # Mode B: full production from pages
    ap.add_argument("--unique-png-root", default="", help="Directory containing unique_png/<page_id>.png (canonical pages).")
    ap.add_argument(
        "--paddle-vl15-jsonl",
        default="",
        help=(
            "Comma-separated Paddle VL1.5 markdown export JSONL(.gz) paths (from newspaper-parsing). "
            "For large Torch runs this can be very large; prefer --paddle-run-root to read per-page .md lazily."
        ),
    )
    ap.add_argument(
        "--paddle-run-root",
        default="",
        help=(
            "Root directory containing Paddle VL1.5 shard run dirs (Torch default: /scratch/$USER/paddleocr_vl15/runs). "
            "If provided (or if the default exists), the exporter reads per-page Markdown files lazily instead of loading "
            "all text from --paddle-vl15-jsonl into memory."
        ),
    )
    ap.add_argument(
        "--paddle-run-prefix",
        default="layout_bagging_unique_png_sharded_20260220_200103_manifest_",
        help="Shard run directory prefix under --paddle-run-root.",
    )
    ap.add_argument(
        "--paddle-layout-variant",
        default="pld_v3_thr03",
        help="Layout variant name for artifacts/reports/paddle_layout_<variant>.tsv under each run dir.",
    )
    ap.add_argument(
        "--paddle-vl15-variant",
        default="pvl15_docparser_v15",
        help="VL1.5 variant directory under outputs/sources/paddle_vl15/ containing per-page Markdown.",
    )
    ap.add_argument(
        "--vlm-page-roots",
        default="",
        help="Comma-separated directories to search recursively for per-page <page_id>.vlm.json files.",
    )
    ap.add_argument(
        "--openai-box-results-jsonl",
        default="",
        help="Comma-separated OpenAI per-box openai_results_shard*.jsonl (optionally .gz) to aggregate into per-page text.",
    )
    ap.add_argument(
        "--gemini-box-results-jsonl",
        default="",
        help="Comma-separated Gemini per-box gemini_results_shard*.jsonl (optionally .gz) to aggregate into per-page text.",
    )
    ap.add_argument(
        "--vlm-source-priority",
        default="paddle_vl15_md,vlm_page,gemini_box_results,openai_box_results",
        help=(
            "Comma-separated source priority list. Known labels: paddle_vl15_md, vlm_page, gemini_box_results, openai_box_results. "
            "First available wins per page."
        ),
    )

    ap.add_argument(
        "--missing-page-policy",
        choices=["require", "skip", "ignore"],
        default="skip",
        help="How to handle issues that still have pages with missing transcript text after all fallbacks.",
    )
    ap.add_argument(
        "--max-issue-chars",
        type=int,
        default=0,
        help="Hard cap on concatenated issue transcript chars (0 disables). Exceeding cap FAILS (no clipping).",
    )
    ap.add_argument("--issues-per-shard", type=int, default=5000, help="Maximum issue requests per output shard file.")
    ap.add_argument(
        "--max-bytes-per-shard",
        type=int,
        default=0,
        help=(
            "Max bytes per provider request shard file (OpenAI/Gemini). "
            "0 disables byte-based splitting. Recommended for OpenAI: ~180_000_000."
        ),
    )
    ap.add_argument("--max-issues", type=int, default=0, help="Optional cap on exported issues (0 = all).")
    ap.add_argument(
        "--issues",
        default="",
        help="Optional issue_id allowlist: comma-separated or @file (one issue_id per line).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    req_dir = out_dir / "requests"
    inputs_dir = out_dir / "inputs"
    req_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt_path).expanduser().resolve()
    if not prompt_path.is_file():
        raise SystemExit(f"--prompt-path not found: {prompt_path}")
    prompt_prefix = prompt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not prompt_prefix:
        raise SystemExit(f"--prompt-path is empty: {prompt_path}")
    (out_dir / "prompt_used.txt").write_text(prompt_prefix + "\n", encoding="utf-8")

    issues_filter = _parse_issues_spec(str(args.issues))
    max_issues = int(args.max_issues) if int(args.max_issues) > 0 else None
    max_issue_chars = int(args.max_issue_chars)
    issues_per_shard = int(args.issues_per_shard)
    max_bytes_per_shard = int(args.max_bytes_per_shard)
    if issues_per_shard <= 0:
        raise SystemExit("--issues-per-shard must be > 0")
    if max_bytes_per_shard < 0:
        raise SystemExit("--max-bytes-per-shard must be >= 0")

    want_openai = str(args.provider) in {"openai", "both"}
    want_gemini = str(args.provider) in {"gemini", "both"}
    if not (want_openai or want_gemini):
        raise SystemExit("--provider must be one of: openai, gemini, both")

    # Provenance / config snapshot
    (inputs_dir / "export_config.json").write_text(
        json.dumps(
            {
                "prompt_path": str(prompt_path),
                "provider": str(args.provider),
                "openai_model": str(args.openai_model),
                "issue_txt_dir": str(args.issue_txt_dir),
                "unique_png_root": str(args.unique_png_root),
                "paddle_vl15_jsonl": str(args.paddle_vl15_jsonl),
                "paddle_run_root": str(args.paddle_run_root),
                "paddle_run_prefix": str(args.paddle_run_prefix),
                "paddle_layout_variant": str(args.paddle_layout_variant),
                "paddle_vl15_variant": str(args.paddle_vl15_variant),
                "vlm_page_roots": str(args.vlm_page_roots),
                "openai_box_results_jsonl": str(args.openai_box_results_jsonl),
                "gemini_box_results_jsonl": str(args.gemini_box_results_jsonl),
                "vlm_source_priority": str(args.vlm_source_priority),
                "missing_page_policy": str(args.missing_page_policy),
                "max_issue_chars": max_issue_chars,
                "issues_per_shard": issues_per_shard,
                "max_bytes_per_shard": max_bytes_per_shard,
                "max_issues": max_issues,
                "issues_filter": sorted(issues_filter) if issues_filter else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Mode A: issue transcript directory (smoke tests)
    issue_txt_dir = Path(str(args.issue_txt_dir)).expanduser().resolve() if str(args.issue_txt_dir).strip() else None
    if issue_txt_dir is not None:
        _eprint(f"mode=issue_txt_dir dir={issue_txt_dir}")
        issue_rows: list[tuple[str, str]] = []
        for issue_id, text in _iter_issue_txt_dir(issue_txt_dir):
            if issues_filter and issue_id not in issues_filter:
                continue
            issue_rows.append((issue_id, text))
        issue_rows = sorted(issue_rows, key=lambda x: x[0])
        if max_issues is not None:
            issue_rows = issue_rows[:max_issues]
        if not issue_rows:
            raise SystemExit("No issue transcripts found after filters.")

        req_rows: list[dict[str, Any]] = []
        gemini_rows: list[dict[str, Any]] = []
        map_rows: list[dict[str, Any]] = []
        for issue_id, text in issue_rows:
            _bounded_text_or_fail(text, max_issue_chars, label=f"issue_id={issue_id} issue_txt")
            prompt = prompt_prefix + "\n\n" + f"ISSUE_ID: {issue_id}\n\nISSUE_TRANSCRIPT:\n" + text.strip()
            if want_openai:
                req_rows.append(_make_openai_request(issue_id, prompt, model=str(args.openai_model)))
            if want_gemini:
                gemini_rows.append(_make_gemini_request(issue_id, prompt))
            map_rows.append(
                {
                    "id": issue_id,
                    "custom_id": issue_id,
                    "issue_id": issue_id,
                    "issue_date": (issue_id.split("__", 1)[1] if "__" in issue_id else ""),
                    "slug": (issue_id.split("__", 1)[0] if "__" in issue_id else ""),
                    "source_mode": "issue_txt_dir",
                    "source_path": str(issue_txt_dir / f"{issue_id}.txt"),
                    "issue_chars": len(text),
                    "issue_sha256": _sha256(text),
                    "exported_at": int(dt.datetime.now().timestamp()),
                }
            )

        # Shard write (single shard is fine for smoke tests)
        if want_openai:
            req_path = req_dir / "openai_requests_shard000.jsonl"
            _write_jsonl(req_path, req_rows)
            _eprint(f"wrote_requests {req_path} rows={len(req_rows)}")
        if want_gemini:
            gemini_path = req_dir / "gemini_requests_shard000.jsonl"
            _write_jsonl(gemini_path, gemini_rows)
            _eprint(f"wrote_requests {gemini_path} rows={len(gemini_rows)}")
        map_path = req_dir / "mapping_shard000.jsonl"
        _write_jsonl(map_path, map_rows)
        _eprint(f"wrote_mapping {map_path} rows={len(map_rows)}")
        return

    # Mode B: page aggregation
    unique_png_root = Path(str(args.unique_png_root)).expanduser().resolve() if str(args.unique_png_root).strip() else None
    if unique_png_root is None:
        raise SystemExit("Must provide either --issue-txt-dir (smoke) or --unique-png-root (full page aggregation).")

    def _expand_dirs(raw: str) -> list[Path]:
        """
        Expand a comma-separated list of directory paths, supporting shell-like globs.

        Example:
          --vlm-page-roots '/scratch/.../part_*/vlm_out_gemini'
        """
        out: list[Path] = []
        for tok in str(raw).split(","):
            t = tok.strip()
            if not t:
                continue
            t = os.path.expanduser(t)
            if any(ch in t for ch in "*?[]"):
                matches = sorted(glob.glob(t))
            else:
                matches = [t]
            for m in matches:
                p = Path(m).resolve()
                if p.is_dir():
                    out.append(p)
                else:
                    _eprint(f"warn: expected directory, got: {p}")
        # Dedupe preserving order.
        dedup: list[Path] = []
        seen: set[str] = set()
        for p in out:
            k = str(p)
            if k in seen:
                continue
            seen.add(k)
            dedup.append(p)
        return dedup

    def _expand_jsonl_inputs(raw: str, *, patterns: list[str]) -> list[Path]:
        out: list[Path] = []
        for tok in str(raw).split(","):
            t = tok.strip()
            if not t:
                continue
            t = os.path.expanduser(t)
            if any(ch in t for ch in "*?[]"):
                for m in sorted(glob.glob(t)):
                    pm = Path(m).resolve()
                    if pm.is_dir():
                        for pat in patterns:
                            out.extend(sorted(pm.glob(pat)))
                    else:
                        out.append(pm)
                continue
            p = Path(t).resolve()
            if p.is_dir():
                for pat in patterns:
                    out.extend(sorted(p.glob(pat)))
                continue
            out.append(p)
        # Dedupe while preserving order.
        dedup: list[Path] = []
        seen: set[str] = set()
        for p in out:
            k = str(p)
            if k in seen:
                continue
            seen.add(k)
            dedup.append(p)
        return dedup

    paddle_paths = _expand_jsonl_inputs(str(args.paddle_vl15_jsonl), patterns=["*.jsonl", "*.jsonl.gz"])
    vlm_roots = _expand_dirs(str(args.vlm_page_roots))
    openai_box_paths = _expand_jsonl_inputs(
        str(args.openai_box_results_jsonl),
        patterns=["openai_results_shard*.jsonl", "openai_results_shard*.jsonl.gz", "*.jsonl", "*.jsonl.gz"],
    )
    gemini_box_paths = _expand_jsonl_inputs(
        str(args.gemini_box_results_jsonl),
        patterns=["gemini_results_shard*.jsonl", "gemini_results_shard*.jsonl.gz", "*.jsonl", "*.jsonl.gz"],
    )
    priority = [s.strip() for s in str(args.vlm_source_priority).split(",") if s.strip()]
    if not priority:
        raise SystemExit("--vlm-source-priority must be non-empty")

    _eprint(f"mode=page_aggregation unique_png_root={unique_png_root}")

    # Canonical page list from unique_png.
    page_ids: list[str] = []
    bad_page_ids: list[str] = []
    page_meta: dict[str, dict[str, Any]] = {}
    for pid, _png in _iter_unique_png_pages(unique_png_root):
        parsed = _parse_page_id(pid)
        if not parsed:
            bad_page_ids.append(pid)
            continue
        page_ids.append(pid)
        page_meta[pid] = parsed
    if bad_page_ids:
        (inputs_dir / "skipped_unparseable_page_ids.txt").write_text("\n".join(bad_page_ids) + "\n", encoding="utf-8")
        _eprint(f"warn: skipped_unparseable_pages={len(bad_page_ids)} (see inputs/skipped_unparseable_page_ids.txt)")
    if not page_ids:
        raise SystemExit(f"No parseable page_ids found under unique_png_root={unique_png_root}")

    # Prefer reading per-page Paddle markdown lazily (to keep RAM bounded).
    paddle_ok_md: dict[str, Path] = {}
    paddle_mem: dict[str, PageText] = {}

    paddle_run_root = None
    if str(args.paddle_run_root).strip():
        paddle_run_root = Path(str(args.paddle_run_root)).expanduser().resolve()
    else:
        user = os.environ.get("USER", "").strip() or "unknown"
        paddle_run_root = Path(f"/scratch/{user}/paddleocr_vl15/runs").resolve()

    if paddle_run_root.is_dir():
        paddle_ok_md = _build_paddle_md_ok_index(
            run_root=paddle_run_root,
            run_prefix=str(args.paddle_run_prefix),
            layout_variant=str(args.paddle_layout_variant),
            vl15_variant=str(args.paddle_vl15_variant),
        )
        if not paddle_ok_md:
            _eprint(f"warn: paddle run root exists but no ok markdown found: {paddle_run_root}")
    elif paddle_paths:
        _eprint("warn: using --paddle-vl15-jsonl as an in-memory paddle source (may require lots of RAM).")
        paddle_mem = _load_paddle_vl15_jsonl(paddle_paths)
    else:
        _eprint("warn: no Paddle source found (no --paddle-run-root and no --paddle-vl15-jsonl); Paddle disabled.")

    paddle_ok_set: set[str] = set(paddle_ok_md.keys()) if paddle_ok_md else set(paddle_mem.keys())

    # Load VLM sources only for pages missing Paddle.
    wanted_vlm_page_ids: set[str] = {pid for pid in page_ids if pid not in paddle_ok_set}
    want_vlm_page = "vlm_page" in priority and bool(vlm_roots)
    want_gemini_box = "gemini_box_results" in priority and bool(gemini_box_paths)
    want_openai_box = "openai_box_results" in priority and bool(openai_box_paths)

    vlm_map = (
        _load_vlm_page_roots(vlm_roots, source_label="vlm_page", wanted_page_ids=wanted_vlm_page_ids) if want_vlm_page else {}
    )
    gemini_box_map = _load_gemini_box_results_jsonl(gemini_box_paths, wanted_page_ids=wanted_vlm_page_ids) if want_gemini_box else {}
    openai_box_map = _load_openai_box_results_jsonl(openai_box_paths, wanted_page_ids=wanted_vlm_page_ids) if want_openai_box else {}

    # Select best VLM fallback per page (Paddle handled lazily via paddle_ok_md / paddle_mem).
    vlm_selected_pages: dict[str, PageText] = {}
    missing_pages: list[dict[str, Any]] = []

    def pick_vlm(pid: str) -> PageText | None:
        for src in priority:
            if src == "paddle_vl15_md":
                continue
            elif src == "vlm_page":
                pt = vlm_map.get(pid)
                if pt and pt.text.strip():
                    return pt
            elif src == "gemini_box_results":
                pt = gemini_box_map.get(pid)
                if pt and pt.text.strip():
                    return pt
            elif src == "openai_box_results":
                pt = openai_box_map.get(pid)
                if pt and pt.text.strip():
                    return pt
            else:
                raise SystemExit(f"Unknown source label in --vlm-source-priority: {src}")
        return None

    for pid in sorted(page_ids):
        if pid in paddle_ok_set:
            continue
        pt = pick_vlm(pid)
        if pt is None:
            meta = page_meta.get(pid) or {}
            missing_pages.append(
                {
                    "page_id": pid,
                    "issue_id": _norm(meta.get("issue_id")),
                    "issue_date": _norm(meta.get("issue_date")),
                    "newspaper_slug": _norm(meta.get("slug")),
                    "page_num": int(meta.get("page_num") or 0),
                }
            )
            continue
        vlm_selected_pages[pid] = pt

    if missing_pages:
        _write_csv(
            out_dir / "missing_pages.csv",
            missing_pages,
            fieldnames=["page_id", "issue_id", "issue_date", "newspaper_slug", "page_num"],
        )
        _eprint(f"missing_pages={len(missing_pages)} (see missing_pages.csv)")

    # Group into issues (canonical pages define issue membership).
    issue_to_pages: dict[str, list[str]] = {}
    for pid in sorted(page_ids):
        meta = page_meta[pid]
        issue_to_pages.setdefault(_norm(meta.get("issue_id")), []).append(pid)

    # Export loop: one request per issue
    missing_policy: MissingIssuePolicy = str(args.missing_page_policy)  # type: ignore[assignment]
    skipped_issues: list[dict[str, Any]] = []
    req_shard_idx = 0
    req_in_shard = 0
    bytes_in_shard_openai = 0
    bytes_in_shard_gemini = 0
    openai_f = (
        (req_dir / f"openai_requests_shard{req_shard_idx:03d}.jsonl").open("w", encoding="utf-8") if want_openai else None
    )
    gemini_f = (
        (req_dir / f"gemini_requests_shard{req_shard_idx:03d}.jsonl").open("w", encoding="utf-8") if want_gemini else None
    )
    map_f = (req_dir / f"mapping_shard{req_shard_idx:03d}.jsonl").open("w", encoding="utf-8")

    def rotate_if_needed(next_openai_line: str | None, next_gemini_line: str | None) -> None:
        nonlocal req_shard_idx, req_in_shard, bytes_in_shard_openai, bytes_in_shard_gemini, openai_f, gemini_f, map_f
        if req_in_shard <= 0:
            return
        if req_in_shard < issues_per_shard:
            limit = int(max_bytes_per_shard)
            if limit > 0:
                if next_openai_line and (bytes_in_shard_openai + len(next_openai_line.encode("utf-8")) > limit):
                    pass
                elif next_gemini_line and (bytes_in_shard_gemini + len(next_gemini_line.encode("utf-8")) > limit):
                    pass
                else:
                    return
            else:
                return

        if openai_f:
            openai_f.close()
        if gemini_f:
            gemini_f.close()
        map_f.close()
        req_shard_idx += 1
        req_in_shard = 0
        bytes_in_shard_openai = 0
        bytes_in_shard_gemini = 0
        openai_f = (
            (req_dir / f"openai_requests_shard{req_shard_idx:03d}.jsonl").open("w", encoding="utf-8") if want_openai else None
        )
        gemini_f = (
            (req_dir / f"gemini_requests_shard{req_shard_idx:03d}.jsonl").open("w", encoding="utf-8") if want_gemini else None
        )
        map_f = (req_dir / f"mapping_shard{req_shard_idx:03d}.jsonl").open("w", encoding="utf-8")

    issues_written = 0
    for issue_id in sorted(issue_to_pages.keys()):
        if issues_filter and issue_id not in issues_filter:
            continue

        pids = issue_to_pages[issue_id]
        missing_in_issue: list[str] = [
            pid for pid in pids if (pid not in paddle_ok_set and pid not in paddle_mem and pid not in vlm_selected_pages)
        ]
        if missing_in_issue:
            rec = {
                "issue_id": issue_id,
                "missing_pages": len(missing_in_issue),
                "total_pages": len(pids),
                "missing_page_ids_preview": ",".join(missing_in_issue[:10]),
            }
            if missing_policy == "require":
                raise SystemExit(f"Missing page transcripts for issue_id={issue_id}: {rec}")
            if missing_policy == "skip":
                skipped_issues.append({**rec, "reason": "missing_page_text"})
                continue
            # ignore: proceed with partial issue transcript (NOT recommended, but sometimes necessary)

        pages: list[PageText] = []
        load_missing: list[str] = []
        for pid in sorted(pids, key=lambda x: int(page_meta[x].get("page_num") or 0)):
            pt: PageText | None = None
            if pid in paddle_ok_md:
                md_path = paddle_ok_md[pid]
                try:
                    text = md_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    text = ""
                if text.strip():
                    meta = page_meta.get(pid) or {}
                    pt = PageText(
                        page_id=pid,
                        issue_id=_norm(meta.get("issue_id")),
                        issue_date=_norm(meta.get("issue_date")),
                        newspaper_slug=_norm(meta.get("slug")).lower(),
                        page_num=int(meta.get("page_num") or 0),
                        text=text,
                        text_sha256=_sha256(text),
                        text_chars=len(text),
                        source="paddle_vl15_md",
                        source_path=str(md_path),
                    )
            elif pid in paddle_mem:
                pt = paddle_mem.get(pid)
            else:
                pt = vlm_selected_pages.get(pid)

            if pt is None:
                load_missing.append(pid)
                continue
            pages.append(pt)

        if load_missing:
            rec = {
                "issue_id": issue_id,
                "missing_pages": len(load_missing),
                "total_pages": len(pids),
                "missing_page_ids_preview": ",".join(load_missing[:10]),
            }
            if missing_policy == "require":
                raise SystemExit(f"Missing page transcripts for issue_id={issue_id}: {rec}")
            if missing_policy == "skip":
                skipped_issues.append({**rec, "reason": "missing_page_text_after_load"})
                continue
            # ignore: proceed with partial issue transcript

        pages_sorted = sorted(pages, key=lambda x: (x.page_num, x.page_id))
        issue_date = issue_id.split("__", 1)[1] if "__" in issue_id else ""
        slug = issue_id.split("__", 1)[0] if "__" in issue_id else ""

        issue_transcript = _build_issue_transcript(pages_sorted, max_issue_chars=max_issue_chars, issue_id=issue_id)
        prompt = (
            prompt_prefix
            + "\n\n"
            + f"ISSUE_ID: {issue_id}\nISSUE_DATE: {issue_date}\nNEWSPAPER_SLUG: {slug}\n\nISSUE_TRANSCRIPT:\n"
            + issue_transcript
        )

        openai_line_s = None
        if want_openai:
            req_obj = _make_openai_request(
                issue_id,
                prompt,
                model=str(args.openai_model),
                reasoning_effort=str(args.openai_reasoning_effort),
            )
            openai_line_s = json.dumps(req_obj, ensure_ascii=False) + "\n"
        gemini_line_s = None
        if want_gemini:
            gemini_obj = _make_gemini_request(
                issue_id,
                prompt,
                response_mime_type=str(args.gemini_response_mime_type),
                thinking_budget=args.gemini_thinking_budget,
                thinking_level=str(args.gemini_thinking_level),
            )
            gemini_line_s = json.dumps(gemini_obj, ensure_ascii=False) + "\n"

        rotate_if_needed(openai_line_s, gemini_line_s)

        if openai_f and openai_line_s is not None:
            openai_f.write(openai_line_s)
            bytes_in_shard_openai += len(openai_line_s.encode("utf-8"))
        if gemini_f and gemini_line_s is not None:
            gemini_f.write(gemini_line_s)
            bytes_in_shard_gemini += len(gemini_line_s.encode("utf-8"))

        mapping = {
            "id": issue_id,
            "custom_id": issue_id,
            "issue_id": issue_id,
            "issue_date": issue_date,
            "slug": slug,
            "exported_at": int(dt.datetime.now().timestamp()),
            "input_stats": {
                "issue_id": issue_id,
                "pages_total": len(pids),
                "pages_with_text": len(pages_sorted),
                "issue_chars": len(issue_transcript),
                "issue_sha256": _sha256(issue_transcript),
            },
            "source_pages": [
                {
                    "page_id": p.page_id,
                    "page_num": p.page_num,
                    "text_source": p.source,
                    "text_chars": p.text_chars,
                    "text_sha256": p.text_sha256,
                    "source_path": p.source_path,
                }
                for p in pages_sorted
            ],
        }
        map_f.write(json.dumps(mapping, ensure_ascii=False) + "\n")

        req_in_shard += 1
        issues_written += 1
        rotate_if_needed(None, None)

        if max_issues is not None and issues_written >= max_issues:
            break

    if openai_f:
        openai_f.close()
    if gemini_f:
        gemini_f.close()
    map_f.close()

    if skipped_issues:
        _write_csv(
            out_dir / "skipped_issues.csv",
            skipped_issues,
            fieldnames=["issue_id", "missing_pages", "total_pages", "missing_page_ids_preview", "reason"],
        )
        _eprint(f"skipped_issues={len(skipped_issues)} (see skipped_issues.csv)")

    # Write a top-level provenance snapshot for the run.
    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "prompt_path": str(prompt_path),
        "prompt_sha256": _sha256(prompt_prefix),
        "unique_png_root": str(unique_png_root),
        "pages_total": len(page_ids),
        "pages_missing_text": len(missing_pages),
        "issues_total": len(issue_to_pages),
        "issues_written": issues_written,
        "issues_skipped": len(skipped_issues),
        "source_counts": {
            "paddle_vl15_md_pages": len(paddle_ok_set),
            "vlm_page_pages": len([1 for p in vlm_selected_pages.values() if p.source == "vlm_page"]),
            "gemini_box_results_pages": len([1 for p in vlm_selected_pages.values() if p.source == "gemini_box_results"]),
            "openai_box_results_pages": len([1 for p in vlm_selected_pages.values() if p.source == "openai_box_results"]),
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _eprint(f"done issues_written={issues_written} shards={req_shard_idx+1}")


if __name__ == "__main__":
    main()
