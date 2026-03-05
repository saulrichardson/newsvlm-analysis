#!/usr/bin/env python3
"""
Build OpenAI batch classification requests directly from raw VLM outputs.

Source-first path:
  raw VLM page transcripts (*.vlm.json) -> normalized page text -> requests.

Request granularity can be:
  - page: one request per page
  - issue: one request per issue containing all pages for that issue
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


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

ALLOWED_YNU = {"yes", "no", "uncertain"}
ALLOWED_ENACT = {"enacted", "proposed", "unknown"}
ALLOWED_ZONING_DOMAIN = {"zoning", "other_law", "none", "uncertain"}
ALLOWED_DOCUMENT_FORM = {"ordinance", "amendment_or_rezoning", "legal_notice", "uncertain", "n/a"}
ALLOWED_SCOPE = {"comprehensive_citywide_code", "noncomprehensive_zoning_ordinance", "unknown", "n/a"}
ALLOWED_EVIDENCE_STRENGTH = {"strong", "moderate", "weak", "none"}
ALLOWED_FRAGMENT = {"yes", "no", "uncertain", "n/a"}
ALLOWED_PAGE_CLASS = {
    "zoning_ordinance_comprehensive",
    "zoning_ordinance_noncomprehensive",
    "zoning_amendment_or_rezoning",
    "zoning_legal_notice",
    "building_code_or_other_law",
    "zoning_narrative_nonverbatim",
    "non_zoning",
    "uncertain",
}
PAGE_CLASS_TO_LEGAL_OBJECT_TYPE = {
    "zoning_ordinance_comprehensive": "full_ordinance",
    "zoning_ordinance_noncomprehensive": "full_ordinance",
    "zoning_amendment_or_rezoning": "amendment",
    "zoning_legal_notice": "notice",
    "building_code_or_other_law": "building_code_or_other_law",
    "zoning_narrative_nonverbatim": "zoning_narrative_nonverbatim",
    "non_zoning": "non_zoning",
    "uncertain": "uncertain",
}
FRAGMENT_TO_COMPLETENESS = {
    "yes": "partial",
    "no": "complete",
    "uncertain": "uncertain",
    "n/a": "uncertain",
}
ALLOWED_REQUEST_GRANULARITY = {"page", "issue"}
ALLOWED_REASONING_EFFORT = {"minimal", "low", "medium", "high"}


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _resolve_vast_to_scratch(path_str: str) -> Path:
    p = _norm(path_str)
    if not p:
        return Path("")
    candidate = Path(p).expanduser()
    if candidate.is_file():
        return candidate
    if p.startswith("/vast/sxr203/"):
        return Path("/scratch/sxr203/" + p[len("/vast/sxr203/") :])
    legacy_abs = "/Users/saulrichardson/projects/newspaper-parsing/newspaper-parsing-local/"
    current_abs = "/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/"
    if p.startswith(legacy_abs):
        mapped = Path(current_abs + p[len(legacy_abs) :])
        if mapped.is_file():
            return mapped
    if p.startswith("newspaper-parsing-local/"):
        mapped = Path(current_abs + p[len("newspaper-parsing-local/") :])
        if mapped.is_file():
            return mapped
    return Path(p)


def _parse_page_id(page_id: str) -> dict[str, Any] | None:
    """
    Parse:
      <slug>-<mon>-<dd>-<yyyy>-p-<num>
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


def _box_sort_key(box: dict[str, Any]) -> tuple[int, str]:
    bid = _norm(box.get("id"))
    m = re.search(r"(\d+)$", bid)
    if m:
        return (int(m.group(1)), bid)
    return (10**9, bid)


def _extract_clean_text_from_vlm(path: Path) -> str:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
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
        # Keep textual content unless the OCR/transcription step marked it unusable.
        if status in {"unreadable", "error"}:
            continue
        parts.append(txt)
    text = "\n\n".join(parts).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _bounded_text_or_fail(text: str, max_chars: int, label: str) -> str:
    s = str(text or "")
    if max_chars > 0 and len(s) > max_chars:
        raise ValueError(
            f"{label} length {len(s)} exceeds hard cap {max_chars}. "
            "Clipping/condensing is disabled; increase cap or reduce input scope."
        )
    return s


def _build_prompt(page_row: dict[str, Any], max_page_chars: int, prompt_path: Path) -> str:
    if not prompt_path.is_file():
        raise SystemExit(f"Missing prompt template: {prompt_path}")
    page_text = _bounded_text_or_fail(
        _norm(page_row.get("text")),
        int(max_page_chars),
        label=f"page_id={_norm(page_row.get('page_id'))} page_text",
    )
    template = prompt_path.read_text(encoding="utf-8")
    required_placeholders = [
        "{{page_id}}",
        "{{issue_id}}",
        "{{issue_date}}",
        "{{newspaper_slug}}",
        "{{page_text}}",
    ]
    missing_placeholders = [ph for ph in required_placeholders if ph not in template]
    if missing_placeholders:
        raise SystemExit(
            "Prompt template is missing required placeholders: "
            + ", ".join(missing_placeholders)
            + f" (template={prompt_path})"
        )
    rendered = (
        template.replace("{{page_id}}", _norm(page_row.get("page_id")))
        .replace("{{issue_id}}", _norm(page_row.get("issue_id")))
        .replace("{{issue_date}}", _norm(page_row.get("issue_date")))
        .replace("{{newspaper_slug}}", _norm(page_row.get("newspaper_slug")))
        .replace("{{page_text}}", page_text)
    )
    unresolved = re.findall(r"\{\{[^{}]+\}\}", rendered)
    if unresolved:
        raise SystemExit(
            "Prompt template has unresolved placeholders after rendering: "
            + ", ".join(sorted(set(unresolved)))
            + f" (template={prompt_path}, page_id={_norm(page_row.get('page_id'))})"
        )
    return rendered.strip()


def _build_issue_prompt(
    *,
    issue_id: str,
    issue_date: str,
    newspaper_slug: str,
    issue_pages: list[dict[str, Any]],
    max_page_chars: int,
    max_issue_chars: int,
    prompt_path: Path,
) -> str:
    if not prompt_path.is_file():
        raise SystemExit(f"Missing prompt template: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    issue_template_placeholders = [
        "{{issue_id}}",
        "{{issue_date}}",
        "{{newspaper_slug}}",
        "{{page_blocks}}",
    ]
    page_template_placeholders = [
        "{{page_id}}",
        "{{issue_id}}",
        "{{issue_date}}",
        "{{newspaper_slug}}",
        "{{page_text}}",
    ]
    is_issue_template = all(ph in template for ph in issue_template_placeholders)
    is_page_template = all(ph in template for ph in page_template_placeholders)
    if not is_issue_template and not is_page_template:
        raise SystemExit(
            "Issue-mode prompt template must be either:\n"
            f"  (A) issue-batch placeholders {issue_template_placeholders}\n"
            f"  (B) page placeholders {page_template_placeholders}\n"
            f"Found template={prompt_path}"
        )

    blocks: list[str] = []
    for page in issue_pages:
        page_text = _bounded_text_or_fail(
            _norm(page.get("text")),
            int(max_page_chars),
            label=f"issue_id={issue_id} page_id={_norm(page.get('page_id'))} page_text",
        )
        blocks.append(
            "\n".join(
                [
                    "=== PAGE START ===",
                    f"page_id={_norm(page.get('page_id'))}",
                    f"page_num={int(page.get('page_num') or 0)}",
                    "page_text:",
                    page_text,
                    "=== PAGE END ===",
                ]
            )
        )

    page_blocks = "\n\n".join(blocks).strip()
    page_blocks = _bounded_text_or_fail(
        page_blocks,
        int(max_issue_chars),
        label=f"issue_id={issue_id} concatenated_issue_text",
    )
    if is_issue_template:
        rendered = (
            template.replace("{{issue_id}}", _norm(issue_id))
            .replace("{{issue_date}}", _norm(issue_date))
            .replace("{{newspaper_slug}}", _norm(newspaper_slug))
            .replace("{{page_blocks}}", page_blocks)
        )
    else:
        # Adapter mode: feed whole issue text into page-style prompt without editing prompt source.
        synthetic_page_id = f"{_norm(issue_id)}__all_pages"
        rendered = (
            template.replace("{{page_id}}", synthetic_page_id)
            .replace("{{issue_id}}", _norm(issue_id))
            .replace("{{issue_date}}", _norm(issue_date))
            .replace("{{newspaper_slug}}", _norm(newspaper_slug))
            .replace("{{page_text}}", page_blocks)
        )
    unresolved = re.findall(r"\{\{[^{}]+\}\}", rendered)
    if unresolved:
        raise SystemExit(
            "Issue prompt template has unresolved placeholders after rendering: "
            + ", ".join(sorted(set(unresolved)))
            + f" (template={prompt_path}, issue_id={issue_id})"
        )
    return rendered.strip()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export or run page-level OpenAI classification from raw VLM pages.")
    ap.add_argument(
        "--vlm-manifests",
        required=True,
        help="Comma-separated manifest.jsonl paths from VLM outputs.",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for page text inventory + OpenAI batch requests/results.",
    )
    ap.add_argument(
        "--newspapers",
        default="",
        help="Optional comma-separated newspaper slugs to include.",
    )
    ap.add_argument("--max-pages", type=int, default=0, help="Optional cap on pages (0 = all).")
    ap.add_argument("--max-issues", type=int, default=0, help="Optional cap on issues (0 = all).")
    ap.add_argument(
        "--request-granularity",
        default="issue",
        choices=sorted(ALLOWED_REQUEST_GRANULARITY),
        help="Request granularity: page (one request per page) or issue (one request per issue with all pages).",
    )
    ap.add_argument(
        "--max-page-chars",
        type=int,
        default=0,
        help="Hard cap for per-page text in prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument(
        "--max-issue-chars",
        type=int,
        default=0,
        help="Hard cap for concatenated issue text in issue-level mode. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument("--model", default="openai:gpt-5")
    ap.add_argument(
        "--reasoning-effort",
        default="",
        help="Optional OpenAI Responses reasoning effort: minimal|low|medium|high.",
    )
    ap.add_argument(
        "--prompt-template",
        default="prompts/prompt_v5_issue_batch.txt",
        help="Prompt template path. Page mode requires {{page_id}}, {{issue_id}}, {{issue_date}}, {{newspaper_slug}}, {{page_text}}. Issue mode requires {{issue_id}}, {{issue_date}}, {{newspaper_slug}}, {{page_blocks}}.",
    )
    ap.add_argument("--requests-per-shard", type=int, default=5000)
    ap.add_argument("--run-gateway", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--gateway-runner",
        default=str(Path(__file__).resolve().parents[1] / "scripts" / "run_openai_requests_via_gateway.py"),
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default=str(Path(__file__).resolve().parents[1] / "agent-gateway" / "src"),
    )
    ap.add_argument("--gov-env-path", default=".env")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _iter_manifest_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _parse_newspapers_spec(raw: str) -> set[str]:
    spec = _norm(raw)
    if not spec:
        return set()
    if spec.startswith("@"):
        p = Path(spec[1:]).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Newspaper list file not found: {p}")
        vals: list[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = _norm(line).lower()
            if not s or s.startswith("#"):
                continue
            vals.append(s)
        return {v for v in vals if v}
    return {x.strip().lower() for x in spec.split(",") if x.strip()}


def _parse_manifest_spec(raw: str) -> list[Path]:
    """
    Parse manifest inputs supplied as either:
      1) comma-separated paths
      2) @file (one path per line, '#' comments allowed)
    """
    spec = _norm(raw)
    if not spec:
        return []

    vals: list[str] = []
    if spec.startswith("@"):
        p = Path(spec[1:]).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Manifest list file not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            s = _norm(line)
            if not s or s.startswith("#"):
                continue
            vals.append(s)
    else:
        vals = [x.strip() for x in spec.split(",") if x.strip()]

    out: list[Path] = []
    for s in vals:
        out.append(Path(s).expanduser().resolve())
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_openai_request_shards(
    path_dir: Path,
    req_rows: list[tuple[str, str]],
    model: str,
    per_shard: int,
    reasoning_effort: str,
) -> list[Path]:
    path_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    if per_shard <= 0:
        per_shard = max(1, len(req_rows))

    model_name = str(model).strip()
    model_provider = ""
    if ":" in model_name:
        provider, rest = model_name.split(":", 1)
        if provider.lower() == "openai" and rest.strip():
            model_name = rest.strip()
            model_provider = "openai"
        else:
            model_provider = provider.lower()
    if not model_provider:
        model_provider = "openai"

    effort = _norm(reasoning_effort).lower()
    if effort and effort not in ALLOWED_REASONING_EFFORT:
        raise SystemExit(
            f"Unsupported --reasoning-effort '{reasoning_effort}'. "
            f"Allowed: {sorted(ALLOWED_REASONING_EFFORT)}"
        )

    shard_idx = 0
    for i in range(0, len(req_rows), per_shard):
        shard = req_rows[i : i + per_shard]
        out = path_dir / f"openai_requests_shard{shard_idx:03d}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for custom_id, prompt in shard:
                req = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model_name,
                        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                        "stream": False,
                    },
                }
                if model_provider == "openai" and effort:
                    req["body"]["reasoning"] = {"effort": effort}
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        out_paths.append(out)
        shard_idx += 1
    return out_paths


def _write_mapping_shards(path_dir: Path, mapping_rows: list[dict[str, Any]], per_shard: int) -> list[Path]:
    path_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    if per_shard <= 0:
        per_shard = max(1, len(mapping_rows))

    shard_idx = 0
    for i in range(0, len(mapping_rows), per_shard):
        shard = mapping_rows[i : i + per_shard]
        out = path_dir / f"mapping_shard{shard_idx:03d}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in shard:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_paths.append(out)
        shard_idx += 1
    return out_paths


def _extract_openai_output_text(body: dict[str, Any]) -> str:
    out = body.get("output")
    if not isinstance(out, list):
        return ""
    parts: list[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "output_text":
                txt = c.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
    return "\n".join(parts).strip()


def _repair_truncated_json(candidate: str) -> str:
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
    return sorted(candidates, key=lambda x: len(json.dumps(x, ensure_ascii=False)), reverse=True)[0]


def _read_result_jsonl(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(results_dir.glob("openai_results_shard*.jsonl")):
        for raw in p.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            cid = obj.get("custom_id")
            if isinstance(cid, str) and cid:
                out[cid] = obj
    return out


def _read_error_jsonl(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(results_dir.glob("openai_errors_shard*.jsonl")):
        for raw in p.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            cid = obj.get("custom_id")
            if isinstance(cid, str) and cid:
                out[cid] = obj
    return out


def _run_gateway(
    *,
    request_dir: Path,
    output_dir: Path,
    runner_path: Path,
    model: str,
    gateway_pythonpath: Path | None,
    timeout_s: float,
    concurrency: int,
    gov_env_path: Path,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    if not runner_path.is_file():
        raise SystemExit(f"Gateway runner not found: {runner_path}")

    env = dict(os.environ)
    if gateway_pythonpath is not None:
        pp = str(gateway_pythonpath)
        prior = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = pp if not prior else f"{pp}:{prior}"

    parsed = _parse_env_file(gov_env_path)
    for key in ("OPENAI_KEY", "GEMINI_KEY", "CLAUDE_KEY"):
        if not env.get(key) and _norm(parsed.get(key)):
            env[key] = _norm(parsed.get(key))
    if not env.get("OPENAI_KEY"):
        alias = _norm(parsed.get("CODEX_API_KEY") or parsed.get("OPENAI_API_KEY"))
        if alias:
            env["OPENAI_KEY"] = alias
    if not env.get("OPENAI_KEY"):
        raise SystemExit("OPENAI_KEY missing. Set in env or --gov-env-path file.")

    cmd = [
        sys.executable,
        str(runner_path),
        "--request-dir",
        str(request_dir),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model),
        "--max-concurrency",
        str(int(concurrency)),
        "--timeout",
        str(float(timeout_s)),
        "--max-retries",
        "1",
    ]
    if bool(skip_existing):
        cmd.append("--skip-existing")
    if bool(dry_run):
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True, env=env)


def _coerce_result(obj: dict[str, Any], page_meta: dict[str, Any]) -> dict[str, Any]:
    is_verbatim = _norm(obj.get("is_verbatim_legal_text")).lower()
    if is_verbatim not in ALLOWED_YNU:
        is_verbatim = "uncertain"

    zoning_presence = _norm(obj.get("zoning_legal_text_presence")).lower()
    if zoning_presence not in ALLOWED_YNU:
        zoning_presence = "uncertain"

    zoning_domain = _norm(obj.get("zoning_legal_domain")).lower()
    if zoning_domain not in ALLOWED_ZONING_DOMAIN:
        zoning_domain = "uncertain"

    zoning_form = _norm(obj.get("zoning_legal_document_form")).lower()
    if zoning_form not in ALLOWED_DOCUMENT_FORM:
        zoning_form = "uncertain"

    ordinance_scope = _norm(obj.get("ordinance_scope_signal")).lower()
    if ordinance_scope not in ALLOWED_SCOPE:
        ordinance_scope = "unknown"

    enactment_strength = _norm(obj.get("enactment_evidence_strength")).lower()
    if enactment_strength not in ALLOWED_EVIDENCE_STRENGTH:
        enactment_strength = "none"

    is_fragment = _norm(obj.get("is_fragment")).lower()
    if is_fragment not in ALLOWED_FRAGMENT:
        is_fragment = "uncertain"

    page_class = _norm(obj.get("page_class")).lower()
    if page_class not in ALLOWED_PAGE_CLASS:
        page_class = "uncertain"

    contains = _norm(obj.get("contains_verbatim_zoning_law_language")).lower()
    if contains not in ALLOWED_YNU:
        contains = zoning_presence

    legal_object_type = PAGE_CLASS_TO_LEGAL_OBJECT_TYPE.get(page_class, "uncertain")
    page_completeness = FRAGMENT_TO_COMPLETENESS.get(is_fragment, "uncertain")

    enact = _norm(obj.get("enactment_status_signal")).lower()
    if enact not in ALLOWED_ENACT:
        enact = "unknown"

    conf = obj.get("confidence_0_to_1")
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    conf_f = max(0.0, min(1.0, conf_f))
    eq = obj.get("evidence_quotes")
    if isinstance(eq, list):
        evidence_quotes = [str(x).strip() for x in eq if str(x).strip()][:3]
    else:
        evidence_quotes = []
    return {
        "page_id": _norm(obj.get("page_id")) or _norm(page_meta.get("page_id")),
        "issue_id": _norm(page_meta.get("issue_id")),
        "issue_date": _norm(page_meta.get("issue_date")),
        "newspaper_slug": _norm(page_meta.get("newspaper_slug")),
        "page_num": int(page_meta.get("page_num") or 0),
        "contains_verbatim_zoning_law_language": contains,
        "is_verbatim_legal_text": is_verbatim,
        "zoning_legal_text_presence": zoning_presence,
        "zoning_legal_domain": zoning_domain,
        "zoning_legal_document_form": zoning_form,
        "ordinance_scope_signal": ordinance_scope,
        "enactment_evidence_strength": enactment_strength,
        "is_fragment": is_fragment,
        "page_class": page_class,
        "legal_object_type": legal_object_type,
        "page_completeness": page_completeness,
        "enactment_status_signal": enact,
        "confidence_0_to_1": conf_f,
        "evidence_quotes": evidence_quotes,
        "notes": _norm(obj.get("notes")),
    }


def _extract_issue_page_objects(parsed: dict[str, Any], expected_page_ids: set[str]) -> dict[str, dict[str, Any]]:
    # Accept either:
    #  - {"page_outputs": [ ... ]}
    #  - {"pages": [ ... ]}
    #  - {"results": [ ... ]}
    #  - plain list-like under common keys
    #  - single-page object
    page_lists: list[Any] = []
    for k in ("page_outputs", "pages", "results", "classifications", "page_results"):
        v = parsed.get(k)
        if isinstance(v, list):
            page_lists.append(v)
    if not page_lists and parsed.get("page_id"):
        page_lists = [[parsed]]

    out: dict[str, dict[str, Any]] = {}
    for lst in page_lists:
        for item in lst:
            if not isinstance(item, dict):
                continue
            pid = _norm(item.get("page_id")).lower()
            if not pid:
                continue
            if expected_page_ids and pid not in expected_page_ids:
                continue
            out[pid] = item
    return out


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    req_dir = out_dir / "requests"
    res_dir = out_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    req_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = _parse_manifest_spec(str(args.vlm_manifests))
    if not manifest_paths:
        raise SystemExit("No --vlm-manifests supplied.")
    prompt_template_path = Path(str(args.prompt_template)).expanduser()
    if not prompt_template_path.is_absolute():
        prompt_template_path = (Path(__file__).resolve().parents[1] / prompt_template_path).resolve()
    if not prompt_template_path.is_file():
        raise SystemExit(f"Prompt template not found: {prompt_template_path}")

    wanted_newspapers = _parse_newspapers_spec(str(args.newspapers))
    max_pages = int(args.max_pages)
    max_issues = int(args.max_issues)
    request_granularity = _norm(args.request_granularity).lower()
    if request_granularity not in ALLOWED_REQUEST_GRANULARITY:
        raise SystemExit(f"Unsupported --request-granularity: {request_granularity}")
    if request_granularity == "issue" and max_pages > 0:
        raise SystemExit("--max-pages is page-mode only. Use --max-issues for issue mode.")

    rows = _iter_manifest_rows(manifest_paths)
    # Deduplicate by page_id across manifests (keep first seen row).
    # If manifests are ordered as openai first, gemini second, this preserves openai when duplicated.
    dedup_rows: list[dict[str, Any]] = []
    seen_page_ids: set[str] = set()
    for r in rows:
        pid = _norm(r.get("page_id")).lower()
        if not pid or pid in seen_page_ids:
            continue
        seen_page_ids.add(pid)
        dedup_rows.append(r)
    rows = dedup_rows
    page_rows: list[dict[str, Any]] = []
    for r in rows:
        page_id = _norm(r.get("page_id")).lower()
        if not page_id:
            continue
        pid = _parse_page_id(page_id)
        if pid is None:
            continue
        slug = _norm(pid.get("slug")).lower()
        if wanted_newspapers and slug not in wanted_newspapers:
            continue
        output_path = _resolve_vast_to_scratch(_norm(r.get("output_path")))
        if not output_path.is_file():
            continue
        text = _extract_clean_text_from_vlm(output_path)
        page_rows.append(
            {
                "page_id": page_id,
                "issue_id": _norm(pid.get("issue_id")),
                "issue_date": _norm(pid.get("issue_date")),
                "newspaper_slug": slug,
                "page_num": int(pid.get("page_num") or 0),
                "vlm_path": str(output_path),
                "text": text,
                "text_chars": int(len(text)),
                "text_sha256": _sha256(text),
            }
        )
    page_rows = sorted(page_rows, key=lambda x: (_norm(x.get("newspaper_slug")), _norm(x.get("issue_date")), int(x.get("page_num") or 0), _norm(x.get("page_id"))))
    if max_pages > 0:
        page_rows = page_rows[:max_pages]
    if not page_rows:
        raise SystemExit("No page rows found after filters.")

    _write_jsonl(out_dir / "page_texts_clean.jsonl", page_rows)

    req_rows: list[tuple[str, str]] = []
    mapping_rows: list[dict[str, Any]] = []
    page_map: dict[str, dict[str, Any]] = {}
    issue_map: dict[str, dict[str, Any]] = {}
    if request_granularity == "page":
        for r in page_rows:
            cid = f"page_classify_v5::{_norm(r.get('page_id'))}"
            req_rows.append((cid, _build_prompt(r, int(args.max_page_chars), prompt_template_path)))
            page_map[cid] = r
            mapping_rows.append(
                {
                    "custom_id": cid,
                    "doc_id": _norm(r.get("page_id")),
                    "newspaper_slug": _norm(r.get("newspaper_slug")),
                    "issue_id": _norm(r.get("issue_id")),
                    "issue_date": _norm(r.get("issue_date")),
                    "page_id": _norm(r.get("page_id")),
                    "page_num": int(r.get("page_num") or 0),
                    "text_chars": int(r.get("text_chars") or 0),
                    "text_sha256": _norm(r.get("text_sha256")),
                    "vlm_path": _norm(r.get("vlm_path")),
                }
            )
    else:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for r in page_rows:
            key = (
                _norm(r.get("newspaper_slug")).lower(),
                _norm(r.get("issue_date")),
                _norm(r.get("issue_id")),
            )
            grouped.setdefault(key, []).append(r)
        issue_rows = []
        for (_slug, _date, _iid), pages in grouped.items():
            pages_sorted = sorted(pages, key=lambda x: (int(x.get("page_num") or 0), _norm(x.get("page_id"))))
            issue_rows.append(
                {
                    "newspaper_slug": _slug,
                    "issue_date": _date,
                    "issue_id": _iid,
                    "pages": pages_sorted,
                }
            )
        issue_rows = sorted(issue_rows, key=lambda x: (_norm(x.get("newspaper_slug")), _norm(x.get("issue_date")), _norm(x.get("issue_id"))))
        if max_issues > 0:
            issue_rows = issue_rows[:max_issues]
        if not issue_rows:
            raise SystemExit("No issue rows found after grouping.")
        for issue in issue_rows:
            cid = f"issue_classify_v5::{_norm(issue.get('issue_id'))}"
            prompt = _build_issue_prompt(
                issue_id=_norm(issue.get("issue_id")),
                issue_date=_norm(issue.get("issue_date")),
                newspaper_slug=_norm(issue.get("newspaper_slug")),
                issue_pages=list(issue.get("pages") or []),
                max_page_chars=int(args.max_page_chars),
                max_issue_chars=int(args.max_issue_chars),
                prompt_path=prompt_template_path,
            )
            req_rows.append((cid, prompt))
            issue_map[cid] = issue
            for r in list(issue.get("pages") or []):
                mapping_rows.append(
                    {
                        "custom_id": cid,
                        "doc_id": _norm(r.get("page_id")),
                        "newspaper_slug": _norm(r.get("newspaper_slug")),
                        "issue_id": _norm(r.get("issue_id")),
                        "issue_date": _norm(r.get("issue_date")),
                        "page_id": _norm(r.get("page_id")),
                        "page_num": int(r.get("page_num") or 0),
                        "text_chars": int(r.get("text_chars") or 0),
                        "text_sha256": _norm(r.get("text_sha256")),
                        "vlm_path": _norm(r.get("vlm_path")),
                    }
                )

    shard_paths = _write_openai_request_shards(
        req_dir,
        req_rows,
        model=str(args.model),
        per_shard=int(args.requests_per_shard),
        reasoning_effort=str(args.reasoning_effort),
    )
    mapping_paths = _write_mapping_shards(req_dir, mapping_rows, per_shard=int(args.requests_per_shard))

    if bool(args.run_gateway):
        _run_gateway(
            request_dir=req_dir,
            output_dir=res_dir,
            runner_path=Path(args.gateway_runner).expanduser().resolve(),
            model=str(args.model),
            gateway_pythonpath=Path(args.gateway_pythonpath).expanduser().resolve() if _norm(args.gateway_pythonpath) else None,
            timeout_s=float(args.timeout),
            concurrency=int(args.concurrency),
            gov_env_path=Path(args.gov_env_path).expanduser().resolve(),
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
        )

        result_map = _read_result_jsonl(res_dir)
        error_map = _read_error_jsonl(res_dir)
        out_rows: list[dict[str, Any]] = []
        if request_granularity == "page":
            for cid, meta in page_map.items():
                had_error = 0
                parse_ok = 0
                output_text = ""
                if cid in result_map:
                    body = (((result_map[cid].get("response") or {}).get("body")) or {})
                    output_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
                elif cid in error_map:
                    had_error = 1
                    body = (((error_map[cid].get("response") or {}).get("body")) or {})
                    if isinstance(body, dict):
                        output_text = _norm(((body.get("error") or {}).get("message")))
                parsed = _parse_json_from_text(output_text)
                if isinstance(parsed, dict) and parsed:
                    parse_ok = 1
                coerced = _coerce_result(parsed if isinstance(parsed, dict) else {}, meta)
                coerced["parse_ok"] = parse_ok
                coerced["had_error"] = had_error
                out_rows.append(coerced)
        else:
            for cid, issue in issue_map.items():
                had_error = 0
                output_text = ""
                if cid in result_map:
                    body = (((result_map[cid].get("response") or {}).get("body")) or {})
                    output_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
                elif cid in error_map:
                    had_error = 1
                    body = (((error_map[cid].get("response") or {}).get("body")) or {})
                    if isinstance(body, dict):
                        output_text = _norm(((body.get("error") or {}).get("message")))
                parsed = _parse_json_from_text(output_text)
                expected_page_ids = {
                    _norm(r.get("page_id")).lower() for r in list(issue.get("pages") or []) if _norm(r.get("page_id"))
                }
                parsed_by_page = _extract_issue_page_objects(parsed if isinstance(parsed, dict) else {}, expected_page_ids)
                for meta in list(issue.get("pages") or []):
                    pid = _norm(meta.get("page_id")).lower()
                    page_obj = parsed_by_page.get(pid) or {}
                    parse_ok = 1 if page_obj else 0
                    coerced = _coerce_result(page_obj, meta)
                    coerced["parse_ok"] = parse_ok
                    coerced["had_error"] = had_error
                    out_rows.append(coerced)
        out_rows = sorted(out_rows, key=lambda x: (_norm(x.get("newspaper_slug")), _norm(x.get("issue_date")), int(x.get("page_num") or 0), _norm(x.get("page_id"))))
        _write_jsonl(out_dir / "page_classification_outputs.jsonl", out_rows)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vlm_manifests": [str(x) for x in manifest_paths],
        "newspapers_filter": sorted(list(wanted_newspapers)),
        "request_granularity": request_granularity,
        "page_rows": int(len(page_rows)),
        "issue_rows": int(len(issue_map)),
        "request_rows": int(len(req_rows)),
        "request_shards": [str(x) for x in shard_paths],
        "mapping_shards": [str(x) for x in mapping_paths],
        "model": str(args.model),
        "reasoning_effort": _norm(args.reasoning_effort).lower(),
        "prompt_template_path": str(prompt_template_path),
        "prompt_template_sha256": hashlib.sha256(prompt_template_path.read_bytes()).hexdigest(),
        "run_gateway": bool(args.run_gateway),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"done_export_page_classification_requests_from_vlm pages={len(page_rows)} out={out_dir}")


if __name__ == "__main__":
    main()
