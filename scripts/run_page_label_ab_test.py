#!/usr/bin/env python3
"""
Run A/B page-classification prompts on a labeled page set.

Purpose:
- Select a deterministic ~N page sample from an existing labeled CSV.
- Read full page transcript text from organized side-by-side artifacts.
- Run two prompt variants against the same pages.
- Compare predictions against existing labels and write metrics + confusion tables.

This script does not clip page text. If a positive --max-page-chars cap is provided
and exceeded, that page is skipped and logged as a hard input failure.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_LEGAL_OBJECT_TYPE = {
    "full_ordinance",
    "amendment",
    "proposal",
    "notice",
    "building_code_or_other_law",
    "zoning_narrative_nonverbatim",
    "non_zoning",
    "uncertain",
}
ALLOWED_YNU = {"yes", "no", "uncertain"}
ALLOWED_ENACT = {"enacted", "proposed", "unknown"}
ALLOWED_PAGE_COMPLETENESS = {"partial", "complete", "uncertain"}


PRED_TO_BUCKET = {
    "full_ordinance": "full_ordinance",
    "amendment": "amendment",
    "proposal": "proposal_full_text",
    "notice": "notice",
    "building_code_or_other_law": "building_code_non_zoning",
    "zoning_narrative_nonverbatim": "narrative_or_nonverbatim",
    "non_zoning": "non_zoning",
    "uncertain": "uncertain_other",
}


GOLD_TO_EVAL = {
    "full_ordinance": "full_ordinance",
    "partial_ordinance_enacted": "full_ordinance",
    "amendment": "amendment",
    "proposal_full_text": "proposal_full_text",
    "proposal_amendment": "proposal_full_text",
    "notice": "notice",
    "building_code_non_zoning": "building_code_non_zoning",
    "other_jurisdiction_law": "building_code_non_zoning",
    "narrative_or_nonverbatim": "narrative_or_nonverbatim",
    "non_zoning": "non_zoning",
    "uncertain_other": "uncertain_other",
}


def _norm(x: Any) -> str:
    return str(x or "").strip()


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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


def _coerce_result(obj: dict[str, Any], *, page_id: str) -> dict[str, Any]:
    legal_object_type = _norm(obj.get("legal_object_type")).lower()
    if legal_object_type not in ALLOWED_LEGAL_OBJECT_TYPE:
        legal_object_type = "uncertain"
    contains = _norm(obj.get("contains_verbatim_zoning_law_language")).lower()
    if contains not in ALLOWED_YNU:
        contains = "uncertain"
    verbatim = _norm(obj.get("is_verbatim_legal_text")).lower()
    if verbatim not in ALLOWED_YNU:
        verbatim = "uncertain"
    enact = _norm(obj.get("enactment_status_signal")).lower()
    if enact not in ALLOWED_ENACT:
        enact = "unknown"
    page_completeness = _norm(obj.get("page_completeness")).lower()
    if page_completeness not in ALLOWED_PAGE_COMPLETENESS:
        page_completeness = "uncertain"
    conf = obj.get("confidence_0_to_1")
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    conf_f = max(0.0, min(1.0, conf_f))
    return {
        "page_id": page_id,
        "legal_object_type": legal_object_type,
        "page_completeness": page_completeness,
        "contains_verbatim_zoning_law_language": contains,
        "is_verbatim_legal_text": verbatim,
        "enactment_status_signal": enact,
        "confidence_0_to_1": conf_f,
        "notes": _norm(obj.get("notes")),
    }


def _build_prompt_variant_a(*, page_id: str, issue_id: str, issue_date: str, newspaper_slug: str, text: str) -> str:
    schema = {
        "page_id": "string",
        "contains_verbatim_zoning_law_language": "yes",
        "is_verbatim_legal_text": "yes",
        "legal_object_type": "full_ordinance",
        "page_completeness": "partial",
        "enactment_status_signal": "enacted",
        "confidence_0_to_1": 0.85,
        "evidence_quotes": ["short quote 1", "short quote 2"],
        "notes": "brief rationale",
    }
    lines: list[str] = []
    lines.append("Classify one newspaper page for zoning-law extraction.")
    lines.append("Important: this is OCR/VLM transcript text from scanned newspaper pages.")
    lines.append("Expect OCR artifacts (misspellings, broken lines, stray columns).")
    lines.append("Classify by legal meaning, not OCR polish.")
    lines.append("Return ONLY JSON. No markdown.")
    lines.append("")
    lines.append("Use exactly one legal_object_type from:")
    lines.extend(
        [
            "- full_ordinance",
            "- amendment",
            "- proposal",
            "- notice",
            "- building_code_or_other_law",
            "- zoning_narrative_nonverbatim",
            "- non_zoning",
            "- uncertain",
        ]
    )
    lines.append("")
    lines.append("Policy constraints:")
    lines.append("- Proposal/hearing/consideration language is not enacted full ordinance.")
    lines.append("- Narrative reporting about zoning is not verbatim legal text.")
    lines.append("- Building code text not specific to zoning -> building_code_or_other_law.")
    lines.append("- If mixed or ambiguous, use uncertain and explain.")
    lines.append("")
    lines.append("Return fields:")
    lines.append("- contains_verbatim_zoning_law_language: yes|no|uncertain")
    lines.append("- is_verbatim_legal_text: yes|no|uncertain")
    lines.append("- legal_object_type: one of the 8 values above")
    lines.append("- page_completeness: partial|complete|uncertain")
    lines.append("- enactment_status_signal: enacted|proposed|unknown")
    lines.append("- confidence_0_to_1: 0..1")
    lines.append("- evidence_quotes: 1-3 short snippets from page text")
    lines.append("")
    lines.append("JSON schema example:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Page metadata: page_id={page_id} issue_id={issue_id} issue_date={issue_date} newspaper_slug={newspaper_slug}"
    )
    lines.append("")
    lines.append("Page text:")
    lines.append(text)
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _build_prompt_variant_b(*, page_id: str, issue_id: str, issue_date: str, newspaper_slug: str, text: str) -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "page_label_variant_b_prompt.txt"
    if not prompt_path.is_file():
        raise SystemExit(f"Missing prompt template: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    rendered = (
        template.replace("{{page_id}}", page_id)
        .replace("{{issue_id}}", issue_id)
        .replace("{{issue_date}}", issue_date)
        .replace("{{newspaper_slug}}", newspaper_slug)
        .replace("{{page_text}}", text)
    )
    return rendered.strip()


def _build_prompt_variant_c(*, page_id: str, issue_id: str, issue_date: str, newspaper_slug: str, text: str) -> str:
    schema = {
        "page_id": "string",
        "contains_verbatim_zoning_law_language": "yes",
        "is_verbatim_legal_text": "yes",
        "law_domain": "zoning",
        "instrument_signal": "full_ordinance",
        "legal_object_type": "full_ordinance",
        "page_completeness": "partial",
        "enactment_status_signal": "enacted",
        "confidence_0_to_1": 0.85,
        "evidence_quotes": ["short quote 1", "short quote 2"],
        "notes": "brief rationale",
    }
    lines: list[str] = []
    lines.append("Precision-first classification of one OCR newspaper page.")
    lines.append("False positives for full/amendment are costly. If uncertain, output uncertain.")
    lines.append("Return ONLY JSON. No markdown.")
    lines.append("")
    lines.append("Step 1: Verbatim zoning-law presence")
    lines.append("- yes: page contains legal ordinance/amendment clause language for zoning.")
    lines.append("- no: no zoning-law clauses.")
    lines.append("- uncertain: ambiguous OCR or mixed layout.")
    lines.append("")
    lines.append("Step 2: Law domain")
    lines.append("- zoning")
    lines.append("- building_or_other_law")
    lines.append("- non_law")
    lines.append("- uncertain")
    lines.append("")
    lines.append("Step 3: Instrument signal")
    lines.append("- full_ordinance")
    lines.append("- amendment")
    lines.append("- proposal")
    lines.append("- notice")
    lines.append("- none")
    lines.append("- uncertain")
    lines.append("")
    lines.append("Step 4: Final legal_object_type (must follow rules)")
    lines.extend(
        [
            "- full_ordinance",
            "- amendment",
            "- proposal",
            "- notice",
            "- building_code_or_other_law",
            "- zoning_narrative_nonverbatim",
            "- non_zoning",
            "- uncertain",
        ]
    )
    lines.append("")
    lines.append("Hard rules:")
    lines.append("- If law_domain is building_or_other_law, final type must be building_code_or_other_law (or uncertain).")
    lines.append("- If contains_verbatim_zoning_law_language is not yes, do not output full_ordinance or amendment.")
    lines.append("- Proposal/hearing/recommendation language without in-force/adopted cue => proposal, not full_ordinance.")
    lines.append("- Narrative summaries without clause text => zoning_narrative_nonverbatim.")
    lines.append("- Enactment signal:")
    lines.append("  - enacted only if explicit in-force/adopted/effective cue is present on-page.")
    lines.append("  - proposed if pending/proposed/hearing language appears.")
    lines.append("  - otherwise unknown.")
    lines.append("")
    lines.append("OCR note: text may be broken; infer from legal structure and cues, but stay conservative.")
    lines.append("")
    lines.append("Return fields:")
    lines.append("- contains_verbatim_zoning_law_language: yes|no|uncertain")
    lines.append("- is_verbatim_legal_text: yes|no|uncertain")
    lines.append("- law_domain: zoning|building_or_other_law|non_law|uncertain")
    lines.append("- instrument_signal: full_ordinance|amendment|proposal|notice|none|uncertain")
    lines.append("- legal_object_type: one of the 8 values above")
    lines.append("- page_completeness: partial|complete|uncertain")
    lines.append("- enactment_status_signal: enacted|proposed|unknown")
    lines.append("- confidence_0_to_1: 0..1")
    lines.append("- evidence_quotes: 1-3 short snippets")
    lines.append("- notes: one short sentence")
    lines.append("")
    lines.append("JSON schema example:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Page metadata: page_id={page_id} issue_id={issue_id} issue_date={issue_date} newspaper_slug={newspaper_slug}"
    )
    lines.append("")
    lines.append("Page text:")
    lines.append(text)
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _build_prompt_variant_d(*, page_id: str, issue_id: str, issue_date: str, newspaper_slug: str, text: str) -> str:
    schema = {
        "page_id": "string",
        "contains_verbatim_zoning_law_language": "yes",
        "is_verbatim_legal_text": "yes",
        "page_scope": "zoning_ordinance",
        "legal_object_type": "full_ordinance",
        "page_completeness": "partial",
        "enactment_status_signal": "enacted",
        "confidence_0_to_1": 0.9,
        "evidence_quotes": ["short quote 1", "short quote 2"],
        "notes": "brief rationale",
    }
    lines: list[str] = []
    lines.append("Precision-first classification of one OCR newspaper page.")
    lines.append("Return ONLY JSON. No markdown.")
    lines.append("")
    lines.append("Step 1: pick one page_scope:")
    lines.append("- zoning_ordinance")
    lines.append("- zoning_amendment")
    lines.append("- zoning_proposal_or_notice")
    lines.append("- other_municipal_law")
    lines.append("- zoning_narrative")
    lines.append("- non_zoning")
    lines.append("- uncertain")
    lines.append("")
    lines.append("Step 2: map to final legal_object_type:")
    lines.append("- zoning_ordinance -> full_ordinance")
    lines.append("- zoning_amendment -> amendment")
    lines.append("- zoning_proposal_or_notice -> proposal OR notice")
    lines.append("- other_municipal_law -> building_code_or_other_law")
    lines.append("- zoning_narrative -> zoning_narrative_nonverbatim")
    lines.append("- non_zoning -> non_zoning")
    lines.append("- uncertain -> uncertain")
    lines.append("")
    lines.append("Contrastive examples (guidance):")
    lines.append("1) 'An ordinance adopting the Building Code ... electrical/plumbing ...' => other_municipal_law.")
    lines.append("2) 'Notice of public hearing on proposed zoning amendment ... first reading ...' => zoning_proposal_or_notice (proposal/notice).")
    lines.append("3) 'Section X. R-1 district uses ... lot area ... setbacks ...' => zoning_ordinance.")
    lines.append("4) 'Ordinance amending Section 12 of zoning ordinance; rezoning parcel ...' => zoning_amendment.")
    lines.append("5) 'Article discussing zoning dispute without legal clause text' => zoning_narrative.")
    lines.append("")
    lines.append("Hard precision rules:")
    lines.append("- If text is legal but subject is building code/traffic/plumbing/fiscal/utility and not zoning law, NEVER output full/amendment/proposal/notice zoning classes.")
    lines.append("- Proposal/hearing/first-reading/pending language is not enacted full ordinance.")
    lines.append("- If mixed or ambiguous, use uncertain rather than full/amendment.")
    lines.append("")
    lines.append("Return fields:")
    lines.append("- contains_verbatim_zoning_law_language: yes|no|uncertain")
    lines.append("- is_verbatim_legal_text: yes|no|uncertain")
    lines.append("- page_scope (one of seven values above)")
    lines.append("- legal_object_type: one of the 8 values above")
    lines.append("- page_completeness: partial|complete|uncertain")
    lines.append("- enactment_status_signal: enacted|proposed|unknown")
    lines.append("- confidence_0_to_1: 0..1")
    lines.append("- evidence_quotes: 1-3 snippets")
    lines.append("- notes")
    lines.append("")
    lines.append("JSON schema example:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Page metadata: page_id={page_id} issue_id={issue_id} issue_date={issue_date} newspaper_slug={newspaper_slug}"
    )
    lines.append("")
    lines.append("Page text:")
    lines.append(text)
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _build_prompt_variant_e(*, page_id: str, issue_id: str, issue_date: str, newspaper_slug: str, text: str) -> str:
    schema = {
        "page_id": "string",
        "contains_verbatim_zoning_law_language": "yes",
        "is_verbatim_legal_text": "yes",
        "zoning_evidence_checklist": {
            "zoning_subject_explicit": "yes",
            "clause_structure_present": "yes",
            "regulatory_content_present": "yes",
        },
        "legal_object_type": "full_ordinance",
        "page_completeness": "partial",
        "enactment_status_signal": "enacted",
        "confidence_0_to_1": 0.9,
        "evidence_quotes": ["short quote 1", "short quote 2"],
        "notes": "brief rationale",
    }
    lines: list[str] = []
    lines.append("Classify one OCR newspaper page with precision-first zoning checklist.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Allowed classes:")
    lines.extend(
        [
            "- full_ordinance",
            "- amendment",
            "- proposal",
            "- notice",
            "- building_code_or_other_law",
            "- zoning_narrative_nonverbatim",
            "- non_zoning",
            "- uncertain",
        ]
    )
    lines.append("")
    lines.append("Zoning evidence checklist (required for full/amendment):")
    lines.append("1) zoning_subject_explicit: does text explicitly refer to zoning ordinance/district/zone/rezoning/land-use zoning rules?")
    lines.append("2) clause_structure_present: legal clause structure (section/article/ordinance language)?")
    lines.append("3) regulatory_content_present: actual zoning rules (uses, districts, lot/setback/height/parking, nonconforming, definitions, enforcement)?")
    lines.append("")
    lines.append("Hard rules:")
    lines.append("- full_ordinance or amendment requires ALL THREE checklist items = yes.")
    lines.append("- If legal clauses exist but zoning subject is not explicit, classify building_code_or_other_law.")
    lines.append("- If proposal/hearing/first-reading/pending adoption dominates, use proposal or notice.")
    lines.append("- Narrative discussion without verbatim clauses -> zoning_narrative_nonverbatim.")
    lines.append("- If uncertain on checklist, use uncertain.")
    lines.append("")
    lines.append("Return fields:")
    lines.append("- contains_verbatim_zoning_law_language: yes|no|uncertain")
    lines.append("- is_verbatim_legal_text: yes|no|uncertain")
    lines.append("- zoning_evidence_checklist: zoning_subject_explicit/clause_structure_present/regulatory_content_present as yes|no|uncertain")
    lines.append("- legal_object_type")
    lines.append("- page_completeness: partial|complete|uncertain")
    lines.append("- enactment_status_signal: enacted|proposed|unknown")
    lines.append("- confidence_0_to_1")
    lines.append("- evidence_quotes")
    lines.append("- notes")
    lines.append("")
    lines.append("JSON schema example:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Page metadata: page_id={page_id} issue_id={issue_id} issue_date={issue_date} newspaper_slug={newspaper_slug}"
    )
    lines.append("")
    lines.append("Page text:")
    lines.append(text)
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


@dataclass
class PageRow:
    page_id: str
    issue_id: str
    issue_date: str
    newspaper_slug: str
    gold_bucket: str
    text_path: Path
    text: str


def _sample_stratified(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.copy()
    df = df.copy().sort_values(
        ["page_class_bucket", "newspaper_slug", "issue_date", "issue_id", "page_num_in_issue", "page_id"]
    )
    counts = df["page_class_bucket"].value_counts().to_dict()
    total = float(len(df))
    classes = sorted(counts.keys())
    if sample_size <= len(classes):
        # If sample is smaller than number of classes, take one deterministic row
        # per top classes by prevalence to avoid an impossible min-1-all-class allocation.
        keep = sorted(classes, key=lambda k: (-counts[k], k))[:sample_size]
        parts = [df[df["page_class_bucket"] == k].head(1) for k in keep]
        return pd.concat(parts, ignore_index=True).reset_index(drop=True)
    target: dict[str, int] = {}
    for k, n in counts.items():
        t = int(round(sample_size * (n / total)))
        target[k] = max(1, t)
    # adjust to exact sample_size
    cur = sum(target.values())
    order = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    while cur > sample_size:
        for k, _n in order:
            if cur <= sample_size:
                break
            if target[k] > 1:
                target[k] -= 1
                cur -= 1
    while cur < sample_size:
        for k, _n in order:
            if cur >= sample_size:
                break
            target[k] += 1
            cur += 1
    parts: list[pd.DataFrame] = []
    for k, t in target.items():
        g = df[df["page_class_bucket"] == k].head(t)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    if len(out) > sample_size:
        out = out.head(sample_size)
    if len(out) < sample_size:
        used = set(out["page_id"].astype(str))
        rem = df[~df["page_id"].astype(str).isin(used)].head(sample_size - len(out))
        out = pd.concat([out, rem], ignore_index=True)
    return out.reset_index(drop=True)


def _index_page_txt(side_by_side_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    pat = re.compile(r"^page_\d+__(.+)\.txt$")
    for p in side_by_side_root.glob("**/page_*__*.txt"):
        m = pat.match(p.name)
        if not m:
            continue
        pid = _norm(m.group(1))
        if pid and pid not in out:
            out[pid] = p
    return out


def _run_gateway(
    *,
    request_dir: Path,
    output_dir: Path,
    model: str,
    runner_path: Path,
    gateway_pythonpath: Path,
    gov_env_path: Path,
    timeout: float,
    concurrency: int,
    skip_existing: bool,
) -> None:
    env = dict(os.environ)
    prior = _norm(env.get("PYTHONPATH"))
    env["PYTHONPATH"] = str(gateway_pythonpath) if not prior else f"{gateway_pythonpath}:{prior}"
    parsed = _parse_env_file(gov_env_path)
    for key in ("OPENAI_KEY", "GEMINI_KEY", "CLAUDE_KEY"):
        if not env.get(key) and _norm(parsed.get(key)):
            env[key] = _norm(parsed.get(key))
    if not env.get("OPENAI_KEY"):
        alias = _norm(parsed.get("CODEX_API_KEY") or parsed.get("OPENAI_API_KEY"))
        if alias:
            env["OPENAI_KEY"] = alias
    if not env.get("OPENAI_KEY") and str(model).startswith("openai:"):
        raise SystemExit("OPENAI_KEY missing in env and gov env file.")

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
        str(float(timeout)),
        "--max-retries",
        "1",
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    subprocess.run(cmd, check=True, env=env)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run page-classification A/B test on labeled pages.")
    ap.add_argument(
        "--label-csv",
        default="reports/tmp/hardened_classification_refresh_5papers_20260222/page_classification_precision_v1/page_classification_index_v1.csv",
        help="CSV with page_id + page_class_bucket labels.",
    )
    ap.add_argument(
        "--side-by-side-root",
        default="reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3_run_20260220_021657/organized_side_by_side/by_newspaper",
        help="Root containing page_###__<page_id>.txt files.",
    )
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--model", default="openai:gpt-5-mini")
    ap.add_argument(
        "--variants",
        default="A,B",
        help="Comma-separated prompt variants to run from {A,B,C,D,E}. Example: B,E",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-page-chars", type=int, default=0, help="Hard per-page char cap; 0 disables cap.")
    ap.add_argument(
        "--gateway-runner",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/experimental/scripts/run_openai_requests_via_gateway.py",
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/agent-gateway/src",
    )
    ap.add_argument(
        "--gov-env-path",
        default="/Users/saulrichardson/Dropbox/Remote Work Startups/main/.env",
    )
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-output-tokens", type=int, default=1200)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-run", action="store_true", help="Only write request files.")
    return ap.parse_args()


def _binary_metrics(df: pd.DataFrame, pos_label: str) -> dict[str, float]:
    y = (df["gold_eval"] == pos_label).astype(int)
    p = (df["pred_eval"] == pos_label).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "target": pos_label,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_csv = Path(args.label_csv).expanduser().resolve()
    side_root = Path(args.side_by_side_root).expanduser().resolve()
    if not label_csv.is_file():
        raise SystemExit(f"Missing label CSV: {label_csv}")
    if not side_root.is_dir():
        raise SystemExit(f"Missing side-by-side root: {side_root}")

    df = pd.read_csv(label_csv)
    required = {"page_id", "issue_id", "issue_date", "newspaper_slug", "page_class_bucket"}
    miss = sorted(required - set(df.columns))
    if miss:
        raise SystemExit(f"Label CSV missing columns: {miss}")
    df = df[df["page_id"].astype(str).str.len() > 0].copy()
    df["page_class_bucket"] = df["page_class_bucket"].astype(str)
    df = _sample_stratified(df, int(args.sample_size))

    page_map = _index_page_txt(side_root)
    rows: list[PageRow] = []
    hard_fail_rows: list[dict[str, Any]] = []
    for r in df.itertuples(index=False):
        page_id = _norm(getattr(r, "page_id", ""))
        text_path = page_map.get(page_id)
        if text_path is None:
            hard_fail_rows.append({"page_id": page_id, "reason": "missing_page_txt"})
            continue
        text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        if int(args.max_page_chars) > 0 and len(text) > int(args.max_page_chars):
            hard_fail_rows.append(
                {"page_id": page_id, "reason": f"text_len_exceeds_cap:{len(text)}>{int(args.max_page_chars)}"}
            )
            continue
        rows.append(
            PageRow(
                page_id=page_id,
                issue_id=_norm(getattr(r, "issue_id", "")),
                issue_date=_norm(getattr(r, "issue_date", "")),
                newspaper_slug=_norm(getattr(r, "newspaper_slug", "")),
                gold_bucket=_norm(getattr(r, "page_class_bucket", "")),
                text_path=text_path,
                text=text,
            )
        )

    if not rows:
        raise SystemExit("No rows to process after loading and hard-cap checks.")

    sample_df = pd.DataFrame(
        [
            {
                "page_id": x.page_id,
                "issue_id": x.issue_id,
                "issue_date": x.issue_date,
                "newspaper_slug": x.newspaper_slug,
                "gold_bucket": x.gold_bucket,
                "text_path": str(x.text_path),
                "text_chars": len(x.text),
            }
            for x in rows
        ]
    )
    sample_df.to_csv(out_dir / "sample_pages.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    if hard_fail_rows:
        pd.DataFrame(hard_fail_rows).to_csv(out_dir / "sample_hard_failures.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    variant_builders = {
        "A": _build_prompt_variant_a,
        "B": _build_prompt_variant_b,
        "C": _build_prompt_variant_c,
        "D": _build_prompt_variant_d,
        "E": _build_prompt_variant_e,
    }
    selected_variants = [_norm(x).upper() for x in str(args.variants).split(",") if _norm(x)]
    selected_variants = [v for v in selected_variants if v in variant_builders]
    if not selected_variants:
        raise SystemExit("No valid variants selected. Use --variants from {A,B,C,D,E}.")

    for vid in selected_variants:
        prompt_builder = variant_builders[vid]
        req_dir = out_dir / f"variant_{vid}" / "requests"
        res_dir = out_dir / f"variant_{vid}" / "results"
        req_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)

        req_rows: list[dict[str, Any]] = []
        mapping_rows: list[dict[str, Any]] = []
        for x in rows:
            cid = f"ab_page_{vid}::{x.page_id}"
            prompt = prompt_builder(
                page_id=x.page_id,
                issue_id=x.issue_id,
                issue_date=x.issue_date,
                newspaper_slug=x.newspaper_slug,
                text=x.text,
            )
            req_rows.append(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": str(args.model).split(":", 1)[1] if ":" in str(args.model) else str(args.model),
                        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                        "max_output_tokens": int(args.max_output_tokens),
                        "stream": False,
                    },
                }
            )
            mapping_rows.append(
                {
                    "custom_id": cid,
                    "page_id": x.page_id,
                    "issue_id": x.issue_id,
                    "issue_date": x.issue_date,
                    "newspaper_slug": x.newspaper_slug,
                    "gold_bucket": x.gold_bucket,
                    "text_chars": len(x.text),
                    "text_sha256": hashlib.sha256(x.text.encode("utf-8")).hexdigest(),
                }
            )
        _write_jsonl(req_dir / "openai_requests_shard000.jsonl", req_rows)
        _write_jsonl(req_dir / "mapping_shard000.jsonl", mapping_rows)

        if not args.no_run:
            _run_gateway(
                request_dir=req_dir,
                output_dir=res_dir,
                model=str(args.model),
                runner_path=Path(args.gateway_runner).expanduser().resolve(),
                gateway_pythonpath=Path(args.gateway_pythonpath).expanduser().resolve(),
                gov_env_path=Path(args.gov_env_path).expanduser().resolve(),
                timeout=float(args.timeout),
                concurrency=int(args.concurrency),
                skip_existing=bool(args.skip_existing),
            )

    if args.no_run:
        print(f"prepared_requests_only out={out_dir}")
        return

    # Parse and score each variant.
    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for vid in selected_variants:
        req_dir = out_dir / f"variant_{vid}" / "requests"
        res_dir = out_dir / f"variant_{vid}" / "results"
        mapping = [json.loads(x) for x in (req_dir / "mapping_shard000.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        map_df = pd.DataFrame(mapping)
        result_map = _read_result_jsonl(res_dir)
        error_map = _read_error_jsonl(res_dir)

        parsed_rows: list[dict[str, Any]] = []
        for r in mapping:
            cid = _norm(r.get("custom_id"))
            page_id = _norm(r.get("page_id"))
            output_text = ""
            had_error = 0
            if cid in result_map:
                body = (((result_map[cid].get("response") or {}).get("body")) or {})
                output_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
            elif cid in error_map:
                had_error = 1
                body = (((error_map[cid].get("response") or {}).get("body")) or {})
                if isinstance(body, dict):
                    output_text = _norm(((body.get("error") or {}).get("message")))
            parsed = _parse_json_from_text(output_text)
            parse_ok = 1 if isinstance(parsed, dict) and parsed else 0
            obj = _coerce_result(parsed if isinstance(parsed, dict) else {}, page_id=page_id)
            obj["custom_id"] = cid
            obj["parse_ok"] = parse_ok
            obj["had_error"] = had_error
            parsed_rows.append(obj)

        pred_df = pd.DataFrame(parsed_rows)
        pred_df["pred_bucket"] = pred_df["legal_object_type"].map(PRED_TO_BUCKET).fillna("uncertain_other")
        joined = map_df.merge(pred_df, on=["custom_id", "page_id"], how="left")
        joined["gold_eval"] = joined["gold_bucket"].map(GOLD_TO_EVAL).fillna("uncertain_other")
        joined["pred_eval"] = joined["pred_bucket"].map(GOLD_TO_EVAL).fillna("uncertain_other")
        joined["is_match"] = joined["gold_eval"] == joined["pred_eval"]

        joined.to_csv(out_dir / f"variant_{vid}_joined_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        review_df = joined[
            joined["pred_eval"].isin(["full_ordinance", "amendment", "proposal_full_text", "notice"])
            & (joined["pred_eval"] != joined["gold_eval"])
        ].copy()
        if not review_df.empty:
            review_df = review_df.sort_values(["pred_eval", "confidence_0_to_1"], ascending=[True, False])
        review_df.to_csv(out_dir / f"variant_{vid}_priority_review.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        # confusion table
        conf = (
            joined.groupby(["gold_eval", "pred_eval"], dropna=False)["page_id"]
            .count()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        conf.to_csv(out_dir / f"variant_{vid}_confusion.csv", index=False, quoting=csv.QUOTE_MINIMAL)

        acc = float(joined["is_match"].mean())
        parse_ok_share = float(joined["parse_ok"].mean())
        error_share = float(joined["had_error"].mean())
        summary_rows.append(
            {
                "variant": vid,
                "n_pages": int(len(joined)),
                "accuracy_eval_bucket": acc,
                "parse_ok_share": parse_ok_share,
                "error_share": error_share,
            }
        )

        targets = sorted(set(joined["gold_eval"]))
        for t in targets:
            m = _binary_metrics(joined.rename(columns={"gold_eval": "gold_eval", "pred_eval": "pred_eval"}), t)
            m["variant"] = vid
            metric_rows.append(m)

    summary_df = pd.DataFrame(summary_rows).sort_values("variant")
    summary_df.to_csv(out_dir / "ab_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    metric_df = pd.DataFrame(metric_rows).sort_values(["variant", "target"])
    metric_df.to_csv(out_dir / "ab_binary_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "label_csv": str(label_csv),
        "side_by_side_root": str(side_root),
        "sample_size_requested": int(args.sample_size),
        "sample_size_used": int(len(rows)),
        "model": str(args.model),
        "gov_env_path": str(Path(args.gov_env_path).expanduser().resolve()),
        "variants": selected_variants,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best = summary_df.sort_values("accuracy_eval_bucket", ascending=False).head(1)
    print("done_page_label_ab_test")
    print(summary_df.to_string(index=False))
    if not best.empty:
        print(f"best_variant={best.iloc[0]['variant']} acc={best.iloc[0]['accuracy_eval_bucket']:.3f}")


if __name__ == "__main__":
    main()
