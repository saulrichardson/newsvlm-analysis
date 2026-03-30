#!/usr/bin/env python3
"""
Run full-context LLM Q&A over newspaper timelines.

Design goals:
- Whole-context by default (all selected issues + full transcripts in one request).
- No clipping/condensing fallback. If --max-panel-chars > 0 and exceeded, fail.
- Compare multiple workflow modes (full-only, amendment-only, full+amendment, etc.).
- Compare multiple question packs.
- Run via gateway and emit parsed/summary artifacts for manual validation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

from fullnewspaper_v3_common import (
    eprint,
    extract_openai_output_text,
    norm_str,
    parse_json_from_text,
    read_error_jsonl,
    read_result_jsonl,
    run_gateway_requests,
)


WORKFLOW_MODE_LABELS: dict[str, set[str]] = {
    "full_only": {"full_ordinance"},
    "amendment_only": {"amendment"},
    "full_plus_amendment": {"full_ordinance", "amendment"},
    "all_legal_plus_context": {
        "full_ordinance",
        "amendment",
        "notice",
        "building_code_or_other_law",
        "zoning_narrative_nonverbatim",
        "uncertain",
    },
}

QUESTION_PACKS: dict[str, list[dict[str, str]]] = {
    "core_v1": [
        {
            "id": "q1_first_full_ordinance",
            "question": (
                "Identify the earliest issue that most likely contains full-ordinance text in this workflow scope. "
                "If absent, return not_found."
            ),
        },
        {
            "id": "q2_amendment_waves",
            "question": (
                "Describe major amendment waves over time (dates, substantive focus, and whether changes appear "
                "expansive, restrictive, clarifying, or mixed)."
            ),
        },
        {
            "id": "q3_regulatory_focus_shift",
            "question": (
                "How did topical emphasis shift over time across: land use restrictions, bulk standards, procedural "
                "governance, parking/transport, nonconforming use, definitions, environmental protection, flexible zoning, "
                "historic preservation, inclusionary/affordable housing, enforcement, miscellaneous site rules?"
            ),
        },
        {
            "id": "q4_numeric_constraints",
            "question": (
                "What direction do numeric constraints appear to move over time for lot size, height, setback, and parking? "
                "Cite issue evidence and note uncertainty."
            ),
        },
        {
            "id": "q5_candidate_measures",
            "question": (
                "Propose 3 concrete panel-level measures to track this newspaper's zoning trajectory, each with a "
                "definition and why it is useful."
            ),
        },
    ],
    "measurement_v1": [
        {
            "id": "q1_measure_first_signal",
            "question": (
                "Return first zoning signal date and issue_id in scope, and classify signal type as "
                "mention|amendment|full_ordinance|none."
            ),
        },
        {
            "id": "q2_measure_first_full",
            "question": (
                "Return first full-ordinance date and issue_id in scope. If uncertain, explain what evidence is missing."
            ),
        },
        {
            "id": "q3_measure_stage_profile",
            "question": (
                "Split the timeline in issue-order terciles and summarize dominant categories in early/mid/late stages."
            ),
        },
        {
            "id": "q4_measure_amendment_intensity",
            "question": (
                "Quantify amendment intensity qualitatively by decade in scope (low/medium/high) with supporting evidence."
            ),
        },
        {
            "id": "q5_measure_reliability_notes",
            "question": (
                "List the strongest risks to interpretation caused by OCR artifacts, missing context, or ambiguous legal status."
            ),
        },
    ],
    "contract_v2_demo": [
        {
            "id": "q1_first_enacted_full_date",
            "question": (
                "Give the earliest date in scope with enacted full-ordinance language. "
                "Return a scalar date answer and cite issue_id/date evidence."
            ),
        },
        {
            "id": "q2_max_min_lot_size_single_family_sqft",
            "question": (
                "What is the maximum minimum lot size requirement (single-family context) observed in scope? "
                "Return a scalar numeric answer in sqft and cite evidence."
            ),
        },
        {
            "id": "q3_parking_requirement_direction",
            "question": (
                "Classify overall parking requirement direction over time as up|down|flat|mixed|uncertain "
                "and provide brief evidence."
            ),
        },
        {
            "id": "q4_full_vs_amendment_counts_by_decade",
            "question": (
                "Return decade-level counts for full_ordinance and amendment rows in scope as timeseries records."
            ),
        },
        {
            "id": "q5_min_lot_size_timeseries",
            "question": (
                "Return dated observations for minimum lot size (single-family context where available) "
                "as a timeseries with numeric values and quotes."
            ),
        },
    ],
    "single_family_zoning_v1": [
        {
            "id": "q1_min_lot_size_bounds_over_time",
            "question": (
                "Return answer_type=timeseries. For each period, emit records for single-family lot-size bounds: "
                "one record with district_or_context=smallest_minimum and one with district_or_context=largest_minimum. "
                "Set value_number to lot size, unit=sqft, value_text to district name, and include issue/date/quote evidence."
            ),
        },
        {
            "id": "q2_front_setback_bounds_over_time",
            "question": (
                "Return answer_type=timeseries. For each period, emit records for single-family front-setback bounds: "
                "one record with district_or_context=smallest_minimum and one with district_or_context=largest_minimum. "
                "Set value_number to setback, unit=ft, value_text to district name, and include issue/date/quote evidence."
            ),
        },
        {
            "id": "q3_tightening_source",
            "question": (
                "Return answer_type=timeseries. For each period, emit one record with district_or_context=tightening_source and "
                "value_category in {strictest,least_strict,both,none,uncertain}. "
                "Classify based on whether tightening comes from the strictest requirement, least strict requirement, both, or neither."
            ),
        },
        {
            "id": "q4_internal_segmentation_gaps",
            "question": (
                "Return answer_type=timeseries. For each period, emit two numeric gap records: "
                "district_or_context=lot_size_gap with unit=sqft and "
                "district_or_context=front_setback_gap with unit=ft. "
                "Each value_number is (largest bound minus smallest bound) among single-family districts."
            ),
        },
        {
            "id": "q5_site_plan_review_discretion",
            "question": (
                "Return answer_type=timeseries. For each period, emit one record with district_or_context=site_plan_review and "
                "value_category in {required_general_residential,required_single_family_pathway,not_required,uncertain}. "
                "Put trigger + decision body in value_text and cite issue/date/quote."
            ),
        },
        {
            "id": "q6_use_variance_availability",
            "question": (
                "Return answer_type=timeseries. For each period, emit one record with district_or_context=use_variance and "
                "value_category in {available,not_available,uncertain}. "
                "Put legal standard + deciding body in value_text and cite issue/date/quote."
            ),
        },
        {
            "id": "q7_procedure_substitutes_for_flexibility",
            "question": (
                "Return answer_type=categorical with value_category in {yes,no,mixed,uncertain}. "
                "Assess whether the timeline shows more discretionary procedure (site-plan review) alongside less flexibility "
                "(use-variance elimination/tightening). Use answer for a short justification with evidence citations."
            ),
        },
        {
            "id": "q8_major_break_events",
            "question": (
                "Return answer_type=timeseries. Emit up to three major-break records with "
                "district_or_context in {major_break_lot_size,major_break_front_setback,major_break_site_plan_review}. "
                "Each record should include date/issue_id, short one-sentence change summary in value_text, and quote evidence."
            ),
        },
    ],
}

QUESTION_EXPECTED_TYPES: dict[str, dict[str, str]] = {
    "contract_v2_demo": {
        "q1_first_enacted_full_date": "date",
        "q2_max_min_lot_size_single_family_sqft": "number",
        "q3_parking_requirement_direction": "categorical",
        "q4_full_vs_amendment_counts_by_decade": "timeseries",
        "q5_min_lot_size_timeseries": "timeseries",
    },
    "single_family_zoning_v1": {
        "q1_min_lot_size_bounds_over_time": "timeseries",
        "q2_front_setback_bounds_over_time": "timeseries",
        "q3_tightening_source": "timeseries",
        "q4_internal_segmentation_gaps": "timeseries",
        "q5_site_plan_review_discretion": "timeseries",
        "q6_use_variance_availability": "timeseries",
        "q7_procedure_substitutes_for_flexibility": "categorical",
        "q8_major_break_events": "timeseries",
    },
}

TYPED_CONTRACT_PACKS = {"contract_v2_demo", "single_family_zoning_v1"}

SCALAR_ALLOWED_CATEGORIES: dict[str, dict[str, set[str]]] = {
    "contract_v2_demo": {
        "q3_parking_requirement_direction": {"up", "down", "flat", "mixed", "uncertain"},
    },
    "single_family_zoning_v1": {
        "q7_procedure_substitutes_for_flexibility": {"yes", "no", "mixed", "uncertain"},
    },
}

RECORD_ALLOWED_CATEGORIES: dict[str, dict[str, set[str]]] = {
    "single_family_zoning_v1": {
        "q3_tightening_source": {"strictest", "least_strict", "both", "none", "uncertain"},
        "q5_site_plan_review_discretion": {
            "required_general_residential",
            "required_single_family_pathway",
            "not_required",
            "uncertain",
        },
        "q6_use_variance_availability": {"available", "not_available", "uncertain"},
    }
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run full-context panel Q&A workflows via gateway.")
    ap.add_argument(
        "--package-dir",
        default="/Users/saulrichardson/Dropbox/Inclusionary Zoning/saul/newspapers/page_artifact_packets_v5_full_20260225",
        help="Folder containing classified_pages/ and metadata_core/.",
    )
    ap.add_argument(
        "--metadata-dir",
        default="",
        help="Override metadata folder (default: <package-dir>/metadata_core).",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/panel_qa_workflow_v1",
    )
    ap.add_argument("--newspapers", default="", help="Comma-separated slugs or @file path.")
    ap.add_argument(
        "--top-newspapers",
        type=int,
        default=0,
        help="If --newspapers omitted, auto-pick top N by (full+amendment issue count). 0 means all newspapers with full/amendment rows.",
    )
    ap.add_argument(
        "--workflow-modes",
        default="full_only,amendment_only,full_plus_amendment",
        help=f"Comma-separated modes. Allowed: {','.join(sorted(WORKFLOW_MODE_LABELS))}",
    )
    ap.add_argument(
        "--question-packs",
        default="core_v1,measurement_v1",
        help=f"Comma-separated packs. Allowed: {','.join(sorted(QUESTION_PACKS))}",
    )
    ap.add_argument("--model", default="openai:gpt-5-mini")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=360.0)
    ap.add_argument(
        "--max-panel-chars",
        type=int,
        default=0,
        help="Hard cap only. 0 means no cap. If positive and exceeded, run fails (no clipping).",
    )
    ap.add_argument(
        "--max-panel-tokens",
        type=int,
        default=0,
        help="Hard token cap per request. 0 means disabled. If positive, requests are split into chronological chunks up to this budget.",
    )
    ap.add_argument(
        "--token-encoding",
        default="o200k_base",
        help="Tokenizer encoding name used for --max-panel-tokens budgeting (default: o200k_base).",
    )
    ap.add_argument(
        "--gateway-runner",
        default=str(Path(__file__).resolve().parents[1] / "scripts" / "run_openai_requests_via_gateway.py"),
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default=str(Path(__file__).resolve().parents[1] / "agent-gateway" / "src"),
    )
    ap.add_argument(
        "--gov-env-path",
        default="/Users/saulrichardson/projects/credit-agreement-extraction/.env",
        help="Env file used to resolve gateway API keys.",
    )
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in norm_str(raw).split(",") if x.strip()]


def _parse_slug_spec(raw: str) -> set[str]:
    spec = norm_str(raw)
    if not spec:
        return set()
    if spec.startswith("@"):
        p = Path(spec[1:]).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Newspaper list file not found: {p}")
        vals = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = norm_str(line).lower()
            if not s or s.startswith("#"):
                continue
            vals.append(s)
        return set(vals)
    return {x.strip().lower() for x in spec.split(",") if x.strip()}


def _norm_json_obj(raw: str) -> dict[str, Any] | None:
    s = norm_str(raw)
    if not s:
        return None
    s = re.sub(r"\bNaN\b", "null", s)
    s = re.sub(r"\bInfinity\b", "null", s)
    s = re.sub(r"\b-Infinity\b", "null", s)
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _answer_to_text(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return norm_str(val)
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return norm_str(str(val))


def _safe_optional_float(val: Any) -> float | None:
    if val is None:
        return None
    s = norm_str(val)
    if not s:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _extract_iso_date(text: Any) -> str:
    s = norm_str(text)
    if not s:
        return ""
    m = re.search(r"\b(18|19|20)\d{2}-\d{2}-\d{2}\b", s)
    return m.group(0) if m else ""


def _extract_first_number(text: Any) -> float | None:
    s = norm_str(text)
    if not s:
        return None
    m = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return None


def _coerce_category_value(raw: Any, allowed: set[str], *, default: str = "uncertain") -> str:
    s = norm_str(raw).lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"__+", "_", s).strip("_")
    if not allowed:
        return s
    if s in allowed:
        return s
    # Minimal synonym normalization.
    aliases = {
        "unknown": "uncertain",
        "na": "uncertain",
        "n_a": "uncertain",
        "not_sure": "uncertain",
        "unclear": "uncertain",
        "required": "required_general_residential",
        "notrequired": "not_required",
        "notavailable": "not_available",
        "available_yes": "available",
        "available_no": "not_available",
        "strict": "strictest",
        "least": "least_strict",
    }
    s2 = aliases.get(s, s)
    if s2 in allowed:
        return s2
    return default if default in allowed else sorted(allowed)[0]


def _coerce_district_context(question_id: str, raw: Any, value_text: Any) -> str:
    s = norm_str(raw).lower().replace("-", "_").replace(" ", "_")
    t = norm_str(value_text).lower()
    if question_id in {"q1_min_lot_size_bounds_over_time", "q2_front_setback_bounds_over_time"}:
        if s in {"smallest_minimum", "largest_minimum"}:
            return s
        if "small" in s or "small" in t or "least" in t or "minimum" in t:
            return "smallest_minimum"
        if "large" in s or "max" in s or "strict" in t:
            return "largest_minimum"
    if question_id == "q4_internal_segmentation_gaps":
        if s in {"lot_size_gap", "front_setback_gap"}:
            return s
        if "setback" in s or "setback" in t:
            return "front_setback_gap"
        if "lot" in s or "lot" in t:
            return "lot_size_gap"
    if question_id == "q8_major_break_events":
        allowed = {"major_break_lot_size", "major_break_front_setback", "major_break_site_plan_review"}
        if s in allowed:
            return s
        if "setback" in s or "setback" in t:
            return "major_break_front_setback"
        if "site" in s or "plan" in s or "review" in s or "site" in t or "plan" in t:
            return "major_break_site_plan_review"
        if "lot" in s or "lot" in t:
            return "major_break_lot_size"
    return norm_str(raw)


def _coerce_answer_type(raw: Any, expected: str) -> str:
    s = norm_str(raw).lower()
    allowed = {"number", "text", "categorical", "timeseries", "date", "boolean", "distribution", "list", "uncertain"}
    if expected:
        return expected
    return s if s in allowed else ""


def _as_iso_date(text: Any) -> str:
    return _extract_iso_date(text)


def _extract_json_array_for_key(text: str, key: str) -> list[Any] | None:
    s = norm_str(text)
    if not s:
        return None
    key_token = f'"{norm_str(key)}"'
    i = s.find(key_token)
    if i < 0:
        return None
    j = s.find("[", i)
    if j < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    end = -1
    for k in range(j, len(s)):
        ch = s[k]
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
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = k
                break
    snippet = s[j : end + 1] if end >= 0 else s[j:]
    if not snippet:
        return None
    # If the array is truncated, close open strings/brackets/braces before parsing.
    if end < 0:
        stack: list[str] = []
        in_str = False
        esc = False
        for ch in snippet:
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
            if ch in "[{":
                stack.append(ch)
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
        if in_str:
            snippet += '"'
        if stack:
            snippet += "".join("]" if ch == "[" else "}" for ch in reversed(stack))
    try:
        arr = json.loads(snippet)
    except Exception:
        return None
    return arr if isinstance(arr, list) else None


def _page_num_from_id(page_id: str) -> int:
    s = norm_str(page_id).lower()
    m = re.search(r"-p-(\d+)$", s)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _load_dataset(metadata_dir: Path, package_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues_path = metadata_dir / "issues.csv"
    pages_path = metadata_dir / "pages.csv"
    if not issues_path.is_file() or not pages_path.is_file():
        raise SystemExit(f"Missing metadata files in {metadata_dir}; need issues.csv and pages.csv")
    issues = pd.read_csv(issues_path, low_memory=False)
    pages = pd.read_csv(pages_path, low_memory=False)
    if "issue_id" not in issues.columns:
        raise SystemExit(f"issues.csv missing issue_id: {issues_path}")
    if "issue_id" not in pages.columns or "page_id" not in pages.columns:
        raise SystemExit(f"pages.csv missing issue_id/page_id: {pages_path}")
    issues["newspaper_slug"] = issues["newspaper_slug"].astype(str).str.lower().str.strip()
    issues["primary_label"] = issues["primary_label"].astype(str).str.strip()
    issues["issue_date"] = issues["issue_date"].astype(str).str.strip()
    pages["issue_id"] = pages["issue_id"].astype(str).str.strip()
    pages["page_id"] = pages["page_id"].astype(str).str.strip()
    pages["primary_label"] = pages["primary_label"].astype(str).str.strip()
    pages["label"] = pages["label"].astype(str).str.strip()
    pages["page_num"] = pages["page_id"].map(_page_num_from_id)
    pages["transcript_abs_path"] = pages["transcript_rel_path"].map(lambda p: str((package_dir / norm_str(p)).resolve()))
    pages["classification_abs_path"] = pages["classification_rel_path"].map(lambda p: str((package_dir / norm_str(p)).resolve()))
    return issues, pages


def _choose_newspapers(issues: pd.DataFrame, requested: set[str], top_n: int) -> list[str]:
    if requested:
        return sorted([x for x in requested if x in set(issues["newspaper_slug"].astype(str))])
    d = issues[issues["primary_label"].isin({"full_ordinance", "amendment"})].copy()
    if d.empty:
        return []
    g = (
        d.groupby(["newspaper_slug", "primary_label"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )
    if "full_ordinance" not in g.columns:
        g["full_ordinance"] = 0
    if "amendment" not in g.columns:
        g["amendment"] = 0
    g["score"] = g["full_ordinance"].astype(int) + g["amendment"].astype(int)
    g = g.sort_values(["score", "full_ordinance", "amendment", "newspaper_slug"], ascending=[False, False, False, True]).reset_index(drop=True)
    if top_n > 0:
        g = g.head(int(top_n))
    return [str(x) for x in g["newspaper_slug"].tolist()]


def _build_issue_rows_for_mode(
    *,
    newspaper_slug: str,
    issues: pd.DataFrame,
    pages: pd.DataFrame,
    allowed_labels: set[str],
) -> list[dict[str, Any]]:
    i = issues[(issues["newspaper_slug"] == newspaper_slug) & (issues["primary_label"].isin(allowed_labels))].copy()
    if i.empty:
        return []
    i = i.sort_values(["issue_date", "issue_id"]).reset_index(drop=True)
    page_map = pages[pages["issue_id"].isin(set(i["issue_id"]))].copy()
    page_map = page_map.sort_values(["issue_id", "page_num", "page_id"]).reset_index(drop=True)

    out: list[dict[str, Any]] = []
    for r in i.itertuples(index=False):
        issue_id = str(getattr(r, "issue_id"))
        pp = page_map[page_map["issue_id"] == issue_id]
        page_rows: list[dict[str, Any]] = []
        chunks: list[str] = []
        for p in pp.itertuples(index=False):
            transcript_path = Path(str(getattr(p, "transcript_abs_path")))
            txt = _read_text(transcript_path)
            class_path = Path(str(getattr(p, "classification_abs_path")))
            class_obj = _norm_json_obj(_read_text(class_path)) or {}
            page_rows.append(
                {
                    "page_id": str(getattr(p, "page_id")),
                    "page_num": int(getattr(p, "page_num") or 0),
                    "label": str(getattr(p, "label") or getattr(p, "primary_label") or ""),
                    "classification": class_obj,
                    "transcript_chars": len(txt),
                }
            )
            block = "\n".join(
                [
                    f"[PAGE] page_id={str(getattr(p, 'page_id'))} page_num={int(getattr(p, 'page_num') or 0)} label={str(getattr(p, 'label') or getattr(p, 'primary_label') or '')}",
                    "[TRANSCRIPT]",
                    txt,
                ]
            )
            chunks.append(block)
        issue_text = "\n\n".join(chunks).strip()
        out.append(
            {
                "issue_id": issue_id,
                "issue_date": str(getattr(r, "issue_date") or ""),
                "primary_label": str(getattr(r, "primary_label") or ""),
                "page_count": int(len(page_rows)),
                "issue_text": issue_text,
                "issue_chars": len(issue_text),
                "pages": page_rows,
            }
        )
    return out


def _question_pack_text(name: str) -> str:
    if name not in QUESTION_PACKS:
        raise SystemExit(f"Unknown question pack '{name}'. Allowed: {sorted(QUESTION_PACKS)}")
    lines: list[str] = []
    type_hints = QUESTION_EXPECTED_TYPES.get(name, {})
    for q in QUESTION_PACKS[name]:
        t = norm_str(type_hints.get(q["id"]))
        if t:
            lines.append(f"- {q['id']} [{t}]: {q['question']}")
        else:
            lines.append(f"- {q['id']}: {q['question']}")
    return "\n".join(lines)


def _schema_template(newspaper_slug: str, workflow_mode: str, question_pack: str) -> dict[str, Any]:
    qids = [x["id"] for x in QUESTION_PACKS[question_pack]]
    return {
        "newspaper_slug": newspaper_slug,
        "workflow_mode": workflow_mode,
        "question_pack": question_pack,
        "coverage": {
            "issue_count": 0,
            "start_issue_date": "",
            "end_issue_date": "",
            "labels_in_scope": [],
        },
        "timeline_facts": {
            "first_zoning_signal": {
                "issue_id": "",
                "issue_date": "",
                "signal_type": "none",
                "confidence_0_to_1": 0.0,
                "evidence_quotes": [],
            },
            "first_full_ordinance": {
                "status": "not_found",
                "issue_id": "",
                "issue_date": "",
                "confidence_0_to_1": 0.0,
                "evidence_quotes": [],
            },
            "amendment_waves": [],
            "regulatory_focus_shifts": [],
            "numeric_trajectory": {
                "lot_size": {"direction": "uncertain", "notes": ""},
                "height": {"direction": "uncertain", "notes": ""},
                "setback": {"direction": "uncertain", "notes": ""},
                "parking": {"direction": "uncertain", "notes": ""},
            },
        },
        "answers": [
            {
                "question_id": qids[0] if qids else "q1",
                "answer": "",
                "confidence_0_to_1": 0.0,
                "evidence_issue_ids": [],
                "evidence_quotes": [],
            }
        ],
        "quality_notes": [],
    }


def _build_prompt(
    *,
    newspaper_slug: str,
    workflow_mode: str,
    question_pack: str,
    issue_rows: list[dict[str, Any]],
    max_panel_chars: int,
) -> tuple[str, int]:
    total_chars = int(sum(int(x.get("issue_chars") or 0) for x in issue_rows))
    if int(max_panel_chars) > 0 and total_chars > int(max_panel_chars):
        raise SystemExit(
            f"Panel newspaper={newspaper_slug} mode={workflow_mode} chars={total_chars} exceeds hard cap {int(max_panel_chars)}. "
            "No clipping fallback is enabled."
        )

    min_date = min((norm_str(x.get("issue_date")) for x in issue_rows if norm_str(x.get("issue_date"))), default="")
    max_date = max((norm_str(x.get("issue_date")) for x in issue_rows if norm_str(x.get("issue_date"))), default="")
    labels = sorted({norm_str(x.get("primary_label")) for x in issue_rows if norm_str(x.get("primary_label"))})
    schema = _schema_template(newspaper_slug, workflow_mode, question_pack)

    lines: list[str] = []
    lines.append("You are answering zoning-law timeline questions from OCR transcripts of newspaper pages.")
    lines.append("Return ONLY one JSON object. Do not include markdown.")
    lines.append("")
    lines.append("Important constraints:")
    lines.append("- These are OCR transcripts and can contain OCR errors.")
    lines.append("- Use only issue_ids present in the timeline when citing evidence_issue_ids.")
    lines.append("- If uncertain, say uncertain explicitly; do not fabricate.")
    lines.append("- Preserve chronology when reasoning about first events and changes over time.")
    lines.append("- Focus on verbatim legal language when possible; distinguish narrative reporting from legal text.")
    lines.append("- first_zoning_signal.signal_type must be one of: none|mention|amendment|full_ordinance.")
    lines.append("- first_full_ordinance.status must be one of: found|not_found|uncertain.")
    lines.append("- Keep output concise to avoid truncation: target <= 8,000 characters total.")
    lines.append("- Keep each answer field short (about 60-120 words) and cap evidence_issue_ids to <= 6 per answer.")
    lines.append("- Never leave answers blank. If a question is not answerable in scope, use: 'not_in_scope: <brief reason>'.")
    lines.append("- Do not use whitespace-only placeholder answers.")
    lines.append("- Prefer numeric and categorical outputs when possible so results are directly usable for downstream plots.")
    if question_pack in TYPED_CONTRACT_PACKS:
        lines.append("- For each answer, set answer_type and fill typed value fields consistently.")
        lines.append("- Use records[] for dated observations. Keep records sorted by date ascending.")
        lines.append("- Each record must include date, issue_id, issue_date, and newspaper_slug.")
        lines.append("- If no observation exists, return empty records[] and explain briefly in answer.")
        lines.append("- For answer_type=date, fill value_date as YYYY-MM-DD (non-empty when known).")
        lines.append("- For answer_type=number, fill value_number (non-null when known) and unit where applicable.")
        lines.append("- For answer_type=categorical, fill value_category with the category value.")
        lines.append("- Do not place structured JSON blobs inside answer; use the typed fields and records.")
    if question_pack == "single_family_zoning_v1":
        lines.append("- Periods should be chronological bins based on issue dates in scope (prefer decade bins if feasible).")
        lines.append("- For lot-size/setback bound questions, emit two records per period using district_or_context labels:")
        lines.append("  smallest_minimum and largest_minimum; set value_number to the requirement and value_text to district name.")
        lines.append("- For tightening-source, set value_category to strictest|least_strict|both|none|uncertain for each period.")
        lines.append("- For segmentation gaps, emit records with district_or_context=lot_size_gap or front_setback_gap and numeric value_number.")
        lines.append("- For site-plan review, set value_category to required_general_residential|required_single_family_pathway|not_required|uncertain.")
        lines.append("- For use-variance, set value_category to available|not_available|uncertain.")
        lines.append("- For major break events, use district_or_context labels: major_break_lot_size|major_break_front_setback|major_break_site_plan_review.")
        lines.append("- If a requested record is not observable in period text, omit that record and note uncertainty in answer.")
    lines.append("")
    lines.append("Question pack:")
    lines.append(_question_pack_text(question_pack))
    lines.append("")
    lines.append("JSON schema template:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Scope metadata: newspaper_slug={newspaper_slug} workflow_mode={workflow_mode} "
        f"issue_count={len(issue_rows)} date_range={min_date}..{max_date} labels={','.join(labels)}"
    )
    lines.append("")
    lines.append("Chronological issue timeline (full text, no clipping):")
    for idx, r in enumerate(issue_rows, start=1):
        lines.append("")
        lines.append(
            f"[ISSUE {idx}] issue_id={norm_str(r.get('issue_id'))} date={norm_str(r.get('issue_date'))} "
            f"primary_label={norm_str(r.get('primary_label'))} page_count={int(r.get('page_count') or 0)}"
        )
        lines.append("[ISSUE_FULL_TEXT_START]")
        lines.append(norm_str(r.get("issue_text")))
        lines.append("[ISSUE_FULL_TEXT_END]")
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip(), total_chars


def _get_token_encoder(name: str) -> Any:
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(str(name or "o200k_base"))
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def _count_tokens(text: str, encoder: Any) -> int:
    s = norm_str(text)
    if not s:
        return 0
    if encoder is None:
        # Fallback approximation when tokenizer package is unavailable.
        return int(math.ceil(len(s) / 4.0))
    try:
        return int(len(encoder.encode(s)))
    except Exception:
        return int(math.ceil(len(s) / 4.0))


def _chunk_rows_by_token_budget(
    *,
    newspaper_slug: str,
    workflow_mode: str,
    question_pack: str,
    issue_rows: list[dict[str, Any]],
    max_panel_chars: int,
    max_panel_tokens: int,
    token_encoder: Any,
) -> list[dict[str, Any]]:
    if not issue_rows:
        return []
    if int(max_panel_tokens) <= 0:
        prompt, prompt_chars = _build_prompt(
            newspaper_slug=newspaper_slug,
            workflow_mode=workflow_mode,
            question_pack=question_pack,
            issue_rows=issue_rows,
            max_panel_chars=int(max_panel_chars),
        )
        return [
            {
                "chunk_index": 1,
                "issue_rows": issue_rows,
                "prompt": prompt,
                "prompt_chars": int(prompt_chars),
                "prompt_tokens": int(_count_tokens(prompt, token_encoder)),
            }
        ]

    chunks: list[dict[str, Any]] = []
    n = len(issue_rows)
    start = 0
    while start < n:
        best_end = -1
        best_prompt = ""
        best_chars = 0
        best_tokens = 0
        for end in range(start, n):
            candidate_rows = issue_rows[start : end + 1]
            prompt, prompt_chars = _build_prompt(
                newspaper_slug=newspaper_slug,
                workflow_mode=workflow_mode,
                question_pack=question_pack,
                issue_rows=candidate_rows,
                max_panel_chars=int(max_panel_chars),
            )
            prompt_tokens = _count_tokens(prompt, token_encoder)
            if int(prompt_tokens) <= int(max_panel_tokens):
                best_end = end
                best_prompt = prompt
                best_chars = int(prompt_chars)
                best_tokens = int(prompt_tokens)
                continue
            break

        if best_end < start:
            one = issue_rows[start]
            one_prompt, one_chars = _build_prompt(
                newspaper_slug=newspaper_slug,
                workflow_mode=workflow_mode,
                question_pack=question_pack,
                issue_rows=[one],
                max_panel_chars=int(max_panel_chars),
            )
            one_tokens = _count_tokens(one_prompt, token_encoder)
            raise SystemExit(
                f"Cannot fit single issue within token budget: newspaper={newspaper_slug} mode={workflow_mode} "
                f"pack={question_pack} issue_id={norm_str(one.get('issue_id'))} tokens={one_tokens} cap={int(max_panel_tokens)}"
            )

        chunk_rows = issue_rows[start : best_end + 1]
        chunks.append(
            {
                "chunk_index": int(len(chunks) + 1),
                "issue_rows": chunk_rows,
                "prompt": best_prompt,
                "prompt_chars": int(best_chars),
                "prompt_tokens": int(best_tokens),
            }
        )
        start = best_end + 1

    total_chunks = int(len(chunks))
    for c in chunks:
        c["chunk_count"] = total_chunks
    return chunks


def _panel_qa_text_config(question_pack: str) -> dict[str, Any]:
    if question_pack not in QUESTION_PACKS:
        raise SystemExit(f"Unknown question pack for schema: {question_pack}")
    qids = [q["id"] for q in QUESTION_PACKS[question_pack]]
    contract_mode = question_pack in TYPED_CONTRACT_PACKS
    answer_properties: dict[str, Any] = {
        "question_id": {"type": "string", "enum": qids},
        "answer": {"type": "string", "minLength": 1, "pattern": "\\S"},
        "confidence_0_to_1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence_issue_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_quotes": {"type": "array", "items": {"type": "string"}},
    }
    answer_required = [
        "question_id",
        "answer",
        "confidence_0_to_1",
        "evidence_issue_ids",
        "evidence_quotes",
    ]
    if contract_mode:
        answer_properties.update(
            {
                "answer_type": {
                    "type": "string",
                    "enum": ["number", "text", "categorical", "timeseries", "date", "boolean", "distribution", "list", "uncertain"],
                },
                "value_number": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "value_text": {"type": "string"},
                "value_category": {"type": "string"},
                "value_date": {"type": "string"},
                "value_boolean": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                "unit": {"type": "string"},
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "date": {"type": "string"},
                            "value_number": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "value_category": {"type": "string"},
                            "value_text": {"type": "string"},
                            "unit": {"type": "string"},
                            "district_or_context": {"type": "string"},
                            "newspaper_slug": {"type": "string"},
                            "issue_id": {"type": "string"},
                            "issue_date": {"type": "string"},
                            "page_ids": {"type": "array", "items": {"type": "string"}},
                            "quote": {"type": "string"},
                            "confidence_0_to_1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": [
                            "date",
                            "value_number",
                            "value_text",
                            "unit",
                            "district_or_context",
                            "newspaper_slug",
                            "issue_id",
                            "issue_date",
                            "page_ids",
                            "quote",
                            "confidence_0_to_1",
                        ],
                    },
                },
            }
        )
        answer_required.extend(
            [
                "answer_type",
                "value_number",
                "value_text",
                "value_category",
                "value_date",
                "value_boolean",
                "unit",
                "records",
            ]
        )
    return {
        "format": {
            "type": "json_schema",
            "name": f"panel_qa_{question_pack}_output",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "newspaper_slug": {"type": "string"},
                    "workflow_mode": {"type": "string"},
                    "question_pack": {"type": "string"},
                    "coverage": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "issue_count": {"type": "integer", "minimum": 0},
                            "start_issue_date": {"type": "string"},
                            "end_issue_date": {"type": "string"},
                            "labels_in_scope": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["issue_count", "start_issue_date", "end_issue_date", "labels_in_scope"],
                    },
                    "timeline_facts": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "first_zoning_signal": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "issue_id": {"type": "string"},
                                    "issue_date": {"type": "string"},
                                    "signal_type": {
                                        "type": "string",
                                        "enum": ["none", "mention", "amendment", "full_ordinance"],
                                    },
                                    "confidence_0_to_1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                    "evidence_quotes": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["issue_id", "issue_date", "signal_type", "confidence_0_to_1", "evidence_quotes"],
                            },
                            "first_full_ordinance": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "status": {"type": "string", "enum": ["found", "not_found", "uncertain"]},
                                    "issue_id": {"type": "string"},
                                    "issue_date": {"type": "string"},
                                    "confidence_0_to_1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                    "evidence_quotes": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["status", "issue_id", "issue_date", "confidence_0_to_1", "evidence_quotes"],
                            },
                            "amendment_waves": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                            "regulatory_focus_shifts": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                            "numeric_trajectory": {"type": "object", "additionalProperties": True},
                        },
                        "required": [
                            "first_zoning_signal",
                            "first_full_ordinance",
                            "amendment_waves",
                            "regulatory_focus_shifts",
                            "numeric_trajectory",
                        ],
                    },
                    "answers": {
                        "type": "array",
                        "minItems": len(qids),
                        "maxItems": len(qids),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": answer_properties,
                            "required": answer_required,
                        },
                    },
                    "quality_notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "newspaper_slug",
                    "workflow_mode",
                    "question_pack",
                    "coverage",
                    "timeline_facts",
                    "answers",
                    "quality_notes",
                ],
            },
        }
    }


def _write_openai_requests_with_reasoning(path: Path, rows: list[tuple[str, str, str]], *, model: str, reasoning_effort: str = "high") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_name = str(model).strip()
    provider = ""
    if ":" in model_name:
        p, rest = model_name.split(":", 1)
        provider = p.lower().strip()
        model_name = rest.strip() if rest.strip() else model_name
    if not provider:
        provider = "openai"
    effort = norm_str(reasoning_effort).lower()
    if effort and effort not in {"minimal", "low", "medium", "high"}:
        raise SystemExit(f"Invalid reasoning effort '{reasoning_effort}'")

    with path.open("w", encoding="utf-8") as f:
        for custom_id, prompt, question_pack in rows:
            body: dict[str, Any] = {
                "model": model_name,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                "stream": False,
            }
            if provider == "openai" and effort:
                body["reasoning"] = {"effort": effort}
            if provider == "openai":
                body["text"] = _panel_qa_text_config(str(question_pack))
            req = {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def _coerce_parsed_json(
    obj: dict[str, Any] | None,
    *,
    newspaper_slug: str,
    workflow_mode: str,
    question_pack: str,
    issue_ids: set[str],
    issue_label_by_id: dict[str, str],
    labels_in_scope: list[str],
    issue_count: int,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    out = obj if isinstance(obj, dict) else {}
    answers = out.get("answers")
    if not isinstance(answers, list):
        answers = []
    qid_expected = [q["id"] for q in QUESTION_PACKS[question_pack]]
    expected_types = {k: norm_str(v).lower() for k, v in QUESTION_EXPECTED_TYPES.get(question_pack, {}).items()}
    scalar_allowed_by_q = SCALAR_ALLOWED_CATEGORIES.get(question_pack, {})
    record_allowed_by_q = RECORD_ALLOWED_CATEGORIES.get(question_pack, {})
    qmap: dict[str, dict[str, Any]] = {}
    for a in answers:
        if not isinstance(a, dict):
            continue
        qid = norm_str(a.get("question_id"))
        if not qid:
            continue
        expected_type = expected_types.get(qid, "")
        ev_ids = a.get("evidence_issue_ids")
        if not isinstance(ev_ids, list):
            ev_ids = []
        ev_ids = [norm_str(x) for x in ev_ids if norm_str(x) in issue_ids]
        ev_q = a.get("evidence_quotes")
        if not isinstance(ev_q, list):
            ev_q = []
        ans_text = _answer_to_text(a.get("answer"))
        if not norm_str(ans_text):
            ans_text = "not_answered_by_model"
        recs_raw = a.get("records")
        recs_out: list[dict[str, Any]] = []
        if isinstance(recs_raw, list):
            for rr in recs_raw:
                if not isinstance(rr, dict):
                    continue
                rr_issue_id = norm_str(rr.get("issue_id"))
                if rr_issue_id and rr_issue_id not in issue_ids:
                    continue
                rr_issue_date = norm_str(rr.get("issue_date"))
                rr_date = _as_iso_date(rr.get("date")) or _as_iso_date(rr_issue_date)
                rr_value_text = norm_str(rr.get("value_text"))
                rr_unit = norm_str(rr.get("unit"))
                rr_context = _coerce_district_context(qid, rr.get("district_or_context"), rr_value_text)
                rr_value_category = norm_str(rr.get("value_category"))
                allowed_record_cats = record_allowed_by_q.get(qid, set())
                if allowed_record_cats:
                    rr_value_category = _coerce_category_value(
                        rr_value_category or rr_value_text,
                        allowed_record_cats,
                        default="uncertain",
                    )
                rr_value_number = _safe_optional_float(rr.get("value_number"))
                if rr_value_number is None and expected_type in {"number", "timeseries"}:
                    rr_value_number = _extract_first_number(rr_value_text)
                rr_page_ids = rr.get("page_ids")
                if not isinstance(rr_page_ids, list):
                    rr_page_ids = []
                recs_out.append(
                    {
                        "date": rr_date,
                        "value_number": rr_value_number,
                        "value_category": rr_value_category,
                        "value_text": rr_value_text,
                        "unit": rr_unit,
                        "district_or_context": rr_context,
                        "newspaper_slug": norm_str(rr.get("newspaper_slug")) or newspaper_slug,
                        "issue_id": rr_issue_id,
                        "issue_date": rr_issue_date,
                        "page_ids": [norm_str(x) for x in rr_page_ids if norm_str(x)],
                        "quote": norm_str(rr.get("quote")),
                        "confidence_0_to_1": float(max(0.0, min(1.0, float(rr.get("confidence_0_to_1") or 0.0)))),
                    }
                )
        recs_out = sorted(
            recs_out,
            key=lambda r: (
                _as_iso_date(r.get("date")) or "9999-99-99",
                norm_str(r.get("issue_date")) or "9999-99-99",
                norm_str(r.get("issue_id")),
                norm_str(r.get("district_or_context")),
            ),
        )
        atype = _coerce_answer_type(a.get("answer_type"), expected_type)
        v_num = _safe_optional_float(a.get("value_number"))
        v_text = norm_str(a.get("value_text"))
        v_cat = norm_str(a.get("value_category"))
        v_date = _as_iso_date(a.get("value_date"))
        v_bool = (bool(a.get("value_boolean")) if isinstance(a.get("value_boolean"), bool) else None)
        v_unit = norm_str(a.get("unit"))

        # Contract fallback: recover typed scalar fields from answer text when model leaves typed slots blank.
        if atype == "date" and not v_date:
            v_date = _extract_iso_date(ans_text)
        if atype == "number" and v_num is None:
            v_num = _extract_first_number(ans_text)
        if atype == "categorical":
            allowed_scalar_cats = scalar_allowed_by_q.get(qid, set())
            if allowed_scalar_cats:
                v_cat = _coerce_category_value(v_cat or ans_text, allowed_scalar_cats, default="uncertain")
            elif not v_cat:
                v_cat = norm_str(ans_text).split()[0].lower() if norm_str(ans_text) else ""
        if atype == "boolean" and v_bool is None:
            low = norm_str(ans_text).lower()
            if low.startswith("yes") or low.startswith("true"):
                v_bool = True
            elif low.startswith("no") or low.startswith("false"):
                v_bool = False
        if atype == "timeseries" and recs_out:
            if not v_date:
                v_date = _as_iso_date(recs_out[0].get("date"))
            if v_num is None:
                nums = [_safe_optional_float(r.get("value_number")) for r in recs_out]
                nums = [n for n in nums if n is not None]
                if nums:
                    v_num = float(nums[0])
        if ans_text == "not_answered_by_model":
            if v_date:
                ans_text = v_date
            elif v_num is not None:
                ans_text = str(int(v_num)) if float(v_num).is_integer() else str(v_num)
            elif v_cat:
                ans_text = v_cat
            elif recs_out:
                ans_text = "see records"
            elif v_text:
                ans_text = v_text
        if atype == "timeseries" and norm_str(ans_text) in {"", "not_answered_by_model"} and recs_out:
            ans_text = f"{len(recs_out)} records"
        qmap[qid] = {
            "question_id": qid,
            "answer": ans_text,
            "confidence_0_to_1": float(max(0.0, min(1.0, float(a.get("confidence_0_to_1") or 0.0)))),
            "evidence_issue_ids": ev_ids,
            "evidence_quotes": [norm_str(x) for x in ev_q if norm_str(x)][:6],
            "answer_type": atype,
            "value_number": v_num,
            "value_text": v_text,
            "value_category": v_cat,
            "value_date": v_date,
            "value_boolean": v_bool,
            "unit": v_unit,
            "records": recs_out,
        }
    answers_out: list[dict[str, Any]] = []
    for qid in qid_expected:
        if qid in qmap:
            answers_out.append(qmap[qid])
            continue
        expected_type = expected_types.get(qid, "")
        default_cat = ""
        allowed_scalar = scalar_allowed_by_q.get(qid, set())
        if expected_type == "categorical" and allowed_scalar:
            default_cat = "uncertain" if "uncertain" in allowed_scalar else sorted(allowed_scalar)[0]
        answers_out.append(
            {
                "question_id": qid,
                "answer": "not_answered_by_model",
                "confidence_0_to_1": 0.0,
                "evidence_issue_ids": [],
                "evidence_quotes": [],
                "answer_type": expected_type,
                "value_number": None,
                "value_text": "",
                "value_category": default_cat,
                "value_date": "",
                "value_boolean": None,
                "unit": "",
                "records": [],
            }
        )

    tf = out.get("timeline_facts")
    if not isinstance(tf, dict):
        tf = {}
    ff = tf.get("first_full_ordinance")
    if not isinstance(ff, dict):
        ff = {}
    status = norm_str(ff.get("status")).lower()
    if status not in {"found", "not_found", "uncertain"}:
        status = "uncertain"
    ff_issue = norm_str(ff.get("issue_id"))
    if ff_issue and ff_issue not in issue_ids:
        ff_issue = ""
        status = "uncertain"
    ff_issue_label = norm_str(issue_label_by_id.get(ff_issue, "")).lower() if ff_issue else ""
    if ff_issue and ff_issue_label != "full_ordinance":
        ff_issue = ""
        status = "not_found"
    has_full_in_scope = any(norm_str(v).lower() == "full_ordinance" for v in issue_label_by_id.values())
    if status == "found" and not has_full_in_scope:
        status = "not_found"
        ff_issue = ""
    fzs = tf.get("first_zoning_signal")
    if not isinstance(fzs, dict):
        fzs = {}
    sig_type = norm_str(fzs.get("signal_type")).lower()
    if sig_type not in {"none", "mention", "amendment", "full_ordinance"}:
        sig_type = "none"
    sig_issue = norm_str(fzs.get("issue_id"))
    if sig_issue and sig_issue not in issue_ids:
        sig_issue = ""
        sig_type = "none"
    sig_issue_label = norm_str(issue_label_by_id.get(sig_issue, "")).lower() if sig_issue else ""
    if sig_issue and sig_type == "full_ordinance" and sig_issue_label != "full_ordinance":
        if sig_issue_label == "amendment":
            sig_type = "amendment"
        elif sig_issue_label:
            sig_type = "mention"
        else:
            sig_type = "none"
    ff_issue_date = norm_str(ff.get("issue_date"))
    if status != "found" or not ff_issue:
        ff_issue_date = ""
    sig_issue_date = norm_str(fzs.get("issue_date"))
    if sig_type == "none" or not sig_issue:
        sig_issue_date = ""

    coverage = out.get("coverage")
    if not isinstance(coverage, dict):
        coverage = {}
    quality_notes = out.get("quality_notes")
    if not isinstance(quality_notes, list):
        quality_notes = []

    return {
        "newspaper_slug": newspaper_slug,
        "workflow_mode": workflow_mode,
        "question_pack": question_pack,
        "coverage": {
            "issue_count": int(coverage.get("issue_count") or issue_count),
            "start_issue_date": norm_str(coverage.get("start_issue_date")) or start_date,
            "end_issue_date": norm_str(coverage.get("end_issue_date")) or end_date,
            "labels_in_scope": [norm_str(x) for x in (coverage.get("labels_in_scope") if isinstance(coverage.get("labels_in_scope"), list) else labels_in_scope) if norm_str(x)],
        },
        "timeline_facts": {
            "first_zoning_signal": {
                "issue_id": sig_issue,
                "issue_date": sig_issue_date,
                "signal_type": sig_type,
                "confidence_0_to_1": float(max(0.0, min(1.0, float(fzs.get("confidence_0_to_1") or 0.0)))),
                "evidence_quotes": [norm_str(x) for x in (fzs.get("evidence_quotes") if isinstance(fzs.get("evidence_quotes"), list) else []) if norm_str(x)][:6],
            },
            "first_full_ordinance": {
                "status": status,
                "issue_id": ff_issue,
                "issue_date": ff_issue_date,
                "confidence_0_to_1": float(max(0.0, min(1.0, float(ff.get("confidence_0_to_1") or 0.0)))),
                "evidence_quotes": [norm_str(x) for x in (ff.get("evidence_quotes") if isinstance(ff.get("evidence_quotes"), list) else []) if norm_str(x)][:6],
            },
            "amendment_waves": tf.get("amendment_waves") if isinstance(tf.get("amendment_waves"), list) else [],
            "regulatory_focus_shifts": tf.get("regulatory_focus_shifts") if isinstance(tf.get("regulatory_focus_shifts"), list) else [],
            "numeric_trajectory": tf.get("numeric_trajectory") if isinstance(tf.get("numeric_trajectory"), dict) else {},
        },
        "answers": answers_out,
        "quality_notes": [norm_str(x) for x in quality_notes if norm_str(x)],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve() if norm_str(args.metadata_dir) else (package_dir / "metadata_core")
    output_dir = Path(args.output_dir).expanduser().resolve()
    req_dir = output_dir / "requests"
    res_dir = output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    req_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    issues, pages = _load_dataset(metadata_dir, package_dir)
    requested = _parse_slug_spec(str(args.newspapers))
    slugs = _choose_newspapers(issues, requested=requested, top_n=int(args.top_newspapers))
    if not slugs:
        raise SystemExit("No newspapers selected.")

    modes = _parse_csv_list(str(args.workflow_modes))
    packs = _parse_csv_list(str(args.question_packs))
    bad_modes = [m for m in modes if m not in WORKFLOW_MODE_LABELS]
    bad_packs = [p for p in packs if p not in QUESTION_PACKS]
    if bad_modes:
        raise SystemExit(f"Unknown workflow modes: {bad_modes}. Allowed: {sorted(WORKFLOW_MODE_LABELS)}")
    if bad_packs:
        raise SystemExit(f"Unknown question packs: {bad_packs}. Allowed: {sorted(QUESTION_PACKS)}")
    token_encoder = _get_token_encoder(str(args.token_encoding))
    if int(args.max_panel_tokens) > 0 and token_encoder is None:
        eprint("Warning: tiktoken not available; using chars/4 approximation for token budgeting.")

    req_rows: list[tuple[str, str, str]] = []
    request_meta_rows: list[dict[str, Any]] = []
    prepared_contexts: list[dict[str, Any]] = []
    issue_label_global = {
        str(r.issue_id): norm_str(r.primary_label).lower()
        for r in issues[["issue_id", "primary_label"]].drop_duplicates(subset=["issue_id"]).itertuples(index=False)
    }

    for slug in slugs:
        for mode in modes:
            rows = _build_issue_rows_for_mode(
                newspaper_slug=slug,
                issues=issues,
                pages=pages,
                allowed_labels=WORKFLOW_MODE_LABELS[mode],
            )
            if not rows:
                continue
            labels_in_scope = sorted({norm_str(x.get("primary_label")) for x in rows if norm_str(x.get("primary_label"))})
            start_date = min((norm_str(x.get("issue_date")) for x in rows if norm_str(x.get("issue_date"))), default="")
            end_date = max((norm_str(x.get("issue_date")) for x in rows if norm_str(x.get("issue_date"))), default="")
            total_chars = int(sum(int(x.get("issue_chars") or 0) for x in rows))
            prepared_contexts.append(
                {
                    "newspaper_slug": slug,
                    "workflow_mode": mode,
                    "issue_count": int(len(rows)),
                    "start_issue_date": start_date,
                    "end_issue_date": end_date,
                    "labels_in_scope": labels_in_scope,
                    "total_issue_chars": total_chars,
                }
            )
            for pack in packs:
                chunk_specs = _chunk_rows_by_token_budget(
                    newspaper_slug=slug,
                    workflow_mode=mode,
                    question_pack=pack,
                    issue_rows=rows,
                    max_panel_chars=int(args.max_panel_chars),
                    max_panel_tokens=int(args.max_panel_tokens),
                    token_encoder=token_encoder,
                )
                for c in chunk_specs:
                    c_rows = c.get("issue_rows") if isinstance(c.get("issue_rows"), list) else []
                    c_labels = sorted({norm_str(x.get("primary_label")) for x in c_rows if norm_str(x.get("primary_label"))})
                    c_start = min((norm_str(x.get("issue_date")) for x in c_rows if norm_str(x.get("issue_date"))), default="")
                    c_end = max((norm_str(x.get("issue_date")) for x in c_rows if norm_str(x.get("issue_date"))), default="")
                    c_issue_ids = [norm_str(x.get("issue_id")) for x in c_rows if norm_str(x.get("issue_id"))]
                    chunk_index = int(c.get("chunk_index") or 1)
                    chunk_count = int(c.get("chunk_count") or len(chunk_specs) or 1)
                    if int(args.max_panel_tokens) <= 0 and chunk_count == 1:
                        custom_id = f"panel_qa::{slug}::{mode}::{pack}"
                    else:
                        custom_id = f"panel_qa::{slug}::{mode}::{pack}::chunk{chunk_index:03d}"
                    req_rows.append((custom_id, norm_str(c.get("prompt")), pack))
                    request_meta_rows.append(
                        {
                            "custom_id": custom_id,
                            "newspaper_slug": slug,
                            "workflow_mode": mode,
                            "question_pack": pack,
                            "chunk_index": chunk_index,
                            "chunk_count": chunk_count,
                            "issue_count": int(len(c_rows)),
                            "issue_ids_scope": "|".join(c_issue_ids),
                            "labels_in_scope": "|".join(c_labels),
                            "start_issue_date": c_start,
                            "end_issue_date": c_end,
                            "prompt_chars": int(c.get("prompt_chars") or 0),
                            "prompt_tokens_est": int(c.get("prompt_tokens") or 0),
                        }
                    )

    if not req_rows:
        raise SystemExit("No requests constructed for selected newspapers/modes.")

    _write_openai_requests_with_reasoning(
        req_dir / "openai_requests_shard000.jsonl",
        req_rows,
        model=str(args.model),
        reasoning_effort="high",
    )
    pd.DataFrame.from_records(request_meta_rows).to_csv(output_dir / "request_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    ctx_df = pd.DataFrame.from_records(prepared_contexts)
    if not ctx_df.empty:
        if "labels_in_scope" in ctx_df.columns:
            ctx_df["labels_in_scope"] = ctx_df["labels_in_scope"].map(
                lambda x: "|".join([norm_str(v) for v in x if norm_str(v)]) if isinstance(x, list) else norm_str(x)
            )
        ctx_df = ctx_df.drop_duplicates().sort_values(["newspaper_slug", "workflow_mode"]).reset_index(drop=True)
    ctx_df.to_csv(output_dir / "context_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)

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

    results = read_result_jsonl(res_dir)
    errors = read_error_jsonl(res_dir)
    meta_by_cid = {str(r["custom_id"]): r for r in request_meta_rows}

    parsed_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    for cid, m in sorted(meta_by_cid.items()):
        raw_row = results.get(cid, {})
        body = {}
        if isinstance(raw_row, dict):
            response = raw_row.get("response")
            if isinstance(response, dict):
                body = response.get("body") if isinstance(response.get("body"), dict) else {}
        text = extract_openai_output_text(body) if body else ""
        parsed_raw = parse_json_from_text(text) if text else None
        if isinstance(parsed_raw, dict):
            if not isinstance(parsed_raw.get("answers"), list):
                recovered_answers = _extract_json_array_for_key(text, "answers")
                if isinstance(recovered_answers, list):
                    parsed_raw["answers"] = recovered_answers
            if not isinstance(parsed_raw.get("quality_notes"), list):
                recovered_notes = _extract_json_array_for_key(text, "quality_notes")
                if isinstance(recovered_notes, list):
                    parsed_raw["quality_notes"] = recovered_notes
        labels = [x for x in norm_str(m.get("labels_in_scope")).split("|") if x]
        issue_ids = {x for x in norm_str(m.get("issue_ids_scope")).split("|") if x}
        issue_label_by_id = {iid: issue_label_global.get(iid, "") for iid in issue_ids}
        coerced = _coerce_parsed_json(
            parsed_raw,
            newspaper_slug=str(m["newspaper_slug"]),
            workflow_mode=str(m["workflow_mode"]),
            question_pack=str(m["question_pack"]),
            issue_ids=issue_ids,
            issue_label_by_id=issue_label_by_id,
            labels_in_scope=labels,
            issue_count=int(m.get("issue_count") or 0),
            start_date=norm_str(m.get("start_issue_date")),
            end_date=norm_str(m.get("end_issue_date")),
        )
        parsed_rows.append(
            {
                "custom_id": cid,
                "newspaper_slug": m["newspaper_slug"],
                "workflow_mode": m["workflow_mode"],
                "question_pack": m["question_pack"],
                "prompt_chars": int(m.get("prompt_chars") or 0),
                "prompt_tokens_est": int(m.get("prompt_tokens_est") or 0),
                "chunk_index": int(m.get("chunk_index") or 1),
                "chunk_count": int(m.get("chunk_count") or 1),
                "has_result": int(cid in results),
                "has_error": int(cid in errors),
                "error_message": norm_str((errors.get(cid) or {}).get("error") if isinstance(errors.get(cid), dict) else ""),
                "raw_output_text": text,
                "parsed_json": coerced,
            }
        )
        ff = coerced["timeline_facts"]["first_full_ordinance"]
        fs = coerced["timeline_facts"]["first_zoning_signal"]
        flat_rows.append(
            {
                "custom_id": cid,
                "newspaper_slug": m["newspaper_slug"],
                "workflow_mode": m["workflow_mode"],
                "question_pack": m["question_pack"],
                "chunk_index": int(m.get("chunk_index") or 1),
                "chunk_count": int(m.get("chunk_count") or 1),
                "issue_count": int(coerced["coverage"]["issue_count"]),
                "labels_in_scope": "|".join(coerced["coverage"]["labels_in_scope"]),
                "first_signal_type": fs.get("signal_type", ""),
                "first_signal_issue_id": fs.get("issue_id", ""),
                "first_signal_issue_date": fs.get("issue_date", ""),
                "first_signal_confidence": float(fs.get("confidence_0_to_1") or 0.0),
                "first_full_status": ff.get("status", ""),
                "first_full_issue_id": ff.get("issue_id", ""),
                "first_full_issue_date": ff.get("issue_date", ""),
                "first_full_confidence": float(ff.get("confidence_0_to_1") or 0.0),
                "amendment_wave_count": int(len(coerced["timeline_facts"].get("amendment_waves") or [])),
                "focus_shift_count": int(len(coerced["timeline_facts"].get("regulatory_focus_shifts") or [])),
                "quality_note_count": int(len(coerced.get("quality_notes") or [])),
            }
        )
        for ans in coerced.get("answers") or []:
            qid = norm_str(ans.get("question_id"))
            base_row = {
                "custom_id": cid,
                "newspaper_slug": m["newspaper_slug"],
                "workflow_mode": m["workflow_mode"],
                "question_pack": m["question_pack"],
                "chunk_index": int(m.get("chunk_index") or 1),
                "chunk_count": int(m.get("chunk_count") or 1),
                "question_id": qid,
                "answer": norm_str(ans.get("answer")),
                "answer_type": norm_str(ans.get("answer_type")),
                "value_number": _safe_optional_float(ans.get("value_number")),
                "value_text": norm_str(ans.get("value_text")),
                "value_category": norm_str(ans.get("value_category")),
                "value_date": norm_str(ans.get("value_date")),
                "value_boolean": ans.get("value_boolean") if isinstance(ans.get("value_boolean"), bool) else None,
                "unit": norm_str(ans.get("unit")),
                "record_count": int(len(ans.get("records") or [])) if isinstance(ans.get("records"), list) else 0,
                "records_json": json.dumps(ans.get("records") if isinstance(ans.get("records"), list) else [], ensure_ascii=False),
                "confidence_0_to_1": float(ans.get("confidence_0_to_1") or 0.0),
                "evidence_issue_ids": "|".join([norm_str(x) for x in (ans.get("evidence_issue_ids") or []) if norm_str(x)]),
                "evidence_quotes": " || ".join([norm_str(x) for x in (ans.get("evidence_quotes") or []) if norm_str(x)]),
            }
            answer_rows.append(
                base_row
            )
            value_rows.append(
                {
                    "custom_id": cid,
                    "newspaper_slug": m["newspaper_slug"],
                    "workflow_mode": m["workflow_mode"],
                    "question_pack": m["question_pack"],
                    "chunk_index": int(m.get("chunk_index") or 1),
                    "chunk_count": int(m.get("chunk_count") or 1),
                    "question_id": qid,
                    "source_type": "scalar",
                    "date": norm_str(ans.get("value_date")),
                    "issue_id": "",
                    "issue_date": "",
                    "district_or_context": "",
                    "value_number": _safe_optional_float(ans.get("value_number")),
                    "value_category": norm_str(ans.get("value_category")),
                    "value_text": norm_str(ans.get("value_text") or ans.get("answer")),
                    "value_boolean": ans.get("value_boolean") if isinstance(ans.get("value_boolean"), bool) else None,
                    "unit": norm_str(ans.get("unit")),
                    "confidence_0_to_1": float(ans.get("confidence_0_to_1") or 0.0),
                    "evidence_issue_ids": "|".join([norm_str(x) for x in (ans.get("evidence_issue_ids") or []) if norm_str(x)]),
                    "evidence_quotes": " || ".join([norm_str(x) for x in (ans.get("evidence_quotes") or []) if norm_str(x)]),
                }
            )
            recs = ans.get("records") if isinstance(ans.get("records"), list) else []
            for rr in recs:
                if not isinstance(rr, dict):
                    continue
                rr_row = {
                    "custom_id": cid,
                    "newspaper_slug": m["newspaper_slug"],
                    "workflow_mode": m["workflow_mode"],
                    "question_pack": m["question_pack"],
                    "chunk_index": int(m.get("chunk_index") or 1),
                    "chunk_count": int(m.get("chunk_count") or 1),
                    "question_id": qid,
                    "date": norm_str(rr.get("date")),
                    "issue_id": norm_str(rr.get("issue_id")),
                    "issue_date": norm_str(rr.get("issue_date")),
                    "district_or_context": norm_str(rr.get("district_or_context")),
                    "value_number": _safe_optional_float(rr.get("value_number")),
                    "value_category": norm_str(rr.get("value_category")),
                    "value_text": norm_str(rr.get("value_text")),
                    "value_boolean": rr.get("value_boolean") if isinstance(rr.get("value_boolean"), bool) else None,
                    "unit": norm_str(rr.get("unit")),
                    "quote": norm_str(rr.get("quote")),
                    "confidence_0_to_1": float(rr.get("confidence_0_to_1") or 0.0),
                    "page_ids": "|".join([norm_str(x) for x in (rr.get("page_ids") or []) if norm_str(x)]),
                }
                record_rows.append(rr_row)
                value_rows.append(
                    {
                        "custom_id": cid,
                        "newspaper_slug": m["newspaper_slug"],
                        "workflow_mode": m["workflow_mode"],
                        "question_pack": m["question_pack"],
                        "chunk_index": int(m.get("chunk_index") or 1),
                        "chunk_count": int(m.get("chunk_count") or 1),
                        "question_id": qid,
                        "source_type": "record",
                        "date": rr_row["date"],
                        "issue_id": rr_row["issue_id"],
                        "issue_date": rr_row["issue_date"],
                        "district_or_context": rr_row["district_or_context"],
                        "value_number": rr_row["value_number"],
                        "value_category": rr_row["value_category"],
                        "value_text": rr_row["value_text"],
                        "value_boolean": rr_row["value_boolean"],
                        "unit": rr_row["unit"],
                        "confidence_0_to_1": rr_row["confidence_0_to_1"],
                        "evidence_issue_ids": rr_row["issue_id"],
                        "evidence_quotes": rr_row["quote"],
                    }
                )

    _write_jsonl(output_dir / "panel_qa_parsed_outputs.jsonl", parsed_rows)
    pd.DataFrame.from_records(flat_rows).to_csv(output_dir / "panel_qa_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame.from_records(answer_rows).to_csv(output_dir / "panel_qa_answers.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame.from_records(record_rows).to_csv(output_dir / "panel_qa_records.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame.from_records(value_rows).to_csv(output_dir / "panel_qa_values_long.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # Consistency diagnostics across workflow modes per newspaper+pack.
    s = pd.DataFrame.from_records(flat_rows)
    diag_rows: list[dict[str, Any]] = []
    if not s.empty:
        for (slug, pack, mode), g in s.groupby(["newspaper_slug", "question_pack", "workflow_mode"], dropna=False):
            ff_dates = [norm_str(getattr(r, "first_full_issue_date")) for r in g.itertuples(index=False)]
            ff_status = [norm_str(getattr(r, "first_full_status")) for r in g.itertuples(index=False)]
            amend_waves = [int(getattr(r, "amendment_wave_count")) for r in g.itertuples(index=False)]
            non_empty_ff = sorted({d for d in ff_dates if d})
            diag_rows.append(
                {
                    "newspaper_slug": slug,
                    "question_pack": pack,
                    "workflow_mode": mode,
                    "request_chunks": int(len(g)),
                    "distinct_first_full_dates": int(len(non_empty_ff)),
                    "first_full_dates": "|".join(non_empty_ff),
                    "first_full_status_values": json.dumps(ff_status, ensure_ascii=False),
                    "amendment_wave_count_values": json.dumps(amend_waves, ensure_ascii=False),
                }
            )
    pd.DataFrame.from_records(diag_rows).to_csv(output_dir / "workflow_consistency_diagnostics.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    provenance = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "package_dir": str(package_dir),
        "metadata_dir": str(metadata_dir),
        "output_dir": str(output_dir),
        "model": str(args.model),
        "max_panel_chars": int(args.max_panel_chars),
        "max_panel_tokens": int(args.max_panel_tokens),
        "token_encoding": str(args.token_encoding),
        "workflow_modes": modes,
        "question_packs": packs,
        "newspaper_count": int(len(slugs)),
        "request_count": int(len(req_rows)),
        "result_count": int(len(results)),
        "error_count": int(len(errors)),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    eprint(f"Done. requests={len(req_rows)} results={len(results)} errors={len(errors)} output={output_dir}")


if __name__ == "__main__":
    main()
