#!/usr/bin/env python3
"""
LLM-only issue classification for v3 full-newspaper pipeline.

This stage assigns strict issue classes with gating fields to prevent
narrative-only text being forced into law buckets.

Classification can run raw-first (full issue text as primary evidence) with
cleaned text as optional supporting context.
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
    FIXED_CATEGORY_KEYS,
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


ALLOWED_VARIANTS = {"single_model", "dual_model_with_llm_adjudicator"}
ALLOWED_INPUT_MODE = {"raw_plus_clean", "raw_only"}
ALLOWED_YNU = {"yes", "no", "uncertain"}
ALLOWED_SUFF = {"sufficient", "insufficient", "uncertain"}
ALLOWED_SCOPE = {"law_only", "none", "uncertain"}
ALLOWED_LEGAL_STATUS = {"enacted", "proposed", "notice_only", "uncertain"}
ALLOWED_JUR_TYPE = {"city", "town", "village", "township", "borough", "county", "regional", "state", "unknown"}
ALLOWED_COMPLETENESS = {"complete", "partial_installment", "uncertain"}
ALLOWED_CLASS = {
    "full_code_comprehensive",
    "full_code_recodification",
    "full_code_partial_enacted",
    "full_code_partial_proposed",
    "full_code_proposed",
    "amendment_substantial",
    "amendment_targeted",
    "amendment_proposed",
    "zoning_notice_procedural",
    "zoning_notice_adoption",
    "building_code_non_zoning",
    "not_verbatim_law_text",
    "non_zoning",
}
ENACTED_FULL_CLASSES = {
    "full_code_comprehensive",
    "full_code_recodification",
    "full_code_partial_enacted",
}
ENACTED_AMEND_CLASSES = {
    "amendment_substantial",
    "amendment_targeted",
}
PROPOSED_LAW_CLASSES = {
    "full_code_proposed",
    "full_code_partial_proposed",
    "amendment_proposed",
}
NOTICE_LAW_CLASSES = {
    "zoning_notice_procedural",
    "zoning_notice_adoption",
}
ALLOWED_LAW_CLASSES = ENACTED_FULL_CLASSES | ENACTED_AMEND_CLASSES | PROPOSED_LAW_CLASSES | NOTICE_LAW_CLASSES


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM-only issue classification v3.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Run directory containing panels/<city_key>/issue_texts.jsonl.",
    )
    ap.add_argument(
        "--clean-jsonl",
        default="",
        help="Path to clean_zoning_text.jsonl (default: <run-dir>/fullnewspaper_v3/cleaning/clean_zoning_text.jsonl).",
    )
    ap.add_argument(
        "--law-presence-jsonl",
        default="",
        help="Optional law-presence gate output JSONL from llm_law_presence_v4.py.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-dir>/fullnewspaper_v3/classification).",
    )
    ap.add_argument(
        "--newspapers",
        default="cazenovia-republican,albion-evening-recorder,bountiful-davis-county-clipper,abilene-reporter-news,appleton-post-crescent",
        help="Comma-separated newspaper slugs.",
    )
    ap.add_argument("--variant", default="single_model", choices=sorted(ALLOWED_VARIANTS))
    ap.add_argument(
        "--classification-input-mode",
        default="raw_only",
        choices=sorted(ALLOWED_INPUT_MODE),
        help="raw_only: classify from full issue text (recommended). raw_plus_clean: include clean text as supporting context.",
    )
    ap.add_argument(
        "--require-clean-for-law",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, empty clean text hard-blocks law classes. Default false to avoid cleaning-stage false negatives.",
    )
    ap.add_argument(
        "--strict-contradiction-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, uses stricter enacted/proposed contradiction guardrails.",
    )
    ap.add_argument("--model-primary", default="openai:gpt-5")
    ap.add_argument("--model-secondary", default="gemini:gemini-2.5-flash")
    ap.add_argument("--adjudicator-model", default="")
    ap.add_argument(
        "--max-clean-chars",
        type=int,
        default=0,
        help="Hard character cap for clean text in prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument(
        "--max-raw-chars",
        type=int,
        default=0,
        help="Hard character cap for raw issue text in prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument("--limit-issues", type=int, default=0)
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


def _load_issue_pool(run_dir: Path, newspapers: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    panel_dir = run_dir / "panels"
    for city_dir in sorted(panel_dir.iterdir()):
        p = city_dir / "issue_texts.jsonl"
        if not city_dir.is_dir() or not p.is_file():
            continue
        for r in iter_jsonl(p):
            slug = norm_str(r.get("pub_slug")).lower()
            if not slug:
                issue_id = norm_str(r.get("issue_id"))
                slug = issue_id.rsplit("__", 1)[0].lower() if "__" in issue_id else issue_id.lower()
            if slug in newspapers:
                row = dict(r)
                row["pub_slug"] = slug
                out.append(row)
    out = sorted(out, key=lambda r: (norm_str(r.get("pub_slug")), norm_str(r.get("issue_date")), norm_str(r.get("issue_id"))))
    return out


def _normalize_category_shares(raw: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for k in FIXED_CATEGORY_KEYS:
            out[k] = max(0.0, safe_float(raw.get(k), default=0.0))
    else:
        out = {k: 0.0 for k in FIXED_CATEGORY_KEYS}
    s = sum(out.values())
    if s > 0:
        out = {k: float(v / s) for k, v in out.items()}
    return out


def _build_prompt(
    issue: dict[str, Any],
    clean_row: dict[str, Any],
    law_presence_row: dict[str, Any],
    max_clean: int,
    max_raw: int,
    *,
    input_mode: str,
    strict_contradiction_mode: bool = False,
) -> str:
    schema = {
        "issue_id": norm_str(issue.get("issue_id")),
        "target_city_name": norm_str(issue.get("city_name")),
        "target_state_abbr": norm_str(issue.get("state_abbr")).upper(),
        "jurisdiction_name": norm_str(issue.get("city_name")),
        "jurisdiction_type": "city",
        "jurisdiction_state_abbr": norm_str(issue.get("state_abbr")).upper(),
        "jurisdiction_match_city": "yes",
        "jurisdiction_evidence_quotes": ["short jurisdiction quote"],
        "issue_completeness": "complete",
        "possible_missing_pages": "no",
        "contains_verbatim_zoning_law": "yes",
        "law_text_sufficiency": "sufficient",
        "legal_status": "enacted",
        "enactment_signal": "yes",
        "proposal_signal": "no",
        "non_law_zoning_narrative_present": "no",
        "classification_scope": "law_only",
        "issue_class": "amendment_targeted",
        "confidence_0_to_1": 0.8,
        "evidence_anchors": ["A00001", "A00002"],
        "evidence_quotes": ["short quote 1", "short quote 2"],
        "status_evidence_quotes": ["short enactment/proposal quote"],
        "category_shares": {k: (1.0 / float(len(FIXED_CATEGORY_KEYS))) for k in FIXED_CATEGORY_KEYS},
        "quality_notes": [""],
    }
    lines: list[str] = []
    lines.append("Classify this issue into zoning-law ontology classes.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Allowed issue_class:")
    lines.append(", ".join(sorted(ALLOWED_CLASS)))
    lines.append("")
    lines.append("Critical guardrails:")
    lines.append("- Do NOT classify zoning narrative/reporting as legal ordinance/amendment text.")
    lines.append("- issue_class may be full/amendment/notice ONLY when contains_verbatim_zoning_law=yes and law_text_sufficiency=sufficient.")
    lines.append("- If no sufficient verbatim legal text, use issue_class=not_verbatim_law_text (or non_zoning if no zoning content at all).")
    lines.append("- Use building_code_non_zoning when the legal text is primarily building-code ordinance language rather than zoning ordinance language.")
    lines.append("- Treat sanitary/septic/plumbing/electrical/fire/health code ordinances as building_code_non_zoning unless the page clearly contains zoning district/use/regulation text.")
    lines.append("- Example: 'Septic Tank Code', 'Individual Sewage Disposal Systems', or similar standalone sanitation code should map to building_code_non_zoning, not amendment/full zoning classes.")
    lines.append("- Use full_code_partial_enacted/full_code_partial_proposed when this issue-date appears to contain only a partial installment/continuation of a larger zoning ordinance text.")
    lines.append("- Set issue_completeness to complete, partial_installment, or uncertain based on whether the available issue-date text appears complete.")
    lines.append("- Set possible_missing_pages=yes when the text indicates continuation/missing context or partial publication.")
    lines.append("- Use enacted full/amendment classes ONLY when there is explicit enactment/in-force evidence.")
    lines.append("- If text is proposal/draft/hearing and no enactment evidence, use full_code_proposed or amendment_proposed.")
    lines.append("- If the issue is primarily a hearing/adoption/procedural legal notice, use zoning_notice_procedural or zoning_notice_adoption.")
    lines.append("- Extract the governing legal jurisdiction from the text (e.g., Town of Nelson, City of Albion).")
    lines.append("- Set jurisdiction_match_city=yes only when governing jurisdiction matches the target city for this newspaper panel.")
    lines.append("- If governing jurisdiction is clearly different, set jurisdiction_match_city=no and provide jurisdiction_evidence_quotes.")
    lines.append("- If governing jurisdiction is not the target city (for example neighboring town/village in legal notices), do not assign enacted full/amend zoning classes for the target city.")
    lines.append("- For any law class (full/amendment/notice), provide at least 2 evidence_quotes.")
    lines.append("- Always set legal_status, enactment_signal, and proposal_signal based on the legal text and context.")
    if bool(strict_contradiction_mode):
        lines.append("- Precision mode: proposal and enactment signals cannot both be yes unless clear in-force transition evidence is quoted.")
        lines.append("- If contradiction exists (proposal-heavy evidence and no clear in-force clause), classify as proposed or notice, not enacted.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Issue metadata: issue_id={norm_str(issue.get('issue_id'))} "
        f"date={norm_str(issue.get('issue_date'))} "
        f"city={norm_str(issue.get('city_name'))},{norm_str(issue.get('state_abbr')).upper()} "
        f"newspaper={norm_str(issue.get('pub_slug'))} "
        f"available_page_count={int(issue.get('page_count') or 0)} "
        f"available_page_ids={','.join([norm_str(x) for x in (issue.get('page_ids') if isinstance(issue.get('page_ids'), list) else []) if norm_str(x)])}"
    )
    lines.append("Classify using all available pages for this issue-date (the full text block provided below).")
    lines.append("")
    lines.append("Law-presence pre-gate signal (from prior stage):")
    lines.append(
        f"law_presence={norm_str(law_presence_row.get('law_presence'))} "
        f"evidence_anchor_ids={','.join([norm_str(x) for x in (law_presence_row.get('law_presence_evidence_anchor_ids') if isinstance(law_presence_row.get('law_presence_evidence_anchor_ids'), list) else []) if norm_str(x)])}"
    )
    lines.append("")
    if input_mode == "raw_plus_clean":
        lines.append("Optional supporting context (clean zoning-law candidate from anchor cleaning):")
        clean_txt = norm_str(clean_row.get("clean_zoning_text"))
        lines.append(
            _bounded_text_or_fail(
                clean_txt,
                max_clean,
                label=f"issue_id={norm_str(issue.get('issue_id'))} clean_zoning_text",
            )
            if clean_txt
            else "[clean candidate empty]"
        )
        lines.append("")
    lines.append("Primary evidence: original issue context excerpt (full issue-date text block):")
    lines.append(
        _bounded_text_or_fail(
            norm_str(issue.get("text")),
            max_raw,
            label=f"issue_id={norm_str(issue.get('issue_id'))} raw_issue_text",
        )
    )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _build_adjudication_prompt(
    issue: dict[str, Any],
    clean_row: dict[str, Any],
    law_presence_row: dict[str, Any],
    p_primary: dict[str, Any],
    p_secondary: dict[str, Any],
    max_clean: int,
    max_raw: int,
    *,
    input_mode: str,
    strict_contradiction_mode: bool = False,
) -> str:
    schema = {
        "issue_id": norm_str(issue.get("issue_id")),
        "target_city_name": norm_str(issue.get("city_name")),
        "target_state_abbr": norm_str(issue.get("state_abbr")).upper(),
        "jurisdiction_name": norm_str(issue.get("city_name")),
        "jurisdiction_type": "city",
        "jurisdiction_state_abbr": norm_str(issue.get("state_abbr")).upper(),
        "jurisdiction_match_city": "yes",
        "jurisdiction_evidence_quotes": ["short jurisdiction quote"],
        "issue_completeness": "complete",
        "possible_missing_pages": "no",
        "contains_verbatim_zoning_law": "yes",
        "law_text_sufficiency": "sufficient",
        "legal_status": "enacted",
        "enactment_signal": "yes",
        "proposal_signal": "no",
        "non_law_zoning_narrative_present": "no",
        "classification_scope": "law_only",
        "issue_class": "amendment_targeted",
        "confidence_0_to_1": 0.8,
        "evidence_anchors": ["A00001", "A00002"],
        "evidence_quotes": ["short quote 1", "short quote 2"],
        "status_evidence_quotes": ["short enactment/proposal quote"],
        "category_shares": {k: (1.0 / float(len(FIXED_CATEGORY_KEYS))) for k in FIXED_CATEGORY_KEYS},
        "quality_notes": [""],
    }
    lines: list[str] = []
    lines.append("Adjudicate between two model outputs for zoning issue classification.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Apply the same strict guardrails:")
    lines.append("- narrative must not be classified as ordinance/amendment")
    lines.append("- law class requires sufficient verbatim legal text + >=2 legal evidence quotes")
    lines.append("- building-code legal text should use issue_class=building_code_non_zoning, not full/amend zoning classes")
    lines.append("- explicit sanitary/septic/plumbing/electrical/fire/health code ordinances should be building_code_non_zoning unless zoning district/use regulation text is dominant")
    lines.append("- partial/continued ordinance installments should use full_code_partial_enacted or full_code_partial_proposed")
    lines.append("- choose issue_completeness in {complete, partial_installment, uncertain} and possible_missing_pages in {yes,no,uncertain}")
    lines.append("- enacted classes require explicit enactment evidence")
    lines.append("- proposal/draft/hearing without enactment evidence must be proposed or procedural/adoption notice")
    lines.append("- extract governing jurisdiction and decide if it matches the target city for this panel")
    lines.append("- if governing jurisdiction is not the target city, avoid enacted full/amend zoning classes for this target-city panel")
    if bool(strict_contradiction_mode):
        lines.append("- Precision mode: resolve contradictions toward proposed/notice unless enactment evidence is explicit and strong.")
    lines.append("")
    lines.append("Primary output candidate:")
    lines.append(json.dumps(p_primary, ensure_ascii=False))
    lines.append("")
    lines.append("Secondary output candidate:")
    lines.append(json.dumps(p_secondary, ensure_ascii=False))
    lines.append("")
    lines.append("Issue metadata and text for adjudication:")
    lines.append(
        f"issue_id={norm_str(issue.get('issue_id'))} date={norm_str(issue.get('issue_date'))} newspaper={norm_str(issue.get('pub_slug'))} "
        f"available_page_count={int(issue.get('page_count') or 0)} "
        f"available_page_ids={','.join([norm_str(x) for x in (issue.get('page_ids') if isinstance(issue.get('page_ids'), list) else []) if norm_str(x)])}"
    )
    lines.append("Use all available pages for this issue-date when deciding class/status/completeness.")
    lines.append("")
    lines.append("Law-presence pre-gate signal:")
    lines.append(
        f"law_presence={norm_str(law_presence_row.get('law_presence'))} "
        f"evidence_anchor_ids={','.join([norm_str(x) for x in (law_presence_row.get('law_presence_evidence_anchor_ids') if isinstance(law_presence_row.get('law_presence_evidence_anchor_ids'), list) else []) if norm_str(x)])}"
    )
    lines.append("")
    if input_mode == "raw_plus_clean":
        lines.append("Optional supporting context (clean zoning-law candidate):")
        clean_txt = norm_str(clean_row.get("clean_zoning_text"))
        lines.append(
            _bounded_text_or_fail(
                clean_txt,
                max_clean,
                label=f"issue_id={norm_str(issue.get('issue_id'))} clean_zoning_text",
            )
            if clean_txt
            else "[clean candidate empty]"
        )
        lines.append("")
    lines.append("Primary evidence: original issue context excerpt (full issue-date text block):")
    lines.append(
        _bounded_text_or_fail(
            norm_str(issue.get("text")),
            max_raw,
            label=f"issue_id={norm_str(issue.get('issue_id'))} raw_issue_text",
        )
    )
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _coerce(
    parsed_obj: dict[str, Any],
    issue_id: str,
    clean_chars: float,
    *,
    law_presence_hint: str = "",
    law_presence_anchor_ids: list[str] | None = None,
    require_clean_for_law: bool = False,
    strict_contradiction_mode: bool = False,
) -> dict[str, Any]:
    obj = dict(parsed_obj or {})
    out = {
        "issue_id": norm_str(obj.get("issue_id")) or issue_id,
        "target_city_name": norm_str(obj.get("target_city_name")),
        "target_state_abbr": norm_str(obj.get("target_state_abbr")).upper(),
        "jurisdiction_name": norm_str(obj.get("jurisdiction_name")),
        "jurisdiction_type": enum(obj.get("jurisdiction_type"), ALLOWED_JUR_TYPE, "unknown"),
        "jurisdiction_state_abbr": norm_str(obj.get("jurisdiction_state_abbr")).upper(),
        "jurisdiction_match_city": enum(obj.get("jurisdiction_match_city"), ALLOWED_YNU, "uncertain"),
        "jurisdiction_evidence_quotes": [
            norm_str(x)
            for x in (obj.get("jurisdiction_evidence_quotes") if isinstance(obj.get("jurisdiction_evidence_quotes"), list) else [])
            if norm_str(x)
        ],
        "issue_completeness": enum(obj.get("issue_completeness"), ALLOWED_COMPLETENESS, "uncertain"),
        "possible_missing_pages": enum(obj.get("possible_missing_pages"), ALLOWED_YNU, "uncertain"),
        "contains_verbatim_zoning_law": enum(obj.get("contains_verbatim_zoning_law"), ALLOWED_YNU, "uncertain"),
        "law_text_sufficiency": enum(obj.get("law_text_sufficiency"), ALLOWED_SUFF, "uncertain"),
        "legal_status": enum(obj.get("legal_status"), ALLOWED_LEGAL_STATUS, "uncertain"),
        "enactment_signal": enum(obj.get("enactment_signal"), ALLOWED_YNU, "uncertain"),
        "proposal_signal": enum(obj.get("proposal_signal"), ALLOWED_YNU, "uncertain"),
        "non_law_zoning_narrative_present": enum(obj.get("non_law_zoning_narrative_present"), ALLOWED_YNU, "uncertain"),
        "classification_scope": enum(obj.get("classification_scope"), ALLOWED_SCOPE, "uncertain"),
        "issue_class": enum(obj.get("issue_class"), ALLOWED_CLASS, "not_verbatim_law_text"),
        "confidence_0_to_1": safe_float(obj.get("confidence_0_to_1"), default=0.0),
        "evidence_anchors": [norm_str(x) for x in (obj.get("evidence_anchors") if isinstance(obj.get("evidence_anchors"), list) else []) if norm_str(x)],
        "evidence_quotes": [norm_str(x) for x in (obj.get("evidence_quotes") if isinstance(obj.get("evidence_quotes"), list) else []) if norm_str(x)],
        "status_evidence_quotes": [
            norm_str(x)
            for x in (obj.get("status_evidence_quotes") if isinstance(obj.get("status_evidence_quotes"), list) else [])
            if norm_str(x)
        ],
        "category_shares": _normalize_category_shares(obj.get("category_shares")),
        "quality_notes": [norm_str(x) for x in (obj.get("quality_notes") if isinstance(obj.get("quality_notes"), list) else []) if norm_str(x)],
    }

    # Gate law classes behind strict sufficiency requirement.
    has_law = out["contains_verbatim_zoning_law"] == "yes"
    sufficient = out["law_text_sufficiency"] == "sufficient"
    if out["issue_class"] in ALLOWED_LAW_CLASSES:
        if not (has_law and sufficient and len(out["evidence_quotes"]) >= 2):
            out["issue_class"] = "not_verbatim_law_text"

    if not has_law and out["issue_class"] in ALLOWED_LAW_CLASSES:
        out["issue_class"] = "not_verbatim_law_text"

    if out["classification_scope"] == "none" and out["issue_class"] in ALLOWED_LAW_CLASSES:
        out["issue_class"] = "not_verbatim_law_text"

    # Optional structural guardrail. Default behavior is raw-first classification,
    # so empty cleaned text is non-blocking unless explicitly required.
    if out["issue_class"] in ALLOWED_LAW_CLASSES:
        if bool(require_clean_for_law) and float(clean_chars or 0.0) <= 0.0:
            out["issue_class"] = "not_verbatim_law_text"
            out["quality_notes"].append("clean_text_empty_override")
        elif float(clean_chars or 0.0) <= 0.0:
            out["quality_notes"].append("clean_text_empty_nonblocking")

    # Enactment guardrail: enacted classes require explicit enactment signal.
    if out["issue_class"] in (ENACTED_FULL_CLASSES | ENACTED_AMEND_CLASSES):
        if out["enactment_signal"] != "yes":
            if out["issue_class"] == "full_code_partial_enacted":
                out["issue_class"] = "full_code_partial_proposed"
            elif out["issue_class"] in ENACTED_FULL_CLASSES:
                out["issue_class"] = "full_code_proposed"
            else:
                out["issue_class"] = "amendment_proposed"
            out["quality_notes"].append("downgraded_to_proposed_no_enactment_signal")

    # Proposal guardrail: explicit proposal signal without enactment should not remain enacted.
    if out["proposal_signal"] == "yes" and out["enactment_signal"] != "yes":
        if out["issue_class"] == "full_code_partial_enacted":
            out["issue_class"] = "full_code_partial_proposed"
            out["quality_notes"].append("downgraded_to_partial_proposed_from_proposal_signal")
        elif out["issue_class"] in ENACTED_FULL_CLASSES:
            out["issue_class"] = "full_code_proposed"
            out["quality_notes"].append("downgraded_to_proposed_from_proposal_signal")
        elif out["issue_class"] in ENACTED_AMEND_CLASSES:
            out["issue_class"] = "amendment_proposed"
            out["quality_notes"].append("downgraded_to_proposed_from_proposal_signal")

    # Coherence for explicit partial/full-code classes.
    if out["issue_class"] == "full_code_partial_enacted" and out["enactment_signal"] != "yes":
        out["issue_class"] = "full_code_partial_proposed"
        out["quality_notes"].append("partial_class_downgraded_to_proposed_no_enactment_signal")
    if out["issue_class"] == "full_code_partial_proposed" and out["enactment_signal"] == "yes" and out["proposal_signal"] != "yes":
        out["issue_class"] = "full_code_partial_enacted"
        out["quality_notes"].append("partial_class_upgraded_to_enacted_from_enactment_signal")

    # If model marks partial completeness for full-code classes, keep explicit partial classes.
    if out["issue_completeness"] == "partial_installment":
        if out["issue_class"] in {"full_code_comprehensive", "full_code_recodification"}:
            if out["enactment_signal"] == "yes":
                out["issue_class"] = "full_code_partial_enacted"
            else:
                out["issue_class"] = "full_code_partial_proposed"
            out["quality_notes"].append("coerced_to_partial_class_from_completeness_flag")
        elif out["issue_class"] == "full_code_proposed":
            out["issue_class"] = "full_code_partial_proposed"
            out["quality_notes"].append("coerced_to_partial_proposed_from_completeness_flag")

    # Notice status should not carry enacted/proposed full-amend classes.
    if out["legal_status"] == "notice_only" and out["issue_class"] in ALLOWED_LAW_CLASSES:
        out["issue_class"] = "zoning_notice_adoption" if out["enactment_signal"] == "yes" else "zoning_notice_procedural"
        out["quality_notes"].append("coerced_notice_only_status")

    # Strict contradiction handling: when proposal signal dominates without enactment,
    # avoid enacted classes even if model emitted them.
    if bool(strict_contradiction_mode):
        if out["proposal_signal"] == "yes" and out["enactment_signal"] != "yes":
            if out["issue_class"] == "full_code_partial_enacted":
                out["issue_class"] = "full_code_partial_proposed"
                out["quality_notes"].append("strict_contradiction_downgrade_partial_full")
            elif out["issue_class"] in ENACTED_FULL_CLASSES:
                out["issue_class"] = "full_code_proposed"
                out["quality_notes"].append("strict_contradiction_downgrade_full")
            elif out["issue_class"] in ENACTED_AMEND_CLASSES:
                out["issue_class"] = "amendment_proposed"
                out["quality_notes"].append("strict_contradiction_downgrade_amend")

    if out["issue_class"] in ALLOWED_LAW_CLASSES and out["jurisdiction_match_city"] == "no":
        out["quality_notes"].append("out_of_target_jurisdiction_issue")

    # Carry forward law-presence gate fields for v4 schema.
    lp = enum(law_presence_hint, ALLOWED_YNU, "")
    if not lp:
        lp = out["contains_verbatim_zoning_law"]
    out["law_presence"] = lp
    out["law_presence_evidence_anchor_ids"] = [
        norm_str(x)
        for x in (law_presence_anchor_ids if isinstance(law_presence_anchor_ids, list) else [])
        if norm_str(x)
    ] or list(out["evidence_anchors"])
    out["clean_text_sufficiency"] = out["law_text_sufficiency"]

    return out


def _run_single_pass(
    *,
    issues: list[dict[str, Any]],
    clean_map: dict[str, dict[str, Any]],
    law_presence_map: dict[str, dict[str, Any]],
    out_dir: Path,
    model: str,
    max_clean: int,
    max_raw: int,
    runner: Path,
    gateway_pythonpath: Path | None,
    timeout: float,
    concurrency: int,
    env_path: Path,
    skip_existing: bool,
    dry_run: bool,
    cid_prefix: str,
    classification_input_mode: str,
    require_clean_for_law: bool,
    strict_contradiction_mode: bool,
) -> dict[str, dict[str, Any]]:
    req_rows: list[tuple[str, str]] = []
    meta_by_cid: dict[str, dict[str, Any]] = {}
    for issue in issues:
        iid = norm_str(issue.get("issue_id"))
        slug = norm_str(issue.get("pub_slug")).lower()
        clean = clean_map.get(iid) or {}
        lawp = law_presence_map.get(iid) or {}
        cid = f"{cid_prefix}::{slug}::{iid}"
        prompt = _build_prompt(
            issue,
            clean,
            lawp,
            max_clean=max_clean,
            max_raw=max_raw,
            input_mode=str(classification_input_mode),
            strict_contradiction_mode=bool(strict_contradiction_mode),
        )
        req_rows.append((cid, prompt))
        meta_by_cid[cid] = {"issue": issue, "clean": clean, "law_presence": lawp}

    req_dir = out_dir / "requests"
    res_dir = out_dir / "results"
    write_openai_requests(req_dir / "openai_requests_shard000.jsonl", req_rows, model=model)
    run_gateway_requests(
        request_dir=req_dir,
        output_dir=res_dir,
        runner_path=runner,
        model=model,
        gateway_pythonpath=gateway_pythonpath,
        timeout_s=timeout,
        concurrency=concurrency,
        gov_env_path=env_path,
        skip_existing=skip_existing,
        dry_run=dry_run,
    )
    result_map = read_result_jsonl(res_dir)
    error_map = read_error_jsonl(res_dir)

    out: dict[str, dict[str, Any]] = {}
    for cid, _prompt in req_rows:
        issue = meta_by_cid[cid]["issue"]
        iid = norm_str(issue.get("issue_id"))
        clean = meta_by_cid[cid]["clean"]
        lawp = meta_by_cid[cid]["law_presence"]
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
        coerced = _coerce(
            parsed if isinstance(parsed, dict) else {},
            iid,
            clean_chars=safe_float(clean.get("clean_chars"), default=0.0),
            law_presence_hint=norm_str(lawp.get("law_presence")),
            law_presence_anchor_ids=lawp.get("law_presence_evidence_anchor_ids") if isinstance(lawp.get("law_presence_evidence_anchor_ids"), list) else [],
            require_clean_for_law=bool(require_clean_for_law),
            strict_contradiction_mode=bool(strict_contradiction_mode),
        )
        coerced["parse_ok"] = parse_ok
        coerced["had_error"] = had_error
        out[iid] = coerced
    return out


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    newspapers = [norm_str(x).lower() for x in norm_str(args.newspapers).split(",") if norm_str(x)]
    if not newspapers:
        raise SystemExit("No newspapers provided.")

    clean_path = (
        Path(args.clean_jsonl).expanduser().resolve()
        if norm_str(args.clean_jsonl)
        else run_dir / "fullnewspaper_v3" / "cleaning" / "clean_zoning_text.jsonl"
    )
    law_presence_path = Path(args.law_presence_jsonl).expanduser().resolve() if norm_str(args.law_presence_jsonl) else None
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if norm_str(args.output_dir)
        else run_dir / "fullnewspaper_v3" / "classification"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    issues = _load_issue_pool(run_dir, newspapers)
    if int(args.limit_issues) > 0:
        issues = issues[: int(args.limit_issues)]
    if not issues:
        raise SystemExit("No issues loaded.")

    clean_rows = iter_jsonl(clean_path)
    clean_map = {norm_str(r.get("issue_id")): r for r in clean_rows if norm_str(r.get("issue_id"))}
    law_presence_map: dict[str, dict[str, Any]] = {}
    if law_presence_path is not None and law_presence_path.is_file():
        lp_rows = iter_jsonl(law_presence_path)
        law_presence_map = {norm_str(r.get("issue_id")): r for r in lp_rows if norm_str(r.get("issue_id"))}

    runner = Path(args.gateway_runner).expanduser().resolve()
    gp = Path(args.gateway_pythonpath).expanduser().resolve() if norm_str(args.gateway_pythonpath) else None
    env_path = Path(args.gov_env_path).expanduser().resolve()

    primary = _run_single_pass(
        issues=issues,
        clean_map=clean_map,
        law_presence_map=law_presence_map,
        out_dir=out_dir / "primary",
        model=str(args.model_primary),
        max_clean=int(args.max_clean_chars),
        max_raw=int(args.max_raw_chars),
        runner=runner,
        gateway_pythonpath=gp,
        timeout=float(args.timeout),
        concurrency=int(args.concurrency),
        env_path=env_path,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
        cid_prefix="issue_class_v3_primary",
        classification_input_mode=str(args.classification_input_mode),
        require_clean_for_law=bool(args.require_clean_for_law),
        strict_contradiction_mode=bool(args.strict_contradiction_mode),
    )

    final_by_issue: dict[str, dict[str, Any]] = {}
    secondary: dict[str, dict[str, Any]] = {}

    if args.variant == "single_model":
        final_by_issue = primary
    else:
        secondary = _run_single_pass(
            issues=issues,
            clean_map=clean_map,
            law_presence_map=law_presence_map,
            out_dir=out_dir / "secondary",
            model=str(args.model_secondary),
            max_clean=int(args.max_clean_chars),
            max_raw=int(args.max_raw_chars),
            runner=runner,
            gateway_pythonpath=gp,
            timeout=float(args.timeout),
            concurrency=int(args.concurrency),
            env_path=env_path,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
            cid_prefix="issue_class_v3_secondary",
            classification_input_mode=str(args.classification_input_mode),
            require_clean_for_law=bool(args.require_clean_for_law),
            strict_contradiction_mode=bool(args.strict_contradiction_mode),
        )

        adjudicator_model = norm_str(args.adjudicator_model) or str(args.model_primary)
        adjud_req_rows: list[tuple[str, str]] = []
        adjud_meta: dict[str, dict[str, Any]] = {}
        for issue in issues:
            iid = norm_str(issue.get("issue_id"))
            p = primary.get(iid) or {}
            s = secondary.get(iid) or {}
            # If same class + same gate, accept primary directly.
            same_key = (
                norm_str(p.get("issue_class")) == norm_str(s.get("issue_class"))
                and norm_str(p.get("contains_verbatim_zoning_law")) == norm_str(s.get("contains_verbatim_zoning_law"))
                and norm_str(p.get("law_text_sufficiency")) == norm_str(s.get("law_text_sufficiency"))
            )
            if same_key and p:
                final_by_issue[iid] = p
                continue
            clean = clean_map.get(iid) or {}
            lawp = law_presence_map.get(iid) or {}
            slug = norm_str(issue.get("pub_slug")).lower()
            cid = f"issue_class_v3_adjudicate::{slug}::{iid}"
            prompt = _build_adjudication_prompt(
                issue,
                clean,
                lawp,
                p_primary=p if isinstance(p, dict) else {},
                p_secondary=s if isinstance(s, dict) else {},
                max_clean=int(args.max_clean_chars),
                max_raw=int(args.max_raw_chars),
                input_mode=str(args.classification_input_mode),
                strict_contradiction_mode=bool(args.strict_contradiction_mode),
            )
            adjud_req_rows.append((cid, prompt))
            adjud_meta[cid] = {"issue": issue}

        if adjud_req_rows:
            adj_dir = out_dir / "adjudication"
            write_openai_requests(adj_dir / "requests" / "openai_requests_shard000.jsonl", adjud_req_rows, model=adjudicator_model)
            run_gateway_requests(
                request_dir=adj_dir / "requests",
                output_dir=adj_dir / "results",
                runner_path=runner,
                model=adjudicator_model,
                gateway_pythonpath=gp,
                timeout_s=float(args.timeout),
                concurrency=int(args.concurrency),
                gov_env_path=env_path,
                skip_existing=bool(args.skip_existing),
                dry_run=bool(args.dry_run),
            )
            adj_res = read_result_jsonl(adj_dir / "results")
            adj_err = read_error_jsonl(adj_dir / "results")
            for cid, _prompt in adjud_req_rows:
                issue = adjud_meta[cid]["issue"]
                iid = norm_str(issue.get("issue_id"))
                output_text = ""
                if cid in adj_res:
                    body = (((adj_res[cid].get("response") or {}).get("body")) or {})
                    output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
                elif cid in adj_err:
                    body = (((adj_err[cid].get("response") or {}).get("body")) or {})
                    output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
                parsed = parse_json_from_text(output_text)
                if isinstance(parsed, dict) and parsed:
                    clean = clean_map.get(iid) or {}
                    final_by_issue[iid] = _coerce(
                        parsed,
                        iid,
                        clean_chars=safe_float(clean.get("clean_chars"), default=0.0),
                        law_presence_hint=norm_str((law_presence_map.get(iid) or {}).get("law_presence")),
                        law_presence_anchor_ids=(law_presence_map.get(iid) or {}).get("law_presence_evidence_anchor_ids")
                        if isinstance((law_presence_map.get(iid) or {}).get("law_presence_evidence_anchor_ids"), list)
                        else [],
                        require_clean_for_law=bool(args.require_clean_for_law),
                        strict_contradiction_mode=bool(args.strict_contradiction_mode),
                    )
                else:
                    # fallback to primary if adjudication fails
                    clean = clean_map.get(iid) or {}
                    final_by_issue[iid] = primary.get(iid) or _coerce(
                        {},
                        iid,
                        clean_chars=safe_float(clean.get("clean_chars"), default=0.0),
                        law_presence_hint=norm_str((law_presence_map.get(iid) or {}).get("law_presence")),
                        law_presence_anchor_ids=(law_presence_map.get(iid) or {}).get("law_presence_evidence_anchor_ids")
                        if isinstance((law_presence_map.get(iid) or {}).get("law_presence_evidence_anchor_ids"), list)
                        else [],
                        require_clean_for_law=bool(args.require_clean_for_law),
                        strict_contradiction_mode=bool(args.strict_contradiction_mode),
                    )

    rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for issue in issues:
        iid = norm_str(issue.get("issue_id"))
        slug = norm_str(issue.get("pub_slug")).lower()
        clean = clean_map.get(iid) or {}
        r = final_by_issue.get(iid) or _coerce(
            {},
            iid,
            clean_chars=safe_float(clean.get("clean_chars"), default=0.0),
            require_clean_for_law=bool(args.require_clean_for_law),
        )
        row = {
            "newspaper_slug": slug,
            "city_key": norm_str(issue.get("city_key")),
            "city_name": norm_str(issue.get("city_name")),
            "state_abbr": norm_str(issue.get("state_abbr")).upper(),
            "issue_id": iid,
            "issue_date": norm_str(issue.get("issue_date")),
            "variant": str(args.variant),
            "target_city_name": norm_str(r.get("target_city_name")) or norm_str(issue.get("city_name")),
            "target_state_abbr": norm_str(r.get("target_state_abbr")) or norm_str(issue.get("state_abbr")).upper(),
            "jurisdiction_name": norm_str(r.get("jurisdiction_name")),
            "jurisdiction_type": norm_str(r.get("jurisdiction_type")),
            "jurisdiction_state_abbr": norm_str(r.get("jurisdiction_state_abbr")),
            "jurisdiction_match_city": norm_str(r.get("jurisdiction_match_city")),
            "jurisdiction_evidence_quotes": r.get("jurisdiction_evidence_quotes")
            if isinstance(r.get("jurisdiction_evidence_quotes"), list)
            else [],
            "issue_completeness": norm_str(r.get("issue_completeness")),
            "possible_missing_pages": norm_str(r.get("possible_missing_pages")),
            "law_presence": norm_str(r.get("law_presence")),
            "law_presence_evidence_anchor_ids": r.get("law_presence_evidence_anchor_ids")
            if isinstance(r.get("law_presence_evidence_anchor_ids"), list)
            else [],
            "contains_verbatim_zoning_law": norm_str(r.get("contains_verbatim_zoning_law")),
            "law_text_sufficiency": norm_str(r.get("law_text_sufficiency")),
            "clean_text_sufficiency": norm_str(r.get("clean_text_sufficiency")),
            "legal_status": norm_str(r.get("legal_status")),
            "enactment_signal": norm_str(r.get("enactment_signal")),
            "proposal_signal": norm_str(r.get("proposal_signal")),
            "non_law_zoning_narrative_present": norm_str(r.get("non_law_zoning_narrative_present")),
            "classification_scope": norm_str(r.get("classification_scope")),
            "issue_class": norm_str(r.get("issue_class")),
            "confidence_0_to_1": safe_float(r.get("confidence_0_to_1"), default=0.0),
            "evidence_anchors": r.get("evidence_anchors") if isinstance(r.get("evidence_anchors"), list) else [],
            "evidence_quotes": r.get("evidence_quotes") if isinstance(r.get("evidence_quotes"), list) else [],
            "status_evidence_quotes": r.get("status_evidence_quotes") if isinstance(r.get("status_evidence_quotes"), list) else [],
            "category_shares": r.get("category_shares") if isinstance(r.get("category_shares"), dict) else {k: 0.0 for k in FIXED_CATEGORY_KEYS},
            "quality_notes": r.get("quality_notes") if isinstance(r.get("quality_notes"), list) else [],
            "clean_chars": int(len(norm_str(clean.get("clean_zoning_text")))),
            "clean_kept_anchor_count": int(clean.get("kept_anchor_count") or 0),
            "primary_parse_ok": int((primary.get(iid) or {}).get("parse_ok") or 0),
            "secondary_parse_ok": int((secondary.get(iid) or {}).get("parse_ok") or 0),
            "law_presence_hint_available": int(iid in law_presence_map),
        }
        rows.append(row)
        for cat in FIXED_CATEGORY_KEYS:
            flat_rows.append(
                {
                    "newspaper_slug": slug,
                    "city_key": row["city_key"],
                    "issue_id": iid,
                    "issue_date": row["issue_date"],
                    "issue_class": row["issue_class"],
                    "category": cat,
                    "share": safe_float((row["category_shares"] or {}).get(cat), default=0.0),
                }
            )

    rows = sorted(rows, key=lambda r: (norm_str(r.get("newspaper_slug")), norm_str(r.get("issue_date")), norm_str(r.get("issue_id"))))
    flat_df = pd.DataFrame.from_records(flat_rows)
    main_df = pd.DataFrame.from_records(rows)

    write_jsonl(out_dir / "issue_labels_v3.jsonl", rows)
    write_csv(out_dir / "issue_labels_v3.csv", main_df)
    write_csv(out_dir / "issue_category_shares_v3.csv", flat_df)

    by_class = (
        main_df.groupby(["newspaper_slug", "issue_class"], dropna=False).size().rename("issue_count").reset_index()
        if not main_df.empty
        else pd.DataFrame(columns=["newspaper_slug", "issue_class", "issue_count"])
    )
    write_csv(out_dir / "issue_class_summary.csv", by_class)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "clean_jsonl": str(clean_path),
        "law_presence_jsonl": str(law_presence_path) if law_presence_path is not None else "",
        "newspapers": newspapers,
        "variant": str(args.variant),
        "classification_input_mode": str(args.classification_input_mode),
        "require_clean_for_law": bool(args.require_clean_for_law),
        "strict_contradiction_mode": bool(args.strict_contradiction_mode),
        "model_primary": str(args.model_primary),
        "model_secondary": str(args.model_secondary),
        "adjudicator_model": norm_str(args.adjudicator_model) or str(args.model_primary),
        "issue_count": int(len(rows)),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eprint(f"Done. classification output: {out_dir}")


if __name__ == "__main__":
    main()
