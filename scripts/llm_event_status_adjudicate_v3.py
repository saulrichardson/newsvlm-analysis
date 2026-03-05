#!/usr/bin/env python3
"""
LLM event-level status adjudication for fullnewspaper v3.

Purpose:
- Resolve enacted vs proposal ambiguity after event deduplication.
- Assign event_status and update strict event_type for downstream counting/export.
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


ALLOWED_EVENT_TYPE = {"full_ordinance", "amendment", "proposal", "notice", "building_code", "mixed", "other", "non_law", "uncertain"}
ALLOWED_YNU = {"yes", "no", "uncertain"}
ALLOWED_JUR_TYPE = {"city", "town", "village", "township", "borough", "county", "regional", "state", "unknown"}
ALLOWED_STATUS = {
    "enacted_complete",
    "enacted_partial_installment",
    "proposed_full_text",
    "proposed_partial_installment",
    "notice",
    "uncertain",
}
ALLOWED_DECISION = {"pass", "fail"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM event-level status adjudication v3.")
    ap.add_argument(
        "--classification-jsonl",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/classification/issue_labels_v3.jsonl",
    )
    ap.add_argument(
        "--clean-jsonl",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/cleaning/clean_zoning_text.jsonl",
    )
    ap.add_argument(
        "--event-clusters-csv",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/events/ordinance_event_clusters_v3.csv",
    )
    ap.add_argument(
        "--event-assignments-csv",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/events/issue_event_assignments_v3.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/events_status",
    )
    ap.add_argument("--model", default="openai:gpt-5")
    ap.add_argument(
        "--max-member-clean-chars",
        type=int,
        default=0,
        help="Hard cap for per-member clean text included in prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument(
        "--max-members-per-event",
        type=int,
        default=0,
        help="Hard cap for number of event members included in prompts. 0 means include all members. If positive and exceeded, the run fails (no slicing).",
    )
    ap.add_argument(
        "--strict-qualification-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, applies strict full/amend qualification veto logic.",
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
    s = norm_str(text)
    if max_chars > 0 and len(s) > max_chars:
        raise ValueError(
            f"{label} length {len(s)} exceeds hard cap {max_chars}. "
            "Clipping/condensing is disabled; increase cap or reduce input scope."
        )
    return s


def _build_prompt(
    event_row: dict[str, Any],
    members: list[dict[str, Any]],
    *,
    max_member_chars: int,
    max_members: int,
    strict_qualification_mode: bool,
) -> str:
    slug = norm_str(event_row.get("newspaper_slug"))
    eid = norm_str(event_row.get("ordinance_event_id"))
    schema = {
        "newspaper_slug": slug,
        "ordinance_event_id": eid,
        "target_city_name": norm_str(event_row.get("target_city_name")),
        "target_state_abbr": norm_str(event_row.get("target_state_abbr")).upper(),
        "governing_jurisdiction_name": norm_str(event_row.get("target_city_name")),
        "governing_jurisdiction_type": "city",
        "governing_jurisdiction_state_abbr": norm_str(event_row.get("target_state_abbr")).upper(),
        "jurisdiction_match_city": "yes",
        "jurisdiction_evidence_quotes": ["short jurisdiction quote"],
        "valid_zoning_law_event": "yes",
        "event_type_recommendation": "full_ordinance",
        "event_status": "enacted_complete",
        "qualification_decision": "pass",
        "qualification_reasons": ["short reason"],
        "completeness_confidence": 0.85,
        "event_status_evidence_quotes": ["short quote 1", "short quote 2"],
        "status_notes": [""],
    }
    lines: list[str] = []
    lines.append("Adjudicate legal-event status after deduplication.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Interpretation rules:")
    lines.append("- enacted_complete: enacted/in-force legal event with sufficiently complete publication text.")
    lines.append("- enacted_partial_installment: enacted/in-force legal event but only partial/installment text in this cluster.")
    lines.append("- proposed_full_text: proposal/hearing/draft with broad full-code legal text.")
    lines.append("- proposed_partial_installment: proposal/hearing/draft and only partial/installment legal text.")
    lines.append("- uncertain: insufficient evidence.")
    lines.append("- full_ordinance must be enacted (not proposal-only).")
    lines.append("- Determine the governing legal jurisdiction from the legal text (city/town/village/etc).")
    lines.append("- jurisdiction_match_city=yes only when governing jurisdiction matches the target city for this panel.")
    lines.append("- Treat same-place municipal variants as a match (for example Town/Village/City of the same place name as the target).")
    lines.append("- If jurisdiction differs, set jurisdiction_match_city=no and provide jurisdiction_evidence_quotes.")
    lines.append("- If the text shows conflicting jurisdiction names (for example two different towns/villages), treat jurisdiction as uncertain or no-match; do not pass full_ordinance for the target city.")
    lines.append("- proposal-only clusters must recommend event_type_recommendation=proposal.")
    lines.append("- Keep recommendations within: full_ordinance|amendment|proposal|notice|building_code|mixed|other|non_law|uncertain.")
    lines.append("- Use event_type_recommendation=building_code when this is legal building-code text, not a zoning-law event.")
    lines.append("- Set valid_zoning_law_event=no for building-code-only legal events.")
    lines.append("- Provide at least 2 event_status_evidence_quotes when possible.")
    lines.append("- Set qualification_decision=pass only if recommendation is supported by member composition and quoted evidence.")
    if bool(strict_qualification_mode):
        lines.append("- Strict mode: do not pass full_ordinance when proposal language dominates or full-code evidence is weak.")
        lines.append("- Strict mode: do not pass full_ordinance when jurisdiction_match_city != yes.")
        lines.append("- Strict mode: mixed full/amend/notice composition should fail unless one legal action clearly dominates.")
        lines.append("- Strict mode: if jurisdiction evidence is internally inconsistent, set qualification_decision=fail for full_ordinance.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"[EVENT] newspaper_slug={slug} city_key={norm_str(event_row.get('city_key'))} "
        f"target_city={norm_str(event_row.get('target_city_name'))},{norm_str(event_row.get('target_state_abbr'))} "
        f"ordinance_event_id={eid} event_type={norm_str(event_row.get('event_type'))} "
        f"issue_count={norm_str(event_row.get('issue_count'))} "
        f"first_issue_date={norm_str(event_row.get('first_issue_date'))} "
        f"last_issue_date={norm_str(event_row.get('last_issue_date'))}"
    )
    lines.append(
        "member_class_counts="
        + json.dumps(
            {
                "full_like": int(event_row.get("member_full_like_count") or 0),
                "amend_like": int(event_row.get("member_amend_like_count") or 0),
                "proposal_like": int(event_row.get("member_proposal_like_count") or 0),
                "notice_like": int(event_row.get("member_notice_like_count") or 0),
                "building_like": int(event_row.get("member_building_like_count") or 0),
                "nonlaw_like": int(event_row.get("member_nonlaw_like_count") or 0),
            },
            ensure_ascii=False,
        )
    )
    lines.append(f"event_summary={norm_str(event_row.get('summary'))}")
    lines.append("")
    lines.append("Event members:")
    if int(max_members) > 0 and len(members) > int(max_members):
        raise ValueError(
            f"event_id={eid} has {len(members)} members, exceeding hard cap {int(max_members)}. "
            "Member slicing is disabled; increase --max-members-per-event or set 0 for no cap."
        )
    selected = members
    for m in selected:
        lines.append(
            f"- issue_id={norm_str(m.get('issue_id'))} date={norm_str(m.get('issue_date'))} "
            f"publication_role={norm_str(m.get('publication_role'))} "
            f"class={norm_str(m.get('issue_class'))} "
            f"legal_status={norm_str(m.get('legal_status'))} "
            f"enactment_signal={norm_str(m.get('enactment_signal'))} "
            f"proposal_signal={norm_str(m.get('proposal_signal'))} "
            f"issue_completeness={norm_str(m.get('issue_completeness'))} "
            f"target_city={norm_str(m.get('target_city_name'))},{norm_str(m.get('target_state_abbr'))} "
            f"issue_jurisdiction={norm_str(m.get('issue_jurisdiction_name'))} "
            f"issue_jurisdiction_match_city={norm_str(m.get('issue_jurisdiction_match_city'))}"
        )
        lines.append(
            _bounded_text_or_fail(
                norm_str(m.get("clean_zoning_text")),
                max_member_chars,
                label=f"event_id={eid} issue_id={norm_str(m.get('issue_id'))} clean_zoning_text",
            )
        )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _coerce(parsed: dict[str, Any], event_row: dict[str, Any], *, strict_qualification_mode: bool) -> dict[str, Any]:
    out = dict(parsed or {})
    reco = enum(out.get("event_type_recommendation"), ALLOWED_EVENT_TYPE, "uncertain")
    status = enum(out.get("event_status"), ALLOWED_STATUS, "uncertain")
    decision = enum(out.get("qualification_decision"), ALLOWED_DECISION, "fail")
    reasons = [norm_str(x) for x in (out.get("qualification_reasons") if isinstance(out.get("qualification_reasons"), list) else []) if norm_str(x)]
    conf = safe_float(out.get("completeness_confidence"), default=0.0)
    quotes = [norm_str(x) for x in (out.get("event_status_evidence_quotes") if isinstance(out.get("event_status_evidence_quotes"), list) else []) if norm_str(x)]
    jur_quotes = [norm_str(x) for x in (out.get("jurisdiction_evidence_quotes") if isinstance(out.get("jurisdiction_evidence_quotes"), list) else []) if norm_str(x)]
    notes = [norm_str(x) for x in (out.get("status_notes") if isinstance(out.get("status_notes"), list) else []) if norm_str(x)]
    target_city = norm_str(out.get("target_city_name")) or norm_str(event_row.get("target_city_name"))
    target_state = norm_str(out.get("target_state_abbr")).upper() or norm_str(event_row.get("target_state_abbr")).upper()
    governing_jurisdiction_name = norm_str(out.get("governing_jurisdiction_name"))
    governing_jurisdiction_type = enum(out.get("governing_jurisdiction_type"), ALLOWED_JUR_TYPE, "unknown")
    governing_jurisdiction_state_abbr = norm_str(out.get("governing_jurisdiction_state_abbr")).upper()
    jurisdiction_match_city = enum(out.get("jurisdiction_match_city"), ALLOWED_YNU, "uncertain")
    valid_zoning_law_event = enum(out.get("valid_zoning_law_event"), ALLOWED_YNU, "uncertain")
    original = enum(event_row.get("event_type"), ALLOWED_EVENT_TYPE, "uncertain")
    full_like = int(event_row.get("member_full_like_count") or 0)
    amend_like = int(event_row.get("member_amend_like_count") or 0)
    proposal_like = int(event_row.get("member_proposal_like_count") or 0)
    notice_like = int(event_row.get("member_notice_like_count") or 0)
    building_like = int(event_row.get("member_building_like_count") or 0)

    final = reco
    if status in {"proposed_full_text", "proposed_partial_installment"}:
        if final in {"full_ordinance", "amendment", "uncertain", "mixed", "other"}:
            final = "proposal"
            reasons.append("coerced_to_proposal_by_status")
    elif status in {"enacted_complete", "enacted_partial_installment"}:
        if final not in {"full_ordinance", "amendment", "notice", "mixed", "other", "non_law", "uncertain"}:
            final = original
    else:
        # uncertain status: do not aggressively rewrite type.
        if final == "uncertain":
            final = original

    if bool(strict_qualification_mode):
        if final == "full_ordinance":
            impossible = (
                status not in {"enacted_complete", "enacted_partial_installment"}
                or decision != "pass"
                or full_like <= 0
                or proposal_like > full_like
                or amend_like > full_like
                or jurisdiction_match_city != "yes"
            )
            if impossible:
                if jurisdiction_match_city == "no":
                    final = "other"
                elif status in {"proposed_full_text", "proposed_partial_installment"}:
                    final = "proposal"
                elif amend_like > full_like:
                    final = "amendment"
                elif notice_like > 0 and full_like == 0:
                    final = "notice"
                elif building_like > 0 and full_like == 0 and amend_like == 0:
                    final = "building_code"
                else:
                    final = "uncertain"
                reasons.append("strict_full_qualification_veto")
                if jurisdiction_match_city == "no":
                    reasons.append("strict_full_jurisdiction_mismatch")
        if final == "amendment":
            impossible = (
                decision != "pass"
                or amend_like <= 0
                or (status == "enacted_complete" and amend_like < full_like)
            )
            if impossible:
                if status in {"proposed_full_text", "proposed_partial_installment"}:
                    final = "proposal"
                elif full_like > amend_like and decision == "pass":
                    final = "full_ordinance"
                elif building_like > 0 and amend_like == 0 and full_like == 0:
                    final = "building_code"
                else:
                    final = "uncertain"
                reasons.append("strict_amendment_qualification_veto")
        if final == "amendment" and jurisdiction_match_city == "no":
            final = "other"
            reasons.append("strict_amendment_jurisdiction_mismatch")
        if final == "building_code":
            if valid_zoning_law_event == "yes":
                reasons.append("building_code_marked_despite_valid_zoning_signal")

    return {
        "newspaper_slug": norm_str(event_row.get("newspaper_slug")),
        "city_key": norm_str(event_row.get("city_key")),
        "target_city_name": target_city,
        "target_state_abbr": target_state,
        "ordinance_event_id": norm_str(event_row.get("ordinance_event_id")),
        "governing_jurisdiction_name": governing_jurisdiction_name,
        "governing_jurisdiction_type": governing_jurisdiction_type,
        "governing_jurisdiction_state_abbr": governing_jurisdiction_state_abbr,
        "jurisdiction_match_city": jurisdiction_match_city,
        "jurisdiction_evidence_quotes": jur_quotes,
        "valid_zoning_law_event": valid_zoning_law_event,
        "event_type_raw": original,
        "event_type_original": original,
        "event_type_recommendation": reco,
        "event_type_final": final,
        "event_status": status,
        "qualification_decision": decision,
        "qualification_reasons": reasons,
        "completeness_confidence": max(0.0, min(1.0, conf if conf == conf else 0.0)),
        "event_status_evidence_quotes": quotes,
        "status_notes": notes,
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_rows = iter_jsonl(Path(args.classification_jsonl).expanduser().resolve())
    clean_rows = iter_jsonl(Path(args.clean_jsonl).expanduser().resolve())
    ev_df = pd.read_csv(Path(args.event_clusters_csv).expanduser().resolve())
    as_df = pd.read_csv(Path(args.event_assignments_csv).expanduser().resolve())

    if ev_df.empty or as_df.empty:
        raise SystemExit("Event cluster/assignment inputs are empty.")

    cls_map = {norm_str(r.get("issue_id")): r for r in cls_rows if norm_str(r.get("issue_id"))}
    clean_map = {norm_str(r.get("issue_id")): r for r in clean_rows if norm_str(r.get("issue_id"))}

    reqs: list[tuple[str, str]] = []
    meta: dict[str, dict[str, Any]] = {}
    for r in ev_df.to_dict(orient="records"):
        slug = norm_str(r.get("newspaper_slug"))
        eid = norm_str(r.get("ordinance_event_id"))
        if not slug or not eid:
            continue
        members_df = as_df[
            (as_df["newspaper_slug"].astype(str) == slug)
            & (as_df["ordinance_event_id"].astype(str) == eid)
        ].copy()
        members_df = members_df.sort_values(["issue_date", "issue_id"])
        members: list[dict[str, Any]] = []
        full_like = 0
        amend_like = 0
        proposal_like = 0
        notice_like = 0
        nonlaw_like = 0
        for m in members_df.to_dict(orient="records"):
            iid = norm_str(m.get("issue_id"))
            c = cls_map.get(iid) or {}
            cl = clean_map.get(iid) or {}
            issue_class = norm_str(c.get("issue_class") or m.get("issue_class"))
            if issue_class in {"full_code_comprehensive", "full_code_recodification", "full_code_partial_enacted"}:
                full_like += 1
            elif issue_class in {"amendment_substantial", "amendment_targeted"}:
                amend_like += 1
            elif issue_class in {"full_code_proposed", "full_code_partial_proposed", "amendment_proposed"}:
                proposal_like += 1
            elif issue_class in {"zoning_notice_procedural", "zoning_notice_adoption"}:
                notice_like += 1
            elif issue_class in {"building_code_non_zoning"}:
                # Track separately so status adjudication can route these away from zoning buckets.
                pass
            else:
                nonlaw_like += 1
            members.append(
                {
                    "issue_id": iid,
                    "issue_date": norm_str(m.get("issue_date")),
                    "publication_role": norm_str(m.get("publication_role")),
                    "target_city_name": norm_str(c.get("target_city_name") or m.get("target_city_name")),
                    "target_state_abbr": norm_str(c.get("target_state_abbr") or m.get("target_state_abbr")).upper(),
                    "issue_class": issue_class,
                    "legal_status": norm_str(c.get("legal_status")),
                    "enactment_signal": norm_str(c.get("enactment_signal")),
                    "proposal_signal": norm_str(c.get("proposal_signal")),
                    "issue_completeness": norm_str(c.get("issue_completeness")),
                    "issue_jurisdiction_name": norm_str(c.get("jurisdiction_name") or m.get("issue_jurisdiction_name")),
                    "issue_jurisdiction_match_city": norm_str(c.get("jurisdiction_match_city") or m.get("issue_jurisdiction_match_city")),
                    "clean_zoning_text": norm_str(cl.get("clean_zoning_text")),
                }
            )
        building_like = int(sum(1 for mm in members if norm_str(mm.get("issue_class")) == "building_code_non_zoning"))
        r["member_full_like_count"] = int(full_like)
        r["member_amend_like_count"] = int(amend_like)
        r["member_proposal_like_count"] = int(proposal_like)
        r["member_notice_like_count"] = int(notice_like)
        r["member_building_like_count"] = int(building_like)
        r["member_nonlaw_like_count"] = int(nonlaw_like)
        cid = f"event_status_v3::{slug}::{eid}"
        prompt = _build_prompt(
            r,
            members,
            max_member_chars=int(args.max_member_clean_chars),
            max_members=int(args.max_members_per_event),
            strict_qualification_mode=bool(args.strict_qualification_mode),
        )
        reqs.append((cid, prompt))
        meta[cid] = {"event_row": r}

    req_dir = out_dir / "requests"
    res_dir = out_dir / "results"
    write_openai_requests(req_dir / "openai_requests_shard000.jsonl", reqs, model=str(args.model))
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

    res = read_result_jsonl(res_dir)
    err = read_error_jsonl(res_dir)

    adjud_rows: list[dict[str, Any]] = []
    for cid, _ in reqs:
        event_row = meta[cid]["event_row"]
        output_text = ""
        parse_ok = 0
        if cid in res:
            body = (((res[cid].get("response") or {}).get("body")) or {})
            output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in err:
            body = (((err[cid].get("response") or {}).get("body")) or {})
            output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
        parsed = parse_json_from_text(output_text)
        if isinstance(parsed, dict) and parsed:
            parse_ok = 1
        c = _coerce(
            parsed if isinstance(parsed, dict) else {},
            event_row,
            strict_qualification_mode=bool(args.strict_qualification_mode),
        )
        c["parse_ok"] = parse_ok
        adjud_rows.append(c)

    adj_df = pd.DataFrame.from_records(adjud_rows)
    if adj_df.empty:
        raise SystemExit("No event adjudications parsed.")

    # Merge into event and assignment outputs.
    merge_cols = ["newspaper_slug", "city_key", "ordinance_event_id"]
    ev_out = ev_df.copy()
    ev_out["event_type_original"] = ev_out["event_type"].astype(str)
    for c in (
        "target_city_name",
        "target_state_abbr",
        "governing_jurisdiction_name",
        "governing_jurisdiction_type",
        "governing_jurisdiction_state_abbr",
        "jurisdiction_match_city",
        "jurisdiction_evidence_quotes",
        "valid_zoning_law_event",
    ):
        if c in ev_out.columns:
            ev_out = ev_out.drop(columns=[c])
    ev_out = ev_out.merge(
        adj_df[
            [
                "newspaper_slug",
                "city_key",
                "ordinance_event_id",
                "event_type_recommendation",
                "event_type_final",
                "event_status",
                "qualification_decision",
                "qualification_reasons",
                "completeness_confidence",
                "event_status_evidence_quotes",
                "target_city_name",
                "target_state_abbr",
                "governing_jurisdiction_name",
                "governing_jurisdiction_type",
                "governing_jurisdiction_state_abbr",
                "jurisdiction_match_city",
                "jurisdiction_evidence_quotes",
                "valid_zoning_law_event",
                "status_notes",
                "parse_ok",
            ]
        ],
        on=merge_cols,
        how="left",
    )
    ev_out["event_type"] = ev_out["event_type_final"].fillna(ev_out["event_type"]).astype(str)
    ev_out = ev_out.drop(columns=["event_type_final"])
    ev_out = ev_out.rename(columns={"parse_ok": "event_status_parse_ok"})
    ev_out = ev_out.sort_values(["newspaper_slug", "first_issue_date", "ordinance_event_id"]).reset_index(drop=True)

    as_out = as_df.copy()
    as_out["event_type_original"] = as_out["event_type"].astype(str)
    for c in (
        "target_city_name",
        "target_state_abbr",
        "governing_jurisdiction_name",
        "governing_jurisdiction_type",
        "governing_jurisdiction_state_abbr",
        "jurisdiction_match_city",
        "jurisdiction_evidence_quotes",
        "valid_zoning_law_event",
    ):
        if c in as_out.columns:
            as_out = as_out.drop(columns=[c])
    as_out = as_out.merge(
        adj_df[
            [
                "newspaper_slug",
                "city_key",
                "ordinance_event_id",
                "event_type_recommendation",
                "event_type_final",
                "event_status",
                "qualification_decision",
                "qualification_reasons",
                "completeness_confidence",
                "event_status_evidence_quotes",
                "target_city_name",
                "target_state_abbr",
                "governing_jurisdiction_name",
                "governing_jurisdiction_type",
                "governing_jurisdiction_state_abbr",
                "jurisdiction_match_city",
                "jurisdiction_evidence_quotes",
                "valid_zoning_law_event",
                "status_notes",
            ]
        ],
        on=merge_cols,
        how="left",
    )
    as_out["event_type"] = as_out["event_type_final"].fillna(as_out["event_type"]).astype(str)
    as_out = as_out.drop(columns=["event_type_final"])
    as_out = as_out.sort_values(["newspaper_slug", "issue_date", "issue_id"]).reset_index(drop=True)

    # Summary for QA/reporting.
    summary = (
        ev_out.groupby(["event_type", "event_status", "qualification_decision"], dropna=False)["ordinance_event_id"]
        .nunique()
        .reset_index(name="event_count")
        .sort_values(["event_type", "event_status", "qualification_decision"])
    )

    write_jsonl(out_dir / "event_status_adjudications_v3.jsonl", adjud_rows)
    write_csv(out_dir / "ordinance_event_clusters_status_v3.csv", ev_out)
    write_csv(out_dir / "issue_event_assignments_status_v3.csv", as_out)
    write_csv(out_dir / "event_status_summary_v3.csv", summary)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "classification_jsonl": str(Path(args.classification_jsonl).expanduser().resolve()),
        "clean_jsonl": str(Path(args.clean_jsonl).expanduser().resolve()),
        "event_clusters_csv": str(Path(args.event_clusters_csv).expanduser().resolve()),
        "event_assignments_csv": str(Path(args.event_assignments_csv).expanduser().resolve()),
        "output_dir": str(out_dir),
        "model": str(args.model),
        "strict_qualification_mode": bool(args.strict_qualification_mode),
        "event_count": int(len(ev_out)),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eprint(f"Done. event-status output: {out_dir}")


if __name__ == "__main__":
    main()
