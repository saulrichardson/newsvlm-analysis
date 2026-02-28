#!/usr/bin/env python3
"""
LLM-only newspaper-level event clustering and deduplication for v3 pipeline.
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
    write_csv,
    write_jsonl,
    write_openai_requests,
)


ALLOWED_VARIANTS = {"direct_cluster", "pairwise_then_cluster"}
ALLOWED_EVENT_TYPE = {"full_ordinance", "amendment", "proposal", "notice", "building_code", "mixed", "other", "non_law", "uncertain"}
ALLOWED_ROLE = {"first_publication", "republication", "followup_notice", "distinct_event", "uncertain"}
FULL_CLASSES = {"full_code_comprehensive", "full_code_recodification", "full_code_partial_enacted"}
AMEND_CLASSES = {"amendment_substantial", "amendment_targeted"}
PROPOSAL_CLASSES = {"full_code_proposed", "full_code_partial_proposed", "amendment_proposed"}
NOTICE_CLASSES = {"zoning_notice_procedural", "zoning_notice_adoption"}
BUILDING_CLASSES = {"building_code_non_zoning"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM-only event clustering v3.")
    ap.add_argument(
        "--classification-jsonl",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/classification/issue_labels_v3.jsonl",
    )
    ap.add_argument(
        "--clean-jsonl",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/cleaning/clean_zoning_text.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/events",
    )
    ap.add_argument("--variant", default="direct_cluster", choices=sorted(ALLOWED_VARIANTS))
    ap.add_argument("--model", default="openai:gpt-5")
    ap.add_argument(
        "--max-issue-excerpt-chars",
        type=int,
        default=0,
        help="Hard cap for per-issue clean text included in prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument("--concurrency", type=int, default=2)
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


def _build_direct_prompt(slug: str, issue_rows: list[dict[str, Any]], max_chars: int) -> str:
    schema = {
        "newspaper_slug": slug,
        "events": [
            {
                "ordinance_event_id": "E001",
                "event_type": "full_ordinance",
                "summary": "",
                "first_issue_date": "",
                "last_issue_date": "",
                "issue_ids": ["issue_a", "issue_b"],
                "publication_roles": [{"issue_id": "issue_a", "role": "first_publication"}],
                "confidence_0_to_1": 0.8,
            }
        ],
    }
    lines: list[str] = []
    lines.append("Cluster issue-level zoning records into distinct ordinance/amendment events.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Objective: deduplicate republications and follow-up notices.")
    lines.append("Allowed event_type: full_ordinance|amendment|proposal|notice|building_code|mixed|other|non_law|uncertain")
    lines.append("Allowed publication_roles.role: first_publication|republication|followup_notice|distinct_event|uncertain")
    lines.append("Use full_ordinance/amendment only for enacted legal actions.")
    lines.append("If issues are proposal/draft/hearing without enactment, use event_type=proposal.")
    lines.append("Use event_type=building_code when the cluster is legal building-code language but not a zoning-law event.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(f"Newspaper: {slug}")
    lines.append("Issues:")
    for r in issue_rows:
        lines.append("")
        lines.append(
            f"[ISSUE] issue_id={norm_str(r.get('issue_id'))} "
            f"date={norm_str(r.get('issue_date'))} "
            f"target_city={norm_str(r.get('target_city_name'))},{norm_str(r.get('target_state_abbr'))} "
            f"class={norm_str(r.get('issue_class'))} "
            f"law={norm_str(r.get('contains_verbatim_zoning_law'))} "
            f"suff={norm_str(r.get('law_text_sufficiency'))} "
                f"status={norm_str(r.get('legal_status'))} "
                f"enact={norm_str(r.get('enactment_signal'))} "
                f"proposal={norm_str(r.get('proposal_signal'))} "
                f"completeness={norm_str(r.get('issue_completeness'))} "
                f"jurisdiction={norm_str(r.get('issue_jurisdiction_name'))} "
                f"jurisdiction_match_city={norm_str(r.get('issue_jurisdiction_match_city'))}"
        )
        lines.append(
            _bounded_text_or_fail(
                norm_str(r.get("clean_zoning_text")),
                max_chars,
                label=f"slug={slug} issue_id={norm_str(r.get('issue_id'))} clean_zoning_text",
            )
        )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _build_pairwise_prompt(slug: str, issue_rows: list[dict[str, Any]], max_chars: int) -> str:
    schema = {
        "newspaper_slug": slug,
        "same_event_pairs": [["issue_a", "issue_b"]],
        "notes": "",
    }
    lines: list[str] = []
    lines.append("Identify issue pairs that refer to the same ordinance/amendment event.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Only include a pair when confidence is high that both rows are the same underlying legal event.")
    lines.append("Do not pair unrelated events.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(f"Newspaper: {slug}")
    lines.append("Issues:")
    for r in issue_rows:
        lines.append("")
        lines.append(
            f"[ISSUE] issue_id={norm_str(r.get('issue_id'))} "
            f"date={norm_str(r.get('issue_date'))} "
            f"class={norm_str(r.get('issue_class'))} "
            f"status={norm_str(r.get('legal_status'))} "
            f"enact={norm_str(r.get('enactment_signal'))} "
            f"proposal={norm_str(r.get('proposal_signal'))} "
            f"completeness={norm_str(r.get('issue_completeness'))}"
        )
        lines.append(
            _bounded_text_or_fail(
                norm_str(r.get("clean_zoning_text")),
                max_chars,
                label=f"slug={slug} issue_id={norm_str(r.get('issue_id'))} clean_zoning_text",
            )
        )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _build_cluster_label_prompt(slug: str, cluster_rows: list[dict[str, Any]], max_chars: int) -> str:
    schema = {
        "newspaper_slug": slug,
        "events": [
            {
                "ordinance_event_id": "E001",
                "event_type": "amendment",
                "summary": "",
                "first_issue_date": "",
                "last_issue_date": "",
                "issue_ids": ["issue_a", "issue_b"],
                "publication_roles": [{"issue_id": "issue_a", "role": "first_publication"}],
                "confidence_0_to_1": 0.8,
            }
        ],
    }
    lines: list[str] = []
    lines.append("Label precomputed issue clusters as ordinance/amendment events.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Allowed event_type: full_ordinance|amendment|proposal|notice|building_code|mixed|other|non_law|uncertain")
    lines.append("Allowed publication_roles.role: first_publication|republication|followup_notice|distinct_event|uncertain")
    lines.append("Use event_type=building_code when this cluster is a building-code legal action, not zoning-law.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(f"Newspaper: {slug}")
    lines.append("Cluster candidates:")
    for cr in cluster_rows:
        lines.append("")
        lines.append(f"[CLUSTER] ordinance_event_id={norm_str(cr.get('ordinance_event_id'))}")
        for r in cr.get("issues", []):
            lines.append(
                f"  - issue_id={norm_str(r.get('issue_id'))} "
                f"date={norm_str(r.get('issue_date'))} "
                f"class={norm_str(r.get('issue_class'))} "
                f"status={norm_str(r.get('legal_status'))} "
                f"enact={norm_str(r.get('enactment_signal'))} "
                f"proposal={norm_str(r.get('proposal_signal'))} "
                f"completeness={norm_str(r.get('issue_completeness'))} "
                f"target_city={norm_str(r.get('target_city_name'))},{norm_str(r.get('target_state_abbr'))} "
                f"jurisdiction={norm_str(r.get('issue_jurisdiction_name'))} "
                f"jurisdiction_match_city={norm_str(r.get('issue_jurisdiction_match_city'))}"
            )
            lines.append(
                "    "
                + _bounded_text_or_fail(
                    norm_str(r.get("clean_zoning_text")),
                    max_chars,
                    label=f"slug={slug} event_id={norm_str(cr.get('ordinance_event_id'))} issue_id={norm_str(r.get('issue_id'))} clean_zoning_text",
                ).replace("\n", "\n    ")
            )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _parse_events(parsed_obj: dict[str, Any], issue_ids: set[str]) -> list[dict[str, Any]]:
    events_raw = parsed_obj.get("events") if isinstance(parsed_obj.get("events"), list) else []
    out: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for i, ev in enumerate(events_raw, start=1):
        if not isinstance(ev, dict):
            continue
        eid = norm_str(ev.get("ordinance_event_id")) or f"E{i:03d}"
        ids = []
        if isinstance(ev.get("issue_ids"), list):
            for x in ev.get("issue_ids"):
                sid = norm_str(x)
                if sid and sid in issue_ids and sid not in ids:
                    ids.append(sid)
        if not ids:
            continue
        role_map: dict[str, str] = {}
        roles_raw = ev.get("publication_roles")
        if isinstance(roles_raw, list):
            for rr in roles_raw:
                if not isinstance(rr, dict):
                    continue
                sid = norm_str(rr.get("issue_id"))
                if sid in ids:
                    role_map[sid] = enum(rr.get("role"), ALLOWED_ROLE, "uncertain")
        pub_roles = [{"issue_id": sid, "role": role_map.get(sid, "uncertain")} for sid in ids]
        used_ids.update(ids)
        out.append(
            {
                "ordinance_event_id": eid,
                "event_type": enum(ev.get("event_type"), ALLOWED_EVENT_TYPE, "uncertain"),
                "summary": norm_str(ev.get("summary")),
                "first_issue_date": norm_str(ev.get("first_issue_date")),
                "last_issue_date": norm_str(ev.get("last_issue_date")),
                "issue_ids": ids,
                "publication_roles": pub_roles,
                "confidence_0_to_1": float(ev.get("confidence_0_to_1") or 0.0),
            }
        )
    # Guarantee coverage: assign missing issues as singleton uncertain events.
    for sid in sorted(issue_ids - used_ids):
        out.append(
            {
                "ordinance_event_id": f"E_MISS_{sid}",
                "event_type": "uncertain",
                "summary": "",
                "first_issue_date": "",
                "last_issue_date": "",
                "issue_ids": [sid],
                "publication_roles": [{"issue_id": sid, "role": "uncertain"}],
                "confidence_0_to_1": 0.0,
            }
        )
    return out


class _UF:
    def __init__(self, nodes: list[str]) -> None:
        self.p = {n: n for n in nodes}

    def find(self, x: str) -> str:
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _flatten_events(slug: str, city_key: str, events: list[dict[str, Any]], issue_map: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    assign_rows: list[dict[str, Any]] = []
    for ev in events:
        eid = norm_str(ev.get("ordinance_event_id"))
        ids = [norm_str(x) for x in (ev.get("issue_ids") if isinstance(ev.get("issue_ids"), list) else []) if norm_str(x)]
        if not ids:
            continue
        role_map = {}
        for pr in ev.get("publication_roles") if isinstance(ev.get("publication_roles"), list) else []:
            if isinstance(pr, dict):
                role_map[norm_str(pr.get("issue_id"))] = enum(pr.get("role"), ALLOWED_ROLE, "uncertain")
        dates = [norm_str((issue_map.get(iid) or {}).get("issue_date")) for iid in ids]
        dates = [d for d in dates if d]
        first_date = min(dates) if dates else norm_str(ev.get("first_issue_date"))
        last_date = max(dates) if dates else norm_str(ev.get("last_issue_date"))
        first_issue = issue_map.get(ids[0]) or {}
        event_rows.append(
            {
                "newspaper_slug": slug,
                "city_key": city_key,
                "target_city_name": norm_str(first_issue.get("target_city_name")),
                "target_state_abbr": norm_str(first_issue.get("target_state_abbr")),
                "ordinance_event_id": eid,
                "event_type": enum(ev.get("event_type"), ALLOWED_EVENT_TYPE, "uncertain"),
                "issue_count": int(len(ids)),
                "first_issue_date": first_date,
                "last_issue_date": last_date,
                "summary": norm_str(ev.get("summary")),
                "confidence_0_to_1": float(ev.get("confidence_0_to_1") or 0.0),
            }
        )
        for iid in ids:
            assign_rows.append(
                {
                    "newspaper_slug": slug,
                    "city_key": city_key,
                    "target_city_name": norm_str((issue_map.get(iid) or {}).get("target_city_name")),
                    "target_state_abbr": norm_str((issue_map.get(iid) or {}).get("target_state_abbr")),
                    "issue_id": iid,
                    "issue_date": norm_str((issue_map.get(iid) or {}).get("issue_date")),
                    "issue_class": norm_str((issue_map.get(iid) or {}).get("issue_class")),
                    "issue_jurisdiction_name": norm_str((issue_map.get(iid) or {}).get("issue_jurisdiction_name")),
                    "issue_jurisdiction_type": norm_str((issue_map.get(iid) or {}).get("issue_jurisdiction_type")),
                    "issue_jurisdiction_state_abbr": norm_str((issue_map.get(iid) or {}).get("issue_jurisdiction_state_abbr")),
                    "issue_jurisdiction_match_city": norm_str((issue_map.get(iid) or {}).get("issue_jurisdiction_match_city")),
                    "issue_completeness": norm_str((issue_map.get(iid) or {}).get("issue_completeness")),
                    "ordinance_event_id": eid,
                    "event_type": enum(ev.get("event_type"), ALLOWED_EVENT_TYPE, "uncertain"),
                    "publication_role": role_map.get(iid, "uncertain"),
                }
            )
    return event_rows, assign_rows


def _choose_event_type_from_member_classes(raw_event_type: str, issue_classes: list[str]) -> str:
    """
    Enforce event-type coherence with member issue classes.

    This is a deterministic guardrail on top of LLM clustering:
    - full_ordinance requires at least one full_code_* issue class
    - amendment requires at least one amendment_* issue class
    - proposal is used when cluster members are proposal-only classes
    - building_code is used when cluster members are legal building-code only
    """
    et = enum(raw_event_type, ALLOWED_EVENT_TYPE, "uncertain")
    cls = [norm_str(x) for x in issue_classes if norm_str(x)]
    cls_set = set(cls)
    has_full = any(c in FULL_CLASSES for c in cls_set)
    has_amend = any(c in AMEND_CLASSES for c in cls_set)
    has_proposal = any(c in PROPOSAL_CLASSES for c in cls_set)
    has_notice = any(c in NOTICE_CLASSES for c in cls_set)
    has_building = any(c in BUILDING_CLASSES for c in cls_set)

    if has_building and not has_full and not has_amend and not has_proposal and not has_notice:
        if et in {"full_ordinance", "amendment", "proposal", "notice", "mixed", "other", "uncertain"}:
            return "building_code"
        return et

    # Mixed legal composition should not be counted as a pure full/amend event.
    if has_full and has_amend:
        return "mixed"

    # Proposal-only clusters should not be counted as enacted full/amend events.
    if has_proposal and not has_full and not has_amend:
        if et in {"full_ordinance", "amendment", "uncertain", "other", "mixed"}:
            return "proposal"
        return et

    # Coerce impossible full/amend assignments.
    if et == "full_ordinance" and not has_full:
        if has_amend:
            return "amendment"
        if has_proposal:
            return "proposal"
        if has_notice:
            return "notice"
        if has_building:
            return "building_code"
        return "uncertain"
    if et == "amendment" and not has_amend:
        if has_full:
            return "full_ordinance"
        if has_proposal:
            return "proposal"
        if has_notice:
            return "notice"
        if has_building:
            return "building_code"
        return "uncertain"

    # If LLM is vague, prefer class-supported legal type.
    if et in {"uncertain", "other", "mixed"}:
        if has_full and not has_amend:
            return "full_ordinance"
        if has_amend and not has_full:
            return "amendment"
        if has_proposal and not has_full and not has_amend:
            return "proposal"
        if has_notice and not has_full and not has_amend:
            return "notice"
        if has_building and not has_full and not has_amend and not has_proposal and not has_notice:
            return "building_code"
    return et


def _harmonize_event_types(event_df: pd.DataFrame, assign_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if event_df.empty or assign_df.empty:
        return event_df, assign_df, {"changed_event_type_rows": 0, "changed_events": 0}

    df = assign_df.copy()
    e = event_df.copy()

    key_cols = ["newspaper_slug", "ordinance_event_id"]
    if any(c not in df.columns for c in key_cols) or any(c not in e.columns for c in key_cols):
        return event_df, assign_df, {"changed_event_type_rows": 0, "changed_events": 0}

    mapping: dict[tuple[str, str], str] = {}
    changed_events = 0
    for (slug, eid), grp in df.groupby(key_cols, dropna=False):
        slug_n = norm_str(slug)
        eid_n = norm_str(eid)
        if not slug_n or not eid_n:
            continue
        current = norm_str(grp["event_type"].iloc[0]) if "event_type" in grp.columns else "uncertain"
        classes = grp["issue_class"].astype(str).tolist() if "issue_class" in grp.columns else []
        revised = _choose_event_type_from_member_classes(current, classes)
        mapping[(slug_n, eid_n)] = revised
        if revised != current:
            changed_events += 1

    if not mapping:
        return event_df, assign_df, {"changed_event_type_rows": 0, "changed_events": 0}

    before = df["event_type"].astype(str).copy()
    df["event_type"] = df.apply(
        lambda r: mapping.get((norm_str(r.get("newspaper_slug")), norm_str(r.get("ordinance_event_id"))), norm_str(r.get("event_type"))),
        axis=1,
    )
    changed_rows = int((before != df["event_type"].astype(str)).sum())

    e["event_type"] = e.apply(
        lambda r: mapping.get((norm_str(r.get("newspaper_slug")), norm_str(r.get("ordinance_event_id"))), norm_str(r.get("event_type"))),
        axis=1,
    )
    return e, df, {"changed_event_type_rows": changed_rows, "changed_events": changed_events}


def main() -> None:
    args = _parse_args()
    cls_path = Path(args.classification_jsonl).expanduser().resolve()
    clean_path = Path(args.clean_jsonl).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_rows = iter_jsonl(cls_path)
    clean_rows = iter_jsonl(clean_path)
    clean_map = {norm_str(r.get("issue_id")): r for r in clean_rows if norm_str(r.get("issue_id"))}
    if not cls_rows:
        raise SystemExit("No classification rows found.")

    # join clean text
    issue_rows: list[dict[str, Any]] = []
    for r in cls_rows:
        iid = norm_str(r.get("issue_id"))
        issue_rows.append(
            {
                "newspaper_slug": norm_str(r.get("newspaper_slug")).lower(),
                "city_key": norm_str(r.get("city_key")),
                "target_city_name": norm_str(r.get("target_city_name") or r.get("city_name")),
                "target_state_abbr": norm_str(r.get("target_state_abbr") or r.get("state_abbr")).upper(),
                "issue_id": iid,
                "issue_date": norm_str(r.get("issue_date")),
                "issue_class": norm_str(r.get("issue_class")),
                "contains_verbatim_zoning_law": norm_str(r.get("contains_verbatim_zoning_law")),
                "law_text_sufficiency": norm_str(r.get("law_text_sufficiency")),
                "legal_status": norm_str(r.get("legal_status")),
                "enactment_signal": norm_str(r.get("enactment_signal")),
                "proposal_signal": norm_str(r.get("proposal_signal")),
                "issue_completeness": norm_str(r.get("issue_completeness")),
                "issue_jurisdiction_name": norm_str(r.get("jurisdiction_name")),
                "issue_jurisdiction_type": norm_str(r.get("jurisdiction_type")),
                "issue_jurisdiction_state_abbr": norm_str(r.get("jurisdiction_state_abbr")),
                "issue_jurisdiction_match_city": norm_str(r.get("jurisdiction_match_city")),
                "clean_zoning_text": norm_str((clean_map.get(iid) or {}).get("clean_zoning_text")),
            }
        )

    by_slug: dict[str, list[dict[str, Any]]] = {}
    for r in issue_rows:
        by_slug.setdefault(norm_str(r.get("newspaper_slug")), []).append(r)
    for slug in by_slug:
        by_slug[slug] = sorted(by_slug[slug], key=lambda x: (norm_str(x.get("issue_date")), norm_str(x.get("issue_id"))))

    runner = Path(args.gateway_runner).expanduser().resolve()
    gp = Path(args.gateway_pythonpath).expanduser().resolve() if norm_str(args.gateway_pythonpath) else None
    env_path = Path(args.gov_env_path).expanduser().resolve()

    all_events_rows: list[dict[str, Any]] = []
    all_assign_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    if args.variant == "direct_cluster":
        reqs: list[tuple[str, str]] = []
        for slug, rows in sorted(by_slug.items()):
            cid = f"event_cluster_v3_direct::{slug}"
            reqs.append((cid, _build_direct_prompt(slug, rows, max_chars=int(args.max_issue_excerpt_chars))))
        write_openai_requests(out_dir / "direct" / "requests" / "openai_requests_shard000.jsonl", reqs, model=str(args.model))
        run_gateway_requests(
            request_dir=out_dir / "direct" / "requests",
            output_dir=out_dir / "direct" / "results",
            runner_path=runner,
            model=str(args.model),
            gateway_pythonpath=gp,
            timeout_s=float(args.timeout),
            concurrency=int(args.concurrency),
            gov_env_path=env_path,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
        )
        res = read_result_jsonl(out_dir / "direct" / "results")
        err = read_error_jsonl(out_dir / "direct" / "results")
        for cid, _prompt in reqs:
            slug = cid.split("::", 1)[1]
            output_text = ""
            if cid in res:
                body = (((res[cid].get("response") or {}).get("body")) or {})
                output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
            elif cid in err:
                body = (((err[cid].get("response") or {}).get("body")) or {})
                output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
            parsed = parse_json_from_text(output_text)
            issue_ids = {norm_str(r.get("issue_id")) for r in by_slug.get(slug, []) if norm_str(r.get("issue_id"))}
            events = _parse_events(parsed if isinstance(parsed, dict) else {}, issue_ids)
            city_key = norm_str(by_slug.get(slug, [{}])[0].get("city_key"))
            issue_map = {norm_str(r.get("issue_id")): r for r in by_slug.get(slug, [])}
            event_rows, assign_rows = _flatten_events(slug, city_key, events, issue_map)
            all_events_rows.extend(event_rows)
            all_assign_rows.extend(assign_rows)
            raw_rows.append({"newspaper_slug": slug, "variant": "direct_cluster", "parsed_json": parsed if isinstance(parsed, dict) else {}})
    else:
        # pairwise_then_cluster
        reqs: list[tuple[str, str]] = []
        for slug, rows in sorted(by_slug.items()):
            cid = f"event_cluster_v3_pair::{slug}"
            reqs.append((cid, _build_pairwise_prompt(slug, rows, max_chars=int(args.max_issue_excerpt_chars))))
        write_openai_requests(out_dir / "pairwise" / "requests" / "openai_requests_shard000.jsonl", reqs, model=str(args.model))
        run_gateway_requests(
            request_dir=out_dir / "pairwise" / "requests",
            output_dir=out_dir / "pairwise" / "results",
            runner_path=runner,
            model=str(args.model),
            gateway_pythonpath=gp,
            timeout_s=float(args.timeout),
            concurrency=int(args.concurrency),
            gov_env_path=env_path,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
        )
        res = read_result_jsonl(out_dir / "pairwise" / "results")
        err = read_error_jsonl(out_dir / "pairwise" / "results")

        cluster_label_reqs: list[tuple[str, str]] = []
        cluster_meta: dict[str, dict[str, Any]] = {}
        for cid, _prompt in reqs:
            slug = cid.split("::", 1)[1]
            output_text = ""
            if cid in res:
                body = (((res[cid].get("response") or {}).get("body")) or {})
                output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
            elif cid in err:
                body = (((err[cid].get("response") or {}).get("body")) or {})
                output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
            parsed = parse_json_from_text(output_text)
            pairs = []
            if isinstance(parsed, dict) and isinstance(parsed.get("same_event_pairs"), list):
                for pp in parsed.get("same_event_pairs"):
                    if isinstance(pp, list) and len(pp) == 2:
                        a, b = norm_str(pp[0]), norm_str(pp[1])
                        if a and b and a != b:
                            pairs.append((a, b))
            ids = [norm_str(r.get("issue_id")) for r in by_slug.get(slug, []) if norm_str(r.get("issue_id"))]
            uf = _UF(ids)
            valid_set = set(ids)
            for a, b in pairs:
                if a in valid_set and b in valid_set:
                    uf.union(a, b)
            groups: dict[str, list[str]] = {}
            for iid in ids:
                groups.setdefault(uf.find(iid), []).append(iid)
            clusters: list[dict[str, Any]] = []
            for i, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True), start=1):
                e_id = f"E{i:03d}"
                issues_for_cluster = [r for r in by_slug.get(slug, []) if norm_str(r.get("issue_id")) in set(members)]
                issues_for_cluster = sorted(issues_for_cluster, key=lambda x: (norm_str(x.get("issue_date")), norm_str(x.get("issue_id"))))
                clusters.append({"ordinance_event_id": e_id, "issues": issues_for_cluster})
            cid2 = f"event_cluster_v3_pair_label::{slug}"
            cluster_label_reqs.append((cid2, _build_cluster_label_prompt(slug, clusters, max_chars=int(args.max_issue_excerpt_chars))))
            cluster_meta[cid2] = {"slug": slug, "clusters": clusters}

        write_openai_requests(out_dir / "pairwise_label" / "requests" / "openai_requests_shard000.jsonl", cluster_label_reqs, model=str(args.model))
        run_gateway_requests(
            request_dir=out_dir / "pairwise_label" / "requests",
            output_dir=out_dir / "pairwise_label" / "results",
            runner_path=runner,
            model=str(args.model),
            gateway_pythonpath=gp,
            timeout_s=float(args.timeout),
            concurrency=int(args.concurrency),
            gov_env_path=env_path,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
        )
        res2 = read_result_jsonl(out_dir / "pairwise_label" / "results")
        err2 = read_error_jsonl(out_dir / "pairwise_label" / "results")
        for cid2, _prompt2 in cluster_label_reqs:
            slug = cluster_meta[cid2]["slug"]
            output_text = ""
            if cid2 in res2:
                body = (((res2[cid2].get("response") or {}).get("body")) or {})
                output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
            elif cid2 in err2:
                body = (((err2[cid2].get("response") or {}).get("body")) or {})
                output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
            parsed = parse_json_from_text(output_text)
            issue_ids = {norm_str(r.get("issue_id")) for r in by_slug.get(slug, []) if norm_str(r.get("issue_id"))}
            events = _parse_events(parsed if isinstance(parsed, dict) else {}, issue_ids)
            city_key = norm_str(by_slug.get(slug, [{}])[0].get("city_key"))
            issue_map = {norm_str(r.get("issue_id")): r for r in by_slug.get(slug, [])}
            event_rows, assign_rows = _flatten_events(slug, city_key, events, issue_map)
            all_events_rows.extend(event_rows)
            all_assign_rows.extend(assign_rows)
            raw_rows.append({"newspaper_slug": slug, "variant": "pairwise_then_cluster", "parsed_json": parsed if isinstance(parsed, dict) else {}})

    ev_df = pd.DataFrame.from_records(all_events_rows).sort_values(["newspaper_slug", "first_issue_date", "ordinance_event_id"]).reset_index(drop=True)
    as_df = pd.DataFrame.from_records(all_assign_rows).sort_values(["newspaper_slug", "issue_date", "issue_id"]).reset_index(drop=True)

    # Deterministic guardrail for coherent event types.
    ev_df, as_df, harmonize_stats = _harmonize_event_types(ev_df, as_df)
    write_csv(out_dir / "ordinance_event_clusters_v3.csv", ev_df)
    write_csv(out_dir / "issue_event_assignments_v3.csv", as_df)
    write_jsonl(out_dir / "event_cluster_parsed_outputs.jsonl", raw_rows)

    # Main contract file requested by plan.
    write_csv(out_dir / "ordinance_event_clusters_v3_flat.csv", as_df)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "classification_jsonl": str(cls_path),
        "clean_jsonl": str(clean_path),
        "output_dir": str(out_dir),
        "variant": str(args.variant),
        "model": str(args.model),
        "newspaper_count": int(len(by_slug)),
        "issue_count": int(len(issue_rows)),
        "event_count": int(len(ev_df)),
        "event_type_harmonization": harmonize_stats,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eprint(f"Done. events output: {out_dir}")


if __name__ == "__main__":
    main()
