#!/usr/bin/env python3
"""
LLM collision re-adjudication pass for deduplicated events.

Purpose:
- Detect likely duplicate event ids that survived clustering.
- Ask the LLM whether to merge exact/similar collision groups.
- Emit remapped event assignments and clusters.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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
    write_csv,
    write_jsonl,
    write_openai_requests,
)


ALLOWED_DECISION = {"merge", "keep_separate", "uncertain"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM event collision re-adjudication v4.")
    ap.add_argument("--classification-jsonl", required=True)
    ap.add_argument("--clean-jsonl", required=True)
    ap.add_argument("--event-clusters-csv", required=True)
    ap.add_argument("--event-assignments-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="openai:gpt-5")
    ap.add_argument(
        "--max-events-per-group",
        type=int,
        default=0,
        help="Hard cap for number of candidate events in a collision group prompt. 0 means include all. If positive and exceeded, the run fails (no truncation).",
    )
    ap.add_argument(
        "--max-clean-excerpt-chars",
        type=int,
        default=0,
        help="Hard cap for per-event clean text included in prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
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


def _normalize_for_hash(text: str) -> str:
    s = norm_str(text).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = s.strip()
    return s


def _bounded_text_or_fail(text: str, max_chars: int, label: str) -> str:
    s = norm_str(text)
    if max_chars > 0 and len(s) > max_chars:
        raise ValueError(
            f"{label} length {len(s)} exceeds hard cap {max_chars}. "
            "Clipping/condensing is disabled; increase cap or reduce input scope."
        )
    return s


def _build_prompt(slug: str, group_id: str, rows: list[dict[str, Any]], max_chars: int) -> str:
    schema = {
        "newspaper_slug": slug,
        "collision_group_id": group_id,
        "merge_decision": "merge",
        "canonical_event_id": "E001",
        "merge_event_ids": ["E001", "E002"],
        "rationale": "short reason",
    }
    lines: list[str] = []
    lines.append("Decide whether these event ids represent the same underlying legal action.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Rules:")
    lines.append("- merge: same legal action, repeated publication/reprint/installment continuation.")
    lines.append("- keep_separate: distinct legal actions.")
    lines.append("- uncertain: insufficient evidence.")
    lines.append("- Only choose canonical_event_id from provided event ids.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(f"newspaper_slug={slug} collision_group_id={group_id}")
    lines.append("Candidate events:")
    for r in rows:
        lines.append(
            f"- event_id={norm_str(r.get('ordinance_event_id'))} type={norm_str(r.get('event_type'))} "
            f"status={norm_str(r.get('event_status'))} issues={int(r.get('issue_count') or 0)} "
            f"first={norm_str(r.get('first_issue_date'))} last={norm_str(r.get('last_issue_date'))}"
        )
        lines.append(
            _bounded_text_or_fail(
                norm_str(r.get("clean_text_excerpt")),
                max_chars,
                label=f"slug={slug} collision_group={group_id} event_id={norm_str(r.get('ordinance_event_id'))} clean_text",
            )
        )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _coerce(parsed_obj: dict[str, Any], slug: str, group_id: str, allowed_ids: set[str]) -> dict[str, Any]:
    obj = dict(parsed_obj or {})
    decision = enum(obj.get("merge_decision"), ALLOWED_DECISION, "uncertain")
    canonical = norm_str(obj.get("canonical_event_id"))
    merge_ids = []
    if isinstance(obj.get("merge_event_ids"), list):
        for x in obj.get("merge_event_ids"):
            eid = norm_str(x)
            if eid and eid in allowed_ids and eid not in merge_ids:
                merge_ids.append(eid)
    if canonical not in allowed_ids:
        canonical = ""
    if decision == "merge":
        if not canonical and merge_ids:
            canonical = merge_ids[0]
        if canonical and canonical not in merge_ids:
            merge_ids = [canonical] + merge_ids
        merge_ids = [x for x in merge_ids if x in allowed_ids]
        if len(merge_ids) < 2 or not canonical:
            decision = "uncertain"
    return {
        "newspaper_slug": slug,
        "collision_group_id": group_id,
        "merge_decision": decision,
        "canonical_event_id": canonical,
        "merge_event_ids": merge_ids,
        "rationale": norm_str(obj.get("rationale")),
    }


def _build_updated_clusters(assign_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    if assign_df.empty:
        return events_df.copy()
    agg_spec: dict[str, Any] = {
        "issue_count": ("issue_id", "nunique"),
        "first_issue_date": ("issue_date", "min"),
        "last_issue_date": ("issue_date", "max"),
        "event_type": ("event_type", lambda s: s.astype(str).mode().iloc[0] if not s.empty else "uncertain"),
    }
    if "qualification_decision" in assign_df.columns:
        agg_spec["qualification_decision"] = (
            "qualification_decision",
            lambda s: s.astype(str).mode().iloc[0] if not s.empty else "",
        )
    if "event_status" in assign_df.columns:
        agg_spec["event_status"] = (
            "event_status",
            lambda s: s.astype(str).mode().iloc[0] if not s.empty else "",
        )
    grp = (
        assign_df.groupby(["newspaper_slug", "city_key", "ordinance_event_id"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )
    if "summary" in events_df.columns:
        m = (
            events_df.sort_values(["newspaper_slug", "first_issue_date", "ordinance_event_id"])
            .drop_duplicates(subset=["newspaper_slug", "ordinance_event_id"], keep="first")
            [["newspaper_slug", "ordinance_event_id", "summary"]]
        )
        grp = grp.merge(m, on=["newspaper_slug", "ordinance_event_id"], how="left")
    return grp.sort_values(["newspaper_slug", "first_issue_date", "ordinance_event_id"]).reset_index(drop=True)


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

    clean_map = {norm_str(r.get("issue_id")): norm_str(r.get("clean_zoning_text")) for r in clean_rows if norm_str(r.get("issue_id"))}
    cls_map = {norm_str(r.get("issue_id")): r for r in cls_rows if norm_str(r.get("issue_id"))}

    as_df = as_df.copy()
    ev_df = ev_df.copy()
    for c in ("newspaper_slug", "ordinance_event_id", "issue_id", "issue_date", "event_type", "city_key"):
        if c in as_df.columns:
            as_df[c] = as_df[c].map(norm_str)
    for c in ("newspaper_slug", "ordinance_event_id", "event_type", "first_issue_date", "last_issue_date", "city_key"):
        if c in ev_df.columns:
            ev_df[c] = ev_df[c].map(norm_str)

    # Build event-level clean text hashes.
    event_text_rows: list[dict[str, Any]] = []
    for (slug, eid), grp in as_df.groupby(["newspaper_slug", "ordinance_event_id"], dropna=False):
        members = grp.sort_values(["issue_date", "issue_id"])
        parts = [norm_str(clean_map.get(norm_str(iid))) for iid in members["issue_id"].astype(str).tolist()]
        parts = [p for p in parts if p]
        combined = "\n".join(parts).strip()
        normed = _normalize_for_hash(combined)
        if not normed:
            continue
        h = hashlib.sha256(normed.encode("utf-8")).hexdigest()
        meta = ev_df[(ev_df["newspaper_slug"] == slug) & (ev_df["ordinance_event_id"] == eid)]
        event_text_rows.append(
            {
                "newspaper_slug": slug,
                "ordinance_event_id": eid,
                "event_type": norm_str(meta["event_type"].iloc[0]) if not meta.empty else norm_str(grp["event_type"].iloc[0]),
                "event_status": norm_str(meta["event_status"].iloc[0]) if ("event_status" in meta.columns and not meta.empty) else "",
                "issue_count": int(grp["issue_id"].nunique()),
                "first_issue_date": norm_str(meta["first_issue_date"].iloc[0]) if not meta.empty else norm_str(members["issue_date"].min()),
                "last_issue_date": norm_str(meta["last_issue_date"].iloc[0]) if not meta.empty else norm_str(members["issue_date"].max()),
                "clean_text_hash": h,
                "clean_text_excerpt": combined,
            }
        )
    txt_df = pd.DataFrame.from_records(event_text_rows)
    if txt_df.empty:
        write_csv(out_dir / "collision_groups_v4.csv", pd.DataFrame())
        write_jsonl(out_dir / "collision_adjudications_v4.jsonl", [])
        write_csv(out_dir / "issue_event_assignments_v3_rechecked.csv", as_df)
        write_csv(out_dir / "ordinance_event_clusters_v3_rechecked.csv", ev_df)
        return

    collision_groups = (
        txt_df.groupby(["newspaper_slug", "clean_text_hash"], dropna=False)["ordinance_event_id"]
        .nunique()
        .reset_index(name="event_n")
    )
    collision_groups = collision_groups[collision_groups["event_n"] > 1].copy()

    reqs: list[tuple[str, str]] = []
    meta_by_cid: dict[str, dict[str, Any]] = {}
    group_rows: list[dict[str, Any]] = []
    for row in collision_groups.itertuples(index=False):
        slug = norm_str(row.newspaper_slug)
        h = norm_str(row.clean_text_hash)
        subset = txt_df[(txt_df["newspaper_slug"] == slug) & (txt_df["clean_text_hash"] == h)].copy()
        subset = subset.sort_values(["first_issue_date", "ordinance_event_id"])
        if int(args.max_events_per_group) > 0 and len(subset) > int(args.max_events_per_group):
            raise ValueError(
                f"slug={slug} collision_group={h[:12]} has {len(subset)} candidate events, exceeding hard cap "
                f"{int(args.max_events_per_group)}. Group truncation is disabled; increase cap or set 0 for no cap."
            )
        if len(subset) < 2:
            continue
        gid = f"{slug}::{h[:12]}"
        rows = subset.to_dict(orient="records")
        reqs.append((f"event_collision_v4::{gid}", _build_prompt(slug, gid, rows, max_chars=int(args.max_clean_excerpt_chars))))
        meta_by_cid[f"event_collision_v4::{gid}"] = {"slug": slug, "group_id": gid, "rows": rows}
        for r in rows:
            group_rows.append({"collision_group_id": gid, **r})

    write_csv(out_dir / "collision_groups_v4.csv", pd.DataFrame.from_records(group_rows))

    if not reqs:
        write_jsonl(out_dir / "collision_adjudications_v4.jsonl", [])
        write_csv(out_dir / "issue_event_assignments_v3_rechecked.csv", as_df)
        write_csv(out_dir / "ordinance_event_clusters_v3_rechecked.csv", ev_df)
        return

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
    remap: dict[tuple[str, str], str] = {}
    for cid, _ in reqs:
        meta = meta_by_cid[cid]
        rows = meta["rows"]
        allowed = {norm_str(r.get("ordinance_event_id")) for r in rows if norm_str(r.get("ordinance_event_id"))}
        output_text = ""
        if cid in res:
            body = (((res[cid].get("response") or {}).get("body")) or {})
            output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in err:
            body = (((err[cid].get("response") or {}).get("body")) or {})
            output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
        parsed = parse_json_from_text(output_text)
        adj = _coerce(
            parsed if isinstance(parsed, dict) else {},
            slug=meta["slug"],
            group_id=meta["group_id"],
            allowed_ids=allowed,
        )
        adjud_rows.append(adj)
        if adj["merge_decision"] == "merge" and adj["canonical_event_id"]:
            can = norm_str(adj["canonical_event_id"])
            for eid in adj["merge_event_ids"]:
                if norm_str(eid) and norm_str(eid) != can:
                    remap[(meta["slug"], norm_str(eid))] = can

    out_assign = as_df.copy()
    out_assign["ordinance_event_id_original"] = out_assign["ordinance_event_id"]
    out_assign["ordinance_event_id"] = out_assign.apply(
        lambda r: remap.get((norm_str(r.get("newspaper_slug")), norm_str(r.get("ordinance_event_id"))), norm_str(r.get("ordinance_event_id"))),
        axis=1,
    )

    # Harmonize event type/status to canonical id within each (slug, event_id) group.
    for col in ("event_type", "event_status", "qualification_decision"):
        if col in out_assign.columns:
            out_assign[col] = (
                out_assign.groupby(["newspaper_slug", "ordinance_event_id"], dropna=False)[col]
                .transform(lambda s: s.astype(str).mode().iloc[0] if not s.empty else "")
            )

    out_events = _build_updated_clusters(out_assign, ev_df)
    write_jsonl(out_dir / "collision_adjudications_v4.jsonl", adjud_rows)
    write_csv(out_dir / "issue_event_assignments_v3_rechecked.csv", out_assign.sort_values(["newspaper_slug", "issue_date", "issue_id"]))
    write_csv(out_dir / "ordinance_event_clusters_v3_rechecked.csv", out_events)

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
        "collision_groups": int(len(collision_groups)),
        "merge_decisions": int(len([r for r in adjud_rows if norm_str(r.get("merge_decision")) == "merge"])),
        "remapped_event_ids": int(len(remap)),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eprint(f"Done. collision recheck output: {out_dir}")


if __name__ == "__main__":
    main()
