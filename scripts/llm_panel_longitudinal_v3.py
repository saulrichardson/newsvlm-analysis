#!/usr/bin/env python3
"""
LLM-only full-panel longitudinal extraction (v3).
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
    as_issue_id_list,
    eprint,
    enum,
    extract_openai_output_text,
    iter_jsonl,
    load_issue_rows_for_city,
    norm_str,
    parse_json_from_text,
    read_error_jsonl,
    read_result_jsonl,
    run_gateway_requests,
    safe_float,
    safe_int,
    write_csv,
    write_jsonl,
    write_openai_requests,
)


ALLOWED_SIGNAL = {"none", "mention", "amendment", "full_ordinance"}
ALLOWED_FIRST_FULL_STATUS = {"found", "not_found", "uncertain"}
ALLOWED_DIR = {"up", "down", "flat", "mixed", "uncertain"}
ALLOWED_WAVE = {"expansion", "tightening", "clarification", "reclassification", "mixed", "none", "uncertain"}
ALLOWED_PHASE = {
    "initial_establishment",
    "formalization_enforcement",
    "expansion_reclassification",
    "flexible_special_use",
    "infrastructure_environment",
    "maturation_implementation",
    "other_unclear",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM-only panel longitudinal extraction v3.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Run dir containing panels/<city_key>/issue_texts.jsonl for metadata/text.",
    )
    ap.add_argument(
        "--classification-jsonl",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/classification/issue_labels_v3.jsonl",
    )
    ap.add_argument(
        "--clean-jsonl",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/cleaning/clean_zoning_text.jsonl",
    )
    ap.add_argument(
        "--event-assignments-csv",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/events/issue_event_assignments_v3.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/fullnewspaper_v3/longitudinal",
    )
    ap.add_argument("--model", default="openai:gpt-5")
    ap.add_argument(
        "--max-panel-chars",
        type=int,
        default=0,
        help="Hard cap for per-panel raw text characters in a single prompt. 0 means no local cap. If positive and exceeded, the run fails (no overflow chunking).",
    )
    ap.add_argument(
        "--qualified-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only events passing qualification_decision=pass when available.",
    )
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300.0)
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


def _schema_template(city_key: str, city_name: str, state_abbr: str) -> dict[str, Any]:
    return {
        "city_key": city_key,
        "city_name": city_name,
        "state_abbr": state_abbr.lower(),
        "first_zoning_signal": {
            "issue_id": "",
            "issue_date": "",
            "signal_type": "none",
            "confidence_0_to_1": 0.0,
            "evidence": "",
        },
        "first_full_ordinance": {
            "issue_id": "",
            "issue_date": "",
            "confidence_0_to_1": 0.0,
            "evidence": "",
            "status": "not_found",
        },
        "evolution_phases": [
            {
                "phase_order": 1,
                "phase_label": "initial_establishment",
                "start_issue_date": "",
                "end_issue_date": "",
                "dominant_categories": ["procedural_governance"],
                "complexity_direction": "mixed",
                "evidence_issue_ids": [""],
            }
        ],
        "amendment_waves": [
            {
                "wave_order": 1,
                "start_issue_date": "",
                "end_issue_date": "",
                "change_type": "mixed",
                "affected_categories": ["procedural_governance"],
                "evidence_issue_ids": [""],
            }
        ],
        "category_trajectory": {
            k: {"early_share": 0.0, "mid_share": 0.0, "late_share": 0.0, "direction": "flat"} for k in FIXED_CATEGORY_KEYS
        },
        "complexity_trajectory": {
            "regulatory_complexity_trend": "mixed",
            "procedural_intensity_trend": "mixed",
            "dimensional_specificity_trend": "mixed",
        },
        "numeric_trajectory": {
            "lot_size": {"direction": "uncertain", "evidence": ""},
            "height": {"direction": "uncertain", "evidence": ""},
            "setback": {"direction": "uncertain", "evidence": ""},
            "parking": {"direction": "uncertain", "evidence": ""},
        },
        "narrative_summary": "",
        "quality_notes": [""],
    }


def _build_panel_prompt(slug: str, city_key: str, city_name: str, state_abbr: str, rows: list[dict[str, Any]]) -> tuple[str, int]:
    schema = _schema_template(city_key, city_name, state_abbr)
    lines: list[str] = []
    lines.append("You are analyzing one full newspaper timeline to measure zoning emergence and evolution.")
    lines.append("Return ONLY one JSON object.")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Use only provided issue_ids in evidence_issue_ids.")
    lines.append("- Distinguish first zoning signal from first full ordinance.")
    lines.append("- Keep category_trajectory keys exactly as provided.")
    lines.append("- Stage units are issue-order terciles within this newspaper timeline.")
    lines.append("- Do not treat zoning narrative as equivalent to verbatim legal adoption.")
    lines.append("- Use trend labels only from: up|down|flat|mixed|uncertain")
    lines.append("")
    lines.append("JSON schema template:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(f"Newspaper={slug} city={city_name},{state_abbr.upper()} city_key={city_key}")
    lines.append(f"Issue_count={len(rows)}")
    lines.append("")
    lines.append("Chronological timeline:")
    chars = 0
    for i, r in enumerate(rows, start=1):
        raw = norm_str(r.get("raw_text"))
        clean = norm_str(r.get("clean_zoning_text"))
        chars += len(raw)
        lines.append("")
        lines.append(
            f"[ISSUE {i}] issue_id={norm_str(r.get('issue_id'))} date={norm_str(r.get('issue_date'))} "
            f"issue_class={norm_str(r.get('issue_class'))} event_id={norm_str(r.get('ordinance_event_id'))} "
            f"publication_role={norm_str(r.get('publication_role'))} "
            f"event_type={norm_str(r.get('event_type'))} event_status={norm_str(r.get('event_status'))} "
            f"qualification_decision={norm_str(r.get('qualification_decision'))} "
            f"contains_law={norm_str(r.get('contains_verbatim_zoning_law'))} suff={norm_str(r.get('law_text_sufficiency'))}"
        )
        lines.append("[RAW_ISSUE_TEXT]")
        lines.append(raw)
        lines.append("[CLEAN_LEGAL_TEXT]")
        lines.append(clean)
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip(), chars


def _build_chunk_planner_prompt(slug: str, rows: list[dict[str, Any]], target_chunks: int) -> str:
    schema = {"newspaper_slug": slug, "chunks": [{"chunk_id": 1, "start_index_1based": 1, "end_index_1based": 10}]}
    lines: list[str] = []
    lines.append("Plan contiguous timeline chunks for a long newspaper panel.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Goal: split issue sequence into chunks preserving chronology.")
    lines.append("Choose about target chunk count and contiguous ranges that cover all issues with no overlap/gaps.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(f"newspaper_slug={slug} target_chunks={target_chunks}")
    lines.append("Issue index:")
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"{i}. issue_id={norm_str(r.get('issue_id'))} date={norm_str(r.get('issue_date'))} class={norm_str(r.get('issue_class'))}"
        )
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _coerce_output(parsed_obj: dict[str, Any], city_key: str, city_name: str, state_abbr: str, issue_ids: set[str]) -> dict[str, Any]:
    out = dict(parsed_obj or {})
    out["city_key"] = norm_str(out.get("city_key")) or city_key
    out["city_name"] = norm_str(out.get("city_name")) or city_name
    out["state_abbr"] = (norm_str(out.get("state_abbr")) or state_abbr).lower()

    fzs = out.get("first_zoning_signal") if isinstance(out.get("first_zoning_signal"), dict) else {}
    out["first_zoning_signal"] = {
        "issue_id": norm_str(fzs.get("issue_id")),
        "issue_date": norm_str(fzs.get("issue_date")),
        "signal_type": enum(fzs.get("signal_type"), ALLOWED_SIGNAL, "none"),
        "confidence_0_to_1": safe_float(fzs.get("confidence_0_to_1"), default=0.0),
        "evidence": norm_str(fzs.get("evidence")),
    }
    if out["first_zoning_signal"]["issue_id"] and out["first_zoning_signal"]["issue_id"] not in issue_ids:
        out["first_zoning_signal"]["issue_id"] = ""
        out["first_zoning_signal"]["signal_type"] = "none"

    ffo = out.get("first_full_ordinance") if isinstance(out.get("first_full_ordinance"), dict) else {}
    out["first_full_ordinance"] = {
        "issue_id": norm_str(ffo.get("issue_id")),
        "issue_date": norm_str(ffo.get("issue_date")),
        "confidence_0_to_1": safe_float(ffo.get("confidence_0_to_1"), default=0.0),
        "evidence": norm_str(ffo.get("evidence")),
        "status": enum(ffo.get("status"), ALLOWED_FIRST_FULL_STATUS, "uncertain"),
    }
    if out["first_full_ordinance"]["status"] == "found" and out["first_full_ordinance"]["issue_id"] not in issue_ids:
        out["first_full_ordinance"]["status"] = "uncertain"
        out["first_full_ordinance"]["issue_id"] = ""

    phases = out.get("evolution_phases") if isinstance(out.get("evolution_phases"), list) else []
    ph_out: list[dict[str, Any]] = []
    for i, ph in enumerate(phases, start=1):
        if not isinstance(ph, dict):
            continue
        dom = ph.get("dominant_categories")
        if not isinstance(dom, list):
            dom = [norm_str(dom)] if norm_str(dom) else []
        dom = [norm_str(x) for x in dom if norm_str(x)]
        ev = [x for x in as_issue_id_list(ph.get("evidence_issue_ids")) if x in issue_ids]
        ph_out.append(
            {
                "phase_order": safe_int(ph.get("phase_order"), default=i),
                "phase_label": enum(ph.get("phase_label"), ALLOWED_PHASE, "other_unclear"),
                "start_issue_date": norm_str(ph.get("start_issue_date")),
                "end_issue_date": norm_str(ph.get("end_issue_date")),
                "dominant_categories": dom,
                "complexity_direction": enum(ph.get("complexity_direction"), ALLOWED_DIR, "uncertain"),
                "evidence_issue_ids": ev,
            }
        )
    out["evolution_phases"] = ph_out

    waves = out.get("amendment_waves") if isinstance(out.get("amendment_waves"), list) else []
    w_out: list[dict[str, Any]] = []
    for i, w in enumerate(waves, start=1):
        if not isinstance(w, dict):
            continue
        aff = w.get("affected_categories")
        if not isinstance(aff, list):
            aff = [norm_str(aff)] if norm_str(aff) else []
        aff = [norm_str(x) for x in aff if norm_str(x)]
        ev = [x for x in as_issue_id_list(w.get("evidence_issue_ids")) if x in issue_ids]
        w_out.append(
            {
                "wave_order": safe_int(w.get("wave_order"), default=i),
                "start_issue_date": norm_str(w.get("start_issue_date")),
                "end_issue_date": norm_str(w.get("end_issue_date")),
                "change_type": enum(w.get("change_type"), ALLOWED_WAVE, "uncertain"),
                "affected_categories": aff,
                "evidence_issue_ids": ev,
            }
        )
    out["amendment_waves"] = w_out

    ct = out.get("category_trajectory") if isinstance(out.get("category_trajectory"), dict) else {}
    ct_out: dict[str, Any] = {}
    for k in FIXED_CATEGORY_KEYS:
        kv = ct.get(k) if isinstance(ct.get(k), dict) else {}
        ct_out[k] = {
            "early_share": safe_float(kv.get("early_share"), default=0.0),
            "mid_share": safe_float(kv.get("mid_share"), default=0.0),
            "late_share": safe_float(kv.get("late_share"), default=0.0),
            "direction": enum(kv.get("direction"), ALLOWED_DIR, "uncertain"),
        }
    out["category_trajectory"] = ct_out

    cx = out.get("complexity_trajectory") if isinstance(out.get("complexity_trajectory"), dict) else {}
    out["complexity_trajectory"] = {
        "regulatory_complexity_trend": enum(cx.get("regulatory_complexity_trend"), ALLOWED_DIR, "uncertain"),
        "procedural_intensity_trend": enum(cx.get("procedural_intensity_trend"), ALLOWED_DIR, "uncertain"),
        "dimensional_specificity_trend": enum(cx.get("dimensional_specificity_trend"), ALLOWED_DIR, "uncertain"),
    }

    nx = out.get("numeric_trajectory") if isinstance(out.get("numeric_trajectory"), dict) else {}
    nx_out: dict[str, Any] = {}
    for metric in ("lot_size", "height", "setback", "parking"):
        mv = nx.get(metric) if isinstance(nx.get(metric), dict) else {}
        nx_out[metric] = {
            "direction": enum(mv.get("direction"), ALLOWED_DIR, "uncertain"),
            "evidence": norm_str(mv.get("evidence")),
        }
    out["numeric_trajectory"] = nx_out

    out["narrative_summary"] = norm_str(out.get("narrative_summary"))
    qn = out.get("quality_notes")
    out["quality_notes"] = [norm_str(x) for x in qn if norm_str(x)] if isinstance(qn, list) else []
    return out


def _run_single_request(prompt: str, custom_id: str, tmp_dir: Path, model: str, args: argparse.Namespace) -> dict[str, Any] | None:
    req_dir = tmp_dir / "requests"
    res_dir = tmp_dir / "results"
    write_openai_requests(req_dir / "openai_requests_shard000.jsonl", [(custom_id, prompt)], model=model)
    run_gateway_requests(
        request_dir=req_dir,
        output_dir=res_dir,
        runner_path=Path(args.gateway_runner).expanduser().resolve(),
        model=model,
        gateway_pythonpath=Path(args.gateway_pythonpath).expanduser().resolve() if norm_str(args.gateway_pythonpath) else None,
        timeout_s=float(args.timeout),
        concurrency=1,
        gov_env_path=Path(args.gov_env_path).expanduser().resolve(),
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )
    res = read_result_jsonl(res_dir)
    err = read_error_jsonl(res_dir)
    output_text = ""
    if custom_id in res:
        body = (((res[custom_id].get("response") or {}).get("body")) or {})
        output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
    elif custom_id in err:
        body = (((err[custom_id].get("response") or {}).get("body")) or {})
        output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
    parsed = parse_json_from_text(output_text)
    return parsed if isinstance(parsed, dict) else None


def _flatten_outputs(parsed_rows: list[dict[str, Any]], out_dir: Path) -> None:
    summary_rows: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    wave_rows: list[dict[str, Any]] = []
    cat_rows: list[dict[str, Any]] = []
    cpx_rows: list[dict[str, Any]] = []
    num_rows: list[dict[str, Any]] = []

    for r in parsed_rows:
        meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
        pj = r.get("parsed_json") if isinstance(r.get("parsed_json"), dict) else {}
        slug = norm_str(meta.get("newspaper_slug"))
        city_key = norm_str(meta.get("city_key"))
        city_name = norm_str(meta.get("city_name"))
        state = norm_str(meta.get("state_abbr"))

        fzs = pj.get("first_zoning_signal") if isinstance(pj.get("first_zoning_signal"), dict) else {}
        ffo = pj.get("first_full_ordinance") if isinstance(pj.get("first_full_ordinance"), dict) else {}
        cpx = pj.get("complexity_trajectory") if isinstance(pj.get("complexity_trajectory"), dict) else {}
        qn = pj.get("quality_notes") if isinstance(pj.get("quality_notes"), list) else []

        summary_rows.append(
            {
                "newspaper_slug": slug,
                "city_key": city_key,
                "city_name": city_name,
                "state_abbr": state,
                "issue_count": int(meta.get("issue_count") or 0),
                "panel_chars_original": int(meta.get("panel_chars_original") or 0),
                "panel_chars_used": int(meta.get("panel_chars_used") or 0),
                "overflow_applied": int(bool(meta.get("overflow_applied"))),
                "clipped_issue_count": int(meta.get("clipped_issue_count") or 0),
                "first_signal_issue_id": norm_str(fzs.get("issue_id")),
                "first_signal_date": norm_str(fzs.get("issue_date")),
                "first_signal_type": norm_str(fzs.get("signal_type")),
                "first_full_issue_id": norm_str(ffo.get("issue_id")),
                "first_full_date": norm_str(ffo.get("issue_date")),
                "first_full_status": norm_str(ffo.get("status")),
                "regulatory_complexity_trend": norm_str(cpx.get("regulatory_complexity_trend")),
                "procedural_intensity_trend": norm_str(cpx.get("procedural_intensity_trend")),
                "dimensional_specificity_trend": norm_str(cpx.get("dimensional_specificity_trend")),
                "quality_notes_count": int(len([x for x in qn if norm_str(x)])),
            }
        )
        first_rows.append(
            {
                "newspaper_slug": slug,
                "city_key": city_key,
                "city_name": city_name,
                "state_abbr": state,
                "first_signal_issue_id": norm_str(fzs.get("issue_id")),
                "first_signal_date": norm_str(fzs.get("issue_date")),
                "first_signal_type": norm_str(fzs.get("signal_type")),
                "first_full_issue_id": norm_str(ffo.get("issue_id")),
                "first_full_date": norm_str(ffo.get("issue_date")),
                "first_full_status": norm_str(ffo.get("status")),
            }
        )

        for ph in pj.get("evolution_phases") if isinstance(pj.get("evolution_phases"), list) else []:
            if not isinstance(ph, dict):
                continue
            phase_rows.append(
                {
                    "newspaper_slug": slug,
                    "city_key": city_key,
                    "city_name": city_name,
                    "state_abbr": state,
                    "phase_order": safe_int(ph.get("phase_order"), default=0),
                    "phase_label": norm_str(ph.get("phase_label")),
                    "start_issue_date": norm_str(ph.get("start_issue_date")),
                    "end_issue_date": norm_str(ph.get("end_issue_date")),
                    "dominant_categories": "|".join([norm_str(x) for x in (ph.get("dominant_categories") if isinstance(ph.get("dominant_categories"), list) else []) if norm_str(x)]),
                    "complexity_direction": norm_str(ph.get("complexity_direction")),
                    "evidence_issue_ids": "|".join([norm_str(x) for x in as_issue_id_list(ph.get("evidence_issue_ids")) if norm_str(x)]),
                }
            )

        for w in pj.get("amendment_waves") if isinstance(pj.get("amendment_waves"), list) else []:
            if not isinstance(w, dict):
                continue
            wave_rows.append(
                {
                    "newspaper_slug": slug,
                    "city_key": city_key,
                    "city_name": city_name,
                    "state_abbr": state,
                    "wave_order": safe_int(w.get("wave_order"), default=0),
                    "start_issue_date": norm_str(w.get("start_issue_date")),
                    "end_issue_date": norm_str(w.get("end_issue_date")),
                    "change_type": norm_str(w.get("change_type")),
                    "affected_categories": "|".join([norm_str(x) for x in (w.get("affected_categories") if isinstance(w.get("affected_categories"), list) else []) if norm_str(x)]),
                    "evidence_issue_ids": "|".join([norm_str(x) for x in as_issue_id_list(w.get("evidence_issue_ids")) if norm_str(x)]),
                }
            )

        ct = pj.get("category_trajectory") if isinstance(pj.get("category_trajectory"), dict) else {}
        for cat in FIXED_CATEGORY_KEYS:
            v = ct.get(cat) if isinstance(ct.get(cat), dict) else {}
            cat_rows.append(
                {
                    "newspaper_slug": slug,
                    "city_key": city_key,
                    "city_name": city_name,
                    "state_abbr": state,
                    "category": cat,
                    "early_share": safe_float(v.get("early_share"), default=0.0),
                    "mid_share": safe_float(v.get("mid_share"), default=0.0),
                    "late_share": safe_float(v.get("late_share"), default=0.0),
                    "direction": norm_str(v.get("direction")),
                }
            )

        cpx_rows.append(
            {
                "newspaper_slug": slug,
                "city_key": city_key,
                "city_name": city_name,
                "state_abbr": state,
                "regulatory_complexity_trend": norm_str(cpx.get("regulatory_complexity_trend")),
                "procedural_intensity_trend": norm_str(cpx.get("procedural_intensity_trend")),
                "dimensional_specificity_trend": norm_str(cpx.get("dimensional_specificity_trend")),
            }
        )

        nx = pj.get("numeric_trajectory") if isinstance(pj.get("numeric_trajectory"), dict) else {}
        for metric in ("lot_size", "height", "setback", "parking"):
            mv = nx.get(metric) if isinstance(nx.get(metric), dict) else {}
            num_rows.append(
                {
                    "newspaper_slug": slug,
                    "city_key": city_key,
                    "city_name": city_name,
                    "state_abbr": state,
                    "metric": metric,
                    "direction": norm_str(mv.get("direction")),
                    "evidence": norm_str(mv.get("evidence")),
                }
            )

    write_csv(out_dir / "city_longitudinal_summary.csv", pd.DataFrame.from_records(summary_rows))
    write_csv(out_dir / "city_first_zoning_events.csv", pd.DataFrame.from_records(first_rows))
    write_csv(out_dir / "city_evolution_phases.csv", pd.DataFrame.from_records(phase_rows))
    write_csv(out_dir / "city_amendment_waves.csv", pd.DataFrame.from_records(wave_rows))
    write_csv(out_dir / "city_category_trajectory.csv", pd.DataFrame.from_records(cat_rows))
    write_csv(out_dir / "city_complexity_trajectory.csv", pd.DataFrame.from_records(cpx_rows))
    write_csv(out_dir / "city_numeric_trajectory.csv", pd.DataFrame.from_records(num_rows))


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    cls_path = Path(args.classification_jsonl).expanduser().resolve()
    clean_path = Path(args.clean_jsonl).expanduser().resolve()
    event_path = Path(args.event_assignments_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_rows = iter_jsonl(cls_path)
    clean_rows = iter_jsonl(clean_path)
    ev_df = pd.read_csv(event_path) if event_path.is_file() else pd.DataFrame(columns=["issue_id", "ordinance_event_id", "publication_role"])
    if not cls_rows:
        raise SystemExit("No classification rows found.")

    clean_map = {norm_str(r.get("issue_id")): r for r in clean_rows if norm_str(r.get("issue_id"))}
    ev_map: dict[str, dict[str, Any]] = {}
    for r in ev_df.itertuples(index=False):
        iid = norm_str(getattr(r, "issue_id", ""))
        if not iid:
            continue
        ev_map[iid] = {
            "ordinance_event_id": norm_str(getattr(r, "ordinance_event_id", "")),
            "publication_role": norm_str(getattr(r, "publication_role", "")),
            "event_type": norm_str(getattr(r, "event_type", "")),
            "event_status": norm_str(getattr(r, "event_status", "")),
            "qualification_decision": norm_str(getattr(r, "qualification_decision", "")),
        }

    # Build issue metadata map from run_dir panels.
    issue_meta_map: dict[str, dict[str, Any]] = {}
    panel_dir = run_dir / "panels"
    for city in sorted(panel_dir.iterdir()):
        if not city.is_dir():
            continue
        rows = load_issue_rows_for_city(run_dir, city.name)
        for r in rows:
            iid = norm_str(r.get("issue_id"))
            if iid:
                issue_meta_map[iid] = r

    panels: dict[str, list[dict[str, Any]]] = {}
    panel_meta: dict[str, dict[str, Any]] = {}
    for r in cls_rows:
        iid = norm_str(r.get("issue_id"))
        slug = norm_str(r.get("newspaper_slug")).lower()
        meta = issue_meta_map.get(iid) or {}
        city_key = norm_str(r.get("city_key")) or norm_str(meta.get("city_key"))
        city_name = norm_str(meta.get("city_name"))
        state = norm_str(meta.get("state_abbr")).lower()
        evr = ev_map.get(iid) or {}
        if bool(args.qualified_only):
            if not evr:
                continue
            if norm_str(evr.get("qualification_decision")) not in {"", "pass"}:
                continue
            if norm_str(evr.get("event_type")) not in {"full_ordinance", "amendment", "proposal", "notice"}:
                continue
        panel_meta[slug] = {"city_key": city_key, "city_name": city_name, "state_abbr": state}
        panels.setdefault(slug, []).append(
            {
                "newspaper_slug": slug,
                "city_key": city_key,
                "city_name": city_name,
                "state_abbr": state,
                "issue_id": iid,
                "issue_date": norm_str(r.get("issue_date") or meta.get("issue_date")),
                "issue_class": norm_str(r.get("issue_class")),
                "contains_verbatim_zoning_law": norm_str(r.get("contains_verbatim_zoning_law")),
                "law_text_sufficiency": norm_str(r.get("law_text_sufficiency")),
                "clean_zoning_text": norm_str((clean_map.get(iid) or {}).get("clean_zoning_text")),
                "raw_text": norm_str(meta.get("text")),
                "ordinance_event_id": norm_str(evr.get("ordinance_event_id")),
                "publication_role": norm_str(evr.get("publication_role")),
                "event_type": norm_str(evr.get("event_type")),
                "event_status": norm_str(evr.get("event_status")),
                "qualification_decision": norm_str(evr.get("qualification_decision")),
            }
        )
    for slug in panels:
        panels[slug] = sorted(panels[slug], key=lambda x: (norm_str(x.get("issue_date")), norm_str(x.get("issue_id"))))

    req_rows: list[tuple[str, str]] = []
    req_meta: dict[str, dict[str, Any]] = {}

    overflow_slugs: list[str] = []
    for slug, rows in sorted(panels.items()):
        pm = panel_meta.get(slug) or {}
        prompt, panel_chars = _build_panel_prompt(
            slug,
            city_key=norm_str(pm.get("city_key")),
            city_name=norm_str(pm.get("city_name")),
            state_abbr=norm_str(pm.get("state_abbr")),
            rows=rows,
        )
        if int(args.max_panel_chars) > 0 and panel_chars > int(args.max_panel_chars):
            raise SystemExit(
                f"Panel for newspaper '{slug}' has {panel_chars} chars, exceeding hard cap {int(args.max_panel_chars)}. "
                "Overflow chunking is disabled; increase --max-panel-chars or reduce scope."
            )
        cid = f"panel_longitudinal_v3::{slug}"
        req_rows.append((cid, prompt))
        req_meta[cid] = {
            "newspaper_slug": slug,
            "city_key": norm_str(pm.get("city_key")),
            "city_name": norm_str(pm.get("city_name")),
            "state_abbr": norm_str(pm.get("state_abbr")),
            "issue_count": int(len(rows)),
            "panel_chars_original": int(panel_chars),
            "panel_chars_used": int(panel_chars),
            "overflow_applied": False,
            "clipped_issue_count": 0,
            "issue_ids": [norm_str(x.get("issue_id")) for x in rows if norm_str(x.get("issue_id"))],
        }

    # Run non-overflow single-shot requests in one batch.
    if req_rows:
        write_openai_requests(out_dir / "requests" / "openai_requests_shard000.jsonl", req_rows, model=str(args.model))
        run_gateway_requests(
            request_dir=out_dir / "requests",
            output_dir=out_dir / "results",
            runner_path=Path(args.gateway_runner).expanduser().resolve(),
            model=str(args.model),
            gateway_pythonpath=Path(args.gateway_pythonpath).expanduser().resolve() if norm_str(args.gateway_pythonpath) else None,
            timeout_s=float(args.timeout),
            concurrency=int(args.concurrency),
            gov_env_path=Path(args.gov_env_path).expanduser().resolve(),
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
        )

    res_map = read_result_jsonl(out_dir / "results")
    err_map = read_error_jsonl(out_dir / "results")
    parsed_rows: list[dict[str, Any]] = []

    for cid, _prompt in req_rows:
        meta = req_meta[cid]
        output_text = ""
        if cid in res_map:
            body = (((res_map[cid].get("response") or {}).get("body")) or {})
            output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in err_map:
            body = (((err_map[cid].get("response") or {}).get("body")) or {})
            output_text = norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
        parsed = parse_json_from_text(output_text)
        parse_ok = int(isinstance(parsed, dict) and bool(parsed))
        coerced = _coerce_output(
            parsed if isinstance(parsed, dict) else {},
            city_key=meta["city_key"],
            city_name=meta["city_name"],
            state_abbr=meta["state_abbr"],
            issue_ids=set(meta["issue_ids"]),
        )
        parsed_rows.append(
            {
                "custom_id": cid,
                "newspaper_slug": meta["newspaper_slug"],
                "parse_ok": parse_ok,
                "meta": meta,
                "parsed_json": coerced,
            }
        )

    # Overflow fallback with LLM chunk planner + merge.
    for slug in overflow_slugs:
        rows = panels[slug]
        pm = panel_meta.get(slug) or {}
        city_key = norm_str(pm.get("city_key"))
        city_name = norm_str(pm.get("city_name"))
        state = norm_str(pm.get("state_abbr"))
        issue_ids = [norm_str(x.get("issue_id")) for x in rows if norm_str(x.get("issue_id"))]
        panel_chars = sum(len(norm_str(x.get("raw_text"))) for x in rows)
        target_chunks = max(2, int(math.ceil(panel_chars / max(1, int(args.max_panel_chars)))))

        # 1) planner
        planner_prompt = _build_chunk_planner_prompt(slug, rows, target_chunks=target_chunks)
        planner_parsed = _run_single_request(
            planner_prompt,
            custom_id=f"panel_longitudinal_v3_planner::{slug}",
            tmp_dir=out_dir / "overflow" / slug / "planner",
            model=str(args.model),
            args=args,
        )
        chunks: list[tuple[int, int]] = []
        if isinstance(planner_parsed, dict) and isinstance(planner_parsed.get("chunks"), list):
            for c in planner_parsed.get("chunks"):
                if not isinstance(c, dict):
                    continue
                s = safe_int(c.get("start_index_1based"), default=0)
                e = safe_int(c.get("end_index_1based"), default=0)
                if s <= 0 or e <= 0:
                    continue
                if s > e:
                    s, e = e, s
                s = max(1, min(s, len(rows)))
                e = max(1, min(e, len(rows)))
                chunks.append((s, e))
        if not chunks:
            # fallback deterministic split only for index ranges.
            step = int(math.ceil(len(rows) / target_chunks))
            step = max(1, step)
            chunks = []
            i = 1
            while i <= len(rows):
                j = min(len(rows), i + step - 1)
                chunks.append((i, j))
                i = j + 1

        # 2) chunk extractions
        chunk_outputs: list[dict[str, Any]] = []
        clipped_total = 0
        for idx, (s, e) in enumerate(chunks, start=1):
            sub = rows[s - 1 : e]
            chunk_prompt, _sub_chars = _build_panel_prompt(
                slug,
                city_key=city_key,
                city_name=city_name,
                state_abbr=state,
                rows=sub,
            )
            parsed_chunk = _run_single_request(
                chunk_prompt,
                custom_id=f"panel_longitudinal_v3_chunk::{slug}::{idx}",
                tmp_dir=out_dir / "overflow" / slug / f"chunk_{idx}",
                model=str(args.model),
                args=args,
            )
            if isinstance(parsed_chunk, dict):
                chunk_outputs.append(parsed_chunk)
            clipped_total += len(sub)

        # 3) merge
        merge_schema = _schema_template(city_key, city_name, state)
        merge_lines = []
        merge_lines.append("Merge chunk-level longitudinal outputs into one final panel-level output.")
        merge_lines.append("Return ONLY JSON.")
        merge_lines.append("")
        merge_lines.append("Use all chunk outputs as evidence; resolve conflicts conservatively.")
        merge_lines.append("Preserve schema keys exactly.")
        merge_lines.append("JSON schema:")
        merge_lines.append(json.dumps(merge_schema, ensure_ascii=False))
        merge_lines.append("")
        merge_lines.append("Chunk outputs:")
        for i, co in enumerate(chunk_outputs, start=1):
            merge_lines.append("")
            merge_lines.append(f"[CHUNK {i}]")
            merge_lines.append(json.dumps(co, ensure_ascii=False))
        merge_lines.append("")
        merge_lines.append("Return JSON now.")
        merged_parsed = _run_single_request(
            "\n".join(merge_lines),
            custom_id=f"panel_longitudinal_v3_merge::{slug}",
            tmp_dir=out_dir / "overflow" / slug / "merge",
            model=str(args.model),
            args=args,
        )
        coerced = _coerce_output(
            merged_parsed if isinstance(merged_parsed, dict) else {},
            city_key=city_key,
            city_name=city_name,
            state_abbr=state,
            issue_ids=set(issue_ids),
        )
        parsed_rows.append(
            {
                "custom_id": f"panel_longitudinal_v3::{slug}",
                "newspaper_slug": slug,
                "parse_ok": int(isinstance(merged_parsed, dict) and bool(merged_parsed)),
                "meta": {
                    "newspaper_slug": slug,
                    "city_key": city_key,
                    "city_name": city_name,
                    "state_abbr": state,
                    "issue_count": int(len(rows)),
                    "panel_chars_original": int(panel_chars),
                    "panel_chars_used": int(panel_chars),
                    "overflow_applied": True,
                    "clipped_issue_count": int(clipped_total),
                    "issue_ids": issue_ids,
                },
                "parsed_json": coerced,
            }
        )

    parsed_rows = sorted(parsed_rows, key=lambda r: norm_str(r.get("newspaper_slug")))
    write_jsonl(out_dir / "panel_outputs_v3.jsonl", parsed_rows)
    _flatten_outputs(parsed_rows, out_dir)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "classification_jsonl": str(cls_path),
        "clean_jsonl": str(clean_path),
        "event_assignments_csv": str(event_path),
        "output_dir": str(out_dir),
        "model": str(args.model),
        "qualified_only": bool(args.qualified_only),
        "newspaper_count": int(len(panels)),
        "overflow_newspaper_count": int(len(overflow_slugs)),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eprint(f"Done. longitudinal output: {out_dir}")


if __name__ == "__main__":
    main()
