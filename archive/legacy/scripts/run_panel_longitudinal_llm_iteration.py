#!/usr/bin/env python3
"""
Iterative panel-level LLM extraction experiments (single-shot, full panel context).

Goal:
  - Test multiple prompt designs against the same city panels.
  - Score output usefulness/consistency.
  - Pick the strongest variant.
  - Run the winner across all pilot cities.

Outputs under <run-dir>/longitudinal_iter (default):
  - city_panel_inventory.csv
  - experiment_manifest.json
  - experiments/<variant>/sample/...
  - variant_scorecard.csv
  - winner_manifest.json
  - final/<winner>/...
  - final/<winner>/city_summary.csv
  - final/<winner>/city_issue_timeline.csv
  - final/<winner>/city_stage_category_shares.csv
  - final/<winner>/city_phase_segments.csv
  - final/<winner>/city_amendment_waves.csv
  - final/<winner>/city_numeric_constraints.csv
  - provenance.json
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_FIXED_CATEGORY_KEYS = [
    "land_use_restrictions",
    "bulk_dimensional_standards",
    "procedural_governance",
    "nonconforming_use",
    "definitions",
    "parking_transport",
    "environmental_protection",
    "historic_preservation",
    "flexible_zoning",
    "inclusionary_affordable_housing",
    "enforcement_penalties",
    "miscellaneous_site_rules",
]

_ALLOWED_SIGNAL = {"none", "mention", "amendment", "full_ordinance"}
_ALLOWED_DOC_ROLE = {"none", "notice", "hearing", "amendment", "full_ordinance", "summary", "other"}
_ALLOWED_DIR = {"up", "down", "flat", "mixed", "uncertain"}
_ALLOWED_PHASE = {
    "initial_establishment",
    "formalization_enforcement",
    "expansion_reclassification",
    "flexible_special_use",
    "infrastructure_environment",
    "maturation_implementation",
    "other_unclear",
}
_ALLOWED_WAVE = {"expansion", "tightening", "clarification", "reclassification", "mixed", "none", "uncertain"}

_NUMERIC_METRICS = [
    "lot_size_sqft",
    "height_ft",
    "front_setback_ft",
    "parking_spaces_per_unit",
]


def _norm_str(x: Any) -> str:
    return str(x or "").strip()


def _safe_float(x: Any, default: float = math.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def _safe_int(x: Any, default: int = 0) -> int:
    v = _safe_float(x, default=math.nan)
    if pd.isna(v):
        return int(default)
    try:
        return int(round(v))
    except Exception:
        return int(default)


def _enum(v: Any, allowed: set[str], default: str) -> str:
    s = _norm_str(v).lower()
    return s if s in allowed else default


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


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
    s = _norm_str(candidate)
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


def _score_parsed_candidate(obj: dict[str, Any]) -> tuple[int, int, int]:
    keys = set(obj.keys())
    primary = {
        "first_events",
        "panel_mix_counts",
        "stage_category_shares",
        "stage_complexity_scores",
        "phase_segments",
        "amendment_waves",
        "district_evolution",
        "numeric_constraints_single_family",
        "narrative_summary",
    }
    secondary = {
        "first_zoning_signal",
        "first_full_ordinance",
        "stage_definition",
        "issue_timeline",
        "category_trajectory",
        "complexity_trajectory",
        "evolution_phases",
    }
    p = sum(1 for k in primary if k in keys)
    s = sum(1 for k in secondary if k in keys)
    return (p, s, len(keys))


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    s = _norm_str(text)
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    candidates: list[dict[str, Any]] = []

    def _add_candidate(obj: Any) -> None:
        if isinstance(obj, dict):
            candidates.append(obj)

    try:
        _add_candidate(json.loads(s))
    except Exception:
        pass
    first = s.find("{")
    if first != -1:
        candidate = s[first:]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        repaired = _repair_truncated_json(candidate)
        try:
            _add_candidate(json.loads(repaired))
        except Exception:
            pass
    # If model output includes trailing text or concatenated objects, recover
    # by decoding the first valid JSON object from the earliest "{" position.
    dec = json.JSONDecoder()
    start = first
    while start != -1:
        try:
            obj, _ = dec.raw_decode(s[start:])
            _add_candidate(obj)
        except Exception:
            pass
        start = s.find("{", start + 1)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        candidate = m.group(0)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        candidate = _repair_truncated_json(candidate)
        try:
            _add_candidate(json.loads(candidate))
        except Exception:
            pass
    if not candidates:
        return None
    candidates_sorted = sorted(
        candidates,
        key=lambda x: (_score_parsed_candidate(x), len(json.dumps(x, ensure_ascii=False))),
        reverse=True,
    )
    return candidates_sorted[0]


def _write_openai_requests(path: Path, rows: list[tuple[str, str]], model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for custom_id, prompt in rows:
            req = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": str(model),
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                    "stream": False,
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def _run_gateway_requests(
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
        if not env.get(key) and _norm_str(parsed.get(key)):
            env[key] = _norm_str(parsed[key])
    if not env.get("OPENAI_KEY"):
        alias = _norm_str(parsed.get("CODEX_API_KEY") or parsed.get("OPENAI_API_KEY"))
        if alias:
            env["OPENAI_KEY"] = alias

    provider = str(model).split(":", 1)[0].lower() if ":" in str(model) else "openai"
    need_key = {"openai": "OPENAI_KEY", "gemini": "GEMINI_KEY", "claude": "CLAUDE_KEY"}.get(provider)

    candidates: list[str] = []
    if need_key:
        seen: set[str] = set()
        for cand in (
            _norm_str(env.get(need_key)),
            _norm_str(parsed.get(need_key)),
            *[_norm_str(parsed.get(f"{need_key}_BACKUP_{i}")) for i in range(1, 6)],
        ):
            if not cand or cand in seen:
                continue
            seen.add(cand)
            candidates.append(cand)
        if not candidates:
            raise SystemExit(f"{need_key} missing for provider={provider}")

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

    if not need_key:
        subprocess.run(cmd, check=True, env=env)
        return

    last_err: Exception | None = None
    for i, key_val in enumerate(candidates, start=1):
        env_try = dict(env)
        env_try[need_key] = key_val
        try:
            print(f"[gateway] trying {need_key} candidate {i}/{len(candidates)}", flush=True)
            subprocess.run(cmd, check=True, env=env_try)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            continue
    raise SystemExit(f"Gateway failed for all {need_key} candidates: {last_err}")


def _read_result_jsonl(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(results_dir.glob("openai_results_shard*.jsonl")):
        for obj in _iter_jsonl(p):
            cid = _norm_str(obj.get("custom_id"))
            if cid:
                out[cid] = obj
    return out


def _read_error_jsonl(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(results_dir.glob("openai_errors_shard*.jsonl")):
        for obj in _iter_jsonl(p):
            cid = _norm_str(obj.get("custom_id"))
            if cid:
                out[cid] = obj
    return out


def _issue_stage_bins(issues_sorted: list[dict[str, Any]]) -> dict[str, list[str]]:
    n = len(issues_sorted)
    idx = list(range(n))
    chunks = np.array_split(np.array(idx), 3)
    early = [str(issues_sorted[i].get("issue_id")) for i in chunks[0].tolist()] if len(chunks) >= 1 else []
    mid = [str(issues_sorted[i].get("issue_id")) for i in chunks[1].tolist()] if len(chunks) >= 2 else []
    late = [str(issues_sorted[i].get("issue_id")) for i in chunks[2].tolist()] if len(chunks) >= 3 else []
    return {"early_issue_ids": early, "mid_issue_ids": mid, "late_issue_ids": late}


def _category_stage_template() -> dict[str, dict[str, float]]:
    return {k: 0.0 for k in _FIXED_CATEGORY_KEYS}


def _build_prompt(
    *,
    variant: str,
    city_key: str,
    city_name: str,
    state_abbr: str,
    issues: list[dict[str, Any]],
    context_budget_chars: int,
) -> tuple[str, dict[str, Any]]:
    issues_sorted = sorted(issues, key=lambda r: (_norm_str(r.get("issue_date")), _norm_str(r.get("issue_id"))))
    total_chars = sum(len(_norm_str(x.get("text"))) for x in issues_sorted)
    if int(context_budget_chars) > 0 and total_chars > int(context_budget_chars):
        raise ValueError(
            f"city_key={city_key} panel chars {total_chars} exceeds hard cap {int(context_budget_chars)}. "
            "Clipping fallback is disabled; increase cap or reduce scope."
        )
    overflow = False
    per_issue_budget = 0

    stage_bins = _issue_stage_bins(issues_sorted)
    issue_ids = [_norm_str(x.get("issue_id")) for x in issues_sorted if _norm_str(x.get("issue_id"))]

    schema_base: dict[str, Any] = {
        "city_key": city_key,
        "city_name": city_name,
        "state_abbr": state_abbr.lower(),
        "stage_definition": {
            "method": "chronological_issue_terciles",
            "early_issue_ids": stage_bins["early_issue_ids"],
            "mid_issue_ids": stage_bins["mid_issue_ids"],
            "late_issue_ids": stage_bins["late_issue_ids"],
        },
        "issue_timeline": [
            {
                "issue_id": "",
                "issue_date": "",
                "signal_type": "none",
                "document_role": "none",
                "zoning_relevance_0_to_1": 0.0,
                "evidence": "",
            }
        ],
        "first_events": {
            "first_zoning_signal": {"issue_id": "", "issue_date": "", "signal_type": "none", "evidence": ""},
            "first_full_ordinance": {"status": "not_found", "issue_id": "", "issue_date": "", "evidence": ""},
        },
        "panel_mix_counts": {
            "full_ordinance_issues": 0,
            "amendment_issues": 0,
            "mention_only_issues": 0,
            "non_zoning_or_uncertain_issues": 0,
        },
        "stage_category_shares": {
            "early": _category_stage_template(),
            "mid": _category_stage_template(),
            "late": _category_stage_template(),
        },
        "stage_complexity_scores": {
            "early": {
                "regulatory_complexity_0_100": 0.0,
                "procedural_intensity_0_100": 0.0,
                "dimensional_specificity_0_100": 0.0,
            },
            "mid": {
                "regulatory_complexity_0_100": 0.0,
                "procedural_intensity_0_100": 0.0,
                "dimensional_specificity_0_100": 0.0,
            },
            "late": {
                "regulatory_complexity_0_100": 0.0,
                "procedural_intensity_0_100": 0.0,
                "dimensional_specificity_0_100": 0.0,
            },
        },
        "phase_segments": [
            {
                "phase_order": 1,
                "phase_archetype": "initial_establishment",
                "phase_label_open": "",
                "start_issue_id": "",
                "end_issue_id": "",
                "dominant_categories": ["land_use_restrictions"],
                "summary": "",
                "evidence_issue_ids": [""],
            }
        ],
        "amendment_waves": [
            {
                "wave_order": 1,
                "start_issue_id": "",
                "end_issue_id": "",
                "change_type": "mixed",
                "affected_categories": ["procedural_governance"],
                "evidence_issue_ids": [""],
            }
        ],
        "district_evolution": {
            "district_count_early": None,
            "district_count_mid": None,
            "district_count_late": None,
            "district_count_direction": "uncertain",
            "evidence_issue_ids": [""],
        },
        "numeric_constraints_single_family": {
            "lot_size_sqft": {
                "early": None,
                "mid": None,
                "late": None,
                "direction": "uncertain",
                "evidence_issue_ids": [""],
                "evidence_quotes": [{"issue_id": "", "quote_excerpt": ""}],
            },
            "height_ft": {
                "early": None,
                "mid": None,
                "late": None,
                "direction": "uncertain",
                "evidence_issue_ids": [""],
                "evidence_quotes": [{"issue_id": "", "quote_excerpt": ""}],
            },
            "front_setback_ft": {
                "early": None,
                "mid": None,
                "late": None,
                "direction": "uncertain",
                "evidence_issue_ids": [""],
                "evidence_quotes": [{"issue_id": "", "quote_excerpt": ""}],
            },
            "parking_spaces_per_unit": {
                "early": None,
                "mid": None,
                "late": None,
                "direction": "uncertain",
                "evidence_issue_ids": [""],
                "evidence_quotes": [{"issue_id": "", "quote_excerpt": ""}],
            },
        },
        "quality_flags": [""],
        "narrative_summary": "",
    }

    lean_mode = variant == "panel_lean_numeric_quote_v1"
    if lean_mode:
        # Avoid extremely long outputs on large-city panels by not requiring
        # one-row-per-issue timeline echo in the response object.
        schema_base.pop("stage_definition", None)
        schema_base.pop("issue_timeline", None)

    if variant == "timeline_evidence_v1":
        schema_base["evidence_registry"] = [
            {
                "claim_id": "claim_1",
                "issue_id": "",
                "quote_excerpt": "",
                "why_it_matters": "",
            }
        ]
    if variant == "hybrid_open_v1":
        schema_base["emergent_patterns"] = [
            {
                "pattern_label": "",
                "description": "",
                "evidence_issue_ids": [""],
                "maps_to_fixed_categories": ["procedural_governance"],
            }
        ]

    lines: list[str] = []
    lines.append("You are analyzing a full chronological city panel of newspaper zoning transcripts.")
    lines.append("Return ONLY one JSON object. No markdown, no backticks, no preface.")
    lines.append("")
    lines.append("Core requirements:")
    lines.append("- You must use all issues provided in this panel, in sequence.")
    if lean_mode:
        lines.append("- Do not return per-issue timeline rows. Return only panel-level structured outputs.")
        lines.append("- panel_mix_counts must classify the full panel and sum exactly to issue_count.")
    else:
        lines.append("- issue_timeline must have exactly one row per issue_id and only use provided issue_ids.")
    lines.append("- signal_type allowed: none|mention|amendment|full_ordinance")
    lines.append("- document_role allowed: none|notice|hearing|amendment|full_ordinance|summary|other")
    lines.append("- phase_archetype allowed: " + "|".join(sorted(_ALLOWED_PHASE)))
    lines.append("- amendment change_type allowed: " + "|".join(sorted(_ALLOWED_WAVE)))
    lines.append("- direction allowed: up|down|flat|mixed|uncertain")
    lines.append("- stage_category_shares must be compositional by stage (sum close to 1 in each stage).")
    lines.append("- If evidence is weak, use uncertain and explain in quality_flags.")
    lines.append("")
    lines.append("Stage bins are pre-defined and must be used exactly:")
    lines.append(json.dumps(stage_bins, ensure_ascii=False))
    lines.append("")
    lines.append("Interpretation guidance:")
    lines.append("- 'mention' means zoning is discussed but not clearly a legal adoption text.")
    lines.append("- 'amendment' means a legal change to existing zoning code/district map.")
    lines.append("- 'full_ordinance' means comprehensive ordinance/adoption text appears.")
    lines.append("- district_evolution should reflect count/variety of district classes, not parcel counts.")
    lines.append("- numeric_constraints_single_family should focus on single-family district rules when present.")
    lines.append("- If a numeric trend is not comparable across the same district family, mark direction as mixed or uncertain.")
    lines.append("- Do not force monotonic numeric trends if evidence conflicts across districts or amendment scopes.")
    lines.append("- For each numeric metric with a non-uncertain direction, provide at least one verbatim evidence quote containing a number and unit.")
    lines.append("")
    if variant == "timeline_evidence_v1":
        lines.append("Extra requirement for this run: include evidence_registry with concise verbatim excerpts and linked issue_id.")
    elif variant == "hybrid_open_v1":
        lines.append("Extra requirement for this run: include emergent_patterns (open discovery), but map each to fixed categories.")
    elif variant == "compact_closed_v2":
        lines.append("Extra requirement for this run: be conservative on numeric directions and prefer uncertain over over-claiming.")
        lines.append("When assigning first_full_ordinance, require clear comprehensive ordinance language; otherwise keep status uncertain/not_found.")
    elif variant == "compact_numeric_quote_v1":
        lines.append("Extra requirement for this run: prioritize numeric reliability with explicit evidence_quotes for each numeric claim.")
        lines.append("If you cannot provide a quote containing both a number and unit, set that metric direction to uncertain.")
    elif variant == "panel_lean_numeric_quote_v1":
        lines.append("Extra requirement for this run: lean panel summary mode (no issue_timeline output).")
        lines.append("Focus output space on first events, stage shares, phases, amendment waves, district evolution, and numeric evidence quotes.")
        lines.append("Use evidence_issue_ids to anchor each substantive claim to concrete issues from this panel.")
    else:
        lines.append("Keep output concise and strictly structured.")
    lines.append("")
    lines.append(f"City: {city_name}, {state_abbr.upper()} ({city_key})")
    lines.append(f"Issue IDs in panel ({len(issue_ids)}): {', '.join(issue_ids)}")
    lines.append(f"Panel chars original: {total_chars}")
    lines.append(f"Overflow applied: {str(overflow).lower()}")
    lines.append("")
    lines.append("Output JSON schema template:")
    lines.append(json.dumps(schema_base, ensure_ascii=False))
    lines.append("")
    lines.append("Chronological issue transcripts:")

    clipped = 0
    used_chars = 0
    for i, issue in enumerate(issues_sorted, start=1):
        iid = _norm_str(issue.get("issue_id"))
        dtv = _norm_str(issue.get("issue_date"))
        lbl = _norm_str(issue.get("classification_label"))
        txt = _norm_str(issue.get("text"))
        used = txt
        was_clipped = False
        used_chars += len(used)
        lines.append("")
        lines.append(f"[ISSUE {i}] issue_id={iid} issue_date={dtv} label={lbl} text_chars={len(txt)} used_chars={len(used)} clipped={str(was_clipped).lower()}")
        lines.append(used)

    lines.append("")
    lines.append("Return JSON now.")

    meta = {
        "variant": variant,
        "issue_count": len(issues_sorted),
        "panel_chars_original": total_chars,
        "panel_chars_used": used_chars,
        "overflow_applied": bool(overflow),
        "per_issue_budget": int(per_issue_budget),
        "clipped_issue_count": int(clipped),
        "stage_bins": stage_bins,
    }
    return "\n".join(lines).strip(), meta


def _coerce_output(parsed: dict[str, Any], issue_ids: set[str]) -> dict[str, Any]:
    out = dict(parsed or {})
    sd = out.get("stage_definition")
    if not isinstance(sd, dict):
        sd = {}
    out["stage_definition"] = {
        "method": _norm_str(sd.get("method")) or "chronological_issue_terciles",
        "early_issue_ids": [x for x in [_norm_str(v) for v in sd.get("early_issue_ids", [])] if x in issue_ids],
        "mid_issue_ids": [x for x in [_norm_str(v) for v in sd.get("mid_issue_ids", [])] if x in issue_ids],
        "late_issue_ids": [x for x in [_norm_str(v) for v in sd.get("late_issue_ids", [])] if x in issue_ids],
    }

    timeline = out.get("issue_timeline")
    if not isinstance(timeline, list):
        timeline = []
    fixed_timeline: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in timeline:
        if not isinstance(row, dict):
            continue
        iid = _norm_str(row.get("issue_id"))
        if not iid or iid not in issue_ids or iid in seen:
            continue
        seen.add(iid)
        fixed_timeline.append(
            {
                "issue_id": iid,
                "issue_date": _norm_str(row.get("issue_date")),
                "signal_type": _enum(row.get("signal_type"), _ALLOWED_SIGNAL, "none"),
                "document_role": _enum(row.get("document_role"), _ALLOWED_DOC_ROLE, "other"),
                "zoning_relevance_0_to_1": max(0.0, min(1.0, _safe_float(row.get("zoning_relevance_0_to_1"), default=0.0))),
                "evidence": _norm_str(row.get("evidence")),
            }
        )
    out["issue_timeline"] = fixed_timeline

    fe = out.get("first_events")
    if not isinstance(fe, dict):
        fe = {}
    fzs = fe.get("first_zoning_signal") if isinstance(fe.get("first_zoning_signal"), dict) else {}
    ffo = fe.get("first_full_ordinance") if isinstance(fe.get("first_full_ordinance"), dict) else {}
    out["first_events"] = {
        "first_zoning_signal": {
            "issue_id": _norm_str(fzs.get("issue_id")),
            "issue_date": _norm_str(fzs.get("issue_date")),
            "signal_type": _enum(fzs.get("signal_type"), _ALLOWED_SIGNAL, "none"),
            "evidence": _norm_str(fzs.get("evidence")),
        },
        "first_full_ordinance": {
            "status": "found" if _norm_str(ffo.get("issue_id")) in issue_ids else _enum(ffo.get("status"), {"found", "not_found", "uncertain"}, "uncertain"),
            "issue_id": _norm_str(ffo.get("issue_id")),
            "issue_date": _norm_str(ffo.get("issue_date")),
            "evidence": _norm_str(ffo.get("evidence")),
        },
    }

    pm = out.get("panel_mix_counts")
    if not isinstance(pm, dict):
        pm = {}
    out["panel_mix_counts"] = {
        "full_ordinance_issues": max(0, _safe_int(pm.get("full_ordinance_issues"), 0)),
        "amendment_issues": max(0, _safe_int(pm.get("amendment_issues"), 0)),
        "mention_only_issues": max(0, _safe_int(pm.get("mention_only_issues"), 0)),
        "non_zoning_or_uncertain_issues": max(0, _safe_int(pm.get("non_zoning_or_uncertain_issues"), 0)),
    }

    sc = out.get("stage_category_shares")
    if not isinstance(sc, dict):
        sc = {}
    sc_fixed: dict[str, dict[str, float]] = {}
    for stage in ("early", "mid", "late"):
        obj = sc.get(stage) if isinstance(sc.get(stage), dict) else {}
        stage_vals = {k: max(0.0, _safe_float(obj.get(k), default=0.0)) for k in _FIXED_CATEGORY_KEYS}
        sc_fixed[stage] = stage_vals
    out["stage_category_shares"] = sc_fixed

    cpx = out.get("stage_complexity_scores")
    if not isinstance(cpx, dict):
        cpx = {}
    cpx_fixed: dict[str, dict[str, float]] = {}
    for stage in ("early", "mid", "late"):
        obj = cpx.get(stage) if isinstance(cpx.get(stage), dict) else {}
        cpx_fixed[stage] = {
            "regulatory_complexity_0_100": max(0.0, min(100.0, _safe_float(obj.get("regulatory_complexity_0_100"), default=math.nan))),
            "procedural_intensity_0_100": max(0.0, min(100.0, _safe_float(obj.get("procedural_intensity_0_100"), default=math.nan))),
            "dimensional_specificity_0_100": max(0.0, min(100.0, _safe_float(obj.get("dimensional_specificity_0_100"), default=math.nan))),
        }
    out["stage_complexity_scores"] = cpx_fixed

    phases = out.get("phase_segments")
    if not isinstance(phases, list):
        phases = []
    fixed_phases: list[dict[str, Any]] = []
    for i, ph in enumerate(phases, start=1):
        if not isinstance(ph, dict):
            continue
        ev = [_norm_str(x) for x in ph.get("evidence_issue_ids", [])] if isinstance(ph.get("evidence_issue_ids"), list) else []
        ev = [x for x in ev if x in issue_ids]
        fixed_phases.append(
            {
                "phase_order": max(1, _safe_int(ph.get("phase_order"), i)),
                "phase_archetype": _enum(ph.get("phase_archetype"), _ALLOWED_PHASE, "other_unclear"),
                "phase_label_open": _norm_str(ph.get("phase_label_open")),
                "start_issue_id": _norm_str(ph.get("start_issue_id")),
                "end_issue_id": _norm_str(ph.get("end_issue_id")),
                "dominant_categories": [x for x in [_norm_str(v) for v in (ph.get("dominant_categories") or [])] if x in _FIXED_CATEGORY_KEYS],
                "summary": _norm_str(ph.get("summary")),
                "evidence_issue_ids": ev,
            }
        )
    out["phase_segments"] = fixed_phases

    waves = out.get("amendment_waves")
    if not isinstance(waves, list):
        waves = []
    fixed_waves: list[dict[str, Any]] = []
    for i, w in enumerate(waves, start=1):
        if not isinstance(w, dict):
            continue
        ev = [_norm_str(x) for x in w.get("evidence_issue_ids", [])] if isinstance(w.get("evidence_issue_ids"), list) else []
        ev = [x for x in ev if x in issue_ids]
        fixed_waves.append(
            {
                "wave_order": max(1, _safe_int(w.get("wave_order"), i)),
                "start_issue_id": _norm_str(w.get("start_issue_id")),
                "end_issue_id": _norm_str(w.get("end_issue_id")),
                "change_type": _enum(w.get("change_type"), _ALLOWED_WAVE, "uncertain"),
                "affected_categories": [x for x in [_norm_str(v) for v in (w.get("affected_categories") or [])] if x in _FIXED_CATEGORY_KEYS],
                "evidence_issue_ids": ev,
            }
        )
    out["amendment_waves"] = fixed_waves

    de = out.get("district_evolution")
    if not isinstance(de, dict):
        de = {}
    out["district_evolution"] = {
        "district_count_early": _safe_int(de.get("district_count_early"), default=-1),
        "district_count_mid": _safe_int(de.get("district_count_mid"), default=-1),
        "district_count_late": _safe_int(de.get("district_count_late"), default=-1),
        "district_count_direction": _enum(de.get("district_count_direction"), _ALLOWED_DIR, "uncertain"),
        "evidence_issue_ids": [x for x in [_norm_str(v) for v in (de.get("evidence_issue_ids") or [])] if x in issue_ids],
    }

    nc = out.get("numeric_constraints_single_family")
    if not isinstance(nc, dict):
        nc = {}
    nc_fix: dict[str, Any] = {}
    for m in _NUMERIC_METRICS:
        obj = nc.get(m) if isinstance(nc.get(m), dict) else {}
        quotes_raw = obj.get("evidence_quotes") if isinstance(obj.get("evidence_quotes"), list) else []
        quotes: list[dict[str, str]] = []
        for q in quotes_raw:
            if not isinstance(q, dict):
                continue
            qid = _norm_str(q.get("issue_id"))
            qtxt = _norm_str(q.get("quote_excerpt"))
            if qid and qid in issue_ids and qtxt:
                quotes.append({"issue_id": qid, "quote_excerpt": qtxt})
        nc_fix[m] = {
            "early": _safe_float(obj.get("early"), default=math.nan),
            "mid": _safe_float(obj.get("mid"), default=math.nan),
            "late": _safe_float(obj.get("late"), default=math.nan),
            "direction": _enum(obj.get("direction"), _ALLOWED_DIR, "uncertain"),
            "evidence_issue_ids": [x for x in [_norm_str(v) for v in (obj.get("evidence_issue_ids") or [])] if x in issue_ids],
            "evidence_quotes": quotes,
        }
    out["numeric_constraints_single_family"] = nc_fix

    qf = out.get("quality_flags")
    out["quality_flags"] = [_norm_str(x) for x in qf if _norm_str(x)] if isinstance(qf, list) else ([_norm_str(qf)] if _norm_str(qf) else [])
    out["narrative_summary"] = _norm_str(out.get("narrative_summary"))

    if isinstance(out.get("evidence_registry"), list):
        reg: list[dict[str, Any]] = []
        for x in out.get("evidence_registry"):
            if not isinstance(x, dict):
                continue
            iid = _norm_str(x.get("issue_id"))
            if iid and iid in issue_ids:
                reg.append(
                    {
                        "claim_id": _norm_str(x.get("claim_id")),
                        "issue_id": iid,
                        "quote_excerpt": _norm_str(x.get("quote_excerpt")),
                        "why_it_matters": _norm_str(x.get("why_it_matters")),
                    }
                )
        out["evidence_registry"] = reg

    if isinstance(out.get("emergent_patterns"), list):
        pats: list[dict[str, Any]] = []
        for x in out.get("emergent_patterns"):
            if not isinstance(x, dict):
                continue
            ev = [v for v in [_norm_str(vv) for vv in (x.get("evidence_issue_ids") or [])] if v in issue_ids]
            maps = [v for v in [_norm_str(vv) for vv in (x.get("maps_to_fixed_categories") or [])] if v in _FIXED_CATEGORY_KEYS]
            pats.append(
                {
                    "pattern_label": _norm_str(x.get("pattern_label")),
                    "description": _norm_str(x.get("description")),
                    "evidence_issue_ids": ev,
                    "maps_to_fixed_categories": maps,
                }
            )
        out["emergent_patterns"] = pats

    return out


def _score_output(parsed: dict[str, Any], issue_ids: list[str]) -> dict[str, Any]:
    iid_set = set(issue_ids)
    timeline = parsed.get("issue_timeline") if isinstance(parsed.get("issue_timeline"), list) else []
    t_ids = [_norm_str(x.get("issue_id")) for x in timeline if isinstance(x, dict)]
    t_ids_valid = [x for x in t_ids if x in iid_set]
    timeline_coverage = (len(set(t_ids_valid)) / max(1, len(iid_set))) if timeline else math.nan

    pm = parsed.get("panel_mix_counts") if isinstance(parsed.get("panel_mix_counts"), dict) else {}
    mix_total = (
        max(0, _safe_int(pm.get("full_ordinance_issues"), 0))
        + max(0, _safe_int(pm.get("amendment_issues"), 0))
        + max(0, _safe_int(pm.get("mention_only_issues"), 0))
        + max(0, _safe_int(pm.get("non_zoning_or_uncertain_issues"), 0))
    )
    panel_mix_completeness = max(0.0, 1.0 - (abs(mix_total - len(iid_set)) / max(1, len(iid_set))))
    coverage_component = timeline_coverage if not math.isnan(timeline_coverage) else panel_mix_completeness

    # Stage share compositional error.
    sc = parsed.get("stage_category_shares") if isinstance(parsed.get("stage_category_shares"), dict) else {}
    stage_errs = []
    for stage in ("early", "mid", "late"):
        obj = sc.get(stage) if isinstance(sc.get(stage), dict) else {}
        s = sum(max(0.0, _safe_float(obj.get(k), default=0.0)) for k in _FIXED_CATEGORY_KEYS)
        stage_errs.append(abs(s - 1.0))
    mean_stage_sum_error = float(np.mean(stage_errs)) if stage_errs else math.nan
    stage_sum_score = max(0.0, 1.0 - min(1.0, mean_stage_sum_error))

    # Evidence validity.
    ev_tot = 0
    ev_valid = 0
    for key in ("phase_segments", "amendment_waves"):
        rows = parsed.get(key) if isinstance(parsed.get(key), list) else []
        for r in rows:
            if not isinstance(r, dict):
                continue
            ev = r.get("evidence_issue_ids") if isinstance(r.get("evidence_issue_ids"), list) else []
            for x in ev:
                ev_tot += 1
                if _norm_str(x) in iid_set:
                    ev_valid += 1
    de = parsed.get("district_evolution") if isinstance(parsed.get("district_evolution"), dict) else {}
    dev = de.get("evidence_issue_ids") if isinstance(de.get("evidence_issue_ids"), list) else []
    for x in dev:
        ev_tot += 1
        if _norm_str(x) in iid_set:
            ev_valid += 1
    nc = parsed.get("numeric_constraints_single_family") if isinstance(parsed.get("numeric_constraints_single_family"), dict) else {}
    for m in _NUMERIC_METRICS:
        obj = nc.get(m) if isinstance(nc.get(m), dict) else {}
        ev = obj.get("evidence_issue_ids") if isinstance(obj.get("evidence_issue_ids"), list) else []
        for x in ev:
            ev_tot += 1
            if _norm_str(x) in iid_set:
                ev_valid += 1
    evidence_valid_share = (ev_valid / ev_tot) if ev_tot > 0 else 1.0

    # Specificity.
    de_dir = _enum(de.get("district_count_direction"), _ALLOWED_DIR, "uncertain")
    district_specific = 1.0 if de_dir in {"up", "down", "flat", "mixed"} else 0.0

    numeric_non_uncertain = 0
    numeric_quote_valid = 0
    for m in _NUMERIC_METRICS:
        obj = nc.get(m) if isinstance(nc.get(m), dict) else {}
        dirv = _enum(obj.get("direction"), _ALLOWED_DIR, "uncertain")
        if dirv != "uncertain":
            numeric_non_uncertain += 1
            quotes = obj.get("evidence_quotes") if isinstance(obj.get("evidence_quotes"), list) else []
            ok = False
            for q in quotes:
                if not isinstance(q, dict):
                    continue
                qid = _norm_str(q.get("issue_id"))
                qtxt = _norm_str(q.get("quote_excerpt"))
                if qid in iid_set and re.search(r"\d", qtxt) and re.search(r"(sq\\.?\\s*ft|feet|foot|ft\\b|spaces?)", qtxt, flags=re.I):
                    ok = True
                    break
            if ok:
                numeric_quote_valid += 1
    numeric_specificity = numeric_non_uncertain / max(1, len(_NUMERIC_METRICS))
    numeric_quote_validity = (numeric_quote_valid / max(1, numeric_non_uncertain)) if numeric_non_uncertain > 0 else 1.0

    # Event specificity.
    fe = parsed.get("first_events") if isinstance(parsed.get("first_events"), dict) else {}
    z = fe.get("first_zoning_signal") if isinstance(fe.get("first_zoning_signal"), dict) else {}
    f = fe.get("first_full_ordinance") if isinstance(fe.get("first_full_ordinance"), dict) else {}
    first_signal_valid = 1.0 if (_norm_str(z.get("issue_id")) in iid_set or _enum(z.get("signal_type"), _ALLOWED_SIGNAL, "none") == "none") else 0.0
    first_full_valid = 1.0 if (_norm_str(f.get("issue_id")) in iid_set or _norm_str(f.get("status")).lower() in {"not_found", "uncertain"}) else 0.0
    event_specificity = 0.5 * first_signal_valid + 0.5 * first_full_valid

    # Phase diversity.
    phases = parsed.get("phase_segments") if isinstance(parsed.get("phase_segments"), list) else []
    phase_labels = [(_enum(x.get("phase_archetype"), _ALLOWED_PHASE, "other_unclear")) for x in phases if isinstance(x, dict)]
    phase_diversity = len(set(phase_labels))
    phase_score = min(1.0, phase_diversity / 3.0)

    overall = (
        0.26 * coverage_component
        + 0.16 * stage_sum_score
        + 0.20 * evidence_valid_share
        + 0.10 * numeric_specificity
        + 0.08 * numeric_quote_validity
        + 0.12 * district_specific
        + 0.12 * event_specificity
    )
    overall = float(max(0.0, min(1.0, overall)))

    return {
        "timeline_coverage": float(timeline_coverage),
        "coverage_component": float(coverage_component),
        "panel_mix_completeness": float(panel_mix_completeness),
        "mean_stage_sum_error": float(mean_stage_sum_error),
        "stage_sum_score": float(stage_sum_score),
        "evidence_valid_share": float(evidence_valid_share),
        "numeric_specificity": float(numeric_specificity),
        "numeric_quote_validity": float(numeric_quote_validity),
        "district_specificity": float(district_specific),
        "event_specificity": float(event_specificity),
        "phase_diversity": int(phase_diversity),
        "phase_score": float(phase_score),
        "overall_quality": float(overall),
    }


def _load_city_inventory(run_dir: Path) -> pd.DataFrame:
    sel_path = run_dir / "panels" / "selected_panel_issues.csv"
    if not sel_path.is_file():
        raise SystemExit(f"Missing {sel_path}")
    d = pd.read_csv(sel_path)
    need = {"city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "issue_id", "text_chars", "issue_date"}
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit(f"selected_panel_issues missing columns: {miss}")
    inv = (
        d.groupby(["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy"], dropna=False)
        .agg(issue_count=("issue_id", "nunique"), panel_chars=("text_chars", "sum"), min_issue_date=("issue_date", "min"), max_issue_date=("issue_date", "max"))
        .reset_index()
        .sort_values(["panel_chars", "issue_count", "city_key"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return inv


def _choose_sample_cities(inv: pd.DataFrame, sample_count: int) -> list[str]:
    keys = inv["city_key"].astype(str).tolist()
    if sample_count <= 0 or sample_count >= len(keys):
        return keys
    # Deterministic spread across panel-size distribution.
    idx = sorted(set(int(round(i * (len(keys) - 1) / max(1, sample_count - 1))) for i in range(sample_count)))
    return [keys[i] for i in idx]


def _read_city_issues(run_dir: Path, city_key: str) -> list[dict[str, Any]]:
    p = run_dir / "panels" / city_key / "issue_texts.jsonl"
    rows = _iter_jsonl(p)
    rows = sorted(rows, key=lambda r: (_norm_str(r.get("issue_date")), _norm_str(r.get("issue_id"))))
    return rows


def _run_variant(
    *,
    run_dir: Path,
    out_dir: Path,
    city_rows: pd.DataFrame,
    variant: str,
    model: str,
    context_budget_chars: int,
    runner_path: Path,
    gateway_pythonpath: Path | None,
    timeout_s: float,
    concurrency: int,
    gov_env_path: Path,
    skip_existing: bool,
    dry_run: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    req_rows: list[tuple[str, str]] = []
    meta_rows: list[dict[str, Any]] = []
    issue_ids_map: dict[str, list[str]] = {}

    for r in city_rows.itertuples(index=False):
        city_key = str(r.city_key)
        issues = _read_city_issues(run_dir, city_key)
        if not issues:
            continue
        prompt, pmeta = _build_prompt(
            variant=variant,
            city_key=city_key,
            city_name=str(r.city_name),
            state_abbr=str(r.state_abbr),
            issues=issues,
            context_budget_chars=int(context_budget_chars),
        )
        cid = f"iter::{variant}::{city_key}"
        req_rows.append((cid, prompt))
        issue_ids_map[city_key] = [_norm_str(x.get("issue_id")) for x in issues if _norm_str(x.get("issue_id"))]
        meta = {
            "custom_id": cid,
            "city_key": city_key,
            "city_name": str(r.city_name),
            "state_abbr": str(r.state_abbr),
            "region": str(r.region),
            "urbanicity_proxy": str(r.urbanicity_proxy),
            "issue_count": int(len(issues)),
            **pmeta,
        }
        meta_rows.append(meta)

    if not req_rows:
        return pd.DataFrame(), []

    req_path = out_dir / "requests" / "openai_requests_shard000.jsonl"
    _write_openai_requests(req_path, req_rows, model=model)

    _run_gateway_requests(
        request_dir=req_path.parent,
        output_dir=out_dir / "results",
        runner_path=runner_path,
        model=model,
        gateway_pythonpath=gateway_pythonpath,
        timeout_s=timeout_s,
        concurrency=concurrency,
        gov_env_path=gov_env_path,
        skip_existing=skip_existing,
        dry_run=dry_run,
    )

    result_map = _read_result_jsonl(out_dir / "results")
    error_map = _read_error_jsonl(out_dir / "results")
    meta_by_cid = {m["custom_id"]: m for m in meta_rows}

    parsed_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for cid, _ in req_rows:
        m = meta_by_cid[cid]
        city_key = str(m["city_key"])
        status_code = 0
        output_text = ""
        if cid in result_map:
            robj = result_map[cid]
            body = (((robj.get("response") or {}).get("body")) or {})
            status_code = int(((robj.get("response") or {}).get("status_code")) or 0)
            output_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in error_map:
            eobj = error_map[cid]
            status_code = int(((eobj.get("response") or {}).get("status_code")) or 0)
            body = (((eobj.get("response") or {}).get("body")) or {})
            output_text = _norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))

        parsed = _parse_json_from_text(output_text)
        parse_valid = isinstance(parsed, dict)
        issue_ids = issue_ids_map.get(city_key, [])
        coerced: dict[str, Any] | None = None
        metrics = {
            "timeline_coverage": math.nan,
            "coverage_component": math.nan,
            "panel_mix_completeness": math.nan,
            "mean_stage_sum_error": math.nan,
            "stage_sum_score": math.nan,
            "evidence_valid_share": math.nan,
            "numeric_specificity": math.nan,
            "numeric_quote_validity": math.nan,
            "district_specificity": math.nan,
            "event_specificity": math.nan,
            "phase_diversity": math.nan,
            "phase_score": math.nan,
            "overall_quality": 0.0,
        }
        if parse_valid and isinstance(parsed, dict):
            coerced = _coerce_output(parsed, set(issue_ids))
            metrics = _score_output(coerced, issue_ids)

        row = {
            **m,
            "variant": variant,
            "status_code": int(status_code),
            "parse_valid": int(bool(parse_valid)),
            **metrics,
        }
        score_rows.append(row)
        parsed_rows.append(
            {
                "custom_id": cid,
                "city_key": city_key,
                "variant": variant,
                "status_code": int(status_code),
                "parse_valid": int(bool(parse_valid)),
                "metrics": metrics,
                "prompt_meta": m,
                "output_text": output_text,
                "parsed_json": coerced if isinstance(coerced, dict) else None,
            }
        )

    score_df = pd.DataFrame.from_records(score_rows).sort_values(["overall_quality", "city_key"], ascending=[False, True]).reset_index(drop=True)
    _write_csv(out_dir / "city_scores.csv", score_df)
    _write_jsonl(out_dir / "panel_parsed_outputs.jsonl", parsed_rows)
    return score_df, parsed_rows


def _flatten_final_outputs(final_dir: Path, parsed_rows: list[dict[str, Any]]) -> None:
    summary_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    wave_rows: list[dict[str, Any]] = []
    num_rows: list[dict[str, Any]] = []

    for rec in parsed_rows:
        city_key = _norm_str(rec.get("city_key"))
        pj = rec.get("parsed_json") if isinstance(rec.get("parsed_json"), dict) else {}
        m = rec.get("prompt_meta") if isinstance(rec.get("prompt_meta"), dict) else {}

        fe = pj.get("first_events") if isinstance(pj.get("first_events"), dict) else {}
        fzs = fe.get("first_zoning_signal") if isinstance(fe.get("first_zoning_signal"), dict) else {}
        ffo = fe.get("first_full_ordinance") if isinstance(fe.get("first_full_ordinance"), dict) else {}
        de = pj.get("district_evolution") if isinstance(pj.get("district_evolution"), dict) else {}
        pm = pj.get("panel_mix_counts") if isinstance(pj.get("panel_mix_counts"), dict) else {}
        metrics = rec.get("metrics") if isinstance(rec.get("metrics"), dict) else {}

        summary_rows.append(
            {
                "city_key": city_key,
                "city_name": _norm_str(m.get("city_name")),
                "state_abbr": _norm_str(m.get("state_abbr")),
                "region": _norm_str(m.get("region")),
                "urbanicity_proxy": _norm_str(m.get("urbanicity_proxy")),
                "issue_count": _safe_int(m.get("issue_count"), 0),
                "panel_chars_original": _safe_int(m.get("panel_chars_original"), 0),
                "panel_chars_used": _safe_int(m.get("panel_chars_used"), 0),
                "overflow_applied": int(bool(m.get("overflow_applied"))),
                "clipped_issue_count": _safe_int(m.get("clipped_issue_count"), 0),
                "first_signal_issue_id": _norm_str(fzs.get("issue_id")),
                "first_signal_date": _norm_str(fzs.get("issue_date")),
                "first_signal_type": _norm_str(fzs.get("signal_type")),
                "first_full_status": _norm_str(ffo.get("status")),
                "first_full_issue_id": _norm_str(ffo.get("issue_id")),
                "first_full_date": _norm_str(ffo.get("issue_date")),
                "full_ordinance_issues": _safe_int(pm.get("full_ordinance_issues"), 0),
                "amendment_issues": _safe_int(pm.get("amendment_issues"), 0),
                "mention_only_issues": _safe_int(pm.get("mention_only_issues"), 0),
                "non_zoning_or_uncertain_issues": _safe_int(pm.get("non_zoning_or_uncertain_issues"), 0),
                "district_count_early": _safe_int(de.get("district_count_early"), -1),
                "district_count_mid": _safe_int(de.get("district_count_mid"), -1),
                "district_count_late": _safe_int(de.get("district_count_late"), -1),
                "district_count_direction": _norm_str(de.get("district_count_direction")),
                "overall_quality": _safe_float(metrics.get("overall_quality")),
                "timeline_coverage": _safe_float(metrics.get("timeline_coverage")),
                "coverage_component": _safe_float(metrics.get("coverage_component")),
                "panel_mix_completeness": _safe_float(metrics.get("panel_mix_completeness")),
                "evidence_valid_share": _safe_float(metrics.get("evidence_valid_share")),
            }
        )

        for t in (pj.get("issue_timeline") or []):
            if not isinstance(t, dict):
                continue
            timeline_rows.append(
                {
                    "city_key": city_key,
                    "issue_id": _norm_str(t.get("issue_id")),
                    "issue_date": _norm_str(t.get("issue_date")),
                    "signal_type": _norm_str(t.get("signal_type")),
                    "document_role": _norm_str(t.get("document_role")),
                    "zoning_relevance_0_to_1": _safe_float(t.get("zoning_relevance_0_to_1")),
                    "evidence": _norm_str(t.get("evidence")),
                }
            )

        sc = pj.get("stage_category_shares") if isinstance(pj.get("stage_category_shares"), dict) else {}
        for stage in ("early", "mid", "late"):
            obj = sc.get(stage) if isinstance(sc.get(stage), dict) else {}
            for cat in _FIXED_CATEGORY_KEYS:
                stage_rows.append(
                    {
                        "city_key": city_key,
                        "stage": stage,
                        "category": cat,
                        "share": _safe_float(obj.get(cat)),
                    }
                )

        for ph in (pj.get("phase_segments") or []):
            if not isinstance(ph, dict):
                continue
            phase_rows.append(
                {
                    "city_key": city_key,
                    "phase_order": _safe_int(ph.get("phase_order"), 0),
                    "phase_archetype": _norm_str(ph.get("phase_archetype")),
                    "phase_label_open": _norm_str(ph.get("phase_label_open")),
                    "start_issue_id": _norm_str(ph.get("start_issue_id")),
                    "end_issue_id": _norm_str(ph.get("end_issue_id")),
                    "dominant_categories": "|".join([_norm_str(x) for x in (ph.get("dominant_categories") or []) if _norm_str(x)]),
                    "summary": _norm_str(ph.get("summary")),
                    "evidence_issue_ids": "|".join([_norm_str(x) for x in (ph.get("evidence_issue_ids") or []) if _norm_str(x)]),
                }
            )

        for w in (pj.get("amendment_waves") or []):
            if not isinstance(w, dict):
                continue
            wave_rows.append(
                {
                    "city_key": city_key,
                    "wave_order": _safe_int(w.get("wave_order"), 0),
                    "start_issue_id": _norm_str(w.get("start_issue_id")),
                    "end_issue_id": _norm_str(w.get("end_issue_id")),
                    "change_type": _norm_str(w.get("change_type")),
                    "affected_categories": "|".join([_norm_str(x) for x in (w.get("affected_categories") or []) if _norm_str(x)]),
                    "evidence_issue_ids": "|".join([_norm_str(x) for x in (w.get("evidence_issue_ids") or []) if _norm_str(x)]),
                }
            )

        nc = pj.get("numeric_constraints_single_family") if isinstance(pj.get("numeric_constraints_single_family"), dict) else {}
        for met in _NUMERIC_METRICS:
            obj = nc.get(met) if isinstance(nc.get(met), dict) else {}
            quotes = obj.get("evidence_quotes") if isinstance(obj.get("evidence_quotes"), list) else []
            quote_compact: list[str] = []
            for q in quotes:
                if not isinstance(q, dict):
                    continue
                qid = _norm_str(q.get("issue_id"))
                qtxt = _norm_str(q.get("quote_excerpt"))
                if qid and qtxt:
                    quote_compact.append(f"{qid}::{qtxt}")
            num_rows.append(
                {
                    "city_key": city_key,
                    "metric": met,
                    "early": _safe_float(obj.get("early")),
                    "mid": _safe_float(obj.get("mid")),
                    "late": _safe_float(obj.get("late")),
                    "direction": _norm_str(obj.get("direction")),
                    "evidence_issue_ids": "|".join([_norm_str(x) for x in (obj.get("evidence_issue_ids") or []) if _norm_str(x)]),
                    "evidence_quotes": " || ".join(quote_compact),
                }
            )

    _write_csv(final_dir / "city_summary.csv", pd.DataFrame.from_records(summary_rows))
    _write_csv(final_dir / "city_issue_timeline.csv", pd.DataFrame.from_records(timeline_rows))
    _write_csv(final_dir / "city_stage_category_shares.csv", pd.DataFrame.from_records(stage_rows))
    _write_csv(final_dir / "city_phase_segments.csv", pd.DataFrame.from_records(phase_rows))
    _write_csv(final_dir / "city_amendment_waves.csv", pd.DataFrame.from_records(wave_rows))
    _write_csv(final_dir / "city_numeric_constraints.csv", pd.DataFrame.from_records(num_rows))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run iterative full-panel LLM extraction experiments.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Run directory containing full city panels.",
    )
    ap.add_argument("--output-dir", default="", help="Output directory (default: <run-dir>/longitudinal_iter).")
    ap.add_argument(
        "--variants",
        default="compact_closed_v1,timeline_evidence_v1,hybrid_open_v1",
        help="Comma-separated prompt variants.",
    )
    ap.add_argument("--sample-city-count", type=int, default=6, help="City count for experiment stage.")
    ap.add_argument(
        "--sample-city-keys",
        default="",
        help="Optional comma-separated explicit city keys for experiment stage (overrides --sample-city-count).",
    )
    ap.add_argument("--model", default="gemini:gemini-2.5-flash")
    ap.add_argument("--model-context-tokens", type=int, default=1_000_000)
    ap.add_argument("--context-utilization", type=float, default=0.70)
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    ap.add_argument("--max-panel-chars", type=int, default=4_000_000)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument(
        "--gateway-runner",
        default=str(Path(__file__).resolve().parents[1] / "scripts" / "run_openai_requests_via_gateway.py"),
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default=str(Path(__file__).resolve().parents[1] / "agent-gateway" / "src"),
    )
    ap.add_argument("--gov-env-path", default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/.env")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-winner-on-all", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force-winner", default="", help="Optional explicit winner variant.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if _norm_str(args.output_dir) else (run_dir / "longitudinal_iter")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [x.strip() for x in _norm_str(args.variants).split(",") if x.strip()]
    if not variants:
        raise SystemExit("No variants provided.")

    inv = _load_city_inventory(run_dir)
    if inv.empty:
        raise SystemExit("No city inventory found.")
    _write_csv(out_dir / "city_panel_inventory.csv", inv)

    explicit_sample = [x.strip() for x in _norm_str(args.sample_city_keys).split(",") if x.strip()]
    if explicit_sample:
        have = set(inv["city_key"].astype(str).tolist())
        missing = [k for k in explicit_sample if k not in have]
        if missing:
            raise SystemExit(f"--sample-city-keys includes unknown city_key(s): {missing}")
        sample_keys = explicit_sample
    else:
        sample_keys = _choose_sample_cities(inv, int(args.sample_city_count))
    sample_df = inv[inv["city_key"].astype(str).isin(set(sample_keys))].copy()
    sample_df = sample_df.sort_values(["panel_chars", "issue_count", "city_key"], ascending=[False, False, True]).reset_index(drop=True)
    _write_csv(out_dir / "sample_city_inventory.csv", sample_df)

    budget_from_ctx = int(math.floor(float(args.model_context_tokens) * float(args.context_utilization) * float(args.chars_per_token)))
    context_budget_chars = int(min(int(args.max_panel_chars), max(200_000, budget_from_ctx)))

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "variants": variants,
        "sample_city_keys": sample_keys,
        "settings": {
            "model": str(args.model),
            "context_budget_chars": int(context_budget_chars),
            "model_context_tokens": int(args.model_context_tokens),
            "context_utilization": float(args.context_utilization),
            "chars_per_token": float(args.chars_per_token),
            "concurrency": int(args.concurrency),
            "timeout": float(args.timeout),
            "skip_existing": bool(args.skip_existing),
        },
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runner_path = Path(args.gateway_runner).expanduser().resolve()
    gateway_pythonpath = Path(args.gateway_pythonpath).expanduser().resolve() if _norm_str(args.gateway_pythonpath) else None
    gov_env_path = Path(args.gov_env_path).expanduser().resolve()

    variant_score_rows: list[dict[str, Any]] = []
    variant_city_scores: dict[str, pd.DataFrame] = {}
    variant_sample_rows: dict[str, list[dict[str, Any]]] = {}

    for variant in variants:
        vdir = out_dir / "experiments" / variant / "sample"
        score_df, parsed_rows = _run_variant(
            run_dir=run_dir,
            out_dir=vdir,
            city_rows=sample_df,
            variant=variant,
            model=str(args.model),
            context_budget_chars=int(context_budget_chars),
            runner_path=runner_path,
            gateway_pythonpath=gateway_pythonpath,
            timeout_s=float(args.timeout),
            concurrency=int(args.concurrency),
            gov_env_path=gov_env_path,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
        )
        if score_df.empty:
            continue
        variant_city_scores[variant] = score_df
        variant_sample_rows[variant] = parsed_rows
        variant_score_rows.append(
            {
                "variant": variant,
                "sample_city_count": int(len(score_df)),
                "parse_valid_share": float(score_df["parse_valid"].mean()),
                "mean_overall_quality": float(score_df["overall_quality"].mean()),
                "median_overall_quality": float(score_df["overall_quality"].median()),
                "mean_coverage_component": float(score_df["coverage_component"].mean()),
                "mean_timeline_coverage": float(score_df["timeline_coverage"].mean()),
                "mean_panel_mix_completeness": float(score_df["panel_mix_completeness"].mean()),
                "mean_stage_sum_error": float(score_df["mean_stage_sum_error"].mean()),
                "mean_evidence_valid_share": float(score_df["evidence_valid_share"].mean()),
                "mean_numeric_specificity": float(score_df["numeric_specificity"].mean()),
                "mean_numeric_quote_validity": float(score_df["numeric_quote_validity"].mean()),
                "mean_district_specificity": float(score_df["district_specificity"].mean()),
                "mean_event_specificity": float(score_df["event_specificity"].mean()),
                "mean_phase_diversity": float(score_df["phase_diversity"].mean()),
            }
        )

    if not variant_score_rows:
        raise SystemExit("No experiment variant completed successfully.")

    scorecard = pd.DataFrame.from_records(variant_score_rows).sort_values(
        ["mean_overall_quality", "parse_valid_share", "mean_coverage_component", "variant"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    _write_csv(out_dir / "variant_scorecard.csv", scorecard)

    forced = _norm_str(args.force_winner)
    if forced:
        if forced not in set(scorecard["variant"].astype(str)):
            raise SystemExit(f"--force-winner variant not found in scorecard: {forced}")
        winner = forced
    else:
        winner = str(scorecard.iloc[0]["variant"])

    winner_manifest = {
        "selected_winner_variant": winner,
        "selection_rule": "max(mean_overall_quality), tie-break parse_valid_share, coverage_component, variant",
        "scorecard_top_rows": scorecard.head(5).to_dict(orient="records"),
    }
    (out_dir / "winner_manifest.json").write_text(json.dumps(winner_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not bool(args.run_winner_on_all):
        print(f"Done experiments only. winner={winner} out_dir={out_dir}")
        return

    final_dir = out_dir / "final" / winner
    score_df_final, parsed_rows_final = _run_variant(
        run_dir=run_dir,
        out_dir=final_dir,
        city_rows=inv,
        variant=winner,
        model=str(args.model),
        context_budget_chars=int(context_budget_chars),
        runner_path=runner_path,
        gateway_pythonpath=gateway_pythonpath,
        timeout_s=float(args.timeout),
        concurrency=int(args.concurrency),
        gov_env_path=gov_env_path,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )

    if not score_df_final.empty:
        _flatten_final_outputs(final_dir, parsed_rows_final)
        summary = {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "run_dir": str(run_dir),
            "output_dir": str(out_dir),
            "winner_variant": winner,
            "final_city_count": int(len(score_df_final)),
            "final_parse_valid_share": float(score_df_final["parse_valid"].mean()),
            "final_mean_overall_quality": float(score_df_final["overall_quality"].mean()),
            "context_budget_chars": int(context_budget_chars),
        }
        (out_dir / "provenance.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "done_iterative_llm "
        f"winner={winner} "
        f"sample_cities={len(sample_df)} "
        f"final_cities={len(score_df_final)} "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
