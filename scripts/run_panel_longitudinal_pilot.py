#!/usr/bin/env python3
"""
Run panel-level longitudinal zoning extraction on top-quality city panels.

Outputs under <run-dir>/longitudinal by default:
  - pilot20_selection_quality.csv
  - pilot20_selection_manifest.json
  - requests/openai_requests_shard000.jsonl
  - results/openai_results_shard000.jsonl
  - panel_longitudinal_parsed_outputs.jsonl
  - city_longitudinal_summary.csv
  - city_first_zoning_events.csv
  - city_evolution_phases.csv
  - city_amendment_waves.csv
  - city_category_trajectory.csv
  - city_complexity_trajectory.csv
  - city_numeric_trajectory.csv
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

_ALLOWED_SIGNAL_TYPES = {"none", "mention", "amendment", "full_ordinance"}
_ALLOWED_FIRST_FULL_STATUS = {"found", "not_found", "uncertain"}
_ALLOWED_DIR = {"up", "down", "flat", "mixed", "uncertain"}
_ALLOWED_WAVE_TYPE = {"expansion", "tightening", "clarification", "reclassification", "mixed", "none", "uncertain"}
_ALLOWED_PHASE_ARCHETYPES = {
    "initial_establishment",
    "formalization_enforcement",
    "expansion_reclassification",
    "flexible_special_use",
    "infrastructure_environment",
    "maturation_implementation",
    "other_unclear",
}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run panel-level longitudinal pilot extraction.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60",
        help="Base run directory containing panels/, llm_panel_vs_issue/, and tables/.",
    )
    ap.add_argument("--output-dir", default="", help="Output directory (default: <run-dir>/longitudinal).")
    ap.add_argument("--panel-count", type=int, default=20)
    ap.add_argument("--selection-mode", default="quality_top", choices=["quality_top", "fixed_keys"])
    ap.add_argument("--quality-rule", default="contamination_parse", choices=["contamination_parse"])
    ap.add_argument(
        "--fixed-city-keys",
        default="",
        help="Comma-separated city keys used when --selection-mode=fixed_keys.",
    )
    ap.add_argument(
        "--fixed-city-keys-file",
        default="",
        help="Optional path with city keys used when --selection-mode=fixed_keys. "
        "Supports JSON (selected_city_keys field), CSV (city_key column), or plain text.",
    )
    ap.add_argument("--model", default="gemini:gemini-2.5-flash")
    ap.add_argument(
        "--max-panel-chars",
        type=int,
        default=0,
        help="Hard cap for panel text in a single prompt. 0 means no local cap. If positive and exceeded, the run fails (no clipping fallback).",
    )
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180.0)
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


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
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
            if not isinstance(c, dict):
                continue
            if c.get("type") == "output_text":
                txt = c.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
    return "\n".join(parts).strip()


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    def _balance_json_delimiters(raw: str) -> str:
        s = str(raw or "")
        stack: list[str] = []
        in_str = False
        escape = False
        for ch in s:
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
        if not stack:
            return s
        out = s
        for opener in reversed(stack):
            out += "}" if opener == "{" else "]"
        return out

    def _try_parse(raw: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    def _repair_common_json_issues(raw: str) -> str:
        fixed = re.sub(
            r'("share_0_to_1"\s*:\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*\n\s*\{',
            r"\1\n    },\n    {",
            raw,
        )
        fixed = re.sub(
            r'("(?:dominant_categories|affected_categories)"\s*:\s*\[[^\]]*\])\s*}\s*,\s*"(evidence_issue_ids)"',
            r'\1, "\2"',
            fixed,
            flags=re.S,
        )
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        return fixed

    s = str(text or "").strip()
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    obj = _try_parse(s)
    if obj is not None:
        return obj
    s_fixed = _repair_common_json_issues(s)
    obj = _try_parse(s_fixed)
    if obj is not None:
        return obj
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        mb = re.search(r"\{.*", s, flags=re.S)
        if not mb:
            return None
        balanced = _balance_json_delimiters(_repair_common_json_issues(mb.group(0)))
        return _try_parse(balanced)
    obj = _try_parse(m.group(0))
    if obj is not None:
        return obj
    repaired = _repair_common_json_issues(m.group(0))
    obj = _try_parse(repaired)
    if obj is not None:
        return obj
    balanced = _balance_json_delimiters(repaired)
    obj = _try_parse(balanced)
    if obj is not None:
        return obj
    mb = re.search(r"\{.*", s, flags=re.S)
    if mb:
        balanced = _balance_json_delimiters(_repair_common_json_issues(mb.group(0)))
        obj = _try_parse(balanced)
        if obj is not None:
            return obj
    return None


def _coerce_parsed_schema(
    *,
    parsed_obj: dict[str, Any],
    city_key: str,
    city_name: str,
    state_abbr: str,
) -> dict[str, Any]:
    out = dict(parsed_obj or {})

    out["city_key"] = _norm_str(out.get("city_key")) or city_key
    out["city_name"] = _norm_str(out.get("city_name")) or city_name
    out["state_abbr"] = (_norm_str(out.get("state_abbr")) or state_abbr).lower()

    fzs = out.get("first_zoning_signal")
    if not isinstance(fzs, dict):
        fzs = {}
    out["first_zoning_signal"] = {
        "issue_id": _norm_str(fzs.get("issue_id")),
        "issue_date": _norm_str(fzs.get("issue_date")),
        "signal_type": _normalize_enum(fzs.get("signal_type"), _ALLOWED_SIGNAL_TYPES, "none"),
        "confidence_0_to_1": _safe_float(fzs.get("confidence_0_to_1"), default=0.0),
        "evidence": _norm_str(fzs.get("evidence")),
    }

    ffo = out.get("first_full_ordinance")
    if not isinstance(ffo, dict):
        ffo = {}
    out["first_full_ordinance"] = {
        "issue_id": _norm_str(ffo.get("issue_id")),
        "issue_date": _norm_str(ffo.get("issue_date")),
        "confidence_0_to_1": _safe_float(ffo.get("confidence_0_to_1"), default=0.0),
        "evidence": _norm_str(ffo.get("evidence")),
        "status": _normalize_enum(ffo.get("status"), _ALLOWED_FIRST_FULL_STATUS, "uncertain"),
    }

    phases = out.get("evolution_phases")
    if not isinstance(phases, list):
        phases = []
    fixed_phases: list[dict[str, Any]] = []
    for i, ph in enumerate(phases, start=1):
        if not isinstance(ph, dict):
            continue
        dom = ph.get("dominant_categories")
        if not isinstance(dom, list):
            dom = [_norm_str(dom)] if _norm_str(dom) else []
        dom = [_norm_str(x) for x in dom if _norm_str(x)]
        ev_ids = _as_issue_id_list(ph.get("evidence_issue_ids"))
        fixed_phases.append(
            {
                "phase_order": int(_safe_float(ph.get("phase_order"), default=float(i))),
                "phase_label": _normalize_enum(ph.get("phase_label"), _ALLOWED_PHASE_ARCHETYPES, "other_unclear"),
                "start_issue_date": _norm_str(ph.get("start_issue_date")),
                "end_issue_date": _norm_str(ph.get("end_issue_date")),
                "dominant_categories": dom,
                "complexity_direction": _normalize_enum(ph.get("complexity_direction"), _ALLOWED_DIR, "uncertain"),
                "evidence_issue_ids": ev_ids,
            }
        )
    out["evolution_phases"] = fixed_phases

    waves = out.get("amendment_waves")
    if not isinstance(waves, list):
        waves = []
    fixed_waves: list[dict[str, Any]] = []
    for i, w in enumerate(waves, start=1):
        if not isinstance(w, dict):
            continue
        aff = w.get("affected_categories")
        if not isinstance(aff, list):
            aff = [_norm_str(aff)] if _norm_str(aff) else []
        aff = [_norm_str(x) for x in aff if _norm_str(x)]
        ev_ids = _as_issue_id_list(w.get("evidence_issue_ids"))
        fixed_waves.append(
            {
                "wave_order": int(_safe_float(w.get("wave_order"), default=float(i))),
                "start_issue_date": _norm_str(w.get("start_issue_date")),
                "end_issue_date": _norm_str(w.get("end_issue_date")),
                "change_type": _normalize_enum(w.get("change_type"), _ALLOWED_WAVE_TYPE, "uncertain"),
                "affected_categories": aff,
                "evidence_issue_ids": ev_ids,
            }
        )
    out["amendment_waves"] = fixed_waves

    ctraj = out.get("category_trajectory")
    if not isinstance(ctraj, dict):
        ctraj = {}
    fixed_ctraj: dict[str, Any] = {}
    for cat in _FIXED_CATEGORY_KEYS:
        cv = ctraj.get(cat)
        if not isinstance(cv, dict):
            cv = {}
        fixed_ctraj[cat] = {
            "early_share": _safe_float(cv.get("early_share"), default=0.0),
            "mid_share": _safe_float(cv.get("mid_share"), default=0.0),
            "late_share": _safe_float(cv.get("late_share"), default=0.0),
            "direction": _normalize_enum(cv.get("direction"), _ALLOWED_DIR, "uncertain"),
        }
    out["category_trajectory"] = fixed_ctraj

    cx = out.get("complexity_trajectory")
    if not isinstance(cx, dict):
        cx = {}
    out["complexity_trajectory"] = {
        "regulatory_complexity_trend": _normalize_enum(cx.get("regulatory_complexity_trend"), _ALLOWED_DIR, "uncertain"),
        "procedural_intensity_trend": _normalize_enum(cx.get("procedural_intensity_trend"), _ALLOWED_DIR, "uncertain"),
        "dimensional_specificity_trend": _normalize_enum(cx.get("dimensional_specificity_trend"), _ALLOWED_DIR, "uncertain"),
    }

    nx = out.get("numeric_trajectory")
    if not isinstance(nx, dict):
        nx = {}
    fixed_nx: dict[str, Any] = {}
    for metric in ("lot_size", "height", "setback", "parking"):
        mv = nx.get(metric)
        if not isinstance(mv, dict):
            mv = {}
        fixed_nx[metric] = {
            "direction": _normalize_enum(mv.get("direction"), _ALLOWED_DIR, "uncertain"),
            "evidence": _norm_str(mv.get("evidence")),
        }
    out["numeric_trajectory"] = fixed_nx

    out["narrative_summary"] = _norm_str(out.get("narrative_summary"))
    qn = out.get("quality_notes")
    if isinstance(qn, list):
        out["quality_notes"] = [_norm_str(x) for x in qn if _norm_str(x)]
    else:
        one = _norm_str(qn)
        out["quality_notes"] = [one] if one else []
    return out


def _write_openai_requests(path: Path, rows: list[tuple[str, str]], *, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cid, prompt in rows:
            body = {
                "model": str(model),
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                "stream": False,
            }
            req = {"custom_id": cid, "method": "POST", "url": "/v1/responses", "body": body}
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
        if not env.get(key) and str(parsed.get(key) or "").strip():
            env[key] = str(parsed[key]).strip()

    if not env.get("OPENAI_KEY"):
        alias = str(parsed.get("CODEX_API_KEY") or parsed.get("OPENAI_API_KEY") or "").strip()
        if alias:
            env["OPENAI_KEY"] = alias

    provider = str(model).split(":", 1)[0].lower() if ":" in str(model) else "openai"
    need_key = {
        "openai": "OPENAI_KEY",
        "gemini": "GEMINI_KEY",
        "claude": "CLAUDE_KEY",
    }.get(provider)

    provider_key_candidates: list[str] = []
    if need_key:
        seen: set[str] = set()
        for cand in (
            str(env.get(need_key) or "").strip(),
            str(parsed.get(need_key) or "").strip(),
            *[str(parsed.get(f"{need_key}_BACKUP_{i}") or "").strip() for i in range(1, 6)],
        ):
            if not cand or cand in seen:
                continue
            seen.add(cand)
            provider_key_candidates.append(cand)
        if not provider_key_candidates:
            raise SystemExit(f"{need_key} is not set and was not found in --gov-env-path for provider '{provider}'.")

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
    for i, key_val in enumerate(provider_key_candidates, start=1):
        env_try = dict(env)
        env_try[need_key] = key_val
        try:
            _eprint(f"Gateway request run with {need_key} candidate {i}/{len(provider_key_candidates)}.")
            subprocess.run(cmd, check=True, env=env_try)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            continue
    raise SystemExit(f"Gateway request run failed for all {need_key} candidates. last_error={last_err}")


def _read_result_jsonl(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(results_dir.glob("openai_results_shard*.jsonl")):
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
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
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get("custom_id")
            if isinstance(cid, str) and cid:
                out[cid] = obj
    return out


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


def _normalize_enum(v: Any, allowed: set[str], default: str) -> str:
    s = _norm_str(v).lower()
    return s if s in allowed else default


def _as_issue_id_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        s = _norm_str(x)
        if s:
            out.append(s)
    return out


def _panel_prompt(
    *,
    city_key: str,
    city_name: str,
    state_abbr: str,
    issues: list[dict[str, Any]],
    max_panel_chars: int,
) -> tuple[str, dict[str, Any]]:
    issues_sorted = sorted(issues, key=lambda r: (_norm_str(r.get("issue_date")), _norm_str(r.get("issue_id"))))
    n = len(issues_sorted)
    panel_chars_original = sum(len(_norm_str(r.get("text"))) for r in issues_sorted)
    if int(max_panel_chars) > 0 and panel_chars_original > int(max_panel_chars):
        raise ValueError(
            f"city_key={city_key} panel chars {panel_chars_original} exceeds hard cap {int(max_panel_chars)}. "
            "Clipping fallback is disabled; increase --max-panel-chars or reduce scope."
        )
    overflow = False
    per_issue_budget = 0

    cat_template = {
        k: {"early_share": 0.0, "mid_share": 0.0, "late_share": 0.0, "direction": "flat"}
        for k in _FIXED_CATEGORY_KEYS
    }
    schema = {
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
        "category_trajectory": cat_template,
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

    lines: list[str] = []
    lines.append("You are analyzing chronological zoning-related newspaper transcripts for one city.")
    lines.append("Return ONLY a single JSON object. No markdown, no code fences, no extra text.")
    lines.append("")
    lines.append("Primary tasks:")
    lines.append("1) Identify when zoning first appears and when full ordinance appears.")
    lines.append("2) Summarize longitudinal evolution phases.")
    lines.append("3) Summarize amendment waves and what changed.")
    lines.append("4) Provide category, complexity, and numeric trajectories.")
    lines.append("")
    lines.append("Hard constraints:")
    lines.append("- Use these fixed categories exactly: " + ", ".join(_FIXED_CATEGORY_KEYS))
    lines.append("- category_trajectory must include all fixed categories.")
    lines.append("- evidence_issue_ids must be real issue_ids from provided issues.")
    lines.append("- signal_type allowed: none|mention|amendment|full_ordinance")
    lines.append("- first_full_ordinance.status allowed: found|not_found|uncertain")
    lines.append("- evolution_phases.phase_label allowed: " + "|".join(sorted(_ALLOWED_PHASE_ARCHETYPES)))
    lines.append("- trend directions allowed: up|down|flat|mixed|uncertain")
    lines.append("- amendment wave change_type allowed: expansion|tightening|clarification|reclassification|mixed|none|uncertain")
    lines.append("- narrative_summary must be 4-8 sentences.")
    lines.append("")
    lines.append("JSON schema template:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(f"City: {city_name}, {state_abbr.upper()} ({city_key})")
    lines.append(f"Issue count: {n}")
    lines.append(f"Panel chars original: {panel_chars_original}")
    lines.append(f"Overflow applied: {str(overflow).lower()}")
    lines.append("")
    lines.append("Chronological issues:")

    clipped_n = 0
    panel_chars_used = 0
    for i, r in enumerate(issues_sorted, start=1):
        iid = _norm_str(r.get("issue_id"))
        dtv = _norm_str(r.get("issue_date"))
        lbl = _norm_str(r.get("classification_label"))
        txt = _norm_str(r.get("text"))
        txt_chars = int(r.get("text_chars") or len(txt) or 0)
        used = txt
        clipped = False
        panel_chars_used += len(used)
        lines.append("")
        lines.append(
            f"[ISSUE {i}] issue_id={iid} date={dtv} label={lbl} text_chars={txt_chars} used_chars={len(used)} clipped={str(clipped).lower()}"
        )
        lines.append(used)

    lines.append("")
    lines.append("Return JSON now.")

    meta = {
        "panel_chars_original": int(panel_chars_original),
        "panel_chars_used": int(panel_chars_used),
        "overflow_applied": bool(overflow),
        "per_issue_budget": int(per_issue_budget),
        "clipped_issue_count": int(clipped_n),
        "issue_count": int(n),
    }
    return "\n".join(lines).strip(), meta


def _load_issue_parse_success(parsed_issue_path: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    if not parsed_issue_path.is_file():
        return out
    for row in _iter_jsonl(parsed_issue_path):
        cid = _norm_str(row.get("custom_id"))
        parts = cid.split("::", 2)
        if len(parts) < 3 or parts[0] != "issue":
            continue
        city_key, issue_id = parts[1], parts[2]
        parsed = row.get("parsed_json")
        if isinstance(parsed, dict) and parsed:
            out.add((city_key, issue_id))
    return out


def _parse_fixed_city_keys(
    *,
    fixed_city_keys: str,
    fixed_city_keys_file: str,
) -> list[str]:
    direct = [x.strip() for x in str(fixed_city_keys).split(",") if x.strip()]
    if direct:
        return direct
    p = Path(str(fixed_city_keys_file).strip()).expanduser().resolve() if str(fixed_city_keys_file).strip() else None
    if not p:
        return []
    if not p.is_file():
        raise SystemExit(f"--fixed-city-keys-file not found: {p}")

    if p.suffix.lower() == ".json":
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Could not parse JSON fixed-city-keys file: {p} ({exc})")
        if isinstance(obj, dict) and isinstance(obj.get("selected_city_keys"), list):
            return [str(x).strip() for x in obj.get("selected_city_keys") if str(x).strip()]
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        raise SystemExit(f"Unsupported JSON structure in {p}; expected selected_city_keys list or top-level list.")

    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        d = pd.read_csv(p, sep=sep)
        if "city_key" not in d.columns:
            raise SystemExit(f"Fixed-city keys file missing city_key column: {p}")
        return [str(x).strip() for x in d["city_key"].tolist() if str(x).strip()]

    out: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.extend([x.strip() for x in s.split(",") if x.strip()])
    return out


def _build_quality_selection(run_dir: Path, panel_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues_path = run_dir / "panels" / "selected_panel_issues.csv"
    contam_path = run_dir / "tables" / "contamination_panel_summary.csv"
    parsed_issue_path = run_dir / "llm_panel_vs_issue" / "parsed_issue_outputs.jsonl"

    if not issues_path.is_file():
        raise SystemExit(f"Missing required file: {issues_path}")
    if not contam_path.is_file():
        raise SystemExit(f"Missing required file: {contam_path}")
    if not parsed_issue_path.is_file():
        raise SystemExit(f"Missing required file: {parsed_issue_path}")

    issues = pd.read_csv(issues_path)
    need = {"city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "issue_id", "text_chars"}
    miss = sorted(c for c in need if c not in issues.columns)
    if miss:
        raise SystemExit(f"selected_panel_issues missing columns: {miss}")

    parse_ok = _load_issue_parse_success(parsed_issue_path)
    parse_df = pd.DataFrame([{"city_key": c, "issue_id": i, "parsed_ok": 1} for c, i in parse_ok])
    if parse_df.empty:
        parse_df = pd.DataFrame(columns=["city_key", "issue_id", "parsed_ok"])

    agg_issue = (
        issues.groupby("city_key", dropna=False)
        .agg(
            city_name=("city_name", "first"),
            state_abbr=("state_abbr", "first"),
            region=("region", "first"),
            urbanicity_proxy=("urbanicity_proxy", "first"),
            selected_issue_count=("issue_id", "nunique"),
            median_issue_text_chars=("text_chars", "median"),
        )
        .reset_index()
    )

    issue_join = issues[["city_key", "issue_id"]].drop_duplicates().merge(parse_df, on=["city_key", "issue_id"], how="left")
    issue_join["parsed_ok"] = issue_join["parsed_ok"].fillna(0).astype(int)
    parsed_ct = issue_join.groupby("city_key", dropna=False)["parsed_ok"].sum().reset_index().rename(columns={"parsed_ok": "parsed_issue_count"})

    contam = pd.read_csv(contam_path)
    need_cont = {"city_key", "post_filter_noise_share", "lexical_noise_share"}
    miss_cont = sorted(c for c in need_cont if c not in contam.columns)
    if miss_cont:
        raise SystemExit(f"contamination_panel_summary missing columns: {miss_cont}")
    contam_city = (
        contam.groupby("city_key", dropna=False)
        .agg(
            post_filter_noise_share=("post_filter_noise_share", "mean"),
            lexical_noise_share=("lexical_noise_share", "mean"),
        )
        .reset_index()
    )

    q = agg_issue.merge(parsed_ct, on="city_key", how="left").merge(contam_city, on="city_key", how="left")
    q["parsed_issue_count"] = q["parsed_issue_count"].fillna(0).astype(int)
    q["parse_success_rate"] = q["parsed_issue_count"] / q["selected_issue_count"].replace(0, math.nan)
    q["is_eligible"] = (q["selected_issue_count"] >= 8).astype(int)

    q = q.sort_values(
        [
            "is_eligible",
            "parse_success_rate",
            "post_filter_noise_share",
            "lexical_noise_share",
            "selected_issue_count",
            "median_issue_text_chars",
            "city_key",
        ],
        ascending=[False, False, True, True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    q["rank_quality"] = range(1, len(q) + 1)
    q["selected_for_pilot"] = 0

    eligible = q[q["is_eligible"] == 1].copy()
    selected = eligible.head(int(panel_count)).copy()
    q.loc[q["city_key"].isin(selected["city_key"]), "selected_for_pilot"] = 1
    return q, selected


def _build_fixed_selection(run_dir: Path, fixed_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues_path = run_dir / "panels" / "selected_panel_issues.csv"
    if not issues_path.is_file():
        raise SystemExit(f"Missing required file: {issues_path}")
    issues = pd.read_csv(issues_path)
    need = {"city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "issue_id", "text_chars"}
    miss = sorted(c for c in need if c not in issues.columns)
    if miss:
        raise SystemExit(f"selected_panel_issues missing columns: {miss}")

    agg_issue = (
        issues.groupby("city_key", dropna=False)
        .agg(
            city_name=("city_name", "first"),
            state_abbr=("state_abbr", "first"),
            region=("region", "first"),
            urbanicity_proxy=("urbanicity_proxy", "first"),
            selected_issue_count=("issue_id", "nunique"),
            median_issue_text_chars=("text_chars", "median"),
        )
        .reset_index()
    )
    agg_issue["parsed_issue_count"] = math.nan
    agg_issue["parse_success_rate"] = math.nan
    agg_issue["post_filter_noise_share"] = math.nan
    agg_issue["lexical_noise_share"] = math.nan
    agg_issue["rank_quality"] = math.nan
    agg_issue["selected_for_pilot"] = 0

    wanted = [str(x).strip() for x in fixed_keys if str(x).strip()]
    if not wanted:
        raise SystemExit("selection_mode=fixed_keys requires --fixed-city-keys or --fixed-city-keys-file.")

    have = set(agg_issue["city_key"].astype(str).tolist())
    missing = [k for k in wanted if k not in have]
    if missing:
        raise SystemExit(f"Fixed city keys not found in selected_panel_issues.csv: {missing}")

    ord_map = {k: i for i, k in enumerate(wanted)}
    selected = agg_issue[agg_issue["city_key"].astype(str).isin(set(wanted))].copy()
    selected["__ord"] = selected["city_key"].map(lambda x: ord_map.get(str(x), 10_000))
    selected = selected.sort_values(["__ord", "city_key"]).drop(columns=["__ord"]).reset_index(drop=True)
    agg_issue.loc[agg_issue["city_key"].astype(str).isin(set(wanted)), "selected_for_pilot"] = 1
    return agg_issue, selected


def _validate_parsed(
    *,
    parsed_obj: dict[str, Any],
    issue_ids: set[str],
) -> dict[str, Any]:
    required_keys = [
        "city_key",
        "city_name",
        "state_abbr",
        "first_zoning_signal",
        "first_full_ordinance",
        "evolution_phases",
        "amendment_waves",
        "category_trajectory",
        "complexity_trajectory",
        "numeric_trajectory",
        "narrative_summary",
        "quality_notes",
    ]
    missing = [k for k in required_keys if k not in parsed_obj]

    cat_obj = parsed_obj.get("category_trajectory") if isinstance(parsed_obj.get("category_trajectory"), dict) else {}
    cat_missing = [k for k in _FIXED_CATEGORY_KEYS if k not in cat_obj]

    fzs = parsed_obj.get("first_zoning_signal") if isinstance(parsed_obj.get("first_zoning_signal"), dict) else {}
    ffo = parsed_obj.get("first_full_ordinance") if isinstance(parsed_obj.get("first_full_ordinance"), dict) else {}

    signal_type = _normalize_enum(fzs.get("signal_type"), _ALLOWED_SIGNAL_TYPES, "none")
    first_full_status = _normalize_enum(ffo.get("status"), _ALLOWED_FIRST_FULL_STATUS, "uncertain")

    fzs_issue = _norm_str(fzs.get("issue_id"))
    ffo_issue = _norm_str(ffo.get("issue_id"))

    first_signal_issue_valid = True
    if signal_type != "none" and fzs_issue:
        first_signal_issue_valid = fzs_issue in issue_ids
    if signal_type != "none" and not fzs_issue:
        first_signal_issue_valid = False

    first_full_issue_valid = True
    if first_full_status == "found":
        first_full_issue_valid = bool(ffo_issue and (ffo_issue in issue_ids))

    ev_ok = True
    phases = parsed_obj.get("evolution_phases") if isinstance(parsed_obj.get("evolution_phases"), list) else []
    phase_label_valid = True
    for ph in phases:
        if isinstance(ph, dict):
            pl_raw = _norm_str(ph.get("phase_label")).lower()
            if pl_raw not in _ALLOWED_PHASE_ARCHETYPES:
                phase_label_valid = False
        ev_ids = _as_issue_id_list(ph.get("evidence_issue_ids") if isinstance(ph, dict) else [])
        if any(x not in issue_ids for x in ev_ids):
            ev_ok = False
            break

    wave_ok = True
    waves = parsed_obj.get("amendment_waves") if isinstance(parsed_obj.get("amendment_waves"), list) else []
    for w in waves:
        ev_ids = _as_issue_id_list(w.get("evidence_issue_ids") if isinstance(w, dict) else [])
        if any(x not in issue_ids for x in ev_ids):
            wave_ok = False
            break

    schema_valid = (len(missing) == 0) and (len(cat_missing) == 0)

    return {
        "missing_top_level_keys": missing,
        "missing_category_keys": cat_missing,
        "signal_type_norm": signal_type,
        "first_full_status_norm": first_full_status,
        "first_signal_issue_valid": bool(first_signal_issue_valid),
        "first_full_issue_valid": bool(first_full_issue_valid),
        "phase_label_valid": bool(phase_label_valid),
        "phase_evidence_valid": bool(ev_ok),
        "wave_evidence_valid": bool(wave_ok),
        "schema_valid": bool(schema_valid),
    }


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "longitudinal")
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_keys = _parse_fixed_city_keys(
        fixed_city_keys=str(args.fixed_city_keys),
        fixed_city_keys_file=str(args.fixed_city_keys_file),
    )

    if str(args.selection_mode) == "fixed_keys":
        q, selected = _build_fixed_selection(run_dir, fixed_keys=fixed_keys)
    else:
        q, selected = _build_quality_selection(run_dir, int(args.panel_count))
    _write_csv(q, out_dir / "pilot20_selection_quality.csv")

    if selected.empty:
        raise SystemExit("No eligible panels found for longitudinal pilot selection.")
    if str(args.selection_mode) != "fixed_keys" and len(selected) < int(args.panel_count):
        raise SystemExit(
            f"Eligible panel count is {len(selected)}, below requested --panel-count={int(args.panel_count)}. "
            "Selection requires at least 8 issues per city."
        )

    req_rows: list[tuple[str, str]] = []
    city_issue_map: dict[str, list[dict[str, Any]]] = {}
    city_prompt_meta: dict[str, dict[str, Any]] = {}

    for r in selected.itertuples(index=False):
        city_key = str(r.city_key)
        issue_jsonl = run_dir / "panels" / city_key / "issue_texts.jsonl"
        if not issue_jsonl.is_file():
            _eprint(f"[warn] missing panel issue_texts for {city_key}: {issue_jsonl}")
            continue
        issues = _iter_jsonl(issue_jsonl)
        if not issues:
            _eprint(f"[warn] empty panel issues for {city_key}")
            continue

        prompt, pmeta = _panel_prompt(
            city_key=city_key,
            city_name=str(r.city_name),
            state_abbr=str(r.state_abbr),
            issues=issues,
            max_panel_chars=int(args.max_panel_chars),
        )
        cid = f"panel_longitudinal::{city_key}"
        req_rows.append((cid, prompt))
        city_issue_map[city_key] = issues
        city_prompt_meta[city_key] = pmeta

    if not req_rows:
        raise SystemExit("No requests prepared for longitudinal pilot.")

    req_path = out_dir / "requests" / "openai_requests_shard000.jsonl"
    _write_openai_requests(req_path, req_rows, model=str(args.model))

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "settings": {
            "panel_count": int(args.panel_count),
            "selection_mode": str(args.selection_mode),
            "quality_rule": str(args.quality_rule),
            "fixed_city_keys_count": int(len(fixed_keys)),
            "model": str(args.model),
            "max_panel_chars": int(args.max_panel_chars),
            "concurrency": int(args.concurrency),
            "timeout": float(args.timeout),
        },
        "selected_city_keys": selected["city_key"].astype(str).tolist(),
        "selected_city_quality_rows": selected[
            [
                c
                for c in [
                    "city_key",
                    "city_name",
                    "state_abbr",
                    "region",
                    "urbanicity_proxy",
                    "selected_issue_count",
                    "parsed_issue_count",
                    "parse_success_rate",
                    "post_filter_noise_share",
                    "lexical_noise_share",
                    "rank_quality",
                ]
                if c in selected.columns
            ]
        ].to_dict(orient="records"),
        "selection_rule": (
            {
                "mode": "fixed_keys",
                "source": (
                    str(Path(str(args.fixed_city_keys_file)).expanduser().resolve())
                    if str(args.fixed_city_keys_file).strip()
                    else "cli:--fixed-city-keys"
                ),
                "keys": fixed_keys,
            }
            if str(args.selection_mode) == "fixed_keys"
            else {
                "eligibility": "selected_issue_count >= 8",
                "ranking_order": [
                    "parse_success_rate desc",
                    "post_filter_noise_share asc",
                    "lexical_noise_share asc",
                    "selected_issue_count desc",
                    "median_issue_text_chars desc",
                    "city_key asc",
                ],
            }
        ),
        "n_requests": int(len(req_rows)),
    }
    (out_dir / "pilot20_selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runner_path = Path(args.gateway_runner).expanduser().resolve()
    gateway_pythonpath = Path(args.gateway_pythonpath).expanduser().resolve() if str(args.gateway_pythonpath).strip() else None
    gov_env_path = Path(args.gov_env_path).expanduser().resolve()

    _run_gateway_requests(
        request_dir=req_path.parent,
        output_dir=out_dir / "results",
        runner_path=runner_path,
        model=str(args.model),
        gateway_pythonpath=gateway_pythonpath,
        timeout_s=float(args.timeout),
        concurrency=int(args.concurrency),
        gov_env_path=gov_env_path,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )

    result_map = _read_result_jsonl(out_dir / "results")
    error_map = _read_error_jsonl(out_dir / "results")

    parsed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    wave_rows: list[dict[str, Any]] = []
    cat_rows: list[dict[str, Any]] = []
    cpx_rows: list[dict[str, Any]] = []
    num_rows: list[dict[str, Any]] = []

    selected_meta = {str(r.city_key): r for r in selected.itertuples(index=False)}

    for cid, _prompt in req_rows:
        city_key = cid.split("::", 1)[1]
        m = selected_meta.get(city_key)
        if m is None:
            continue

        issue_ids = {
            _norm_str(x.get("issue_id"))
            for x in city_issue_map.get(city_key, [])
            if _norm_str(x.get("issue_id"))
        }

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

        parsed_obj = _parse_json_from_text(output_text)
        parse_valid = isinstance(parsed_obj, dict) and bool(parsed_obj)
        if parse_valid and isinstance(parsed_obj, dict):
            parsed_obj = _coerce_parsed_schema(
                parsed_obj=parsed_obj,
                city_key=city_key,
                city_name=str(m.city_name),
                state_abbr=str(m.state_abbr),
            )
        val = {
            "missing_top_level_keys": [],
            "missing_category_keys": list(_FIXED_CATEGORY_KEYS),
            "signal_type_norm": "none",
            "first_full_status_norm": "uncertain",
            "first_signal_issue_valid": False,
            "first_full_issue_valid": False,
            "phase_label_valid": False,
            "phase_evidence_valid": False,
            "wave_evidence_valid": False,
            "schema_valid": False,
        }
        if parse_valid and isinstance(parsed_obj, dict):
            val = _validate_parsed(parsed_obj=parsed_obj, issue_ids=issue_ids)

        parsed_rows.append(
            {
                "custom_id": cid,
                "city_key": city_key,
                "status_code": int(status_code),
                "output_text": output_text,
                "parse_valid": int(bool(parse_valid)),
                "schema_valid": int(bool(val["schema_valid"])),
                "missing_top_level_keys": val["missing_top_level_keys"],
                "missing_category_keys": val["missing_category_keys"],
                "parsed_json": parsed_obj if isinstance(parsed_obj, dict) else None,
            }
        )

        obj = parsed_obj if isinstance(parsed_obj, dict) else {}
        pmeta = city_prompt_meta.get(city_key, {})

        fzs = obj.get("first_zoning_signal") if isinstance(obj.get("first_zoning_signal"), dict) else {}
        ffo = obj.get("first_full_ordinance") if isinstance(obj.get("first_full_ordinance"), dict) else {}

        fzs_issue = _norm_str(fzs.get("issue_id"))
        fzs_date = _norm_str(fzs.get("issue_date"))
        fzs_signal_type = val["signal_type_norm"]
        fzs_conf = _safe_float(fzs.get("confidence_0_to_1"), default=math.nan)
        fzs_ev = _norm_str(fzs.get("evidence"))

        ffo_issue = _norm_str(ffo.get("issue_id"))
        ffo_date = _norm_str(ffo.get("issue_date"))
        ffo_status = val["first_full_status_norm"]
        ffo_conf = _safe_float(ffo.get("confidence_0_to_1"), default=math.nan)
        ffo_ev = _norm_str(ffo.get("evidence"))

        phases = obj.get("evolution_phases") if isinstance(obj.get("evolution_phases"), list) else []
        waves = obj.get("amendment_waves") if isinstance(obj.get("amendment_waves"), list) else []

        ctraj = obj.get("category_trajectory") if isinstance(obj.get("category_trajectory"), dict) else {}
        cx = obj.get("complexity_trajectory") if isinstance(obj.get("complexity_trajectory"), dict) else {}
        nx = obj.get("numeric_trajectory") if isinstance(obj.get("numeric_trajectory"), dict) else {}

        narrative = _norm_str(obj.get("narrative_summary"))
        notes_raw = obj.get("quality_notes") if isinstance(obj.get("quality_notes"), list) else []
        notes = [
            _norm_str(x)
            for x in notes_raw
            if _norm_str(x)
        ]

        first_rows.append(
            {
                "city_key": city_key,
                "city_name": str(m.city_name),
                "state_abbr": str(m.state_abbr),
                "region": str(m.region),
                "urbanicity_proxy": str(m.urbanicity_proxy),
                "first_zoning_issue_id": fzs_issue,
                "first_zoning_issue_date": fzs_date,
                "first_zoning_signal_type": fzs_signal_type,
                "first_zoning_confidence_0_to_1": fzs_conf,
                "first_zoning_evidence": fzs_ev,
                "first_full_status": ffo_status,
                "first_full_issue_id": ffo_issue,
                "first_full_issue_date": ffo_date,
                "first_full_confidence_0_to_1": ffo_conf,
                "first_full_evidence": ffo_ev,
            }
        )

        for ph in phases:
            if not isinstance(ph, dict):
                continue
            phase_rows.append(
                {
                    "city_key": city_key,
                    "city_name": str(m.city_name),
                    "state_abbr": str(m.state_abbr),
                    "region": str(m.region),
                    "urbanicity_proxy": str(m.urbanicity_proxy),
                    "phase_order": int(_safe_float(ph.get("phase_order"), default=0)),
                    "phase_label": _norm_str(ph.get("phase_label")),
                    "start_issue_date": _norm_str(ph.get("start_issue_date")),
                    "end_issue_date": _norm_str(ph.get("end_issue_date")),
                    "dominant_categories": "|".join(_as_issue_id_list(ph.get("dominant_categories"))),
                    "complexity_direction": _normalize_enum(ph.get("complexity_direction"), _ALLOWED_DIR, "uncertain"),
                    "evidence_issue_ids": "|".join(_as_issue_id_list(ph.get("evidence_issue_ids"))),
                }
            )

        for w in waves:
            if not isinstance(w, dict):
                continue
            wave_rows.append(
                {
                    "city_key": city_key,
                    "city_name": str(m.city_name),
                    "state_abbr": str(m.state_abbr),
                    "region": str(m.region),
                    "urbanicity_proxy": str(m.urbanicity_proxy),
                    "wave_order": int(_safe_float(w.get("wave_order"), default=0)),
                    "start_issue_date": _norm_str(w.get("start_issue_date")),
                    "end_issue_date": _norm_str(w.get("end_issue_date")),
                    "change_type": _normalize_enum(w.get("change_type"), _ALLOWED_WAVE_TYPE, "uncertain"),
                    "affected_categories": "|".join(_as_issue_id_list(w.get("affected_categories"))),
                    "evidence_issue_ids": "|".join(_as_issue_id_list(w.get("evidence_issue_ids"))),
                }
            )

        for cat in _FIXED_CATEGORY_KEYS:
            cv = ctraj.get(cat) if isinstance(ctraj.get(cat), dict) else {}
            cat_rows.append(
                {
                    "city_key": city_key,
                    "city_name": str(m.city_name),
                    "state_abbr": str(m.state_abbr),
                    "region": str(m.region),
                    "urbanicity_proxy": str(m.urbanicity_proxy),
                    "category": cat,
                    "early_share": _safe_float(cv.get("early_share"), default=math.nan),
                    "mid_share": _safe_float(cv.get("mid_share"), default=math.nan),
                    "late_share": _safe_float(cv.get("late_share"), default=math.nan),
                    "direction": _normalize_enum(cv.get("direction"), _ALLOWED_DIR, "uncertain"),
                }
            )

        cpx_rows.append(
            {
                "city_key": city_key,
                "city_name": str(m.city_name),
                "state_abbr": str(m.state_abbr),
                "region": str(m.region),
                "urbanicity_proxy": str(m.urbanicity_proxy),
                "regulatory_complexity_trend": _normalize_enum(cx.get("regulatory_complexity_trend"), _ALLOWED_DIR, "uncertain"),
                "procedural_intensity_trend": _normalize_enum(cx.get("procedural_intensity_trend"), _ALLOWED_DIR, "uncertain"),
                "dimensional_specificity_trend": _normalize_enum(cx.get("dimensional_specificity_trend"), _ALLOWED_DIR, "uncertain"),
            }
        )

        for metric in ("lot_size", "height", "setback", "parking"):
            mv = nx.get(metric) if isinstance(nx.get(metric), dict) else {}
            num_rows.append(
                {
                    "city_key": city_key,
                    "city_name": str(m.city_name),
                    "state_abbr": str(m.state_abbr),
                    "region": str(m.region),
                    "urbanicity_proxy": str(m.urbanicity_proxy),
                    "metric": metric,
                    "direction": _normalize_enum(mv.get("direction"), _ALLOWED_DIR, "uncertain"),
                    "evidence": _norm_str(mv.get("evidence")),
                }
            )

        summary_rows.append(
            {
                "city_key": city_key,
                "city_name": str(m.city_name),
                "state_abbr": str(m.state_abbr),
                "region": str(m.region),
                "urbanicity_proxy": str(m.urbanicity_proxy),
                "selected_issue_count": int(m.selected_issue_count),
                "parse_success_rate_selection": _safe_float(m.parse_success_rate, default=math.nan),
                "post_filter_noise_share": _safe_float(m.post_filter_noise_share, default=math.nan),
                "lexical_noise_share": _safe_float(m.lexical_noise_share, default=math.nan),
                "panel_chars_original": int(pmeta.get("panel_chars_original") or 0),
                "panel_chars_used": int(pmeta.get("panel_chars_used") or 0),
                "overflow_applied": int(bool(pmeta.get("overflow_applied"))),
                "clipped_issue_count": int(pmeta.get("clipped_issue_count") or 0),
                "parse_valid": int(bool(parse_valid)),
                "schema_valid": int(bool(val["schema_valid"])),
                "first_signal_issue_valid": int(bool(val["first_signal_issue_valid"])),
                "first_full_issue_valid": int(bool(val["first_full_issue_valid"])),
                "phase_label_valid": int(bool(val["phase_label_valid"])),
                "phase_evidence_valid": int(bool(val["phase_evidence_valid"])),
                "wave_evidence_valid": int(bool(val["wave_evidence_valid"])),
                "n_evolution_phases": int(len(phases)),
                "n_amendment_waves": int(len(waves)),
                "first_zoning_signal_type": fzs_signal_type,
                "first_full_status": ffo_status,
                "regulatory_complexity_trend": _normalize_enum(cx.get("regulatory_complexity_trend"), _ALLOWED_DIR, "uncertain"),
                "procedural_intensity_trend": _normalize_enum(cx.get("procedural_intensity_trend"), _ALLOWED_DIR, "uncertain"),
                "dimensional_specificity_trend": _normalize_enum(cx.get("dimensional_specificity_trend"), _ALLOWED_DIR, "uncertain"),
                "narrative_summary": narrative,
                "quality_notes": " | ".join(notes),
                "missing_top_level_keys": "|".join(val["missing_top_level_keys"]),
                "missing_category_keys": "|".join(val["missing_category_keys"]),
            }
        )

    parsed_out = out_dir / "panel_longitudinal_parsed_outputs.jsonl"
    with parsed_out.open("w", encoding="utf-8") as f:
        for r in parsed_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_df = pd.DataFrame.from_records(summary_rows).sort_values("city_key").reset_index(drop=True)
    first_df = pd.DataFrame.from_records(first_rows).sort_values("city_key").reset_index(drop=True)
    phase_df = pd.DataFrame.from_records(phase_rows).sort_values(["city_key", "phase_order"]).reset_index(drop=True) if phase_rows else pd.DataFrame(columns=[
        "city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "phase_order", "phase_label", "start_issue_date", "end_issue_date", "dominant_categories", "complexity_direction", "evidence_issue_ids"
    ])
    wave_df = pd.DataFrame.from_records(wave_rows).sort_values(["city_key", "wave_order"]).reset_index(drop=True) if wave_rows else pd.DataFrame(columns=[
        "city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "wave_order", "start_issue_date", "end_issue_date", "change_type", "affected_categories", "evidence_issue_ids"
    ])
    cat_df = pd.DataFrame.from_records(cat_rows).sort_values(["city_key", "category"]).reset_index(drop=True)
    cpx_df = pd.DataFrame.from_records(cpx_rows).sort_values(["city_key"]).reset_index(drop=True)
    num_df = pd.DataFrame.from_records(num_rows).sort_values(["city_key", "metric"]).reset_index(drop=True)

    _write_csv(summary_df, out_dir / "city_longitudinal_summary.csv")
    _write_csv(first_df, out_dir / "city_first_zoning_events.csv")
    _write_csv(phase_df, out_dir / "city_evolution_phases.csv")
    _write_csv(wave_df, out_dir / "city_amendment_waves.csv")
    _write_csv(cat_df, out_dir / "city_category_trajectory.csv")
    _write_csv(cpx_df, out_dir / "city_complexity_trajectory.csv")
    _write_csv(num_df, out_dir / "city_numeric_trajectory.csv")

    n_req = len(req_rows)
    n_parse_valid = int(summary_df["parse_valid"].sum()) if not summary_df.empty else 0
    n_schema_valid = int(summary_df["schema_valid"].sum()) if not summary_df.empty else 0
    category_all_keys_share = (
        float((summary_df["missing_category_keys"].astype(str).str.strip() == "").mean()) if not summary_df.empty else math.nan
    )

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "settings": {
            "panel_count": int(args.panel_count),
            "selection_mode": str(args.selection_mode),
            "quality_rule": str(args.quality_rule),
            "model": str(args.model),
            "max_panel_chars": int(args.max_panel_chars),
            "concurrency": int(args.concurrency),
            "timeout": float(args.timeout),
            "skip_existing": bool(args.skip_existing),
            "dry_run": bool(args.dry_run),
        },
        "counts": {
            "requests_prepared": int(n_req),
            "selected_city_count": int(len(selected)),
            "parse_valid_n": int(n_parse_valid),
            "schema_valid_n": int(n_schema_valid),
            "parse_valid_share": float(n_parse_valid / max(1, n_req)),
            "schema_valid_share": float(n_schema_valid / max(1, n_req)),
            "category_all_keys_share": category_all_keys_share,
            "summary_rows": int(len(summary_df)),
            "first_rows": int(len(first_df)),
            "phase_rows": int(len(phase_df)),
            "wave_rows": int(len(wave_df)),
            "category_rows": int(len(cat_df)),
            "complexity_rows": int(len(cpx_df)),
            "numeric_rows": int(len(num_df)),
        },
        "outputs": {
            "pilot20_selection_quality": str(out_dir / "pilot20_selection_quality.csv"),
            "pilot20_selection_manifest": str(out_dir / "pilot20_selection_manifest.json"),
            "requests_jsonl": str(req_path),
            "results_jsonl": str(out_dir / "results" / "openai_results_shard000.jsonl"),
            "panel_longitudinal_parsed_outputs": str(parsed_out),
            "city_longitudinal_summary": str(out_dir / "city_longitudinal_summary.csv"),
            "city_first_zoning_events": str(out_dir / "city_first_zoning_events.csv"),
            "city_evolution_phases": str(out_dir / "city_evolution_phases.csv"),
            "city_amendment_waves": str(out_dir / "city_amendment_waves.csv"),
            "city_category_trajectory": str(out_dir / "city_category_trajectory.csv"),
            "city_complexity_trajectory": str(out_dir / "city_complexity_trajectory.csv"),
            "city_numeric_trajectory": str(out_dir / "city_numeric_trajectory.csv"),
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"selected={len(selected)} "
        f"requests={n_req} "
        f"parse_valid={n_parse_valid}/{n_req} "
        f"schema_valid={n_schema_valid}/{n_req} "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
