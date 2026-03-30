#!/usr/bin/env python3
"""
Extract zoning-only language from full newspaper panel transcripts.

This script is designed to clean panel issue_texts by removing unrelated
newspaper content while preserving zoning ordinance/amendment language.

It supports multiple methods:
  - rules_strict: high precision, lower recall
  - rules_balanced: balanced precision/recall
  - rules_recall: high recall candidate capture
  - llm_hybrid: rules_recall candidates + LLM block-level filtering/cleanup
  - llm_only: LLM-only block selection over ALL blocks (no deterministic candidate filtering)

Core outputs (under --output-dir / <method>/):
  - issue_zoning_extract.jsonl
  - issue_metrics.csv
  - city_metrics.csv
  - panel_zoning_texts/<city_key>.txt

If --experiment is set, writes:
  - experiment/method_comparison.csv
  - experiment/recommended_method.json
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


_METHODS = ("rules_strict", "rules_balanced", "rules_recall", "llm_hybrid", "llm_only")
_ZONING_LABELS = {"full_ordinance", "amendment_substantial", "amendment_targeted"}


# Split signals into (a) hard zoning cues and (b) softer content cues.
# This prevents false positives like "parking spaces" in general news.
_HARD_ZONING_PATTERNS = [
    r"\bzoning\b",
    r"\bzoning\s+ordinance\b",
    r"\bzoning\s+district\b",
    r"\bzone\s+district\b",
    r"\bzoning\s+map\b",
    r"\brezone(?:d|s|ing)?\b",
    r"\brezoning\b",
    r"\breclassification\b",
    r"\breclassif(?:y|ication)\b",
    r"\bmunicipal\s+code\b",
    r"\bcity\s+code\b",
    r"\bland\s+use\b",
    # District codes often used directly in ordinance text.
    r"\bdistrict\s+[A-Z]-\d+\b",
    r"\bR-\d\b|\bC-\d\b|\bM-\d\b",
    r"\bplanned\s+unit\s+development\b|\bPUD\b",
    r"\bboard\s+of\s+zoning\s+appeals\b",
    r"\bzoning\s+appeals?\b",
    r"\bzoning\s+inspector\b",
]

_SOFT_ZONING_PATTERNS = [
    r"\bplanning\s+commission\b",
    r"\bboard\s+of\s+appeals\b",
    r"\bvariance(?:s)?\b",
    r"\bconditional\s+use\b",
    r"\bspecial\s+permit\b",
    r"\bsite\s+plan\b",
    r"\bnonconforming\s+use\b",
    r"\bsetback(?:s)?\b",
    r"\bminimum\s+lot\b|\bmin(?:imum)?\s+lot\s+size\b",
    r"\bmax(?:imum)?\s+height\b",
    # Require more specific parking language than the bare word "parking".
    r"\boff[- ]street\s+parking\b",
    r"\bparking\s+and\s+loading\b",
    r"\bparking\s+requirements?\b",
    r"\bparking\s+spaces?\s+(?:shall|must|required|to\s+be\s+provided|to\s+be\s+provided\s+and\s+maintained)\b",
]

_LEGAL_ZONING_PATTERNS = [
    # Ordinance numbering (full + abbreviated forms).
    r"\bordinance\s+no\.?\s*\d+",
    r"\bord\.?\s*no\.?\s*[0-9][0-9A-Za-z\\-]*",
    r"\ban\s+ordinance\b",
    r"\bthis\s+ordinance\s+shall\b",
    r"\bbe\s+it\s+ordained\b",
    r"\bsection\s+\d+",
    r"\bchapter\s+\d+",
    r"\btitle\s+[ivx0-9]+",
    r"\bcity\s+council\b",
    r"\bpublic\s+notice\b",
    r"\bzoning\s+map\b",
    r"\breclassification\s+from\b",
]

_NOISE_PATTERNS = [
    r"\bsports?\b|\bfootball\b|\bbasketball\b|\bbaseball\b",
    r"\bweather\b|\bforecast\b|\btemperature\b",
    r"\bobituar(?:y|ies)\b|\bfuneral\b|\bbirths?\b",
    r"\bmovie\b|\btheater\b|\btelevision\b|\bcomics?\b",
    r"\bstock\s+exchange\b|\bmarket\b",
    r"\bpolice\b|\bassault\b|\bmurder\b|\brobbery\b",
    # Do NOT match bare "ap" as a word: OCR hyphenation can produce tokens like "ap-\nproached".
    r"\bupi\b|\bassociated\s+press\b|\(\s*ap\s*\)|\bA\.P\.",
    r"\bhospital\s+notes\b",
    r"\bclassified\b|\bwants\s+ads?\b",
]

_EXCLUDE_LEGAL_NONZONING_PATTERNS = [
    r"\bprobate\b",
    r"\bestate\s+of\b",
    r"\bdivorce\b",
    r"\bfictitious\s+business\s+name\b",
    r"\btrustee'?s\s+sale\b",
    r"\bnotice\s+to\s+creditors\b",
    r"\bsubpoena\b",
]

_RE_HARD = [re.compile(p, flags=re.I) for p in _HARD_ZONING_PATTERNS]
_RE_SOFT = [re.compile(p, flags=re.I) for p in _SOFT_ZONING_PATTERNS]
_RE_LEGAL = [re.compile(p, flags=re.I) for p in _LEGAL_ZONING_PATTERNS]
_RE_NOISE = [re.compile(p, flags=re.I) for p in _NOISE_PATTERNS]
_RE_EXCL = [re.compile(p, flags=re.I) for p in _EXCLUDE_LEGAL_NONZONING_PATTERNS]
_RE_WORD = re.compile(r"[A-Za-z0-9]+")


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract zoning-only text from full newspaper panels.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Run directory containing panels/<city_key>/issue_texts.jsonl.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output root (default: <run-dir>/zoning_text_extract).",
    )
    ap.add_argument(
        "--city-keys",
        default="",
        help="Comma-separated city keys. Default: all cities found in panels/.",
    )
    ap.add_argument(
        "--city-keys-file",
        default="",
        help="Optional file with one city key per line.",
    )
    ap.add_argument(
        "--methods",
        default="llm_only",
        help="Comma-separated methods to run. Allowed: rules_strict,rules_balanced,rules_recall,llm_hybrid,llm_only",
    )
    ap.add_argument("--experiment", action="store_true", help="Run all requested methods and produce comparison output.")
    ap.add_argument(
        "--max-issues-per-city",
        type=int,
        default=0,
        help="Optional dev cap. 0 means all issues.",
    )
    ap.add_argument(
        "--min-zoning-issue-kept-chars",
        type=int,
        default=200,
        help="Coverage threshold for zoning-labeled issues in summary scoring.",
    )
    ap.add_argument(
        "--model",
        default="openai:gpt-5-mini",
        help="Model for llm_hybrid and llm_only methods.",
    )
    ap.add_argument("--llm-max-candidate-blocks", type=int, default=24)
    ap.add_argument(
        "--llm-max-block-chars",
        type=int,
        default=2200,
        help="Hard cap for per-block text included in llm_hybrid candidate prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument(
        "--llm-max-issue-candidate-chars",
        type=int,
        default=42000,
        help="Hard cap for total candidate-block text included in llm_hybrid prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument(
        "--llm-only-max-block-chars",
        type=int,
        default=0,
        help="Hard cap for per-block text included in llm_only prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument(
        "--llm-only-max-issue-chars",
        type=int,
        default=0,
        help="Hard cap for total issue text included in llm_only prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
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
    ap.add_argument(
        "--gov-env-path",
        default=".env",
        help="Env file path for provider keys (can point to /Users/.../projects/gov-gpt/.env).",
    )
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _norm_str(x: Any) -> str:
    return str(x or "").strip()


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _load_city_keys(run_dir: Path, city_keys_raw: str, city_keys_file: str) -> list[str]:
    direct = [x.strip() for x in str(city_keys_raw).split(",") if x.strip()]
    from_file: list[str] = []
    if str(city_keys_file).strip():
        p = Path(city_keys_file).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--city-keys-file not found: {p}")
        from_file = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip() and not x.strip().startswith("#")]
    keys = direct or from_file
    if keys:
        return sorted(set(keys))
    panel_dir = run_dir / "panels"
    if not panel_dir.is_dir():
        raise SystemExit(f"Missing panels directory: {panel_dir}")
    out: list[str] = []
    for d in sorted(panel_dir.iterdir()):
        if d.is_dir() and (d / "issue_texts.jsonl").is_file():
            out.append(d.name)
    if not out:
        raise SystemExit(f"No city panels found in {panel_dir}")
    return out


def _split_blocks(text: str) -> list[str]:
    t = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", t)
    out: list[str] = []
    for p in parts:
        s = p.strip()
        if s:
            out.append(s)
    if not out and t.strip():
        out = [t.strip()]
    return out


def _count_hits(patterns: list[re.Pattern[str]], text: str) -> int:
    c = 0
    for rx in patterns:
        c += len(rx.findall(text))
    return c


def _block_features(block_text: str) -> dict[str, Any]:
    t = str(block_text or "")
    t_l = t.lower()
    hard_hits = _count_hits(_RE_HARD, t_l)
    soft_hits = _count_hits(_RE_SOFT, t_l)
    legal_hits = _count_hits(_RE_LEGAL, t_l)
    noise_hits = _count_hits(_RE_NOISE, t_l)
    excl_hits = _count_hits(_RE_EXCL, t_l)
    words = len(_RE_WORD.findall(t))
    has_section_like = bool(re.search(r"\b(section|sec\.|chapter|article)\b", t_l))
    has_ordinance_like = bool(re.search(r"\bordinance\b|\bbe it ordained\b|\bcity council\b", t_l))

    # Weighted score tuned for zoning/legal snippets. Hard cues count more than soft cues.
    score = (
        (3.6 * hard_hits)
        + (1.7 * soft_hits)
        + (1.8 * legal_hits)
        + (0.8 if has_section_like else 0.0)
        + (1.2 if has_ordinance_like else 0.0)
        - (1.5 * noise_hits)
        - (1.1 * excl_hits)
    )
    if words < 20 and hard_hits == 0 and soft_hits == 0 and legal_hits == 0:
        score -= 1.0
    return {
        "hard_hits": int(hard_hits),
        "soft_hits": int(soft_hits),
        "legal_hits": int(legal_hits),
        "noise_hits": int(noise_hits),
        "exclude_hits": int(excl_hits),
        "word_count": int(words),
        "score": float(score),
        "has_section_like": bool(has_section_like),
        "has_ordinance_like": bool(has_ordinance_like),
    }


def _method_keep_decision(method: str, feat: dict[str, Any]) -> bool:
    hard = int(feat["hard_hits"])
    soft = int(feat["soft_hits"])
    legal = int(feat["legal_hits"])
    noise = int(feat["noise_hits"])
    excl = int(feat["exclude_hits"])
    score = float(feat["score"])

    if method == "rules_strict":
        # Strict: require explicit zoning cue OR strong legal structure + multiple soft cues.
        if hard >= 1:
            return (score >= 4.2) and (noise <= 2) and (excl <= 1)
        return (legal >= 2 and soft >= 2 and score >= 3.0 and (noise <= 2) and (excl <= 1))
    if method == "rules_balanced":
        # Balanced: allow legal structure + soft cues, but do not admit "soft-only" blocks.
        if hard >= 1:
            return (score >= 2.2) and (excl <= 2)
        return (legal >= 2 and soft >= 1 and score >= 1.8 and (excl <= 2))
    if method == "rules_recall":
        # Recall: keep anything with hard cue, or legal+soft combos, then neighbor-expand later.
        if hard >= 1:
            return score >= 0.0
        return (legal >= 1 and soft >= 1 and score >= -0.5) or (legal >= 1 and soft >= 3)
    raise ValueError(f"Unknown rule method: {method}")


def _expand_neighbors(blocks: list[str], base_keep: list[bool], block_features: list[dict[str, Any]], window: int = 1) -> list[bool]:
    keep = list(base_keep)
    n = len(blocks)
    for i, k in enumerate(base_keep):
        if not k:
            continue
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if keep[j]:
                continue
            fj = block_features[j]
            # Neighbor inclusion is conservative: only legal-ish and not strongly noisy.
            if (int(fj["legal_hits"]) >= 1 or bool(fj["has_section_like"])) and int(fj["noise_hits"]) <= 2:
                keep[j] = True
    return keep


def _extract_with_rules(issue: dict[str, Any], method: str) -> dict[str, Any]:
    blocks = _split_blocks(_norm_str(issue.get("text")))
    feats = [_block_features(b) for b in blocks]
    keep = [_method_keep_decision(method, f) for f in feats]
    if method == "rules_recall":
        keep = _expand_neighbors(blocks, keep, feats, window=1)

    kept_block_ids: list[int] = [i for i, k in enumerate(keep) if k]
    kept_blocks: list[dict[str, Any]] = []
    for i in kept_block_ids:
        kept_blocks.append(
            {
                "block_id": int(i),
                "text": blocks[i],
                "score": float(feats[i]["score"]),
                "hard_hits": int(feats[i]["hard_hits"]),
                "soft_hits": int(feats[i]["soft_hits"]),
                "legal_hits": int(feats[i]["legal_hits"]),
                "noise_hits": int(feats[i]["noise_hits"]),
                "exclude_hits": int(feats[i]["exclude_hits"]),
            }
        )
    zoning_text = "\n\n".join([x["text"] for x in kept_blocks]).strip()
    return {
        "blocks": blocks,
        "block_features": feats,
        "kept_block_ids": kept_block_ids,
        "kept_blocks": kept_blocks,
        "zoning_text": zoning_text,
    }


def _word_count(text: str) -> int:
    return len(_RE_WORD.findall(str(text or "")))


def _zoning_keyword_hits(text: str) -> int:
    t = str(text or "").lower()
    return _count_hits(_RE_HARD, t) + _count_hits(_RE_SOFT, t) + _count_hits(_RE_LEGAL, t)


def _noise_keyword_hits(text: str) -> int:
    return _count_hits(_RE_NOISE, str(text or "").lower()) + _count_hits(_RE_EXCL, str(text or "").lower())


def _density_per_1k(hits: int, words: int) -> float:
    if words <= 0:
        return math.nan
    return 1000.0 * (float(hits) / float(words))


def _metric_row_from_issue(issue: dict[str, Any], out_text: str, method: str, used_llm: bool, llm_parse_valid: bool, fallback_used: bool) -> dict[str, Any]:
    in_text = _norm_str(issue.get("text"))
    in_chars = len(in_text)
    kept_chars = len(out_text)
    in_words = _word_count(in_text)
    kept_words = _word_count(out_text)

    in_z_hits = _zoning_keyword_hits(in_text)
    kept_z_hits = _zoning_keyword_hits(out_text)
    in_n_hits = _noise_keyword_hits(in_text)
    kept_n_hits = _noise_keyword_hits(out_text)

    label = _norm_str(issue.get("classification_label")).lower()
    is_z_label = int(label in _ZONING_LABELS)
    return {
        "method": method,
        "city_key": _norm_str(issue.get("city_key")),
        "city_name": _norm_str(issue.get("city_name")),
        "state_abbr": _norm_str(issue.get("state_abbr")),
        "issue_id": _norm_str(issue.get("issue_id")),
        "issue_date": _norm_str(issue.get("issue_date")),
        "classification_label": label,
        "is_zoning_label": is_z_label,
        "input_chars": in_chars,
        "kept_chars": kept_chars,
        "keep_ratio": (float(kept_chars) / float(in_chars)) if in_chars > 0 else math.nan,
        "input_words": in_words,
        "kept_words": kept_words,
        "input_zoning_hits": in_z_hits,
        "kept_zoning_hits": kept_z_hits,
        "input_noise_hits": in_n_hits,
        "kept_noise_hits": kept_n_hits,
        "input_zoning_density_per_1k_words": _density_per_1k(in_z_hits, in_words),
        "kept_zoning_density_per_1k_words": _density_per_1k(kept_z_hits, kept_words),
        "input_noise_density_per_1k_words": _density_per_1k(in_n_hits, in_words),
        "kept_noise_density_per_1k_words": _density_per_1k(kept_n_hits, kept_words),
        "used_llm": int(bool(used_llm)),
        "llm_parse_valid": int(bool(llm_parse_valid)),
        "fallback_used": int(bool(fallback_used)),
    }


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
    need_key = {"openai": "OPENAI_KEY", "gemini": "GEMINI_KEY", "claude": "CLAUDE_KEY"}.get(provider)
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
            _eprint(f"Gateway run with {need_key} candidate {i}/{len(provider_key_candidates)}.")
            subprocess.run(cmd, check=True, env=env_try)
            return
        except subprocess.CalledProcessError as exc:
            last_err = exc
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
    s = str(text or "").strip()
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    raw = m.group(0)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _bounded_text_or_fail(text: str, max_chars: int, *, label: str) -> str:
    t = str(text or "")
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    raise SystemExit(
        f"{label} chars={len(t)} exceeds hard cap {int(max_chars)}; no clipping allowed. "
        "Increase the cap or set it to 0 for no local cap."
    )


def _build_llm_prompt(issue: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    schema = {
        "issue_id": _norm_str(issue.get("issue_id")),
        "kept_block_ids": [0],
        "blocks": [
            {
                "block_id": 0,
                "keep": True,
                "clean_text": "only zoning-relevant lines from this block",
            }
        ],
        "issue_notes": "brief caveat or empty string",
    }
    lines: list[str] = []
    lines.append("You are filtering OCR newspaper text to keep only zoning-regulation language.")
    lines.append("Return ONLY JSON. No markdown.")
    lines.append("")
    lines.append("Keep text only if it is specifically zoning/legal code content, such as:")
    lines.append("- ordinance or amendment text for zoning code")
    lines.append("- zoning district/use/bulk/procedure provisions")
    lines.append("- legal notice that changes zoning map/classification")
    lines.append("")
    lines.append("Important edge case: the bare phrase \"parking spaces\" is NOT zoning by itself.")
    lines.append("Only keep parking language when it is clearly a zoning regulation (off-street parking requirements, per dwelling unit, district rules, ordinance text).")
    lines.append("")
    lines.append("Exclude unrelated content, including:")
    lines.append("- general news, sports, weather, obituaries, classifieds")
    lines.append("- unrelated legal notices (probate, trustee sale, fictitious business, etc.)")
    lines.append("")
    lines.append("For each block, set keep=true only if zoning-relevant.")
    lines.append("If keep=true, clean_text should contain only zoning language from that block.")
    lines.append("If keep=false, clean_text must be empty.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Issue metadata: issue_id={_norm_str(issue.get('issue_id'))} date={_norm_str(issue.get('issue_date'))} "
        f"label={_norm_str(issue.get('classification_label'))} city={_norm_str(issue.get('city_name'))},{_norm_str(issue.get('state_abbr')).upper()}"
    )
    lines.append("")
    lines.append("Candidate blocks:")
    for c in candidates:
        lines.append("")
        lines.append(
            f"[BLOCK {int(c['block_id'])}] score={float(c['score']):.2f} "
            f"hard_hits={int(c.get('hard_hits') or 0)} soft_hits={int(c.get('soft_hits') or 0)} "
            f"legal_hits={int(c['legal_hits'])} noise_hits={int(c['noise_hits'])}"
        )
        lines.append(str(c["text"]))
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _build_llm_only_prompt(issue: dict[str, Any], blocks: list[str], *, max_block_chars: int) -> str:
    schema = {
        "issue_id": _norm_str(issue.get("issue_id")),
        "pure_block_ids": [0],
        "mixed_block_ids": [1],
        "issue_notes": "brief caveat or empty string",
    }
    lines: list[str] = []
    lines.append("You are extracting zoning-regulation language from an OCR newspaper issue transcript.")
    lines.append("Return ONLY JSON. No markdown.")
    lines.append("")
    lines.append("Goal: identify verbatim zoning ordinance / amendment / rezoning legal-notice language.")
    lines.append("")
    lines.append("Classify blocks into:")
    lines.append("- pure: the block is entirely (or almost entirely) verbatim ordinance/amendment/legal-notice text. Keep whole block.")
    lines.append("- mixed: the block contains some verbatim ordinance/notice lines PLUS some non-ordinance narrative (lead-in/reporting). Keep, but it will be trimmed later.")
    lines.append("- drop: no verbatim ordinance/amendment/legal-notice language.")
    lines.append("")
    lines.append("Verbatim legal/regulatory text includes:")
    lines.append("- ordinance or amendment text (sections, provisions, 'shall', 'be it ordained', etc.)")
    lines.append("- legal notice of rezoning / reclassification / zoning map changes")
    lines.append("- procedural zoning provisions (variances, appeals, permits) when written as ordinance text")
    lines.append("")
    lines.append("Exclude blocks that are NOT regulatory text, including:")
    lines.append("- general news articles, editorials, sports, weather, obituaries, classifieds")
    lines.append("- legal notices unrelated to zoning (probate, trustee sale, fictitious business name, etc.)")
    lines.append("- commentary about zoning that does NOT quote ordinance/notice text")
    lines.append("")
    lines.append("Important: we want ONLY verbatim ordinance/amendment/legal-notice language in the final corpus.")
    lines.append("This first step is only to locate candidate blocks; trimming to ordinance-only lines happens later.")
    lines.append("")
    lines.append("Parking is zoning-relevant when it is part of requirements/standards (off-street parking, loading, stalls per unit, district rules).")
    lines.append("")
    lines.append("Output:")
    lines.append("- pure_block_ids: list of integer block IDs that are pure verbatim ordinance/notice text")
    lines.append("- mixed_block_ids: list of integer block IDs that contain some verbatim ordinance/notice text but also narrative")
    lines.append("- issue_notes: short caveat if uncertain, else empty string")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Issue metadata: issue_id={_norm_str(issue.get('issue_id'))} date={_norm_str(issue.get('issue_date'))} "
        f"label={_norm_str(issue.get('classification_label'))} city={_norm_str(issue.get('city_name'))},{_norm_str(issue.get('state_abbr')).upper()}"
    )
    lines.append("")
    lines.append("Blocks (split by blank lines):")
    issue_id = _norm_str(issue.get("issue_id"))
    for i, b in enumerate(blocks):
        lines.append("")
        lines.append(f"[BLOCK {int(i)}]")
        lines.append(_bounded_text_or_fail(b, int(max_block_chars), label=f"issue_id={issue_id} block_id={int(i)} block_text"))
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _line_number_block(text: str) -> tuple[list[str], str]:
    """Return (raw_lines, numbered_block_text). Line numbers are 1-based, fixed-width."""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    width = max(3, len(str(len(lines))))
    out: list[str] = []
    for i, line in enumerate(lines, start=1):
        out.append(f"{i:0{width}d}| {line}")
    return lines, "\n".join(out)


def _build_llm_only_trim_prompt(issue: dict[str, Any], blocks: list[dict[str, Any]]) -> str:
    """
    Second-stage verbatim trimming.

    Input: candidate blocks (block_id + original text). Each block is line-numbered in the prompt.
    Output: for each block, line ranges to keep (so reconstruction is verbatim).
    """
    schema = {
        "issue_id": _norm_str(issue.get("issue_id")),
        "blocks": [{"block_id": 0, "keep_line_ranges": [[1, 1]]}],
        "issue_notes": "brief caveat or empty string",
    }
    lines: list[str] = []
    lines.append("You are extracting ONLY verbatim zoning ordinance / amendment / rezoning legal-notice text.")
    lines.append("Return ONLY JSON. No markdown.")
    lines.append("")
    lines.append("You will see candidate blocks with line numbers.")
    lines.append("For each block, choose the line ranges that are verbatim ordinance/amendment/legal-notice language.")
    lines.append("Do NOT include narrative lead-in or reporting about zoning unless it is itself a legal notice/ordinance line.")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Prefer dropping ambiguous lines rather than keeping extra non-ordinance prose.")
    lines.append("- Keep headings like 'ORDINANCE NO. 123', 'AN ORDINANCE...', 'BE IT ORDAINED', 'SECTION 1' if part of the legal text.")
    lines.append("- Keep 'NOTICE OF PUBLIC HEARING' / 'NOTICE OF REZONING' style legal notices when they are formal notice language.")
    lines.append("- Keep zoning standards and use lists (e.g., 'In an A district no building or land shall be used...').")
    lines.append("")
    lines.append("Output format:")
    lines.append("- blocks: list of {block_id, keep_line_ranges}")
    lines.append("- keep_line_ranges: list of [start_line, end_line] (1-based, inclusive)")
    lines.append("- If a block contains NO ordinance/notice lines, return keep_line_ranges as an empty list for that block.")
    lines.append("")
    lines.append("JSON schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Issue metadata: issue_id={_norm_str(issue.get('issue_id'))} date={_norm_str(issue.get('issue_date'))} "
        f"label={_norm_str(issue.get('classification_label'))} city={_norm_str(issue.get('city_name'))},{_norm_str(issue.get('state_abbr')).upper()}"
    )
    lines.append("")
    lines.append("Candidate blocks (line-numbered):")
    for b in blocks:
        bid = int(b["block_id"])
        raw_text = str(b.get("text") or "")
        _raw_lines, numbered = _line_number_block(raw_text)
        lines.append("")
        lines.append(f"[BLOCK {bid}]")
        lines.append(numbered)
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _coerce_llm_only_trim_parse(parsed_obj: dict[str, Any], expected_issue_id: str, candidate_ids: set[int], block_line_counts: dict[int, int]) -> dict[str, Any]:
    out = dict(parsed_obj or {})
    out_issue = _norm_str(out.get("issue_id")) or expected_issue_id
    notes = _norm_str(out.get("issue_notes"))
    blocks_raw = out.get("blocks")
    blocks_out: list[dict[str, Any]] = []
    if isinstance(blocks_raw, list):
        for b in blocks_raw:
            if not isinstance(b, dict):
                continue
            try:
                bid = int(b.get("block_id"))
            except Exception:
                continue
            if bid not in candidate_ids:
                continue
            ranges_raw = b.get("keep_line_ranges")
            keep_ranges: list[list[int]] = []
            if isinstance(ranges_raw, list):
                for rr in ranges_raw:
                    if not isinstance(rr, list) or len(rr) != 2:
                        continue
                    try:
                        s = int(rr[0])
                        e = int(rr[1])
                    except Exception:
                        continue
                    if s <= 0 or e <= 0:
                        continue
                    if s > e:
                        s, e = e, s
                    max_line = int(block_line_counts.get(bid, 0) or 0)
                    if max_line and (s > max_line or e > max_line):
                        continue
                    keep_ranges.append([s, e])
            blocks_out.append({"block_id": bid, "keep_line_ranges": keep_ranges})
    return {"issue_id": out_issue, "blocks": blocks_out, "issue_notes": notes}


def _coerce_llm_only_issue_parse(parsed_obj: dict[str, Any], expected_issue_id: str, n_blocks: int) -> dict[str, Any]:
    out = dict(parsed_obj or {})
    out_issue = _norm_str(out.get("issue_id")) or expected_issue_id

    def _coerce_id_list(val: Any) -> list[int]:
        ids: list[int] = []
        if isinstance(val, list):
            for x in val:
                try:
                    v = int(x)
                except Exception:
                    continue
                if v < 0 or v >= int(n_blocks):
                    continue
                if v not in ids:
                    ids.append(v)
        return ids

    pure_ids = _coerce_id_list(out.get("pure_block_ids"))
    mixed_ids = _coerce_id_list(out.get("mixed_block_ids"))
    kept_ids_raw = out.get("kept_block_ids")
    if kept_ids_raw is None:
        kept_ids_raw = out.get("keep_block_ids")
    kept_ids = _coerce_id_list(kept_ids_raw)

    # Back-compat / robustness:
    # - If pure/mixed are missing but kept ids exist, treat as mixed.
    # - If pure is present but kept contains additional ids, treat unclassified kept ids as mixed.
    if kept_ids:
        if (not pure_ids and not mixed_ids):
            mixed_ids = list(kept_ids)
        else:
            known = set(pure_ids) | set(mixed_ids)
            extras = [x for x in kept_ids if x not in known]
            if extras:
                mixed_ids = sorted(set(mixed_ids) | set(extras))

    # Final kept ids are union.
    kept_union: list[int] = sorted(set(pure_ids) | set(mixed_ids) | set(kept_ids))
    notes = _norm_str(out.get("issue_notes"))
    return {
        "issue_id": out_issue,
        "pure_block_ids": sorted(pure_ids),
        "mixed_block_ids": sorted(mixed_ids),
        "kept_block_ids": kept_union,
        "issue_notes": notes,
    }


def _build_llm_candidate_blocks(
    issue: dict[str, Any],
    recall_extract: dict[str, Any],
    max_blocks: int,
    max_block_chars: int,
    max_issue_chars: int,
) -> list[dict[str, Any]]:
    issue_id = _norm_str(issue.get("issue_id"))
    blocks = recall_extract["blocks"]
    feats = recall_extract["block_features"]
    keep_ids = set(recall_extract["kept_block_ids"])
    cand: list[dict[str, Any]] = []
    # Include all recall-kept blocks plus top-scoring adjacent candidates.
    for i, block_text in enumerate(blocks):
        f = feats[i]
        hard = int(f.get("hard_hits") or 0)
        soft = int(f.get("soft_hits") or 0)
        legal = int(f.get("legal_hits") or 0)
        ok_context = (hard >= 1) or (legal >= 1 and soft >= 1)
        if (i in keep_ids and ok_context) or (ok_context and float(f["score"]) > -0.2):
            bounded_text = _bounded_text_or_fail(
                block_text,
                int(max_block_chars),
                label=f"issue_id={issue_id} block_id={int(i)} candidate_block_text",
            )
            cand.append(
                {
                    "block_id": int(i),
                    "text": bounded_text,
                    "score": float(f["score"]),
                    "hard_hits": int(hard),
                    "soft_hits": int(soft),
                    "legal_hits": int(f["legal_hits"]),
                    "noise_hits": int(f["noise_hits"]),
                }
            )
    cand = sorted(cand, key=lambda x: (x["score"], x.get("hard_hits", 0), x.get("soft_hits", 0), x["legal_hits"]), reverse=True)
    if len(cand) > max_blocks:
        cand = cand[:max_blocks]
    cand = sorted(cand, key=lambda x: x["block_id"])

    # Hard char budget.
    out: list[dict[str, Any]] = []
    used = 0
    for c in cand:
        t = str(c["text"])
        c_len = len(t)
        if int(max_issue_chars) > 0 and used + c_len > int(max_issue_chars):
            raise SystemExit(
                f"issue_id={issue_id} candidate_blocks_chars={used + c_len} exceeds hard cap {int(max_issue_chars)}; "
                "no clipping allowed. Increase the cap or set it to 0 for no local cap."
            )
        out.append(c)
        used += c_len
    return out


def _coerce_llm_issue_parse(parsed_obj: dict[str, Any], expected_issue_id: str, candidate_ids: set[int]) -> dict[str, Any]:
    out = dict(parsed_obj or {})
    out_issue = _norm_str(out.get("issue_id")) or expected_issue_id
    kept_ids_raw = out.get("kept_block_ids")
    kept_ids: list[int] = []
    if isinstance(kept_ids_raw, list):
        for x in kept_ids_raw:
            try:
                v = int(x)
            except Exception:
                continue
            if v in candidate_ids and v not in kept_ids:
                kept_ids.append(v)
    blocks_raw = out.get("blocks")
    blocks: list[dict[str, Any]] = []
    if isinstance(blocks_raw, list):
        for b in blocks_raw:
            if not isinstance(b, dict):
                continue
            try:
                bid = int(b.get("block_id"))
            except Exception:
                continue
            if bid not in candidate_ids:
                continue
            keep = bool(b.get("keep"))
            txt = _norm_str(b.get("clean_text"))
            if not keep:
                txt = ""
            blocks.append({"block_id": bid, "keep": keep, "clean_text": txt})
    notes = _norm_str(out.get("issue_notes"))
    return {
        "issue_id": out_issue,
        "kept_block_ids": kept_ids,
        "blocks": blocks,
        "issue_notes": notes,
    }


def _run_method_rules(city_issues: list[dict[str, Any]], method: str) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    issue_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for issue in city_issues:
        ex = _extract_with_rules(issue, method=method)
        out_text = ex["zoning_text"]
        issue_rows.append(
            {
                "method": method,
                "city_key": _norm_str(issue.get("city_key")),
                "issue_id": _norm_str(issue.get("issue_id")),
                "issue_date": _norm_str(issue.get("issue_date")),
                "classification_label": _norm_str(issue.get("classification_label")),
                "kept_block_ids": ex["kept_block_ids"],
                "kept_blocks": ex["kept_blocks"],
                "zoning_text": out_text,
            }
        )
        metric_rows.append(
            _metric_row_from_issue(
                issue,
                out_text,
                method=method,
                used_llm=False,
                llm_parse_valid=False,
                fallback_used=False,
            )
        )
    return issue_rows, pd.DataFrame(metric_rows)


def _run_method_llm_hybrid(
    *,
    city_issues: list[dict[str, Any]],
    out_dir: Path,
    model: str,
    llm_max_candidate_blocks: int,
    llm_max_block_chars: int,
    llm_max_issue_chars: int,
    gateway_runner: Path,
    gateway_pythonpath: Path | None,
    timeout_s: float,
    concurrency: int,
    gov_env_path: Path,
    skip_existing: bool,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    # Stage 1: rules_recall candidates.
    recall_rows: dict[str, dict[str, Any]] = {}
    req_rows: list[tuple[str, str]] = []
    issue_meta: dict[str, dict[str, Any]] = {}

    for issue in city_issues:
        issue_id = _norm_str(issue.get("issue_id"))
        cid = f"zoning_extract::{_norm_str(issue.get('city_key'))}::{issue_id}"
        rec = _extract_with_rules(issue, method="rules_recall")
        cands = _build_llm_candidate_blocks(
            issue,
            rec,
            max_blocks=int(llm_max_candidate_blocks),
            max_block_chars=int(llm_max_block_chars),
            max_issue_chars=int(llm_max_issue_chars),
        )
        if not cands:
            # Keep deterministic fallback if no viable candidates.
            cands = []
        prompt = _build_llm_prompt(issue, cands)
        req_rows.append((cid, prompt))
        recall_rows[cid] = rec
        issue_meta[cid] = {"issue": issue, "candidates": cands}

    req_path = out_dir / "requests" / "openai_requests_shard000.jsonl"
    _write_openai_requests(req_path, req_rows, model=model)
    _run_gateway_requests(
        request_dir=req_path.parent,
        output_dir=out_dir / "results",
        runner_path=gateway_runner,
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

    issue_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for cid, _prompt in req_rows:
        issue = issue_meta[cid]["issue"]
        rec = recall_rows[cid]
        cands = issue_meta[cid]["candidates"]
        cids = {int(c["block_id"]) for c in cands}

        output_text = ""
        if cid in result_map:
            body = (((result_map[cid].get("response") or {}).get("body")) or {})
            output_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in error_map:
            body = (((error_map[cid].get("response") or {}).get("body")) or {})
            output_text = _norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))

        parsed = _parse_json_from_text(output_text)
        parse_valid = isinstance(parsed, dict) and bool(parsed)
        fallback_used = False
        kept_blocks: list[dict[str, Any]] = []
        llm_issue_notes = ""

        if parse_valid and isinstance(parsed, dict):
            p = _coerce_llm_issue_parse(parsed, expected_issue_id=_norm_str(issue.get("issue_id")), candidate_ids=cids)
            llm_issue_notes = _norm_str(p.get("issue_notes"))
            keep_by_id: dict[int, str] = {}
            for b in p.get("blocks", []):
                if not isinstance(b, dict):
                    continue
                bid = int(b.get("block_id"))
                keep = bool(b.get("keep"))
                ctext = _norm_str(b.get("clean_text"))
                if keep and ctext:
                    keep_by_id[bid] = ctext
            # If block-level flags are missing but kept ids are present, fallback to original candidate text.
            if not keep_by_id and isinstance(p.get("kept_block_ids"), list):
                for bid in p["kept_block_ids"]:
                    for c in cands:
                        if int(c["block_id"]) == int(bid):
                            keep_by_id[int(bid)] = _norm_str(c.get("text"))
                            break

            # Convert to ordered kept block rows.
            for bid in sorted(keep_by_id.keys()):
                text = _norm_str(keep_by_id[bid])
                if not text:
                    continue
                kept_blocks.append({"block_id": int(bid), "text": text})

        if not kept_blocks:
            # Fallback to deterministic recall output.
            fallback_used = True
            for kb in rec["kept_blocks"]:
                kept_blocks.append({"block_id": int(kb["block_id"]), "text": _norm_str(kb.get("text"))})

        zoning_text = "\n\n".join([_norm_str(x.get("text")) for x in kept_blocks if _norm_str(x.get("text"))]).strip()
        issue_rows.append(
            {
                "method": "llm_hybrid",
                "city_key": _norm_str(issue.get("city_key")),
                "issue_id": _norm_str(issue.get("issue_id")),
                "issue_date": _norm_str(issue.get("issue_date")),
                "classification_label": _norm_str(issue.get("classification_label")),
                "kept_block_ids": [int(x["block_id"]) for x in kept_blocks],
                "kept_blocks": kept_blocks,
                "zoning_text": zoning_text,
                "llm_issue_notes": llm_issue_notes,
                "llm_parse_valid": int(parse_valid),
                "fallback_used": int(fallback_used),
            }
        )
        metric_rows.append(
            _metric_row_from_issue(
                issue,
                zoning_text,
                method="llm_hybrid",
                used_llm=True,
                llm_parse_valid=bool(parse_valid),
                fallback_used=bool(fallback_used),
            )
        )

    return issue_rows, pd.DataFrame(metric_rows)


def _run_method_llm_only(
    *,
    city_issues: list[dict[str, Any]],
    out_dir: Path,
    model: str,
    llm_only_max_block_chars: int,
    llm_only_max_issue_chars: int,
    gateway_runner: Path,
    gateway_pythonpath: Path | None,
    timeout_s: float,
    concurrency: int,
    gov_env_path: Path,
    skip_existing: bool,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    # Two-stage LLM-only extraction:
    #   Pass 1: identify candidate blocks containing any verbatim ordinance/notice language.
    #   Pass 2: within those blocks, select exact line ranges that are ordinance/notice (verbatim).

    pass1_req_rows: list[tuple[str, str]] = []
    issue_meta: dict[str, dict[str, Any]] = {}
    for issue in city_issues:
        issue_id = _norm_str(issue.get("issue_id"))
        cid = f"zoning_extract_llm_only_pass1::{_norm_str(issue.get('city_key'))}::{issue_id}"
        blocks = _split_blocks(_norm_str(issue.get("text")))
        issue_chars = sum(len(b) for b in blocks)
        max_issue_chars = int(llm_only_max_issue_chars)
        if max_issue_chars > 0 and issue_chars > max_issue_chars:
            raise SystemExit(
                f"issue_id={issue_id} issue_chars={issue_chars} exceeds hard cap {max_issue_chars}; no clipping allowed. "
                "Increase the cap or set it to 0 for no local cap."
            )
        prompt = _build_llm_only_prompt(issue, blocks, max_block_chars=int(llm_only_max_block_chars))
        pass1_req_rows.append((cid, prompt))
        issue_meta[cid] = {"issue": issue, "blocks": blocks}

    pass1_dir = out_dir / "pass1"
    pass1_req_path = pass1_dir / "requests" / "openai_requests_shard000.jsonl"
    _write_openai_requests(pass1_req_path, pass1_req_rows, model=model)
    _run_gateway_requests(
        request_dir=pass1_req_path.parent,
        output_dir=pass1_dir / "results",
        runner_path=gateway_runner,
        model=model,
        gateway_pythonpath=gateway_pythonpath,
        timeout_s=timeout_s,
        concurrency=concurrency,
        gov_env_path=gov_env_path,
        skip_existing=skip_existing,
        dry_run=dry_run,
    )

    pass1_result_map = _read_result_jsonl(pass1_dir / "results")
    pass1_error_map = _read_error_jsonl(pass1_dir / "results")

    # Collect candidate block IDs per issue.
    issue_candidates: dict[str, dict[str, Any]] = {}
    for cid, _prompt in pass1_req_rows:
        issue = issue_meta[cid]["issue"]
        blocks = issue_meta[cid]["blocks"]

        output_text = ""
        if cid in pass1_result_map:
            body = (((pass1_result_map[cid].get("response") or {}).get("body")) or {})
            output_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in pass1_error_map:
            body = (((pass1_error_map[cid].get("response") or {}).get("body")) or {})
            output_text = _norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))

        parsed = _parse_json_from_text(output_text)
        parse_valid = isinstance(parsed, dict) and bool(parsed)
        kept_block_ids: list[int] = []
        pure_block_ids: list[int] = []
        mixed_block_ids: list[int] = []
        notes = ""
        if parse_valid and isinstance(parsed, dict):
            p = _coerce_llm_only_issue_parse(parsed, expected_issue_id=_norm_str(issue.get("issue_id")), n_blocks=len(blocks))
            notes = _norm_str(p.get("issue_notes"))
            kept_block_ids = sorted({int(x) for x in (p.get("kept_block_ids") or []) if isinstance(x, int)})
            pure_block_ids = sorted({int(x) for x in (p.get("pure_block_ids") or []) if isinstance(x, int)})
            mixed_block_ids = sorted({int(x) for x in (p.get("mixed_block_ids") or []) if isinstance(x, int)})
        issue_candidates[_norm_str(issue.get("issue_id"))] = {
            "city_key": _norm_str(issue.get("city_key")),
            "issue": issue,
            "blocks": blocks,
            "pass1_parse_valid": int(bool(parse_valid)),
            "pass1_notes": notes,
            "candidate_block_ids": kept_block_ids,
            "pure_block_ids": pure_block_ids,
            "mixed_block_ids": mixed_block_ids,
        }

    # Pass 2: trim to ordinance-only line ranges within candidate blocks.
    pass2_req_rows: list[tuple[str, str]] = []
    pass2_meta: dict[str, dict[str, Any]] = {}
    for issue_id, meta in issue_candidates.items():
        # Only mixed blocks require line-level trimming.
        mixed_ids: list[int] = list(meta.get("mixed_block_ids") or [])
        if not mixed_ids:
            continue
        issue = meta["issue"]
        blocks = meta["blocks"]
        cand_blocks: list[dict[str, Any]] = []
        block_line_counts: dict[int, int] = {}
        for bid in mixed_ids:
            if bid < 0 or bid >= len(blocks):
                continue
            txt = _norm_str(blocks[bid])
            if not txt:
                continue
            raw_lines, _numbered = _line_number_block(txt)
            block_line_counts[int(bid)] = int(len(raw_lines))
            cand_blocks.append({"block_id": int(bid), "text": txt})
        if not cand_blocks:
            continue
        cid2 = f"zoning_extract_llm_only_pass2::{_norm_str(issue.get('city_key'))}::{issue_id}"
        prompt2 = _build_llm_only_trim_prompt(issue, cand_blocks)
        pass2_req_rows.append((cid2, prompt2))
        pass2_meta[cid2] = {
            "issue": issue,
            "blocks": blocks,
            "candidate_ids": set(int(x) for x in mixed_ids),
            "block_line_counts": block_line_counts,
        }

    pass2_dir = out_dir / "pass2"
    pass2_req_path = pass2_dir / "requests" / "openai_requests_shard000.jsonl"
    _write_openai_requests(pass2_req_path, pass2_req_rows, model=model)
    _run_gateway_requests(
        request_dir=pass2_req_path.parent,
        output_dir=pass2_dir / "results",
        runner_path=gateway_runner,
        model=model,
        gateway_pythonpath=gateway_pythonpath,
        timeout_s=timeout_s,
        concurrency=concurrency,
        gov_env_path=gov_env_path,
        skip_existing=skip_existing,
        dry_run=dry_run,
    )

    pass2_result_map = _read_result_jsonl(pass2_dir / "results")
    pass2_error_map = _read_error_jsonl(pass2_dir / "results")

    # Index pass2 parse by issue_id for easy join.
    pass2_by_issue: dict[str, dict[str, Any]] = {}
    for cid2, _prompt2 in pass2_req_rows:
        meta2 = pass2_meta[cid2]
        issue = meta2["issue"]
        issue_id = _norm_str(issue.get("issue_id"))
        output_text = ""
        if cid2 in pass2_result_map:
            body = (((pass2_result_map[cid2].get("response") or {}).get("body")) or {})
            output_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid2 in pass2_error_map:
            body = (((pass2_error_map[cid2].get("response") or {}).get("body")) or {})
            output_text = _norm_str((((body.get("error") or {}).get("message")) if isinstance(body, dict) else ""))
        parsed = _parse_json_from_text(output_text)
        parse_valid = isinstance(parsed, dict) and bool(parsed)
        if parse_valid and isinstance(parsed, dict):
            p2 = _coerce_llm_only_trim_parse(
                parsed,
                expected_issue_id=issue_id,
                candidate_ids=set(meta2["candidate_ids"]),
                block_line_counts=meta2["block_line_counts"],
            )
            pass2_by_issue[issue_id] = {"parse_valid": 1, "parsed": p2}
        else:
            pass2_by_issue[issue_id] = {"parse_valid": 0, "parsed": None}

    # Build final issue rows and metrics.
    issue_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    # Preserve original city issue order for stable outputs.
    for issue in city_issues:
        issue_id = _norm_str(issue.get("issue_id"))
        meta = issue_candidates.get(issue_id) or {"candidate_block_ids": [], "pass1_parse_valid": 0, "pass1_notes": ""}
        blocks = _split_blocks(_norm_str(issue.get("text")))
        cand_ids = list(meta.get("candidate_block_ids") or [])
        pure_ids = list(meta.get("pure_block_ids") or [])
        mixed_ids = list(meta.get("mixed_block_ids") or [])

        kept_blocks: list[dict[str, Any]] = []
        kept_block_ids: list[int] = []
        kept_line_ranges_by_block: dict[int, list[list[int]]] = {}

        p2 = pass2_by_issue.get(issue_id) or {}
        p2_valid = int(p2.get("parse_valid") or 0)
        p2_parsed = p2.get("parsed") if isinstance(p2, dict) else None
        p2_notes = ""
        if isinstance(p2_parsed, dict):
            p2_notes = _norm_str(p2_parsed.get("issue_notes"))

        # 1) Keep pure verbatim blocks as-is (verbatim).
        for bid in sorted({int(x) for x in pure_ids if isinstance(x, int)}):
            if bid < 0 or bid >= len(blocks):
                continue
            txt = _norm_str(blocks[bid])
            if not txt:
                continue
            kept_block_ids.append(int(bid))
            kept_blocks.append({"block_id": int(bid), "keep_line_ranges": [], "text": txt})

        # 2) Trim mixed blocks to ordinance-only lines using pass2 output.
        if p2_valid and isinstance(p2_parsed, dict):
            blocks_out = p2_parsed.get("blocks") or []
            if isinstance(blocks_out, list):
                for b in blocks_out:
                    if not isinstance(b, dict):
                        continue
                    try:
                        bid = int(b.get("block_id"))
                    except Exception:
                        continue
                    ranges = b.get("keep_line_ranges") or []
                    if not isinstance(ranges, list) or not ranges:
                        continue
                    if bid < 0 or bid >= len(blocks):
                        continue
                    raw_lines = str(blocks[bid] or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
                    kept_lines: list[str] = []
                    keep_ranges: list[list[int]] = []
                    for rr in ranges:
                        if not isinstance(rr, list) or len(rr) != 2:
                            continue
                        try:
                            s = int(rr[0])
                            e = int(rr[1])
                        except Exception:
                            continue
                        if s <= 0 or e <= 0:
                            continue
                        if s > e:
                            s, e = e, s
                        if s > len(raw_lines) or e > len(raw_lines):
                            continue
                        keep_ranges.append([s, e])
                        kept_lines.extend(raw_lines[s - 1 : e])
                    text = "\n".join([x for x in kept_lines]).strip()
                    if not text:
                        continue
                    kept_block_ids.append(int(bid))
                    kept_line_ranges_by_block[int(bid)] = keep_ranges
                    kept_blocks.append({"block_id": int(bid), "keep_line_ranges": keep_ranges, "text": text})

        # If pass2 fails or yields nothing, leave empty rather than re-introducing narrative.
        # Sort kept blocks by their original order.
        kept_blocks = sorted(kept_blocks, key=lambda x: int(x.get("block_id") or 0))
        kept_block_ids = [int(x.get("block_id")) for x in kept_blocks if isinstance(x, dict) and isinstance(x.get("block_id"), int)]
        zoning_text = "\n\n".join([_norm_str(x.get("text")) for x in kept_blocks if _norm_str(x.get("text"))]).strip()
        llm_issue_notes = "\n".join([x for x in [_norm_str(meta.get("pass1_notes")), _norm_str(p2_notes)] if x]).strip()

        issue_rows.append(
            {
                "method": "llm_only",
                "city_key": _norm_str(issue.get("city_key")),
                "issue_id": issue_id,
                "issue_date": _norm_str(issue.get("issue_date")),
                "classification_label": _norm_str(issue.get("classification_label")),
                "candidate_block_ids": [int(x) for x in cand_ids],
                "kept_block_ids": [int(x) for x in kept_block_ids],
                "kept_blocks": kept_blocks,
                "zoning_text": zoning_text,
                "llm_issue_notes": llm_issue_notes,
                "llm_pass1_parse_valid": int(meta.get("pass1_parse_valid") or 0),
                "llm_pass2_parse_valid": int(p2_valid),
                "fallback_used": 0,
            }
        )
        metric_rows.append(
            _metric_row_from_issue(
                issue,
                zoning_text,
                method="llm_only",
                used_llm=True,
                llm_parse_valid=bool(int(meta.get("pass1_parse_valid") or 0) and int(p2_valid)),
                fallback_used=False,
            )
        )

    return issue_rows, pd.DataFrame(metric_rows)


def _build_city_metrics(issue_metrics: pd.DataFrame, min_zoning_issue_kept_chars: int) -> pd.DataFrame:
    if issue_metrics.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (method, city_key), g in issue_metrics.groupby(["method", "city_key"], dropna=False):
        total_in = float(g["input_chars"].sum())
        total_kept = float(g["kept_chars"].sum())
        in_words = float(g["input_words"].sum())
        kept_words = float(g["kept_words"].sum())
        in_z_hits = float(g["input_zoning_hits"].sum())
        kept_z_hits = float(g["kept_zoning_hits"].sum())
        in_n_hits = float(g["input_noise_hits"].sum())
        kept_n_hits = float(g["kept_noise_hits"].sum())
        zoning_issues = g[g["is_zoning_label"] == 1].copy()
        z_cov = math.nan
        avg_kept_z = math.nan
        if not zoning_issues.empty:
            z_cov = float((zoning_issues["kept_chars"] >= int(min_zoning_issue_kept_chars)).mean())
            avg_kept_z = float(zoning_issues["kept_chars"].mean())
        rows.append(
            {
                "method": _norm_str(method),
                "city_key": _norm_str(city_key),
                "city_name": _norm_str(g["city_name"].dropna().iloc[0]) if not g["city_name"].dropna().empty else "",
                "state_abbr": _norm_str(g["state_abbr"].dropna().iloc[0]) if not g["state_abbr"].dropna().empty else "",
                "issue_count": int(len(g)),
                "total_input_chars": int(total_in),
                "total_kept_chars": int(total_kept),
                "keep_ratio": (total_kept / total_in) if total_in > 0 else math.nan,
                "input_zoning_density_per_1k_words": _density_per_1k(int(in_z_hits), int(in_words)),
                "kept_zoning_density_per_1k_words": _density_per_1k(int(kept_z_hits), int(kept_words)),
                "input_noise_density_per_1k_words": _density_per_1k(int(in_n_hits), int(in_words)),
                "kept_noise_density_per_1k_words": _density_per_1k(int(kept_n_hits), int(kept_words)),
                "zoning_issue_coverage": z_cov,
                "avg_kept_chars_zoning_issues": avg_kept_z,
                "llm_parse_valid_share": float(g["llm_parse_valid"].mean()) if "llm_parse_valid" in g.columns else math.nan,
                "fallback_share": float(g["fallback_used"].mean()) if "fallback_used" in g.columns else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_method_summary(city_metrics: pd.DataFrame) -> pd.DataFrame:
    if city_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method, g in city_metrics.groupby("method", dropna=False):
        in_chars = float(g["total_input_chars"].sum())
        kept_chars = float(g["total_kept_chars"].sum())
        in_z = float((g["input_zoning_density_per_1k_words"] * g["total_input_chars"]).sum())
        kept_z = float((g["kept_zoning_density_per_1k_words"] * g["total_kept_chars"]).sum())
        in_n = float((g["input_noise_density_per_1k_words"] * g["total_input_chars"]).sum())
        kept_n = float((g["kept_noise_density_per_1k_words"] * g["total_kept_chars"]).sum())

        in_z_d = in_z / in_chars if in_chars > 0 else math.nan
        kept_z_d = kept_z / kept_chars if kept_chars > 0 else math.nan
        in_n_d = in_n / in_chars if in_chars > 0 else math.nan
        kept_n_d = kept_n / kept_chars if kept_chars > 0 else math.nan

        z_lift = (kept_z_d / in_z_d) if (in_z_d and math.isfinite(in_z_d) and in_z_d > 0 and math.isfinite(kept_z_d)) else math.nan
        n_drop = (1.0 - (kept_n_d / in_n_d)) if (in_n_d and math.isfinite(in_n_d) and in_n_d > 0 and math.isfinite(kept_n_d)) else math.nan
        cov = float(g["zoning_issue_coverage"].mean()) if "zoning_issue_coverage" in g.columns else math.nan
        keep_ratio = (kept_chars / in_chars) if in_chars > 0 else math.nan

        # Composite score: prioritize preserving zoning issues + purity gains.
        z_term = min(max(z_lift if math.isfinite(z_lift) else 0.0, 0.0), 3.0) / 3.0
        n_term = min(max(n_drop if math.isfinite(n_drop) else 0.0, -1.0), 1.0)
        cov_term = min(max(cov if math.isfinite(cov) else 0.0, 0.0), 1.0)
        score = (0.45 * cov_term) + (0.35 * z_term) + (0.20 * n_term)
        rows.append(
            {
                "method": _norm_str(method),
                "city_count": int(len(g)),
                "total_input_chars": int(in_chars),
                "total_kept_chars": int(kept_chars),
                "keep_ratio": keep_ratio,
                "input_zoning_density_per_1k_words": in_z_d,
                "kept_zoning_density_per_1k_words": kept_z_d,
                "zoning_density_lift": z_lift,
                "input_noise_density_per_1k_words": in_n_d,
                "kept_noise_density_per_1k_words": kept_n_d,
                "noise_density_drop_share": n_drop,
                "mean_zoning_issue_coverage": cov,
                "score": score,
            }
        )
    out = pd.DataFrame(rows).sort_values(["score", "mean_zoning_issue_coverage", "zoning_density_lift"], ascending=[False, False, False]).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def _write_panel_texts(issue_rows: list[dict[str, Any]], out_dir: Path) -> None:
    by_city: dict[str, list[dict[str, Any]]] = {}
    for r in issue_rows:
        by_city.setdefault(_norm_str(r.get("city_key")), []).append(r)
    for city_key, rows in by_city.items():
        rows_s = sorted(rows, key=lambda x: (_norm_str(x.get("issue_date")), _norm_str(x.get("issue_id"))))
        chunks: list[str] = []
        for r in rows_s:
            ztxt = _norm_str(r.get("zoning_text"))
            if not ztxt:
                continue
            chunks.append(
                f"[ISSUE] issue_id={_norm_str(r.get('issue_id'))} date={_norm_str(r.get('issue_date'))} "
                f"label={_norm_str(r.get('classification_label'))}\n{ztxt}"
            )
        p = out_dir / "panel_zoning_texts" / f"{city_key}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n\n" + ("\n\n---\n\n".join(chunks)) if chunks else "", encoding="utf-8")


def _build_provenance(args: argparse.Namespace, run_dir: Path, out_dir: Path, city_keys: list[str], methods: list[str], script_path: Path) -> dict[str, Any]:
    return {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(script_path),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "city_count": int(len(city_keys)),
        "city_keys": city_keys,
        "methods": methods,
        "settings": {
            "experiment": bool(args.experiment),
            "max_issues_per_city": int(args.max_issues_per_city),
            "model": str(args.model),
            "llm_max_candidate_blocks": int(args.llm_max_candidate_blocks),
            "llm_max_block_chars": int(args.llm_max_block_chars),
            "llm_max_issue_candidate_chars": int(args.llm_max_issue_candidate_chars),
            "llm_only_max_block_chars": int(args.llm_only_max_block_chars),
            "llm_only_max_issue_chars": int(args.llm_only_max_issue_chars),
            "concurrency": int(args.concurrency),
            "timeout": float(args.timeout),
            "min_zoning_issue_kept_chars": int(args.min_zoning_issue_kept_chars),
        },
    }


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve() if _norm_str(args.output_dir) else (run_dir / "zoning_text_extract")
    out_root.mkdir(parents=True, exist_ok=True)

    city_keys = _load_city_keys(run_dir, args.city_keys, args.city_keys_file)
    _eprint(f"Loaded {len(city_keys)} city panels.")

    methods = [x.strip() for x in _norm_str(args.methods).split(",") if x.strip()]
    if not methods:
        raise SystemExit("No methods requested. Set --methods.")
    bad = [m for m in methods if m not in _METHODS]
    if bad:
        raise SystemExit(f"Unknown methods: {bad}. Allowed: {', '.join(_METHODS)}")

    all_issue_rows: list[dict[str, Any]] = []
    all_issue_metrics: list[pd.DataFrame] = []
    all_city_metrics: list[pd.DataFrame] = []

    # Load panel issues once.
    city_issues_map: dict[str, list[dict[str, Any]]] = {}
    for city_key in city_keys:
        p = run_dir / "panels" / city_key / "issue_texts.jsonl"
        if not p.is_file():
            _eprint(f"[warn] missing issue_texts: {p}")
            continue
        rows = _iter_jsonl(p)
        if int(args.max_issues_per_city) > 0:
            rows = sorted(rows, key=lambda r: (_norm_str(r.get("issue_date")), _norm_str(r.get("issue_id"))))[: int(args.max_issues_per_city)]
        city_issues_map[city_key] = rows
    city_keys = [k for k in city_keys if k in city_issues_map]
    if not city_keys:
        raise SystemExit("No valid city issue panels loaded.")

    gateway_runner = Path(args.gateway_runner).expanduser().resolve()
    gateway_pythonpath = Path(args.gateway_pythonpath).expanduser().resolve() if _norm_str(args.gateway_pythonpath) else None
    gov_env_path = Path(args.gov_env_path).expanduser().resolve()

    for method in methods:
        _eprint(f"Running method: {method}")
        method_out = out_root / method
        method_out.mkdir(parents=True, exist_ok=True)
        method_issue_rows: list[dict[str, Any]] = []
        method_issue_metrics: list[pd.DataFrame] = []

        for city_key in city_keys:
            issues = city_issues_map[city_key]
            if method == "llm_hybrid":
                city_issue_rows, city_issue_df = _run_method_llm_hybrid(
                    city_issues=issues,
                    out_dir=method_out / "llm" / city_key,
                    model=str(args.model),
                    llm_max_candidate_blocks=int(args.llm_max_candidate_blocks),
                    llm_max_block_chars=int(args.llm_max_block_chars),
                    llm_max_issue_chars=int(args.llm_max_issue_candidate_chars),
                    gateway_runner=gateway_runner,
                    gateway_pythonpath=gateway_pythonpath,
                    timeout_s=float(args.timeout),
                    concurrency=int(args.concurrency),
                    gov_env_path=gov_env_path,
                    skip_existing=bool(args.skip_existing),
                    dry_run=bool(args.dry_run),
                )
            elif method == "llm_only":
                city_issue_rows, city_issue_df = _run_method_llm_only(
                    city_issues=issues,
                    out_dir=method_out / "llm" / city_key,
                    model=str(args.model),
                    llm_only_max_block_chars=int(args.llm_only_max_block_chars),
                    llm_only_max_issue_chars=int(args.llm_only_max_issue_chars),
                    gateway_runner=gateway_runner,
                    gateway_pythonpath=gateway_pythonpath,
                    timeout_s=float(args.timeout),
                    concurrency=int(args.concurrency),
                    gov_env_path=gov_env_path,
                    skip_existing=bool(args.skip_existing),
                    dry_run=bool(args.dry_run),
                )
            else:
                city_issue_rows, city_issue_df = _run_method_rules(issues, method=method)
            method_issue_rows.extend(city_issue_rows)
            if not city_issue_df.empty:
                method_issue_metrics.append(city_issue_df)

        if method_issue_metrics:
            issue_df = pd.concat(method_issue_metrics, ignore_index=True)
        else:
            issue_df = pd.DataFrame()
        city_df = _build_city_metrics(issue_df, min_zoning_issue_kept_chars=int(args.min_zoning_issue_kept_chars))

        _write_jsonl(method_out / "issue_zoning_extract.jsonl", method_issue_rows)
        if not issue_df.empty:
            _write_csv(issue_df, method_out / "issue_metrics.csv")
        if not city_df.empty:
            _write_csv(city_df, method_out / "city_metrics.csv")
        _write_panel_texts(method_issue_rows, method_out)

        all_issue_rows.extend(method_issue_rows)
        if not issue_df.empty:
            all_issue_metrics.append(issue_df)
        if not city_df.empty:
            all_city_metrics.append(city_df)

        # Write city-level consolidated panel text snapshots under each method for easier manual review.
        for city_key in city_keys:
            city_rows = [r for r in method_issue_rows if _norm_str(r.get("city_key")) == city_key]
            city_rows = sorted(city_rows, key=lambda r: (_norm_str(r.get("issue_date")), _norm_str(r.get("issue_id"))))
            out_jsonl = method_out / "city_panels" / city_key / "issue_cleaned.jsonl"
            out_jsonl.parent.mkdir(parents=True, exist_ok=True)
            _write_jsonl(out_jsonl, city_rows)

    if all_issue_metrics:
        all_issue_df = pd.concat(all_issue_metrics, ignore_index=True)
        _write_csv(all_issue_df, out_root / "all_methods_issue_metrics.csv")
    else:
        all_issue_df = pd.DataFrame()

    if all_city_metrics:
        all_city_df = pd.concat(all_city_metrics, ignore_index=True)
        _write_csv(all_city_df, out_root / "all_methods_city_metrics.csv")
    else:
        all_city_df = pd.DataFrame()

    method_summary = _build_method_summary(all_city_df)
    if not method_summary.empty:
        _write_csv(method_summary, out_root / "experiment" / "method_comparison.csv")
        best = method_summary.iloc[0].to_dict()
        reco = {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "recommended_method": _norm_str(best.get("method")),
            "rationale": {
                "score": float(best.get("score", math.nan)),
                "mean_zoning_issue_coverage": float(best.get("mean_zoning_issue_coverage", math.nan)),
                "zoning_density_lift": float(best.get("zoning_density_lift", math.nan)),
                "noise_density_drop_share": float(best.get("noise_density_drop_share", math.nan)),
            },
        }
        p = out_root / "experiment" / "recommended_method.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(reco, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Manual review sample to inspect extraction quality directly in transcript language.
    if not all_issue_df.empty:
        review = all_issue_df.copy()
        review = review.sort_values(
            ["is_zoning_label", "kept_zoning_density_per_1k_words", "keep_ratio"],
            ascending=[False, False, False],
        )
        review = review.head(80)
        _write_csv(
            review[
                [
                    "method",
                    "city_key",
                    "city_name",
                    "state_abbr",
                    "issue_id",
                    "issue_date",
                    "classification_label",
                    "keep_ratio",
                    "input_chars",
                    "kept_chars",
                    "input_zoning_density_per_1k_words",
                    "kept_zoning_density_per_1k_words",
                    "input_noise_density_per_1k_words",
                    "kept_noise_density_per_1k_words",
                ]
            ],
            out_root / "experiment" / "manual_review_candidates.csv",
        )

    prov = _build_provenance(args, run_dir, out_root, city_keys, methods, Path(__file__).resolve())
    (out_root / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _eprint("Done.")
    _eprint(f"Output dir: {out_root}")
    if not method_summary.empty:
        _eprint("Method ranking:")
        for row in method_summary.itertuples(index=False):
            _eprint(
                f"  rank={int(row.rank)} method={row.method} score={float(row.score):.3f} "
                f"coverage={float(row.mean_zoning_issue_coverage):.3f} "
                f"z_lift={float(row.zoning_density_lift):.3f} noise_drop={float(row.noise_density_drop_share):.3f}"
            )


if __name__ == "__main__":
    main()
