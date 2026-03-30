#!/usr/bin/env python3
"""
Shared helpers for full-newspaper v3 LLM-only zoning pipeline scripts.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


FIXED_CATEGORY_KEYS = [
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


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def norm_str(x: Any) -> str:
    return str(x or "").strip()


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def safe_int(x: Any, default: int = 0) -> int:
    v = safe_float(x, default=math.nan)
    if pd.isna(v):
        return int(default)
    try:
        return int(round(v))
    except Exception:
        return int(default)


def enum(v: Any, allowed: set[str], default: str) -> str:
    s = norm_str(v).lower()
    return s if s in allowed else default


def as_issue_id_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        s = norm_str(x)
        if s:
            out.append(s)
    return out


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def parse_env_file(path: Path) -> dict[str, str]:
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


def extract_openai_output_text(body: dict[str, Any]) -> str:
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
    s = norm_str(candidate)
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


def parse_json_from_text(text: str) -> dict[str, Any] | None:
    s = norm_str(text)
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


def write_openai_requests(path: Path, rows: list[tuple[str, str]], *, model: str) -> None:
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


def run_gateway_requests(
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
    gateway_url: str = "",
    port: int = 0,
) -> None:
    if not runner_path.is_file():
        raise SystemExit(f"Gateway runner not found: {runner_path}")

    env = dict(os.environ)
    if gateway_pythonpath is not None:
        pp = str(gateway_pythonpath)
        prior = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = pp if not prior else f"{pp}:{prior}"

    parsed = parse_env_file(gov_env_path)
    for key in ("OPENAI_KEY", "GEMINI_KEY", "CLAUDE_KEY"):
        if not env.get(key) and norm_str(parsed.get(key)):
            env[key] = norm_str(parsed.get(key))

    if not env.get("OPENAI_KEY"):
        alias = norm_str(parsed.get("CODEX_API_KEY") or parsed.get("OPENAI_API_KEY"))
        if alias:
            env["OPENAI_KEY"] = alias

    provider = str(model).split(":", 1)[0].lower() if ":" in str(model) else "openai"
    key_name = {"openai": "OPENAI_KEY", "gemini": "GEMINI_KEY", "claude": "CLAUDE_KEY"}.get(provider)

    key_candidates: list[str] = []
    if key_name:
        seen: set[str] = set()
        for cand in (
            norm_str(env.get(key_name)),
            norm_str(parsed.get(key_name)),
            *[norm_str(parsed.get(f"{key_name}_BACKUP_{i}")) for i in range(1, 6)],
        ):
            if not cand or cand in seen:
                continue
            seen.add(cand)
            key_candidates.append(cand)
        if not key_candidates:
            raise SystemExit(f"Missing provider key for {provider}: {key_name}")

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
    if norm_str(gateway_url):
        cmd.extend(["--gateway-url", str(gateway_url)])
    elif int(port) > 0:
        cmd.extend(["--port", str(int(port))])

    if not key_name:
        subprocess.run(cmd, check=True, env=env)
        return

    last_err: Exception | None = None
    for i, key_val in enumerate(key_candidates, start=1):
        env_try = dict(env)
        env_try[key_name] = key_val
        try:
            eprint(f"Gateway request run with {key_name} candidate {i}/{len(key_candidates)}.")
            subprocess.run(cmd, check=True, env=env_try)
            return
        except subprocess.CalledProcessError as exc:
            last_err = exc
            continue
    raise SystemExit(f"Gateway request run failed for all {key_name} candidates. last_error={last_err}")


def read_result_jsonl(results_dir: Path) -> dict[str, dict[str, Any]]:
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


def read_error_jsonl(results_dir: Path) -> dict[str, dict[str, Any]]:
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


def load_issue_rows_for_city(run_dir: Path, city_key: str) -> list[dict[str, Any]]:
    p = run_dir / "panels" / city_key / "issue_texts.jsonl"
    return iter_jsonl(p)


def find_newspaper_city_map(run_dir: Path, newspapers: list[str]) -> dict[str, str]:
    wanted = {norm_str(x).lower() for x in newspapers if norm_str(x)}
    out: dict[str, str] = {}
    panel_dir = run_dir / "panels"
    for city_path in sorted(panel_dir.iterdir()):
        if not city_path.is_dir():
            continue
        p = city_path / "issue_texts.jsonl"
        if not p.is_file():
            continue
        rows = iter_jsonl(p)
        slugs = {norm_str(r.get("pub_slug")).lower() for r in rows if norm_str(r.get("pub_slug"))}
        for slug in list(wanted):
            if slug in slugs and slug not in out:
                out[slug] = city_path.name
    return out
