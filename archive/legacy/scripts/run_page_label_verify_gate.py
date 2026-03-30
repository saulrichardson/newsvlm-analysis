#!/usr/bin/env python3
"""
Second-pass verifier gate for page classification outputs.

Given a completed A/B run directory and a base variant joined prediction file,
run an LLM verification pass for rows currently predicted as core legal buckets
(full/amendment/proposal/notice), then re-score with verified labels.
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

PRED_TO_BUCKET = {
    "full_ordinance": "full_ordinance",
    "amendment": "amendment",
    "proposal": "proposal_full_text",
    "notice": "notice",
    "building_code_or_other_law": "building_code_non_zoning",
    "zoning_narrative_nonverbatim": "narrative_or_nonverbatim",
    "non_zoning": "non_zoning",
    "uncertain": "uncertain_other",
}

ALLOWED_LEGAL_OBJECT_TYPE = set(PRED_TO_BUCKET.keys())
ALLOWED_YNU = {"yes", "no", "uncertain"}
ALLOWED_ENACT = {"enacted", "proposed", "unknown"}
ALLOWED_PAGE_COMPLETENESS = {"partial", "complete", "uncertain"}


def _norm(x: Any) -> str:
    return str(x or "").strip()


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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
    s = _norm(candidate)
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


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    s = _norm(text)
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\\s*", "", s, flags=re.I)
    s = re.sub(r"\\s*```$", "", s)

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
        candidate = re.sub(r",\\s*([}\\]])", r"\\1", candidate)
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


def _coerce_result(obj: dict[str, Any], *, page_id: str) -> dict[str, Any]:
    legal_object_type = _norm(obj.get("legal_object_type")).lower()
    if legal_object_type not in ALLOWED_LEGAL_OBJECT_TYPE:
        legal_object_type = "uncertain"
    contains = _norm(obj.get("contains_verbatim_zoning_law_language")).lower()
    if contains not in ALLOWED_YNU:
        contains = "uncertain"
    verbatim = _norm(obj.get("is_verbatim_legal_text")).lower()
    if verbatim not in ALLOWED_YNU:
        verbatim = "uncertain"
    enact = _norm(obj.get("enactment_status_signal")).lower()
    if enact not in ALLOWED_ENACT:
        enact = "unknown"
    page_completeness = _norm(obj.get("page_completeness")).lower()
    if page_completeness not in ALLOWED_PAGE_COMPLETENESS:
        page_completeness = "uncertain"
    conf = obj.get("confidence_0_to_1")
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    conf_f = max(0.0, min(1.0, conf_f))
    return {
        "page_id": page_id,
        "legal_object_type": legal_object_type,
        "page_completeness": page_completeness,
        "contains_verbatim_zoning_law_language": contains,
        "is_verbatim_legal_text": verbatim,
        "enactment_status_signal": enact,
        "confidence_0_to_1": conf_f,
        "notes": _norm(obj.get("notes")),
    }


def _read_result_jsonl(results_dir: Path) -> dict[str, dict[str, Any]]:
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


def _read_error_jsonl(results_dir: Path) -> dict[str, dict[str, Any]]:
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


def _run_gateway(
    *,
    request_dir: Path,
    output_dir: Path,
    model: str,
    runner_path: Path,
    gateway_pythonpath: Path,
    gov_env_path: Path,
    timeout: float,
    concurrency: int,
    skip_existing: bool,
) -> None:
    env = dict(os.environ)
    prior = _norm(env.get("PYTHONPATH"))
    env["PYTHONPATH"] = str(gateway_pythonpath) if not prior else f"{gateway_pythonpath}:{prior}"
    parsed = _parse_env_file(gov_env_path)
    for key in ("OPENAI_KEY", "GEMINI_KEY", "CLAUDE_KEY"):
        if not env.get(key) and _norm(parsed.get(key)):
            env[key] = _norm(parsed.get(key))
    if not env.get("OPENAI_KEY"):
        alias = _norm(parsed.get("CODEX_API_KEY") or parsed.get("OPENAI_API_KEY"))
        if alias:
            env["OPENAI_KEY"] = alias

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
        str(float(timeout)),
        "--max-retries",
        "1",
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    subprocess.run(cmd, check=True, env=env)


def _build_verify_prompt(*, row: dict[str, Any], text: str) -> str:
    schema = {
        "page_id": "string",
        "contains_verbatim_zoning_law_language": "yes",
        "is_verbatim_legal_text": "yes",
        "legal_object_type": "full_ordinance",
        "page_completeness": "partial",
        "enactment_status_signal": "enacted",
        "confidence_0_to_1": 0.90,
        "evidence_quotes": ["short quote 1", "short quote 2"],
        "notes": "brief rationale",
    }
    lines: list[str] = []
    lines.append("Second-pass verification for precision on legal-page labeling.")
    lines.append("Return ONLY JSON.")
    lines.append("")
    lines.append("Goal: prevent false positives in full/amendment/proposal/notice for OCR newspaper pages.")
    lines.append("")
    lines.append("Prior model output (for context only; you may disagree):")
    lines.append(
        json.dumps(
            {
                "pred_legal_object_type": _norm(row.get("legal_object_type")),
                "pred_page_completeness": _norm(row.get("page_completeness")),
                "pred_bucket": _norm(row.get("pred_bucket")),
                "pred_enactment_status_signal": _norm(row.get("enactment_status_signal")),
                "pred_confidence": float(row.get("confidence_0_to_1") or 0.0),
                "pred_notes": _norm(row.get("notes")),
            },
            ensure_ascii=False,
        )
    )
    lines.append("")
    lines.append("Use one final class:")
    lines.extend(
        [
            "- full_ordinance",
            "- amendment",
            "- proposal",
            "- notice",
            "- building_code_or_other_law",
            "- zoning_narrative_nonverbatim",
            "- non_zoning",
            "- uncertain",
        ]
    )
    lines.append("")
    lines.append("Hard rules:")
    lines.append("- If legal text is building code / traffic / plumbing / non-zoning municipal law, choose building_code_or_other_law.")
    lines.append("- Do not return full_ordinance unless page has verbatim zoning ordinance clause language.")
    lines.append("- If proposal/hearing/pending adoption text dominates, use proposal.")
    lines.append("- If only meeting/reporting summary and no clause text, use zoning_narrative_nonverbatim.")
    lines.append("- If mixed and ambiguous, use uncertain.")
    lines.append("")
    lines.append("Return fields:")
    lines.append("- contains_verbatim_zoning_law_language: yes|no|uncertain")
    lines.append("- is_verbatim_legal_text: yes|no|uncertain")
    lines.append("- legal_object_type")
    lines.append("- page_completeness: partial|complete|uncertain")
    lines.append("- enactment_status_signal: enacted|proposed|unknown")
    lines.append("- confidence_0_to_1")
    lines.append("- evidence_quotes: 1-3 snippets")
    lines.append("- notes")
    lines.append("")
    lines.append("JSON schema example:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(
        f"Page metadata: page_id={_norm(row.get('page_id'))} issue_id={_norm(row.get('issue_id'))} issue_date={_norm(row.get('issue_date'))} newspaper_slug={_norm(row.get('newspaper_slug'))}"
    )
    lines.append("")
    lines.append("Page text:")
    lines.append(text)
    lines.append("")
    lines.append("Return JSON now.")
    return "\n".join(lines).strip()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run second-pass verifier for a base page-label variant.")
    ap.add_argument("--ab-dir", required=True)
    ap.add_argument("--base-variant", default="B")
    ap.add_argument("--model", default="gemini:gemini-2.5-flash")
    ap.add_argument("--core-buckets", default="full_ordinance,amendment,proposal_full_text,notice")
    ap.add_argument("--output-suffix", default="verify_gate")
    ap.add_argument(
        "--gateway-runner",
        default=str(Path(__file__).resolve().parents[1] / "scripts" / "run_openai_requests_via_gateway.py"),
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default=str(Path(__file__).resolve().parents[1] / "agent-gateway" / "src"),
    )
    ap.add_argument("--gov-env-path", default=".env")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-output-tokens", type=int, default=1200)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-run", action="store_true")
    return ap.parse_args()


def _bin_metrics(df: pd.DataFrame, target: str) -> dict[str, float]:
    y = (df["gold_eval"] == target).astype(int)
    p = (df["pred_eval"] == target).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "target": target,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
    }


def main() -> None:
    args = _parse_args()
    ab_dir = Path(args.ab_dir).expanduser().resolve()
    base_variant = _norm(args.base_variant).upper()
    joined_path = ab_dir / f"variant_{base_variant}_joined_predictions.csv"
    if not joined_path.is_file():
        raise SystemExit(f"Missing joined predictions: {joined_path}")
    sample_pages_path = ab_dir / "sample_pages.csv"
    if not sample_pages_path.is_file():
        raise SystemExit(f"Missing sample pages file: {sample_pages_path}")

    core_buckets = {_norm(x) for x in str(args.core_buckets).split(",") if _norm(x)}
    joined = pd.read_csv(joined_path)
    sample_pages = pd.read_csv(sample_pages_path)
    if "text_path" not in sample_pages.columns:
        raise SystemExit("sample_pages.csv missing text_path column.")
    joined = joined.merge(sample_pages[["page_id", "text_path"]], on="page_id", how="left")
    target = joined[joined["pred_eval"].isin(core_buckets)].copy()
    if target.empty:
        raise SystemExit("No target rows for verification.")

    out_dir = ab_dir / f"{base_variant}_{_norm(args.output_suffix)}"
    req_dir = out_dir / "requests"
    res_dir = out_dir / "results"
    req_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    req_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    for r in target.to_dict(orient="records"):
        text_path = Path(_norm(r.get("text_path"))).expanduser()
        if not text_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        page_id = _norm(r.get("page_id"))
        cid = f"verify::{base_variant}::{page_id}"
        prompt = _build_verify_prompt(row=r, text=text)
        req_rows.append(
            {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": str(args.model).split(":", 1)[1] if ":" in str(args.model) else str(args.model),
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                    "max_output_tokens": int(args.max_output_tokens),
                    "stream": False,
                },
            }
        )
        map_rows.append({"custom_id": cid, "page_id": page_id})
    if not req_rows:
        raise SystemExit("No verification requests generated (missing text_path matches).")

    _write_jsonl(req_dir / "openai_requests_shard000.jsonl", req_rows)
    _write_jsonl(req_dir / "mapping_shard000.jsonl", map_rows)

    if not args.no_run:
        _run_gateway(
            request_dir=req_dir,
            output_dir=res_dir,
            model=str(args.model),
            runner_path=Path(args.gateway_runner).expanduser().resolve(),
            gateway_pythonpath=Path(args.gateway_pythonpath).expanduser().resolve(),
            gov_env_path=Path(args.gov_env_path).expanduser().resolve(),
            timeout=float(args.timeout),
            concurrency=int(args.concurrency),
            skip_existing=bool(args.skip_existing),
        )

    result_map = _read_result_jsonl(res_dir)
    error_map = _read_error_jsonl(res_dir)

    parsed_rows: list[dict[str, Any]] = []
    for m in map_rows:
        cid = _norm(m.get("custom_id"))
        page_id = _norm(m.get("page_id"))
        output_text = ""
        had_error = 0
        if cid in result_map:
            body = (((result_map[cid].get("response") or {}).get("body")) or {})
            output_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in error_map:
            had_error = 1
            body = (((error_map[cid].get("response") or {}).get("body")) or {})
            if isinstance(body, dict):
                output_text = _norm(((body.get("error") or {}).get("message")))
        parsed = _parse_json_from_text(output_text)
        parse_ok = 1 if isinstance(parsed, dict) and parsed else 0
        obj = _coerce_result(parsed if isinstance(parsed, dict) else {}, page_id=page_id)
        obj["custom_id"] = cid
        obj["parse_ok"] = parse_ok
        obj["had_error"] = had_error
        parsed_rows.append(obj)

    verify_df = pd.DataFrame(parsed_rows)
    if verify_df.empty:
        verify_df = pd.DataFrame(
            columns=[
                "page_id",
                "legal_object_type",
                "page_completeness",
                "verify_bucket",
                "enactment_status_signal",
                "confidence_0_to_1",
                "notes",
                "parse_ok",
                "had_error",
            ]
        )
    else:
        verify_df["verify_bucket"] = verify_df["legal_object_type"].map(PRED_TO_BUCKET).fillna("uncertain_other")

    merged = joined.merge(
        verify_df[
            [
                "page_id",
                "legal_object_type",
                "page_completeness",
                "verify_bucket",
                "enactment_status_signal",
                "confidence_0_to_1",
                "notes",
                "parse_ok",
                "had_error",
            ]
        ],
        on="page_id",
        how="left",
        suffixes=("", "_verify"),
    )

    merged["pred_eval_verify"] = merged["pred_eval"]
    mask = merged["verify_bucket"].notna() & (merged["verify_bucket"].astype(str).str.len() > 0)
    merged.loc[mask, "pred_eval_verify"] = merged.loc[mask, "verify_bucket"]

    merged["is_match_base"] = merged["gold_eval"] == merged["pred_eval"]
    merged["is_match_verify"] = merged["gold_eval"] == merged["pred_eval_verify"]

    merged.to_csv(out_dir / "joined_with_verify.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    conf = (
        merged.groupby(["gold_eval", "pred_eval_verify"], dropna=False)["page_id"]
        .count()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    conf.to_csv(out_dir / "verify_confusion.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    summary = pd.DataFrame(
        [
            {
                "base_variant": base_variant,
                "n_total": int(len(merged)),
                "n_verify_rows": int(len(verify_df)),
                "base_accuracy_eval_bucket": float(merged["is_match_base"].mean()),
                "verify_accuracy_eval_bucket": float(merged["is_match_verify"].mean()),
                "verify_parse_ok_share": float(verify_df["parse_ok"].mean()) if len(verify_df) else math.nan,
                "verify_error_share": float(verify_df["had_error"].mean()) if len(verify_df) else math.nan,
            }
        ]
    )
    summary.to_csv(out_dir / "verify_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    metric_rows: list[dict[str, Any]] = []
    base_eval_df = merged.copy()
    verify_eval_df = merged.copy()
    verify_eval_df["pred_eval"] = verify_eval_df["pred_eval_verify"]
    for tgt in sorted(set(merged["gold_eval"])):
        m0 = _bin_metrics(base_eval_df, tgt)
        m0["mode"] = "base"
        m1 = _bin_metrics(verify_eval_df, tgt)
        m1["mode"] = "verify"
        metric_rows.extend([m0, m1])
    pd.DataFrame(metric_rows).to_csv(out_dir / "verify_binary_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ab_dir": str(ab_dir),
        "base_variant": base_variant,
        "model": str(args.model),
        "core_buckets": sorted(core_buckets),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
