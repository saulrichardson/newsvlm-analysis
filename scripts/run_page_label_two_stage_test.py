#!/usr/bin/env python3
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

ALLOWED_STAGE1 = {"zoning_legal", "other_legal", "zoning_narrative", "non_zoning", "uncertain"}
ALLOWED_STAGE2 = {"full_ordinance", "amendment", "proposal", "notice", "uncertain"}
ALLOWED_YNU = {"yes", "no", "uncertain"}
ALLOWED_ENACT = {"enacted", "proposed", "unknown"}


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


def _prompt_stage1(*, page_id: str, issue_id: str, issue_date: str, newspaper_slug: str, text: str) -> str:
    schema = {
        "page_id": "string",
        "coarse_class": "zoning_legal",
        "contains_verbatim_zoning_law_language": "yes",
        "is_verbatim_legal_text": "yes",
        "confidence_0_to_1": 0.9,
        "evidence_quotes": ["short quote 1", "short quote 2"],
        "notes": "brief rationale",
    }
    lines = [
        "Stage 1 (coarse gate): classify one OCR newspaper page.",
        "Return ONLY JSON.",
        "",
        "Choose coarse_class:",
        "- zoning_legal: verbatim zoning-law clauses (ordinance/amendment/proposal/notice text)",
        "- other_legal: verbatim legal text but not zoning (building/traffic/plumbing/other municipal law)",
        "- zoning_narrative: zoning reporting/context, not verbatim law",
        "- non_zoning: unrelated content",
        "- uncertain: ambiguous",
        "",
        "Hard rules:",
        "- Be conservative: if unsure between zoning_legal and other_legal, choose uncertain.",
        "- Narrative summaries are never zoning_legal.",
        "",
        "Also return:",
        "- contains_verbatim_zoning_law_language: yes|no|uncertain",
        "- is_verbatim_legal_text: yes|no|uncertain",
        "- confidence_0_to_1",
        "- evidence_quotes: 1-3 snippets",
        "- notes",
        "",
        "JSON schema example:",
        json.dumps(schema, ensure_ascii=False),
        "",
        f"Page metadata: page_id={page_id} issue_id={issue_id} issue_date={issue_date} newspaper_slug={newspaper_slug}",
        "",
        "Page text:",
        text,
        "",
        "Return JSON now.",
    ]
    return "\n".join(lines).strip()


def _prompt_stage2(*, page_id: str, issue_id: str, issue_date: str, newspaper_slug: str, text: str, stage1_obj: dict[str, Any]) -> str:
    schema = {
        "page_id": "string",
        "zoning_legal_type": "full_ordinance",
        "page_completeness": "partial",
        "enactment_status_signal": "enacted",
        "confidence_0_to_1": 0.9,
        "evidence_quotes": ["short quote 1", "short quote 2"],
        "notes": "brief rationale",
    }
    lines = [
        "Stage 2 (fine typing): classify zoning legal page type from OCR text.",
        "Return ONLY JSON.",
        "",
        "Stage 1 output (context):",
        json.dumps(stage1_obj, ensure_ascii=False),
        "",
        "Choose zoning_legal_type:",
        "- full_ordinance",
        "- amendment",
        "- proposal",
        "- notice",
        "- uncertain",
        "",
        "Rules:",
        "- full_ordinance only for verbatim zoning ordinance clauses (not proposal-only notices).",
        "- amendment only for verbatim amendment clauses.",
        "- proposal for proposed/pending/hearing text.",
        "- notice for procedural legal notices with limited rule text.",
        "- if mixed/ambiguous, uncertain.",
        "",
        "Return fields:",
        "- zoning_legal_type",
        "- page_completeness: partial|complete|uncertain",
        "- enactment_status_signal: enacted|proposed|unknown",
        "- confidence_0_to_1",
        "- evidence_quotes",
        "- notes",
        "",
        "JSON schema example:",
        json.dumps(schema, ensure_ascii=False),
        "",
        f"Page metadata: page_id={page_id} issue_id={issue_id} issue_date={issue_date} newspaper_slug={newspaper_slug}",
        "",
        "Page text:",
        text,
        "",
        "Return JSON now.",
    ]
    return "\n".join(lines).strip()


def _bin_metrics(df: pd.DataFrame, pred_col: str, target: str) -> dict[str, float]:
    y = (df["gold_eval"] == target).astype(int)
    p = (df[pred_col] == target).astype(int)
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


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Two-stage page classification test on existing 100-page sample.")
    ap.add_argument("--ab-dir", required=True, help="Existing AB dir containing sample_pages.csv + variant_B_joined_predictions.csv")
    ap.add_argument("--model", default="gemini:gemini-2.5-flash")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--base-variant", default="B")
    ap.add_argument(
        "--gateway-runner",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/experimental/scripts/run_openai_requests_via_gateway.py",
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/agent-gateway/src",
    )
    ap.add_argument("--gov-env-path", default=".env")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-output-tokens", type=int, default=1000)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    a = _args()
    ab_dir = Path(a.ab_dir).expanduser().resolve()
    base_variant = _norm(a.base_variant).upper()
    sample_path = ab_dir / "sample_pages.csv"
    base_joined = ab_dir / f"variant_{base_variant}_joined_predictions.csv"
    if not sample_path.is_file() or not base_joined.is_file():
        raise SystemExit("Missing sample_pages.csv or base variant joined predictions.")

    out_dir = Path(a.output_dir).expanduser().resolve() if _norm(a.output_dir) else (ab_dir / "two_stage_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(sample_path)
    base = pd.read_csv(base_joined)
    df = base.merge(sample[["page_id", "text_path"]], on="page_id", how="left")

    # Stage 1 requests (all pages)
    s1_req_dir = out_dir / "stage1" / "requests"
    s1_res_dir = out_dir / "stage1" / "results"
    s1_reqs: list[dict[str, Any]] = []
    s1_map: list[dict[str, Any]] = []
    for r in df.to_dict(orient="records"):
        text_path = Path(_norm(r.get("text_path"))).expanduser()
        if not text_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        page_id = _norm(r.get("page_id"))
        cid = f"s1::{page_id}"
        prompt = _prompt_stage1(
            page_id=page_id,
            issue_id=_norm(r.get("issue_id")),
            issue_date=_norm(r.get("issue_date")),
            newspaper_slug=_norm(r.get("newspaper_slug")),
            text=text,
        )
        s1_reqs.append(
            {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": str(a.model).split(":", 1)[1] if ":" in str(a.model) else str(a.model),
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                    "max_output_tokens": int(a.max_output_tokens),
                    "stream": False,
                },
            }
        )
        s1_map.append({"custom_id": cid, "page_id": page_id, "text_path": str(text_path)})
    _write_jsonl(s1_req_dir / "openai_requests_shard000.jsonl", s1_reqs)
    _write_jsonl(s1_req_dir / "mapping_shard000.jsonl", s1_map)
    _run_gateway(
        request_dir=s1_req_dir,
        output_dir=s1_res_dir,
        model=str(a.model),
        runner_path=Path(a.gateway_runner).expanduser().resolve(),
        gateway_pythonpath=Path(a.gateway_pythonpath).expanduser().resolve(),
        gov_env_path=Path(a.gov_env_path).expanduser().resolve(),
        timeout=float(a.timeout),
        concurrency=int(a.concurrency),
        skip_existing=bool(a.skip_existing),
    )

    s1_res_map = _read_result_jsonl(s1_res_dir)
    s1_err_map = _read_error_jsonl(s1_res_dir)
    s1_rows: list[dict[str, Any]] = []
    for m in s1_map:
        cid = _norm(m["custom_id"])
        page_id = _norm(m["page_id"])
        output_text = ""
        had_error = 0
        if cid in s1_res_map:
            body = (((s1_res_map[cid].get("response") or {}).get("body")) or {})
            output_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in s1_err_map:
            had_error = 1
            body = (((s1_err_map[cid].get("response") or {}).get("body")) or {})
            if isinstance(body, dict):
                output_text = _norm(((body.get("error") or {}).get("message")))
        parsed = _parse_json_from_text(output_text)
        coarse = _norm((parsed or {}).get("coarse_class")).lower()
        if coarse not in ALLOWED_STAGE1:
            coarse = "uncertain"
        contains = _norm((parsed or {}).get("contains_verbatim_zoning_law_language")).lower()
        if contains not in ALLOWED_YNU:
            contains = "uncertain"
        verbatim = _norm((parsed or {}).get("is_verbatim_legal_text")).lower()
        if verbatim not in ALLOWED_YNU:
            verbatim = "uncertain"
        try:
            conf = float((parsed or {}).get("confidence_0_to_1", 0.0))
        except Exception:
            conf = 0.0
        s1_rows.append(
            {
                "page_id": page_id,
                "coarse_class": coarse,
                "contains_verbatim_zoning_law_language": contains,
                "is_verbatim_legal_text": verbatim,
                "confidence_s1": max(0.0, min(1.0, conf)),
                "notes_s1": _norm((parsed or {}).get("notes")),
                "parse_ok_s1": 1 if isinstance(parsed, dict) else 0,
                "had_error_s1": had_error,
            }
        )
    s1_df = pd.DataFrame(s1_rows)
    s1_df.to_csv(out_dir / "stage1_outputs.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # Stage 2 requests only for zoning_legal
    s1m = df.merge(s1_df, on="page_id", how="left")
    s2_target = s1m[s1m["coarse_class"] == "zoning_legal"].copy()
    s2_req_dir = out_dir / "stage2" / "requests"
    s2_res_dir = out_dir / "stage2" / "results"
    s2_reqs: list[dict[str, Any]] = []
    s2_map: list[dict[str, Any]] = []
    for r in s2_target.to_dict(orient="records"):
        text_path = Path(_norm(r.get("text_path"))).expanduser()
        if not text_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        page_id = _norm(r.get("page_id"))
        cid = f"s2::{page_id}"
        stage1_obj = {
            "coarse_class": _norm(r.get("coarse_class")),
            "contains_verbatim_zoning_law_language": _norm(r.get("contains_verbatim_zoning_law_language")),
            "is_verbatim_legal_text": _norm(r.get("is_verbatim_legal_text")),
            "confidence": float(r.get("confidence_s1") or 0.0),
            "notes": _norm(r.get("notes_s1")),
        }
        prompt = _prompt_stage2(
            page_id=page_id,
            issue_id=_norm(r.get("issue_id")),
            issue_date=_norm(r.get("issue_date")),
            newspaper_slug=_norm(r.get("newspaper_slug")),
            text=text,
            stage1_obj=stage1_obj,
        )
        s2_reqs.append(
            {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": str(a.model).split(":", 1)[1] if ":" in str(a.model) else str(a.model),
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                    "max_output_tokens": int(a.max_output_tokens),
                    "stream": False,
                },
            }
        )
        s2_map.append({"custom_id": cid, "page_id": page_id})
    _write_jsonl(s2_req_dir / "openai_requests_shard000.jsonl", s2_reqs)
    _write_jsonl(s2_req_dir / "mapping_shard000.jsonl", s2_map)
    if s2_reqs:
        _run_gateway(
            request_dir=s2_req_dir,
            output_dir=s2_res_dir,
            model=str(a.model),
            runner_path=Path(a.gateway_runner).expanduser().resolve(),
            gateway_pythonpath=Path(a.gateway_pythonpath).expanduser().resolve(),
            gov_env_path=Path(a.gov_env_path).expanduser().resolve(),
            timeout=float(a.timeout),
            concurrency=int(a.concurrency),
            skip_existing=bool(a.skip_existing),
        )

    s2_res_map = _read_result_jsonl(s2_res_dir)
    s2_err_map = _read_error_jsonl(s2_res_dir)
    s2_rows: list[dict[str, Any]] = []
    for m in s2_map:
        cid = _norm(m["custom_id"])
        page_id = _norm(m["page_id"])
        output_text = ""
        had_error = 0
        if cid in s2_res_map:
            body = (((s2_res_map[cid].get("response") or {}).get("body")) or {})
            output_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
        elif cid in s2_err_map:
            had_error = 1
            body = (((s2_err_map[cid].get("response") or {}).get("body")) or {})
            if isinstance(body, dict):
                output_text = _norm(((body.get("error") or {}).get("message")))
        parsed = _parse_json_from_text(output_text)
        zt = _norm((parsed or {}).get("zoning_legal_type")).lower()
        if zt not in ALLOWED_STAGE2:
            zt = "uncertain"
        enact = _norm((parsed or {}).get("enactment_status_signal")).lower()
        if enact not in ALLOWED_ENACT:
            enact = "unknown"
        page_completeness = _norm((parsed or {}).get("page_completeness")).lower()
        if page_completeness not in {"partial", "complete", "uncertain"}:
            page_completeness = "uncertain"
        try:
            conf = float((parsed or {}).get("confidence_0_to_1", 0.0))
        except Exception:
            conf = 0.0
        s2_rows.append(
            {
                "page_id": page_id,
                "zoning_legal_type": zt,
                "page_completeness_s2": page_completeness,
                "enactment_status_signal_s2": enact,
                "confidence_s2": max(0.0, min(1.0, conf)),
                "notes_s2": _norm((parsed or {}).get("notes")),
                "parse_ok_s2": 1 if isinstance(parsed, dict) else 0,
                "had_error_s2": had_error,
            }
        )
    s2_df = pd.DataFrame(s2_rows)
    s2_df.to_csv(out_dir / "stage2_outputs.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # Final mapping
    merged = s1m.merge(s2_df, on="page_id", how="left")

    def _final_legal_object_type(r: pd.Series) -> str:
        coarse = _norm(r.get("coarse_class")).lower()
        if coarse == "other_legal":
            return "building_code_or_other_law"
        if coarse == "zoning_narrative":
            return "zoning_narrative_nonverbatim"
        if coarse == "non_zoning":
            return "non_zoning"
        if coarse == "uncertain":
            return "uncertain"
        # zoning_legal -> use stage2
        zt = _norm(r.get("zoning_legal_type")).lower()
        if zt == "full_ordinance":
            return "full_ordinance"
        if zt == "amendment":
            return "amendment"
        if zt == "proposal":
            return "proposal"
        if zt == "notice":
            return "notice"
        return "uncertain"

    def _final_page_completeness(r: pd.Series) -> str:
        legal_object_type = _norm(r.get("pred_legal_object_type_two_stage")).lower()
        if legal_object_type in {"non_zoning", "zoning_narrative_nonverbatim", "building_code_or_other_law", "uncertain"}:
            return "uncertain"
        val = _norm(r.get("page_completeness_s2")).lower()
        if val in {"partial", "complete", "uncertain"}:
            return val
        return "uncertain"

    merged["pred_legal_object_type_two_stage"] = merged.apply(_final_legal_object_type, axis=1)
    merged["pred_page_completeness_two_stage"] = merged.apply(_final_page_completeness, axis=1)
    merged["pred_eval_two_stage"] = merged["pred_legal_object_type_two_stage"].map(PRED_TO_BUCKET).fillna("uncertain_other")
    merged["is_match_two_stage"] = merged["gold_eval"] == merged["pred_eval_two_stage"]

    merged.to_csv(out_dir / "two_stage_joined_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    conf = (
        merged.groupby(["gold_eval", "pred_eval_two_stage"], dropna=False)["page_id"]
        .count()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    conf.to_csv(out_dir / "two_stage_confusion.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "n_pages": int(len(merged)),
        "stage1_zoning_legal_rows": int((merged["coarse_class"] == "zoning_legal").sum()),
        "stage1_parse_ok_share": float(merged["parse_ok_s1"].fillna(0).mean()),
        "stage2_parse_ok_share": float(merged["parse_ok_s2"].fillna(0).mean()) if len(s2_df) else math.nan,
        "accuracy_eval_bucket_two_stage": float(merged["is_match_two_stage"].mean()),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "two_stage_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    metric_rows = []
    for tgt in sorted(set(merged["gold_eval"])):
        m = _bin_metrics(merged, "pred_eval_two_stage", tgt)
        metric_rows.append(m)
    pd.DataFrame(metric_rows).to_csv(out_dir / "two_stage_binary_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # compare against base variant
    merged["is_match_base"] = merged["gold_eval"] == merged["pred_eval"]
    cmp = pd.DataFrame(
        [
            {
                "base_variant": base_variant,
                "base_accuracy_eval_bucket": float(merged["is_match_base"].mean()),
                "two_stage_accuracy_eval_bucket": float(merged["is_match_two_stage"].mean()),
            }
        ]
    )
    cmp.to_csv(out_dir / "compare_vs_base.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ab_dir": str(ab_dir),
        "base_variant": base_variant,
        "model": str(a.model),
        "n_stage1_requests": len(s1_reqs),
        "n_stage2_requests": len(s2_reqs),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
