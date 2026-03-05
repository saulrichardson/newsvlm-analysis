#!/usr/bin/env python3
"""
Run A/B issue-classification prompts on the same issue set and compare outcomes.

Variant A: current issue-level prompt (baseline).
Variant B: stricter ontology prompt for zoning presence + legal document subtype.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


_ALLOWED_A_LABELS = {
    "full_ordinance",
    "amendment_substantial",
    "amendment_targeted",
    "public_hearing",
    "unrelated",
}
_ALLOWED_B_PRIMARY = {
    "unrelated",
    "zoning_discussion",
    "zoning_hearing_notice",
    "zoning_legal_text",
}
_ALLOWED_B_DOC_CLASS = {
    "none",
    "full_code_comprehensive",
    "full_code_recodification",
    "comprehensive_code_excerpt",
    "amendment_substantial_text",
    "amendment_targeted_text",
    "map_rezoning_order",
    "variance_special_use_order",
    "procedural_notice_only",
    "non_zoning_ordinance",
    "uncertain",
}
_ALLOWED_B_LEGACY = _ALLOWED_A_LABELS
_FULL_DOC_CLASSES = {"full_code_comprehensive", "full_code_recodification"}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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


def _inject_provider_envs(env: dict[str, str], env_file_vars: dict[str, str]) -> None:
    # Preserve all keys from env file to help custom gateway behavior.
    for k, v in env_file_vars.items():
        if v and k not in env:
            env[k] = v

    # Primary aliases expected by provider clients.
    if not env.get("OPENAI_API_KEY"):
        for k in ("OPENAI_KEY", "OPENAI_API_KEY", "OPENAI_KEY_BACKUP", "OPENAI_API_KEY_BACKUP"):
            if env.get(k):
                env["OPENAI_API_KEY"] = env[k]
                break
    if not env.get("GEMINI_API_KEY"):
        for k in ("GEMINI_KEY", "GEMINI_API_KEY", "GEMINI_KEY_BACKUP", "GEMINI_API_KEY_BACKUP"):
            if env.get(k):
                env["GEMINI_API_KEY"] = env[k]
                break
    if not env.get("ANTHROPIC_API_KEY"):
        for k in ("CLAUDE_KEY", "ANTHROPIC_API_KEY", "CLAUDE_KEY_BACKUP", "ANTHROPIC_API_KEY_BACKUP"):
            if env.get(k):
                env["ANTHROPIC_API_KEY"] = env[k]
                break


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


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    s = _norm_str(text)
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)

    def _try(raw: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        if isinstance(obj, dict):
            return obj
        return None

    obj = _try(s)
    if obj is not None:
        return obj

    first = s.find("{")
    if first == -1:
        return None
    candidate = s[first:]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    repaired = _repair_truncated_json(candidate)
    return _try(repaired)


def _make_openai_request(custom_id: str, prompt: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-5-nano",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _build_variant_prompt(
    *,
    variant: str,
    prompt_text: str,
    row: pd.Series,
    issue_text: str,
) -> str:
    issue_id = _norm_str(row.get("issue_id"))
    issue_date = _norm_str(row.get("issue_date"))
    city_name = _norm_str(row.get("city_name"))
    state_abbr = _norm_str(row.get("state_abbr")).upper()
    legacy = _norm_str(row.get("classification_label"))
    tchars = _safe_int(row.get("text_chars_actual"), default=len(issue_text))

    if variant == "A":
        return (
            f"{prompt_text}\n\n"
            "ISSUE_SCOPE_NOTE:\n"
            "This issue is provided as one merged pseudo-page for this experiment.\n"
            f"If label is not unrelated, pages_with_zoning should reference PAGE 1 ({issue_id}__alltext).\n\n"
            "ISSUE_PAGES:\n"
            f"PAGE 1 ({issue_id}__alltext):\n{issue_text}\n"
        )

    return (
        f"{prompt_text}\n\n"
        "ISSUE_METADATA:\n"
        f"- issue_id: {issue_id}\n"
        f"- issue_date: {issue_date}\n"
        f"- city: {city_name}, {state_abbr}\n"
        f"- legacy_label_for_context_only: {legacy}\n"
        f"- issue_text_chars: {tchars}\n\n"
        "ISSUE_TEXT:\n"
        f"{issue_text}\n"
    )


def _run_gateway_batch(
    *,
    request_dir: Path,
    output_dir: Path,
    runner_path: Path,
    model: str,
    concurrency: int,
    timeout: float,
    max_retries: int,
    skip_existing: bool,
    gateway_pythonpath: Path | None,
    gov_env_path: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

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
        str(int(max_retries)),
    ]
    if skip_existing:
        cmd.append("--skip-existing")

    env = os.environ.copy()
    file_vars = _parse_env_file(gov_env_path)
    _inject_provider_envs(env, file_vars)
    if gateway_pythonpath:
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{gateway_pythonpath}:{old}" if old else str(gateway_pythonpath)

    subprocess.run(cmd, check=True, env=env)


def _collect_results(result_dir: Path, variant: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in sorted(result_dir.glob("openai_results_shard*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                cid = _norm_str(obj.get("custom_id"))
                if not cid:
                    continue
                parts = cid.split("::", 2)
                if len(parts) != 3:
                    continue
                _, v, issue_id = parts
                if v != variant:
                    continue
                body = obj.get("response", {}).get("body", {})
                out_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
                parsed = _parse_json_from_text(out_text)
                rows.append(
                    {
                        "issue_id": issue_id,
                        "custom_id": cid,
                        "raw_text": out_text,
                        "parsed_json": parsed,
                        "parse_ok": int(parsed is not None),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["issue_id", "custom_id", "raw_text", "parsed_json", "parse_ok"])
    return pd.DataFrame.from_records(rows)


def _normalize_variant_a(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for r in df.itertuples(index=False):
        obj = r.parsed_json if isinstance(r.parsed_json, dict) else {}
        label = _enum(obj.get("label"), _ALLOWED_A_LABELS, "unrelated")
        conf = _safe_float(obj.get("confidence"), default=math.nan)
        present = obj.get("present") if isinstance(obj.get("present"), dict) else {}
        out_rows.append(
            {
                "issue_id": r.issue_id,
                "a_parse_ok": int(r.parse_ok),
                "a_label": label,
                "a_confidence": conf,
                "a_present_full": int(bool(present.get("full_ordinance"))),
                "a_present_amend_substantial": int(bool(present.get("amendment_substantial"))),
                "a_present_amend_targeted": int(bool(present.get("amendment_targeted"))),
                "a_present_public_hearing": int(bool(present.get("public_hearing"))),
                "a_pred_full": int(label == "full_ordinance"),
                "a_pred_notice": int(label == "public_hearing"),
                "a_pred_nonzoning": int(label == "unrelated"),
                "a_raw_text": _norm_str(r.raw_text),
            }
        )
    return pd.DataFrame.from_records(out_rows)


def _normalize_variant_b(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for r in df.itertuples(index=False):
        obj = r.parsed_json if isinstance(r.parsed_json, dict) else {}
        zp = obj.get("zoning_presence") if isinstance(obj.get("zoning_presence"), dict) else {}
        ldc = obj.get("legal_document_class") if isinstance(obj.get("legal_document_class"), dict) else {}
        comps = obj.get("full_code_components_present") if isinstance(obj.get("full_code_components_present"), dict) else {}
        legacy = _enum(obj.get("legacy_projection_label"), _ALLOWED_B_LEGACY, "unrelated")
        primary = _enum(zp.get("primary_class"), _ALLOWED_B_PRIMARY, "unrelated")
        doc_class = _enum(ldc.get("class"), _ALLOWED_B_DOC_CLASS, "uncertain")
        has_any = bool(zp.get("has_any_zoning_reference"))
        comp_count = int(bool(comps.get("district_framework"))) + int(bool(comps.get("use_rules"))) + int(
            bool(comps.get("dimensional_standards"))
        ) + int(bool(comps.get("procedures_enforcement_admin")))

        topics = obj.get("topic_touches")
        if isinstance(topics, list):
            topic_count = int(sum(1 for t in topics if _norm_str(t)))
        else:
            topic_count = 0

        pred_full = int(doc_class in _FULL_DOC_CLASSES)
        pred_notice = int(
            primary == "zoning_hearing_notice"
            or doc_class == "procedural_notice_only"
            or legacy == "public_hearing"
        )
        pred_nonzoning = int(doc_class == "non_zoning_ordinance" or primary == "unrelated" or legacy == "unrelated")

        out_rows.append(
            {
                "issue_id": r.issue_id,
                "b_parse_ok": int(r.parse_ok),
                "b_has_any_zoning_reference": int(has_any),
                "b_primary_class": primary,
                "b_primary_confidence": _safe_float(zp.get("confidence_0_to_1"), default=math.nan),
                "b_doc_class": doc_class,
                "b_doc_confidence": _safe_float(ldc.get("confidence_0_to_1"), default=math.nan),
                "b_doc_scope": _enum(ldc.get("scope"), {"none", "citywide", "district_wide", "parcel_specific", "mixed", "unknown"}, "unknown"),
                "b_doc_operativity": _enum(ldc.get("operativity"), {"none", "operative", "proposed", "unclear"}, "unclear"),
                "b_component_count": comp_count,
                "b_legacy_projection_label": legacy,
                "b_topic_count": topic_count,
                "b_pred_full": pred_full,
                "b_pred_notice": pred_notice,
                "b_pred_nonzoning": pred_nonzoning,
                "b_raw_text": _norm_str(r.raw_text),
            }
        )
    return pd.DataFrame.from_records(out_rows)


def _binary_metrics(df: pd.DataFrame, truth_col: str, pred_col: str) -> dict[str, Any]:
    d = df[[truth_col, pred_col]].copy()
    d[truth_col] = d[truth_col].astype(int)
    d[pred_col] = d[pred_col].astype(int)
    tp = int(((d[truth_col] == 1) & (d[pred_col] == 1)).sum())
    fp = int(((d[truth_col] == 0) & (d[pred_col] == 1)).sum())
    fn = int(((d[truth_col] == 1) & (d[pred_col] == 0)).sum())
    tn = int(((d[truth_col] == 0) & (d[pred_col] == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else math.nan
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else math.nan
    return {
        "n": int(len(d)),
        "positives": int(d[truth_col].sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run zoning issue classification A/B prompt experiment.")
    ap.add_argument(
        "--dataset-csv",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/ontology_ab/ab_diagnostic_dataset_full.csv",
    )
    ap.add_argument("--dataset-split", default="", help="Optional diagnostic_split filter (e.g., pilot60 or full_run).")
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper/ontology_ab/ab_runs/default",
    )
    ap.add_argument(
        "--prompt-a-path",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/prompts/zoning_issue_classifier_prompt_text.txt",
    )
    ap.add_argument(
        "--prompt-b-path",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/prompts/zoning_issue_classifier_prompt_v2_ontology.txt",
    )
    ap.add_argument("--model", default="gemini:gemini-2.5-flash")
    ap.add_argument(
        "--max-issue-chars",
        type=int,
        default=120000,
        help="Hard cap for issue text included in prompts. 0 means no local cap. If positive and exceeded, the run fails (no clipping).",
    )
    ap.add_argument("--max-concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--gateway-runner",
        default=str(Path(__file__).resolve().parents[1] / "scripts" / "run_openai_requests_via_gateway.py"),
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default=str(Path(__file__).resolve().parents[1] / "agent-gateway" / "src"),
    )
    ap.add_argument("--gov-env-path", default=".env")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    dataset_path = Path(args.dataset_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_a_path = Path(args.prompt_a_path).expanduser().resolve()
    prompt_b_path = Path(args.prompt_b_path).expanduser().resolve()
    runner_path = Path(args.gateway_runner).expanduser().resolve()
    gateway_pythonpath = Path(args.gateway_pythonpath).expanduser().resolve() if _norm_str(args.gateway_pythonpath) else None
    gov_env_path = Path(args.gov_env_path).expanduser().resolve()

    if not dataset_path.is_file():
        raise SystemExit(f"Missing dataset: {dataset_path}")
    if not prompt_a_path.is_file():
        raise SystemExit(f"Missing prompt A: {prompt_a_path}")
    if not prompt_b_path.is_file():
        raise SystemExit(f"Missing prompt B: {prompt_b_path}")
    if not runner_path.is_file():
        raise SystemExit(f"Missing gateway runner: {runner_path}")

    ds = pd.read_csv(dataset_path)
    if _norm_str(args.dataset_split):
        ds = ds[ds["diagnostic_split"].astype(str) == _norm_str(args.dataset_split)].copy()
    if ds.empty:
        raise SystemExit("Dataset is empty after filtering.")
    ds = ds.drop_duplicates(subset=["issue_id"]).reset_index(drop=True)

    prompt_a = prompt_a_path.read_text(encoding="utf-8").strip()
    prompt_b = prompt_b_path.read_text(encoding="utf-8").strip()

    req_rows_a: list[dict[str, Any]] = []
    req_rows_b: list[dict[str, Any]] = []
    build_rows: list[dict[str, Any]] = []
    for r in ds.itertuples(index=False):
        issue_id = _norm_str(getattr(r, "issue_id", ""))
        issue_text = _norm_str(getattr(r, "text", ""))
        if not issue_id or not issue_text:
            continue
        max_issue_chars = int(args.max_issue_chars)
        if max_issue_chars > 0 and len(issue_text) > max_issue_chars:
            raise SystemExit(
                f"issue_id={issue_id} chars={len(issue_text)} exceeds hard cap {max_issue_chars}; no clipping allowed. "
                "Re-run with --max-issue-chars 0 (no cap) or increase the cap."
            )

        row_s = pd.Series(r._asdict())
        pa = _build_variant_prompt(variant="A", prompt_text=prompt_a, row=row_s, issue_text=issue_text)
        pb = _build_variant_prompt(variant="B", prompt_text=prompt_b, row=row_s, issue_text=issue_text)

        cid_a = f"ab::A::{issue_id}"
        cid_b = f"ab::B::{issue_id}"
        req_rows_a.append(_make_openai_request(cid_a, pa))
        req_rows_b.append(_make_openai_request(cid_b, pb))
        build_rows.append(
            {
                "issue_id": issue_id,
                "city_key": _norm_str(getattr(r, "city_key", "")),
                "classification_label": _norm_str(getattr(r, "classification_label", "")),
                "manual_bucket": _norm_str(getattr(r, "manual_bucket", "")),
                "text_chars_actual": _safe_int(getattr(r, "text_chars_actual", 0)),
                "text_was_clipped": 0,
                "prompt_chars_a": len(pa),
                "prompt_chars_b": len(pb),
            }
        )

    build_df = pd.DataFrame.from_records(build_rows)
    _write_csv(out_dir / "ab_issue_build_manifest.csv", build_df)

    # Write request shards.
    for variant, req_rows in (("A", req_rows_a), ("B", req_rows_b)):
        req_dir = out_dir / f"variant_{variant}" / "requests"
        _write_jsonl(req_dir / "openai_requests_shard000.jsonl", req_rows)

    # Run variants.
    for variant in ("A", "B"):
        req_dir = out_dir / f"variant_{variant}" / "requests"
        res_dir = out_dir / f"variant_{variant}" / "results"
        _eprint(f"Running variant {variant}: requests={req_dir} results={res_dir}")
        _run_gateway_batch(
            request_dir=req_dir,
            output_dir=res_dir,
            runner_path=runner_path,
            model=str(args.model),
            concurrency=int(args.max_concurrency),
            timeout=float(args.timeout),
            max_retries=int(args.max_retries),
            skip_existing=bool(args.skip_existing),
            gateway_pythonpath=gateway_pythonpath,
            gov_env_path=gov_env_path,
            dry_run=bool(args.dry_run),
        )

    # Parse outputs.
    raw_a = _collect_results(out_dir / "variant_A" / "results", variant="A")
    raw_b = _collect_results(out_dir / "variant_B" / "results", variant="B")
    _write_csv(out_dir / "variant_A_raw_results.csv", raw_a)
    _write_csv(out_dir / "variant_B_raw_results.csv", raw_b)

    norm_a = _normalize_variant_a(raw_a) if not raw_a.empty else pd.DataFrame(columns=["issue_id"])
    norm_b = _normalize_variant_b(raw_b) if not raw_b.empty else pd.DataFrame(columns=["issue_id"])
    merged = ds.merge(norm_a, on="issue_id", how="left").merge(norm_b, on="issue_id", how="left")

    for c in ("a_parse_ok", "b_parse_ok", "a_pred_full", "b_pred_full", "a_pred_notice", "b_pred_notice", "a_pred_nonzoning", "b_pred_nonzoning"):
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)

    _write_csv(out_dir / "ab_outputs_merged.csv", merged)

    # Variant-level summary.
    summary_rows = [
        {
            "variant": "A",
            "n_issues": int(len(merged)),
            "parse_ok_share": float(merged["a_parse_ok"].mean()) if "a_parse_ok" in merged.columns else math.nan,
            "pred_full_share": float(merged["a_pred_full"].mean()) if "a_pred_full" in merged.columns else math.nan,
            "pred_notice_share": float(merged["a_pred_notice"].mean()) if "a_pred_notice" in merged.columns else math.nan,
            "pred_nonzoning_share": float(merged["a_pred_nonzoning"].mean()) if "a_pred_nonzoning" in merged.columns else math.nan,
        },
        {
            "variant": "B",
            "n_issues": int(len(merged)),
            "parse_ok_share": float(merged["b_parse_ok"].mean()) if "b_parse_ok" in merged.columns else math.nan,
            "pred_full_share": float(merged["b_pred_full"].mean()) if "b_pred_full" in merged.columns else math.nan,
            "pred_notice_share": float(merged["b_pred_notice"].mean()) if "b_pred_notice" in merged.columns else math.nan,
            "pred_nonzoning_share": float(merged["b_pred_nonzoning"].mean()) if "b_pred_nonzoning" in merged.columns else math.nan,
        },
    ]
    summary_df = pd.DataFrame.from_records(summary_rows)
    _write_csv(out_dir / "ab_variant_summary.csv", summary_df)

    # Label distribution tables.
    if "a_label" in merged.columns:
        a_dist = merged["a_label"].astype(str).value_counts(dropna=False).rename_axis("a_label").reset_index(name="count")
        _write_csv(out_dir / "ab_variant_a_label_distribution.csv", a_dist)
    if "b_doc_class" in merged.columns:
        b_dist = merged["b_doc_class"].astype(str).value_counts(dropna=False).rename_axis("b_doc_class").reset_index(name="count")
        _write_csv(out_dir / "ab_variant_b_doc_class_distribution.csv", b_dist)
    if "b_primary_class" in merged.columns:
        bprim_dist = (
            merged["b_primary_class"].astype(str).value_counts(dropna=False).rename_axis("b_primary_class").reset_index(name="count")
        )
        _write_csv(out_dir / "ab_variant_b_primary_distribution.csv", bprim_dist)

    # Manual-bucket metrics.
    manual = merged[merged["manual_bucket"].notna() & (merged["manual_bucket"].astype(str).str.strip() != "")].copy()
    metric_rows: list[dict[str, Any]] = []
    if not manual.empty:
        manual["truth_full"] = (manual["manual_bucket"] == "likely_full_zoning_ordinance").astype(int)
        manual["truth_notice"] = (manual["manual_bucket"] == "likely_zoning_notice_hearing").astype(int)
        manual["truth_nonzoning"] = (manual["manual_bucket"] == "likely_non_zoning_notice_or_ordinance").astype(int)
        manual["truth_nonfull"] = (manual["manual_bucket"] != "likely_full_zoning_ordinance").astype(int)

        for variant, pred_full, pred_notice, pred_nonzoning in (
            ("A", "a_pred_full", "a_pred_notice", "a_pred_nonzoning"),
            ("B", "b_pred_full", "b_pred_notice", "b_pred_nonzoning"),
        ):
            if pred_full not in manual.columns:
                continue
            m_full = _binary_metrics(manual, "truth_full", pred_full)
            m_notice = _binary_metrics(manual, "truth_notice", pred_notice)
            m_nonzoning = _binary_metrics(manual, "truth_nonzoning", pred_nonzoning)
            nonfull = manual[manual["truth_nonfull"] == 1]
            ff_rate = float(nonfull[pred_full].mean()) if not nonfull.empty else math.nan
            metric_rows.extend(
                [
                    {"variant": variant, "target": "full_code", **m_full, "nonfull_false_full_rate": ff_rate},
                    {"variant": variant, "target": "hearing_notice", **m_notice, "nonfull_false_full_rate": ff_rate},
                    {"variant": variant, "target": "non_zoning", **m_nonzoning, "nonfull_false_full_rate": ff_rate},
                ]
            )
    metric_df = pd.DataFrame.from_records(metric_rows)
    _write_csv(out_dir / "ab_manual_bucket_metrics.csv", metric_df)

    # Disagreement views.
    dis_cols = [
        "issue_id",
        "city_key",
        "city_name",
        "state_abbr",
        "issue_date",
        "classification_label",
        "manual_bucket",
        "a_label",
        "b_primary_class",
        "b_doc_class",
        "b_legacy_projection_label",
        "a_pred_full",
        "b_pred_full",
    ]
    for c in dis_cols:
        if c not in merged.columns:
            merged[c] = ""
    dis = merged[(merged["a_pred_full"] != merged["b_pred_full"]) | (merged["a_label"] != merged["b_legacy_projection_label"])].copy()
    dis = dis.sort_values(["manual_bucket", "issue_date", "issue_id"])
    _write_csv(out_dir / "ab_disagreements_full_vs_legacy.csv", dis[dis_cols])

    # Keep short text excerpts for manual follow-up.
    ex = dis[["issue_id", "text"]].copy() if "text" in dis.columns else pd.DataFrame(columns=["issue_id", "text"])
    if not ex.empty:
        ex["text_excerpt"] = ex["text"].astype(str).str.slice(0, 1200)
        _write_csv(out_dir / "ab_disagreements_excerpts.csv", ex[["issue_id", "text_excerpt"]])

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_csv": str(dataset_path),
        "dataset_rows": int(len(ds)),
        "dataset_split": _norm_str(args.dataset_split),
        "prompt_a_path": str(prompt_a_path),
        "prompt_b_path": str(prompt_b_path),
        "model": str(args.model),
        "max_issue_chars": int(args.max_issue_chars),
        "max_concurrency": int(args.max_concurrency),
        "timeout": float(args.timeout),
        "max_retries": int(args.max_retries),
        "dry_run": bool(args.dry_run),
    }
    (out_dir / "ab_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _eprint(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
