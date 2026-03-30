#!/usr/bin/env python3
"""
Issue-level numeric/procedure extraction pilot using prompt template + gateway.

One request per issue (all pages in issue). No clipping fallback.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from fullnewspaper_v3_common import (
    eprint,
    extract_openai_output_text,
    norm_str,
    parse_json_from_text,
    read_error_jsonl,
    read_result_jsonl,
    run_gateway_requests,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run issue-level zoning numeric extraction pilot.")
    ap.add_argument(
        "--package-dir",
        default="/Users/saulrichardson/Dropbox/Inclusionary Zoning/saul/newspapers/page_artifact_packets_v5_full_20260225",
    )
    ap.add_argument("--metadata-dir", default="", help="Default: <package-dir>/metadata_core")
    ap.add_argument(
        "--prompt-template",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/prompts/issue_numeric_extraction_v1_prompt.txt",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/issue_numeric_extraction_pilot",
    )
    ap.add_argument(
        "--newspapers",
        default="abilene-reporter-news,albion-evening-recorder,bountiful-davis-county-clipper",
        help="Comma-separated newspaper slugs, or 'all' to include all slugs matching selected labels.",
    )
    ap.add_argument(
        "--labels",
        default="full_ordinance,amendment",
        help="Comma-separated issue labels to include.",
    )
    ap.add_argument("--issues-per-newspaper", type=int, default=8, help="0 means all matching issues.")
    ap.add_argument(
        "--issues-per-label",
        type=int,
        default=4,
        help="If >0 and multiple labels included, cap per label per newspaper before union.",
    )
    ap.add_argument("--max-issue-chars", type=int, default=0, help="0 means no hard cap.")
    ap.add_argument("--model", default="openai:gpt-5-mini")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
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
        default="/Users/saulrichardson/projects/credit-agreement-extraction/.env",
    )
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in norm_str(raw).split(",") if x.strip()]


def _resolve_newspapers(issues: pd.DataFrame, raw: str, labels: list[str]) -> list[str]:
    token = norm_str(raw).lower()
    if token in {"all", "*"}:
        sel = issues[issues["primary_label"].isin(labels)].copy()
        slugs = sorted({norm_str(x).lower() for x in sel["newspaper_slug"].tolist() if norm_str(x)})
        return slugs
    return [x.lower() for x in _parse_csv_list(raw)]


def _load_meta(package_dir: Path, metadata_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues_path = metadata_dir / "issues.csv"
    pages_path = metadata_dir / "pages.csv"
    if not issues_path.is_file() or not pages_path.is_file():
        raise SystemExit(f"Missing issues.csv/pages.csv in {metadata_dir}")
    issues = pd.read_csv(issues_path, low_memory=False)
    pages = pd.read_csv(pages_path, low_memory=False)
    issues["issue_id"] = issues["issue_id"].astype(str).str.strip()
    issues["newspaper_slug"] = issues["newspaper_slug"].astype(str).str.strip().str.lower()
    issues["issue_date"] = issues["issue_date"].astype(str).str.strip()
    issues["primary_label"] = issues["primary_label"].astype(str).str.strip()
    pages["issue_id"] = pages["issue_id"].astype(str).str.strip()
    pages["page_id"] = pages["page_id"].astype(str).str.strip()
    pages["label"] = pages["label"].astype(str).str.strip()
    pages["transcript_abs_path"] = pages["transcript_rel_path"].map(lambda p: str((package_dir / norm_str(p)).resolve()))
    return issues, pages


def _page_num(page_id: str) -> int:
    m = re.search(r"-p-(\d+)$", norm_str(page_id).lower())
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _render_prompt(template: str, *, issue_meta: dict[str, Any], issue_text: str) -> str:
    template_placeholders = set(re.findall(r"\{\{[^{}]+\}\}", template))
    allowed = {"{{ISSUE_METADATA}}", "{{ISSUE_TEXT}}"}
    unexpected = sorted(x for x in template_placeholders if x not in allowed)
    if unexpected:
        raise SystemExit(f"Unexpected template placeholders: {unexpected}")
    out = template.replace("{{ISSUE_METADATA}}", json.dumps(issue_meta, ensure_ascii=False))
    out = out.replace("{{ISSUE_TEXT}}", issue_text)
    return out


def _extract_question_ids_from_template(template: str) -> list[str]:
    """
    Extract question IDs from prompt bullet lines like:
    - q_num_sf_min_lot_area:
    """
    ids = re.findall(r"(?m)^\s*-\s*(q_[a-z0-9_]+)\s*:", template)
    out: list[str] = []
    seen: set[str] = set()
    for qid in ids:
        q = norm_str(qid)
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def _json_schema_text_config(question_ids: list[str]) -> dict[str, Any]:
    question_id_schema: dict[str, Any]
    if question_ids:
        question_id_schema = {"type": "string", "enum": question_ids}
    else:
        question_id_schema = {"type": "string"}
    return {
        "format": {
            "type": "json_schema",
            "name": "issue_numeric_extraction_v1_output",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "issue_id": {"type": "string"},
                    "issue_date": {"type": "string"},
                    "newspaper_slug": {"type": "string"},
                    "observations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "question_id": question_id_schema,
                                "metric": {"type": "string"},
                                "bound_role": {
                                    "type": "string",
                                    "enum": [
                                        "single_value",
                                        "smallest_minimum",
                                        "largest_minimum",
                                        "smallest_maximum",
                                        "largest_maximum",
                                    ],
                                },
                                "district_or_context": {"type": "string"},
                                "value_number": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                                "value_category": {"type": "string"},
                                "unit": {
                                    "type": "string",
                                    "enum": [
                                        "sqft",
                                        "ft",
                                        "stories",
                                        "percent",
                                        "spaces_per_unit",
                                        "du_per_acre",
                                        "usd",
                                        "days",
                                        "none",
                                    ],
                                },
                                "legal_status": {"type": "string", "enum": ["enacted", "proposal", "uncertain"]},
                                "page_id": {"type": "string"},
                                "quote": {"type": "string"},
                                "confidence_0_to_1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                            "required": [
                                "question_id",
                                "metric",
                                "bound_role",
                                "district_or_context",
                                "value_number",
                                "value_category",
                                "unit",
                                "legal_status",
                                "page_id",
                                "quote",
                                "confidence_0_to_1",
                            ],
                        },
                    },
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["issue_id", "issue_date", "newspaper_slug", "observations", "notes"],
            },
        }
    }


def _write_requests(
    path: Path,
    rows: list[tuple[str, str]],
    *,
    model: str,
    reasoning_effort: str,
    question_ids: list[str],
) -> None:
    model_name = norm_str(model)
    provider = "openai"
    if ":" in model_name:
        p, rest = model_name.split(":", 1)
        provider = p.lower().strip() or "openai"
        model_name = rest.strip() if rest.strip() else model_name
    eff = norm_str(reasoning_effort).lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for custom_id, prompt in rows:
            body: dict[str, Any] = {
                "model": model_name,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                "stream": False,
                "text": _json_schema_text_config(question_ids),
            }
            if provider == "openai" and eff in {"minimal", "low", "medium", "high"}:
                body["reasoning"] = {"effort": eff}
            req = {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def _issue_choice(df: pd.DataFrame, labels: list[str], issues_per_newspaper: int, issues_per_label: int) -> pd.DataFrame:
    if not labels:
        return df.iloc[0:0].copy()
    out_parts: list[pd.DataFrame] = []
    for lab in labels:
        d = df[df["primary_label"] == lab].copy()
        d = d.sort_values(["issue_date", "issue_id"]).reset_index(drop=True)
        if issues_per_label > 0:
            d = d.head(int(issues_per_label))
        out_parts.append(d)
    if not out_parts:
        return df.iloc[0:0].copy()
    out = pd.concat(out_parts, ignore_index=True).drop_duplicates(subset=["issue_id"]).sort_values(["issue_date", "issue_id"])
    if issues_per_newspaper > 0:
        out = out.head(int(issues_per_newspaper))
    return out.reset_index(drop=True)


def _normalize_text(s: str) -> str:
    t = norm_str(s).lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _validate_rows(
    obs_rows: list[dict[str, Any]],
    issue_text_by_id: dict[str, str],
    page_ids_by_issue: dict[str, set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in obs_rows:
        issue_id = norm_str(r.get("issue_id"))
        txt_norm = _normalize_text(issue_text_by_id.get(issue_id, ""))
        quote = norm_str(r.get("quote"))
        q_norm = _normalize_text(quote)
        has_quote = int(bool(q_norm and q_norm in txt_norm))
        page_id = norm_str(r.get("page_id"))
        page_ok = int(page_id in page_ids_by_issue.get(issue_id, set()))
        v_num = r.get("value_number")
        v_cat = norm_str(r.get("value_category"))
        xor_ok = int((v_num is None) ^ bool(v_cat))
        rows.append(
            {
                "newspaper_slug": norm_str(r.get("newspaper_slug")),
                "issue_id": issue_id,
                "issue_date": norm_str(r.get("issue_date")),
                "question_id": norm_str(r.get("question_id")),
                "metric": norm_str(r.get("metric")),
                "page_id": page_id,
                "quote_match_in_issue_text": has_quote,
                "page_id_in_issue": page_ok,
                "value_xor_valid": xor_ok,
                "confidence_0_to_1": float(r.get("confidence_0_to_1") or 0.0),
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = _parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve() if norm_str(args.metadata_dir) else (package_dir / "metadata_core")
    prompt_path = Path(args.prompt_template).expanduser().resolve()
    if not prompt_path.is_file():
        raise SystemExit(f"Prompt template missing: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    question_ids = _extract_question_ids_from_template(template)
    if not question_ids:
        eprint("Warning: no q_* question IDs found in prompt; question_id enum will be unconstrained.")
    else:
        eprint(f"Prompt question_id enum size: {len(question_ids)}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    req_dir = output_dir / "requests"
    res_dir = output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    req_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    issues, pages = _load_meta(package_dir, metadata_dir)
    labels = _parse_csv_list(str(args.labels))
    newspapers = _resolve_newspapers(issues, str(args.newspapers), labels)
    if not newspapers:
        raise SystemExit("No newspapers selected.")
    if not labels:
        raise SystemExit("No labels selected.")

    selected_issue_rows: list[dict[str, Any]] = []
    request_rows: list[tuple[str, str]] = []
    issue_text_by_id: dict[str, str] = {}
    page_ids_by_issue: dict[str, set[str]] = {}
    qid_allow = set(question_ids)

    for slug in newspapers:
        s = issues[(issues["newspaper_slug"] == slug) & (issues["primary_label"].isin(labels))].copy()
        if s.empty:
            continue
        chosen = _issue_choice(
            s,
            labels=labels,
            issues_per_newspaper=int(args.issues_per_newspaper),
            issues_per_label=int(args.issues_per_label),
        )
        if chosen.empty:
            continue
        for r in chosen.itertuples(index=False):
            issue_id = str(r.issue_id)
            pp = pages[pages["issue_id"] == issue_id].copy()
            pp["page_num"] = pp["page_id"].map(_page_num)
            pp = pp.sort_values(["page_num", "page_id"]).reset_index(drop=True)
            page_blocks: list[str] = []
            page_ids: set[str] = set()
            for p in pp.itertuples(index=False):
                page_id = str(p.page_id)
                page_ids.add(page_id)
                txt = _read_text(Path(str(p.transcript_abs_path)))
                block = "\n".join(
                    [
                        f"[PAGE] page_id={page_id} page_num={int(getattr(p, 'page_num') or 0)} label={norm_str(getattr(p, 'label'))}",
                        "[TRANSCRIPT]",
                        txt,
                    ]
                )
                page_blocks.append(block)
            issue_text = "\n\n".join(page_blocks).strip()
            if int(args.max_issue_chars) > 0 and len(issue_text) > int(args.max_issue_chars):
                raise SystemExit(
                    f"issue_id={issue_id} chars={len(issue_text)} exceeds hard cap {int(args.max_issue_chars)}; no clipping allowed."
                )
            issue_meta = {
                "issue_id": issue_id,
                "issue_date": norm_str(r.issue_date),
                "newspaper_slug": norm_str(r.newspaper_slug),
                "primary_label": norm_str(r.primary_label),
                "page_count": int(len(pp)),
            }
            prompt = _render_prompt(template, issue_meta=issue_meta, issue_text=issue_text)
            custom_id = f"issue_numeric_v1::{slug}::{issue_id}"
            request_rows.append((custom_id, prompt))
            issue_text_by_id[issue_id] = issue_text
            page_ids_by_issue[issue_id] = page_ids
            selected_issue_rows.append(
                {
                    "custom_id": custom_id,
                    "newspaper_slug": slug,
                    "issue_id": issue_id,
                    "issue_date": norm_str(r.issue_date),
                    "primary_label": norm_str(r.primary_label),
                    "page_count": int(len(pp)),
                    "issue_chars": int(len(issue_text)),
                }
            )

    if not request_rows:
        raise SystemExit("No issue requests built.")

    _write_requests(
        req_dir / "openai_requests_shard000.jsonl",
        request_rows,
        model=str(args.model),
        reasoning_effort=str(args.reasoning_effort),
        question_ids=question_ids,
    )
    pd.DataFrame.from_records(selected_issue_rows).to_csv(output_dir / "issue_request_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)

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

    results = read_result_jsonl(res_dir)
    errors = read_error_jsonl(res_dir)
    meta_by_cid = {r["custom_id"]: r for r in selected_issue_rows}

    parsed_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    dropped_unknown_qid = 0
    for cid, m in sorted(meta_by_cid.items()):
        raw = results.get(cid, {})
        body = {}
        if isinstance(raw, dict):
            resp = raw.get("response")
            if isinstance(resp, dict):
                b = resp.get("body")
                if isinstance(b, dict):
                    body = b
        text = extract_openai_output_text(body) if body else ""
        parsed = parse_json_from_text(text) if text else None
        if not isinstance(parsed, dict):
            parsed = {}
        observations = parsed.get("observations")
        if not isinstance(observations, list):
            observations = []
        parsed_rows.append(
            {
                "custom_id": cid,
                "newspaper_slug": m["newspaper_slug"],
                "issue_id": m["issue_id"],
                "issue_date": m["issue_date"],
                "primary_label": m["primary_label"],
                "has_result": int(cid in results),
                "has_error": int(cid in errors),
                "error_message": norm_str((errors.get(cid) or {}).get("error") if isinstance(errors.get(cid), dict) else ""),
                "raw_output_text": text,
                "parsed_json": parsed,
            }
        )
        for o in observations:
            if not isinstance(o, dict):
                continue
            qid = norm_str(o.get("question_id"))
            if qid_allow and qid not in qid_allow:
                dropped_unknown_qid += 1
                continue
            obs_rows.append(
                {
                    "custom_id": cid,
                    "newspaper_slug": m["newspaper_slug"],
                    "issue_id": m["issue_id"],
                    "issue_date": m["issue_date"],
                    "primary_label": m["primary_label"],
                    "question_id": qid,
                    "metric": norm_str(o.get("metric")),
                    "bound_role": norm_str(o.get("bound_role")),
                    "district_or_context": norm_str(o.get("district_or_context")),
                    "value_number": float(o.get("value_number")) if isinstance(o.get("value_number"), (int, float)) and math.isfinite(float(o.get("value_number"))) else None,
                    "value_category": norm_str(o.get("value_category")),
                    "unit": norm_str(o.get("unit")),
                    "legal_status": norm_str(o.get("legal_status")),
                    "page_id": norm_str(o.get("page_id")),
                    "quote": norm_str(o.get("quote")),
                    "confidence_0_to_1": float(o.get("confidence_0_to_1") or 0.0),
                }
            )

    with (output_dir / "issue_numeric_parsed_outputs.jsonl").open("w", encoding="utf-8") as f:
        for r in parsed_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    obs_df = pd.DataFrame.from_records(obs_rows)
    obs_df.to_csv(output_dir / "issue_numeric_observations.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    val_df = _validate_rows(obs_rows, issue_text_by_id=issue_text_by_id, page_ids_by_issue=page_ids_by_issue)
    val_df.to_csv(output_dir / "issue_numeric_validation_checks.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # Manual-audit packet: focus on rows with low confidence or quote mismatch first.
    if not val_df.empty:
        merged = val_df.merge(
            obs_df[
                [
                    "newspaper_slug",
                    "issue_id",
                    "issue_date",
                    "question_id",
                    "metric",
                    "page_id",
                    "quote",
                    "value_number",
                    "value_category",
                    "unit",
                    "legal_status",
                ]
            ],
            on=["newspaper_slug", "issue_id", "issue_date", "question_id", "metric", "page_id"],
            how="left",
        )
        merged["priority_score"] = (
            (1 - merged["quote_match_in_issue_text"].astype(float))
            + (1 - merged["page_id_in_issue"].astype(float))
            + (1 - merged["value_xor_valid"].astype(float))
            + (1 - merged["confidence_0_to_1"].clip(0, 1))
        )
        merged = merged.sort_values(["priority_score", "confidence_0_to_1"], ascending=[False, True]).reset_index(drop=True)
        merged.to_csv(output_dir / "issue_numeric_manual_audit_packet.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "newspapers": newspapers,
        "labels": labels,
        "requests": int(len(request_rows)),
        "results": int(len(results)),
        "errors": int(len(errors)),
        "observation_rows": int(len(obs_rows)),
        "dropped_unknown_question_id_rows": int(dropped_unknown_qid),
    }
    (output_dir / "provenance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    eprint(f"Done. requests={len(request_rows)} results={len(results)} errors={len(errors)} obs_rows={len(obs_rows)}")


if __name__ == "__main__":
    main()
