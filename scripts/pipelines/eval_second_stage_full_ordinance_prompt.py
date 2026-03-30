#!/usr/bin/env python3
"""
Evaluate a second-stage zoning-law ontology prompt on audited issue transcripts.

The evaluator:
  - builds one OpenAI-batch-shaped request shard from issue transcript .txt files
  - runs the requests synchronously through vendor/agent-gateway
  - parses single-line JSON responses
  - scores predictions against a collapsed manual ontology

This is intended for prompt iteration, not production batch export.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_STAGE2 = {
    "full_ordinance_text",
    "amendment_text",
    "site_specific_zoning_action",
    "other_legislative_zoning_text",
    "not_applicable_printed_zoning_law",
    "uncertain",
}
ALLOWED_FAMILY = {
    "full_code_publication",
    "substantial_amendment",
    "targeted_amendment",
    "limited_scope_ordinance",
    "code_excerpt_or_installment",
    "other_legislative",
    "map_rezoning",
    "variance_special_use",
    "notice_or_summary_only",
    "narrative_or_quote_only",
    "non_zoning_ordinance",
    "uncertain",
}
ALLOWED_OPERATIVITY = {"operative", "proposed", "unclear", "none"}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _safe_float(x: Any, default: float = math.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def _enum(v: Any, allowed: set[str], default: str) -> str:
    s = _norm(v).lower()
    return s if s in allowed else default


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


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
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)

    def _try(raw: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    obj = _try(s)
    if obj is not None:
        return obj

    first = s.find("{")
    if first == -1:
        return None
    candidate = s[first:]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return _try(_repair_truncated_json(candidate))


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
                if isinstance(txt, str) and txt:
                    parts.append(txt)
    return "\n".join(parts).strip()


def _make_openai_request(custom_id: str, prompt: str, model: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "stream": False,
        },
    }


def _manual_stage2_class(row: dict[str, Any]) -> str:
    manual_label = _norm(row.get("manual_label"))
    note = _norm(row.get("note")).lower()
    strict_note = _norm(row.get("strict_operative_target_note")).lower()
    if manual_label == "code_publication_full_issue":
        return "full_ordinance_text"
    # The second-stage ontology treats a fully printed comprehensive ordinance as
    # "full_ordinance_text" even when the earlier fine-grained taxonomy parked it
    # in excerpt/installment because it was proposed rather than operative.
    if manual_label == "code_publication_excerpt_or_installment" and (
        "full proposed zoning ordinance text" in note or "full proposed zoning ordinance text" in strict_note
    ):
        return "full_ordinance_text"
    if manual_label in {"amendment_substantial_text", "amendment_targeted_text"}:
        return "amendment_text"
    if manual_label in {"map_rezoning_order", "variance_special_use_order"}:
        return "site_specific_zoning_action"
    if manual_label in {"code_publication_excerpt_or_installment", "zoning_ordinance_limited_scope"}:
        return "other_legislative_zoning_text"
    if manual_label in {"procedural_notice_only", "non_zoning_ordinance", "none"}:
        return "not_applicable_printed_zoning_law"
    return "uncertain"


def _baseline_stage2_class(model_label: str) -> str:
    row = {"manual_label": model_label, "note": "", "strict_operative_target_note": ""}
    return _manual_stage2_class(row)


def _class_metrics(rows: list[dict[str, Any]], *, truth_key: str, pred_key: str, labels: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label in labels:
        out[label] = _binary_metrics(rows, truth_key=truth_key, pred_key=pred_key, positive_value=label)
    return out


def _binary_metrics(rows: list[dict[str, Any]], *, truth_key: str, pred_key: str, positive_value: str) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        truth = row.get(truth_key) == positive_value
        pred = row.get(pred_key) == positive_value
        if truth and pred:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif truth and not pred:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    f1 = (2 * precision * recall / (precision + recall)) if precision == precision and recall == recall and (precision + recall) else math.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _applicable_binary_metrics(rows: list[dict[str, Any]], *, pred_key: str) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        truth = row.get("manual_stage2_class") in {
            "full_ordinance_text",
            "amendment_text",
            "site_specific_zoning_action",
            "other_legislative_zoning_text",
        }
        pred = row.get(pred_key) in {
            "full_ordinance_text",
            "amendment_text",
            "site_specific_zoning_action",
            "other_legislative_zoning_text",
        }
        if truth and pred:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif truth and not pred:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    f1 = (2 * precision * recall / (precision + recall)) if precision == precision and recall == recall and (precision + recall) else math.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Evaluate second-stage full-ordinance prompt on audited issue transcripts.")
    ap.add_argument("--prompt-path", required=True)
    ap.add_argument("--issue-txt-dir", required=True, help="Directory containing <issue_id>.txt transcripts")
    ap.add_argument("--manual-csv", default=str(repo_root / "reports" / "issue_classifier_v7_manual_validation_20260305.csv"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--openai-model", default="gpt-5-mini", help="Model name embedded in request JSON")
    ap.add_argument("--gateway-model", default="openai:gpt-5-mini", help="Gateway provider:model to execute")
    ap.add_argument("--max-concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gateway-url", default="", help="Existing gateway URL to reuse")
    ap.add_argument("--gateway-port", type=int, default=0, help="Port to start gateway on when auto-starting")
    ap.add_argument(
        "--reuse-existing-gateway",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow auto-start mode to reuse an already-running gateway on the chosen port",
    )
    ap.add_argument(
        "--gateway-runner",
        default=str(repo_root / "scripts" / "run_openai_requests_via_gateway.py"),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    prompt_path = Path(args.prompt_path).expanduser().resolve()
    issue_txt_dir = Path(args.issue_txt_dir).expanduser().resolve()
    manual_csv = Path(args.manual_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    req_dir = out_dir / "requests"
    res_dir = out_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not prompt_path.is_file():
        raise SystemExit(f"Missing prompt: {prompt_path}")
    if not issue_txt_dir.is_dir():
        raise SystemExit(f"Missing issue transcript dir: {issue_txt_dir}")
    if not manual_csv.is_file():
        raise SystemExit(f"Missing manual csv: {manual_csv}")

    prompt_prefix = prompt_path.read_text(encoding="utf-8").strip()
    manual_rows = list(csv.DictReader(manual_csv.open()))
    manual_by_issue = {row["issue_id"]: row for row in manual_rows}

    req_rows: list[dict[str, Any]] = []
    build_rows: list[dict[str, Any]] = []
    for issue_id in sorted(manual_by_issue):
        txt_path = issue_txt_dir / f"{issue_id}.txt"
        if not txt_path.is_file():
            raise SystemExit(f"Missing transcript for {issue_id}: {txt_path}")
        transcript = txt_path.read_text(encoding="utf-8")
        prompt = prompt_prefix + "\n\n" + f"ISSUE_ID: {issue_id}\n\nISSUE_TRANSCRIPT:\n" + transcript.strip()
        req_rows.append(_make_openai_request(issue_id, prompt, str(args.openai_model)))
        build_rows.append(
            {
                "issue_id": issue_id,
                "transcript_chars": len(transcript),
                "prompt_chars": len(prompt),
                "manual_label": manual_by_issue[issue_id]["manual_label"],
                "manual_stage2_class": _manual_stage2_class(manual_by_issue[issue_id]),
            }
        )

    _write_jsonl(req_dir / "openai_requests_shard000.jsonl", req_rows)
    _write_csv(
        out_dir / "build_manifest.csv",
        build_rows,
        fieldnames=["issue_id", "transcript_chars", "prompt_chars", "manual_label", "manual_stage2_class"],
    )

    cmd = [
        sys.executable,
        str(Path(args.gateway_runner).expanduser().resolve()),
        "--request-dir",
        str(req_dir),
        "--output-dir",
        str(res_dir),
        "--model",
        str(args.gateway_model),
        "--max-concurrency",
        str(int(args.max_concurrency)),
        "--timeout",
        str(float(args.timeout)),
        "--max-retries",
        str(int(args.max_retries)),
    ]
    if args.gateway_url:
        cmd.extend(["--gateway-url", str(args.gateway_url)])
    elif int(args.gateway_port):
        cmd.extend(["--port", str(int(args.gateway_port))])
        if args.reuse_existing_gateway:
            cmd.append("--reuse-existing-gateway")
    if args.skip_existing:
        cmd.append("--skip-existing")
    _eprint("running gateway requests")
    subprocess.run(cmd, check=True)

    result_path = res_dir / "openai_results_shard000.jsonl"
    if not result_path.is_file():
        raise SystemExit(f"Missing results file: {result_path}")

    predictions: dict[str, dict[str, Any]] = {}
    parse_failures = 0
    with result_path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            obj = json.loads(s)
            issue_id = _norm(obj.get("custom_id"))
            body = obj.get("response", {}).get("body", {})
            out_text = _extract_openai_output_text(body if isinstance(body, dict) else {})
            parsed = _parse_json_from_text(out_text)
            if not parsed:
                parse_failures += 1
                predictions[issue_id] = {
                    "parse_ok": 0,
                    "candidate_stage2_class": "uncertain",
                    "candidate_instrument_family": "uncertain",
                    "candidate_operativity": "unclear",
                    "candidate_confidence": math.nan,
                    "candidate_evidence_quote": "",
                    "candidate_quality_flags": "",
                    "candidate_rationale": out_text[:1000],
                }
                continue

            ss = parsed.get("second_stage") if isinstance(parsed.get("second_stage"), dict) else {}
            sig = parsed.get("full_ordinance_signals") if isinstance(parsed.get("full_ordinance_signals"), dict) else {}
            flags = parsed.get("quality_flags") if isinstance(parsed.get("quality_flags"), list) else []
            predictions[issue_id] = {
                "parse_ok": 1,
                "candidate_stage2_class": _enum(ss.get("class"), ALLOWED_STAGE2, "uncertain"),
                "candidate_instrument_family": _enum(ss.get("instrument_family"), ALLOWED_FAMILY, "uncertain"),
                "candidate_operativity": _enum(ss.get("operativity"), ALLOWED_OPERATIVITY, "unclear"),
                "candidate_confidence": _safe_float(ss.get("confidence_0_to_1"), default=math.nan),
                "candidate_evidence_quote": _norm(ss.get("evidence_quote")),
                "candidate_quality_flags": "|".join(str(x) for x in flags if _norm(x)),
                "candidate_rationale": _norm(parsed.get("rationale")),
                "signal_official_publication_framing": int(bool(sig.get("official_publication_framing"))),
                "signal_broad_code_structure": int(bool(sig.get("broad_code_structure"))),
                "signal_district_framework_present": int(bool(sig.get("district_framework_present"))),
                "signal_multiple_substantive_components_present": int(bool(sig.get("multiple_substantive_components_present"))),
            }

    rows: list[dict[str, Any]] = []
    for issue_id in sorted(manual_by_issue):
        manual = manual_by_issue[issue_id]
        pred = predictions.get(issue_id)
        if pred is None:
            raise SystemExit(f"Missing prediction for issue_id={issue_id}")
        manual_stage2 = _manual_stage2_class(manual)
        baseline_stage2 = _baseline_stage2_class(manual["model_label"])
        baseline_match = baseline_stage2 == manual_stage2
        candidate_match = pred["candidate_stage2_class"] == manual_stage2
        status_change = "unchanged_wrong"
        if not baseline_match and candidate_match:
            status_change = "improved"
        elif baseline_match and not candidate_match:
            status_change = "worsened"
        elif baseline_match and candidate_match:
            status_change = "unchanged_correct"
        rows.append(
            {
                "issue_id": issue_id,
                "manual_label": manual["manual_label"],
                "manual_stage2_class": manual_stage2,
                "baseline_label": manual["model_label"],
                "baseline_stage2_class": baseline_stage2,
                "baseline_stage2_match": int(baseline_match),
                **pred,
                "candidate_stage2_match": int(candidate_match),
                "status_change": status_change,
                "note": manual.get("note", ""),
                "strict_operative_target_note": manual.get("strict_operative_target_note", ""),
            }
        )

    fieldnames = list(rows[0].keys())
    _write_csv(out_dir / "predictions.csv", rows, fieldnames=fieldnames)

    baseline_acc = sum(row["baseline_stage2_match"] for row in rows) / len(rows)
    candidate_acc = sum(row["candidate_stage2_match"] for row in rows) / len(rows)
    status_counts = Counter(row["status_change"] for row in rows)
    manual_dist = Counter(row["manual_stage2_class"] for row in rows)
    candidate_dist = Counter(row["candidate_stage2_class"] for row in rows)

    full_baseline = _binary_metrics(rows, truth_key="manual_stage2_class", pred_key="baseline_stage2_class", positive_value="full_ordinance_text")
    full_candidate = _binary_metrics(rows, truth_key="manual_stage2_class", pred_key="candidate_stage2_class", positive_value="full_ordinance_text")
    app_baseline = _applicable_binary_metrics(rows, pred_key="baseline_stage2_class")
    app_candidate = _applicable_binary_metrics(rows, pred_key="candidate_stage2_class")
    class_order = [
        "full_ordinance_text",
        "amendment_text",
        "site_specific_zoning_action",
        "other_legislative_zoning_text",
        "not_applicable_printed_zoning_law",
        "uncertain",
    ]
    baseline_class_metrics = _class_metrics(rows, truth_key="manual_stage2_class", pred_key="baseline_stage2_class", labels=class_order)
    candidate_class_metrics = _class_metrics(rows, truth_key="manual_stage2_class", pred_key="candidate_stage2_class", labels=class_order)

    summary = {
        "n_issues": len(rows),
        "parse_failures": parse_failures,
        "baseline_stage2_accuracy": baseline_acc,
        "candidate_stage2_accuracy": candidate_acc,
        "status_counts": dict(status_counts),
        "manual_stage2_distribution": dict(manual_dist),
        "candidate_stage2_distribution": dict(candidate_dist),
        "baseline_full_ordinance_metrics": full_baseline,
        "candidate_full_ordinance_metrics": full_candidate,
        "baseline_applicable_law_metrics": app_baseline,
        "candidate_applicable_law_metrics": app_candidate,
        "baseline_class_metrics": baseline_class_metrics,
        "candidate_class_metrics": candidate_class_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    improved = [row for row in rows if row["status_change"] == "improved"]
    worsened = [row for row in rows if row["status_change"] == "worsened"]
    lines = [
        "# Second-Stage Prompt Evaluation",
        "",
        f"- Prompt: `{prompt_path}`",
        f"- Issue transcripts: `{issue_txt_dir}`",
        f"- Manual source: `{manual_csv}`",
        f"- Gateway model: `{args.gateway_model}`",
        f"- Issues: `{len(rows)}`",
        f"- Parse failures: `{parse_failures}`",
        "",
        "## Headline",
        "",
        f"- Baseline collapsed stage-2 accuracy: `{sum(row['baseline_stage2_match'] for row in rows)}/{len(rows)} = {baseline_acc:.1%}`",
        f"- Candidate stage-2 accuracy: `{sum(row['candidate_stage2_match'] for row in rows)}/{len(rows)} = {candidate_acc:.1%}`",
        f"- Status changes: `{dict(status_counts)}`",
        "",
        "## Full Ordinance",
        "",
        f"- Baseline precision/recall/F1: `{full_baseline['precision']:.3f}` / `{full_baseline['recall']:.3f}` / `{full_baseline['f1']:.3f}`",
        f"- Candidate precision/recall/F1: `{full_candidate['precision']:.3f}` / `{full_candidate['recall']:.3f}` / `{full_candidate['f1']:.3f}`",
        "",
        "## Per-Class Metrics",
        "",
    ]
    for label in class_order:
        bm = baseline_class_metrics[label]
        cm = candidate_class_metrics[label]
        lines.append(
            f"- `{label}` baseline P/R/F1 `{bm['precision']:.3f}` / `{bm['recall']:.3f}` / `{bm['f1']:.3f}`; "
            f"candidate `{cm['precision']:.3f}` / `{cm['recall']:.3f}` / `{cm['f1']:.3f}`"
        )
    lines.extend(
        [
            "",
        "## Any Applicable Printed Zoning Law",
        "",
        f"- Baseline precision/recall/F1: `{app_baseline['precision']:.3f}` / `{app_baseline['recall']:.3f}` / `{app_baseline['f1']:.3f}`",
        f"- Candidate precision/recall/F1: `{app_candidate['precision']:.3f}` / `{app_candidate['recall']:.3f}` / `{app_candidate['f1']:.3f}`",
        "",
        "## Improved Cases",
        "",
        ]
    )
    for row in improved:
        lines.append(
            f"- `{row['issue_id']}`: baseline `{row['baseline_stage2_class']}` -> candidate `{row['candidate_stage2_class']}`; manual `{row['manual_stage2_class']}`"
        )
    lines.extend(["", "## Worsened Cases", ""])
    for row in worsened:
        lines.append(
            f"- `{row['issue_id']}`: baseline `{row['baseline_stage2_class']}` -> candidate `{row['candidate_stage2_class']}`; manual `{row['manual_stage2_class']}`"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _eprint(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
