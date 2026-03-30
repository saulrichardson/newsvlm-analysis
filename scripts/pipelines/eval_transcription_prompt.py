#!/usr/bin/env python3
"""
Evaluate a freeform zoning-law transcription prompt on audited issue transcripts.

This evaluator is intentionally narrow:
  - build OpenAI-Batch-shaped request JSONL from audited issue transcripts
  - run the prompt synchronously through vendor/agent-gateway
  - parse the required SOURCE / ORDINANCE TEXT / NOTES contract
  - score a conservative binary target: does the prompt recover operative printed
    zoning-law text on issues where the audited set says such text exists?

The manual audit file does not contain a standalone "strict operative text exists"
boolean, so the evaluator derives a conservative expectation:
  - clear negative manual classes are negative
  - clear positive manual classes are positive
  - explicit strict-target override notes demote otherwise-positive rows to negative
  - the single "uncertain" manual row is excluded from scoring
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fullnewspaper_v3_common import (
    eprint,
    extract_openai_output_text,
    iter_jsonl,
    norm_str,
    run_gateway_requests,
    write_openai_requests,
)


POSITIVE_MANUAL_LABELS = {
    "amendment_substantial_text",
    "amendment_targeted_text",
    "code_publication_full_issue",
    "code_publication_excerpt_or_installment",
    "map_rezoning_order",
    "variance_special_use_order",
    "zoning_ordinance_limited_scope",
}

NEGATIVE_MANUAL_LABELS = {
    "none",
    "non_zoning_ordinance",
    "procedural_notice_only",
}


def _safe_float(x: Any, default: float = math.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _fill_prompt(template: str, *, source_name: str, transcript: str) -> str:
    if "{SOURCE_NAME}" not in template or "{TRANSCRIPT}" not in template:
        raise SystemExit("Prompt template must contain both {SOURCE_NAME} and {TRANSCRIPT} placeholders.")
    return template.replace("{SOURCE_NAME}", source_name).replace("{TRANSCRIPT}", transcript)


def _strict_target_expected(row: dict[str, Any]) -> tuple[bool | None, str]:
    manual_label = norm_str(row.get("manual_label"))
    strict_note = norm_str(row.get("strict_operative_target_note"))
    if manual_label == "uncertain":
        return None, "manual_uncertain_excluded"
    if manual_label in NEGATIVE_MANUAL_LABELS:
        return False, f"manual_negative_class:{manual_label}"
    if strict_note:
        return False, f"strict_override:{strict_note}"
    if manual_label in POSITIVE_MANUAL_LABELS:
        return True, f"manual_positive_class:{manual_label}"
    return None, f"unmapped_manual_label:{manual_label or 'empty'}"


def _binary_metrics(rows: list[dict[str, Any]], *, truth_key: str, pred_key: str) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        truth = bool(row.get(truth_key))
        pred = bool(row.get(pred_key))
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
    f1 = (2 * precision * recall / (precision + recall)) if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) else math.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _parse_contract(text: str) -> dict[str, Any]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(
        r"^\s*SOURCE:\s*(?P<source>.*?)\n\s*=== ORDINANCE TEXT ===\s*\n(?P<ord>.*?)\n\s*--- NOTES ---\s*\n(?P<notes>.*)\Z",
        raw,
        flags=re.S,
    )
    if not m:
        return {
            "contract_ok": 0,
            "source_line": "",
            "ordinance_text": "",
            "notes_text": "",
        }
    ordinance_text = m.group("ord").strip()
    notes_text = m.group("notes").strip()
    return {
        "contract_ok": 1,
        "source_line": m.group("source").strip(),
        "ordinance_text": ordinance_text,
        "notes_text": notes_text,
    }


def _looks_like_negative_note(notes_text: str) -> bool:
    s = norm_str(notes_text).lower()
    if not s:
        return False
    phrases = [
        "no operative printed zoning law text was detected",
        "does not contain operative printed zoning law",
        "no qualifying text",
        "no operative printed zoning law",
    ]
    return any(p in s for p in phrases)


def _is_substantial_text(text: str, *, min_chars: int = 300) -> bool:
    return len(norm_str(text)) >= int(min_chars)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Evaluate a freeform zoning-law transcription prompt on audited issue transcripts.")
    ap.add_argument("--prompt-path", required=True)
    ap.add_argument("--issue-txt-dir", required=True, help="Directory containing <issue_id>.txt transcripts")
    ap.add_argument("--manual-csv", default=str(repo_root / "reports" / "issue_classifier_v7_manual_validation_20260305.csv"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--env-file", default=str(repo_root / ".env"))
    ap.add_argument("--openai-model", default="gpt-5-mini", help="Model name embedded in request JSON")
    ap.add_argument("--gateway-model", default="openai:gpt-5-mini", help="Gateway provider:model to execute")
    ap.add_argument("--max-concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gateway-runner", default=str(repo_root / "scripts" / "run_openai_requests_via_gateway.py"))
    ap.add_argument("--gateway-pythonpath", default=str(repo_root / "vendor/agent-gateway" / "src"))
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    prompt_path = Path(args.prompt_path).expanduser().resolve()
    issue_txt_dir = Path(args.issue_txt_dir).expanduser().resolve()
    manual_csv = Path(args.manual_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    env_file = Path(args.env_file).expanduser().resolve()
    gateway_runner = Path(args.gateway_runner).expanduser().resolve()
    gateway_pythonpath = Path(args.gateway_pythonpath).expanduser().resolve()

    req_dir = output_dir / "requests"
    res_dir = output_dir / "results"
    text_dir = output_dir / "rendered_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    if not prompt_path.is_file():
        raise SystemExit(f"Missing prompt: {prompt_path}")
    if not issue_txt_dir.is_dir():
        raise SystemExit(f"Missing issue transcript dir: {issue_txt_dir}")
    if not manual_csv.is_file():
        raise SystemExit(f"Missing manual csv: {manual_csv}")
    if not env_file.is_file():
        raise SystemExit(f"Missing env file: {env_file}")

    prompt_template = prompt_path.read_text(encoding="utf-8")
    manual_rows = list(csv.DictReader(manual_csv.open("r", encoding="utf-8")))
    manual_by_issue = {norm_str(row.get("issue_id")): row for row in manual_rows if norm_str(row.get("issue_id"))}

    request_rows: list[tuple[str, str]] = []
    build_rows: list[dict[str, Any]] = []
    expected_rows: dict[str, dict[str, Any]] = {}
    for issue_id in sorted(manual_by_issue):
        txt_path = issue_txt_dir / f"{issue_id}.txt"
        if not txt_path.is_file():
            raise SystemExit(f"Missing transcript for issue_id={issue_id}: {txt_path}")
        transcript = txt_path.read_text(encoding="utf-8", errors="replace")
        expected_positive, expected_reason = _strict_target_expected(manual_by_issue[issue_id])
        prompt = _fill_prompt(prompt_template, source_name=issue_id, transcript=transcript)
        request_rows.append((issue_id, prompt))
        expected_rows[issue_id] = {
            "expected_positive": expected_positive,
            "expected_reason": expected_reason,
            "transcript_chars": len(transcript),
        }
        build_rows.append(
            {
                "issue_id": issue_id,
                "manual_label": norm_str(manual_by_issue[issue_id].get("manual_label")),
                "expected_positive": "" if expected_positive is None else int(bool(expected_positive)),
                "expected_reason": expected_reason,
                "transcript_chars": len(transcript),
                "prompt_chars": len(prompt),
            }
        )

    write_openai_requests(req_dir / "openai_requests_shard000.jsonl", request_rows, model=str(args.openai_model))
    _write_csv(
        output_dir / "build_manifest.csv",
        build_rows,
        fieldnames=["issue_id", "manual_label", "expected_positive", "expected_reason", "transcript_chars", "prompt_chars"],
    )

    eprint("running gateway requests")
    run_gateway_requests(
        request_dir=req_dir,
        output_dir=res_dir,
        runner_path=gateway_runner,
        model=str(args.gateway_model),
        gateway_pythonpath=gateway_pythonpath,
        timeout_s=float(args.timeout),
        concurrency=int(args.max_concurrency),
        gov_env_path=env_file,
        skip_existing=bool(args.skip_existing),
        dry_run=False,
    )

    predictions: dict[str, dict[str, Any]] = {}
    parse_failures = 0
    contract_failures = 0
    for obj in iter_jsonl(res_dir / "openai_results_shard000.jsonl"):
        issue_id = norm_str(obj.get("custom_id"))
        if not issue_id:
            continue
        body = obj.get("response", {}).get("body", {})
        output_text = extract_openai_output_text(body if isinstance(body, dict) else {})
        parsed = _parse_contract(output_text)
        ordinance_text = norm_str(parsed.get("ordinance_text"))
        notes_text = norm_str(parsed.get("notes_text"))
        source_line = norm_str(parsed.get("source_line"))
        contract_ok = int(parsed.get("contract_ok") or 0)
        if not norm_str(output_text):
            parse_failures += 1
        if not contract_ok:
            contract_failures += 1
        (text_dir / f"{issue_id}.txt").write_text(output_text + ("\n" if output_text and not output_text.endswith("\n") else ""), encoding="utf-8")
        predictions[issue_id] = {
            "raw_output_chars": len(output_text),
            "contract_ok": contract_ok,
            "source_line": source_line,
            "ordinance_text": ordinance_text,
            "notes_text": notes_text,
            "output_has_text": int(bool(ordinance_text)),
            "output_chars": len(ordinance_text),
            "substantial_output": int(_is_substantial_text(ordinance_text)),
            "explicit_no_text_note": int(_looks_like_negative_note(notes_text)),
            "pred_positive": int(bool(ordinance_text)),
        }

    rows: list[dict[str, Any]] = []
    for issue_id in sorted(manual_by_issue):
        pred = predictions.get(issue_id)
        if pred is None:
            raise SystemExit(f"Missing prediction for issue_id={issue_id}")
        manual = manual_by_issue[issue_id]
        expected_positive = expected_rows[issue_id]["expected_positive"]
        scoreable = expected_positive is not None
        detection_match = "" if not scoreable else int(bool(pred["pred_positive"]) == bool(expected_positive))
        rows.append(
            {
                "issue_id": issue_id,
                "manual_label": norm_str(manual.get("manual_label")),
                "expected_positive": "" if expected_positive is None else int(bool(expected_positive)),
                "expected_reason": expected_rows[issue_id]["expected_reason"],
                "scoreable": int(scoreable),
                "pred_positive": pred["pred_positive"],
                "detection_match": detection_match,
                "contract_ok": pred["contract_ok"],
                "output_has_text": pred["output_has_text"],
                "substantial_output": pred["substantial_output"],
                "output_chars": pred["output_chars"],
                "raw_output_chars": pred["raw_output_chars"],
                "explicit_no_text_note": pred["explicit_no_text_note"],
                "confidence": _safe_float(manual.get("confidence"), default=math.nan),
                "model_operativity_pred": norm_str(manual.get("operativity_pred")),
                "note": norm_str(manual.get("note")),
                "strict_operative_target_note": norm_str(manual.get("strict_operative_target_note")),
                "source_line": pred["source_line"],
                "ordinance_text_preview": pred["ordinance_text"][:600],
                "notes_text": pred["notes_text"][:600],
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    _write_csv(output_dir / "predictions.csv", rows, fieldnames=fieldnames)

    score_rows = [row for row in rows if row["scoreable"]]
    metrics = _binary_metrics(score_rows, truth_key="expected_positive", pred_key="pred_positive")

    label_stats: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["manual_label"])].append(row)
    for label, grp in sorted(grouped.items()):
        label_stats[label] = {
            "n": len(grp),
            "scoreable_n": sum(int(r["scoreable"]) for r in grp),
            "pred_positive_n": sum(int(r["pred_positive"]) for r in grp),
            "substantial_output_n": sum(int(r["substantial_output"]) for r in grp),
            "median_output_chars": sorted(int(r["output_chars"]) for r in grp)[len(grp) // 2] if grp else 0,
        }

    false_positives = [row for row in score_rows if not row["expected_positive"] and row["pred_positive"]]
    false_negatives = [row for row in score_rows if row["expected_positive"] and not row["pred_positive"]]
    positive_rows = [row for row in score_rows if row["expected_positive"]]
    negative_rows = [row for row in score_rows if not row["expected_positive"]]

    summary = {
        "n_issues": len(rows),
        "scoreable_n": len(score_rows),
        "parse_failures": parse_failures,
        "contract_failures": contract_failures,
        "positive_n": len(positive_rows),
        "negative_n": len(negative_rows),
        "binary_detection_metrics": metrics,
        "positive_with_substantial_output_n": sum(int(r["substantial_output"]) for r in positive_rows),
        "negative_with_any_output_n": sum(int(r["pred_positive"]) for r in negative_rows),
        "manual_label_stats": label_stats,
        "status_counts": dict(Counter(
            "true_positive" if row["expected_positive"] and row["pred_positive"]
            else "false_positive" if (not row["expected_positive"] and row["pred_positive"])
            else "false_negative" if (row["expected_positive"] and not row["pred_positive"])
            else "true_negative"
            for row in score_rows
        )),
        "false_positive_issue_ids": [row["issue_id"] for row in false_positives],
        "false_negative_issue_ids": [row["issue_id"] for row in false_negatives],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Transcription Prompt Evaluation",
        "",
        f"- Prompt: `{prompt_path}`",
        f"- Issue transcripts: `{issue_txt_dir}`",
        f"- Manual source: `{manual_csv}`",
        f"- Gateway model: `{args.gateway_model}`",
        f"- Issues run: `{len(rows)}`",
        f"- Scoreable issues: `{len(score_rows)}`",
        f"- Parse failures (blank output): `{parse_failures}`",
        f"- Contract failures: `{contract_failures}`",
        "",
        "## Headline",
        "",
        f"- Expected operative-text positives: `{len(positive_rows)}`",
        f"- Expected operative-text negatives: `{len(negative_rows)}`",
        f"- Detection precision/recall/F1: `{metrics['precision']:.3f}` / `{metrics['recall']:.3f}` / `{metrics['f1']:.3f}`",
        f"- Positive rows with substantial extracted text (`>=300 chars`): `{sum(int(r['substantial_output']) for r in positive_rows)}/{len(positive_rows) if positive_rows else 0}`",
        f"- Negative rows with any extracted ordinance text: `{sum(int(r['pred_positive']) for r in negative_rows)}/{len(negative_rows) if negative_rows else 0}`",
        "",
        "## Manual Label Stats",
        "",
    ]
    for label, stats in label_stats.items():
        lines.append(
            f"- `{label}`: n=`{stats['n']}`, scoreable=`{stats['scoreable_n']}`, "
            f"pred_positive=`{stats['pred_positive_n']}`, substantial=`{stats['substantial_output_n']}`, "
            f"median_output_chars=`{stats['median_output_chars']}`"
        )
    lines.extend(["", "## False Positives", ""])
    for row in false_positives:
        lines.append(
            f"- `{row['issue_id']}` (`{row['manual_label']}`): output_chars=`{row['output_chars']}`; note={row['note']}"
        )
    lines.extend(["", "## False Negatives", ""])
    for row in false_negatives:
        lines.append(
            f"- `{row['issue_id']}` (`{row['manual_label']}`): strict_note={row['strict_operative_target_note'] or '(none)'}; note={row['note']}"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    eprint(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
