#!/usr/bin/env python3
"""
Evaluate prompt variants for judging whether current full-ordinance markdown
appears complete or missing material.

This is a prompt-iteration harness over the existing exported full-ordinance
markdown files. It intentionally uses the ordinance markdown artifact itself
rather than upstream issue transcripts, because the question is whether the
current public-facing ordinance text appears complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = {
    "complete_or_nearly_complete",
    "likely_incomplete",
    "definitely_incomplete",
    "uncertain",
}
SHARE_BANDS = {
    "none_or_trivial_lt_5",
    "small_5_15",
    "moderate_15_35",
    "large_35_70",
    "very_large_gt_70",
    "unknown",
}
SIGNALS = {
    "explicit_missing_text_marker",
    "explicit_missing_page_language",
    "continued_from_at_start",
    "continued_on_at_end",
    "begins_mid_sentence_or_mid_section",
    "ends_mid_sentence_or_mid_section",
    "numbering_gap_or_major_jump",
    "missing_early_sections_or_articles",
    "missing_late_sections_or_articles",
    "abrupt_truncation",
    "no_clear_gap_signal",
}
AUDIT_CATEGORY_ORDER = [
    "high_confidence_missing_other_pages",
    "incomplete_text_cause_unclear",
    "no_direct_gap_clue",
]


def _norm(x: Any) -> str:
    return str(x or "").strip()


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


def _json_schema_text_config() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "full_ordinance_completeness_eval_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "issue_id": {"type": "string"},
                    "completeness_label": {
                        "type": "string",
                        "enum": sorted(LABELS),
                    },
                    "estimated_missing_share_0_to_1": {
                        "anyOf": [
                            {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            {"type": "null"},
                        ]
                    },
                    "missing_share_band": {
                        "type": "string",
                        "enum": sorted(SHARE_BANDS),
                    },
                    "confidence_0_to_1": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "evidence_signals": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": sorted(SIGNALS),
                        },
                    },
                    "short_explanation": {"type": "string"},
                },
                "required": [
                    "issue_id",
                    "completeness_label",
                    "estimated_missing_share_0_to_1",
                    "missing_share_band",
                    "confidence_0_to_1",
                    "evidence_signals",
                    "short_explanation",
                ],
            },
        }
    }


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Evaluate prompt variants for full-ordinance completeness judgment.")
    ap.add_argument(
        "--audit-csv",
        default=str(repo_root / "reports" / "full_ordinance_missing_page_audit_20260313" / "audit.csv"),
    )
    ap.add_argument(
        "--bundle-root",
        default="/Users/saulrichardson/Dropbox/Inclusionary Zoning/Historical Analysis/raw_data/newspaper_ordinances",
        help="Current full-ordinance Dropbox bundle root that contains parsed/*.md",
    )
    ap.add_argument(
        "--prompt-path",
        action="append",
        default=[],
        help="Prompt path to evaluate. May be repeated. Defaults to the three completeness eval prompts.",
    )
    ap.add_argument(
        "--output-dir",
        default=str(repo_root / "artifacts/scratch" / "full_ordinance_completeness_eval_20260318"),
    )
    ap.add_argument(
        "--gold-csv",
        default="",
        help="Optional manual gold CSV with at least issue_id and manual_completeness_label columns.",
    )
    ap.add_argument(
        "--issue-id-file",
        default="",
        help="Optional text file with one issue_id per line. When provided, restrict evaluation to those issue_ids.",
    )
    ap.add_argument("--sample-per-category", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260318)
    ap.add_argument("--request-model", default="gpt-5.4")
    ap.add_argument(
        "--gateway-model",
        default="openai:gpt-5.4",
        help="Gateway provider:model. Defaults to OpenAI because the strict JSON Schema used here is not Gemini-compatible in this gateway path.",
    )
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gateway-url", default="")
    ap.add_argument("--gateway-port", type=int, default=0)
    ap.add_argument("--reuse-existing-gateway", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--gateway-runner",
        default=str(repo_root / "scripts" / "run_openai_requests_via_gateway.py"),
    )
    return ap.parse_args()


def _load_audit_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [{k: _norm(v) for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        raise SystemExit(f"No audit rows found in {path}")
    return rows


def _load_gold_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [{k: _norm(v) for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        raise SystemExit(f"No gold rows found in {path}")

    out: dict[str, dict[str, str]] = {}
    for row in rows:
        issue_id = row.get("issue_id", "")
        if not issue_id:
            raise SystemExit(f"Gold row missing issue_id in {path}")
        label = row.get("manual_completeness_label", "")
        if label and label not in LABELS:
            raise SystemExit(f"Unsupported manual_completeness_label={label!r} for issue_id={issue_id} in {path}")
        out[issue_id] = row
    return out


def _sample_category(rows: list[dict[str, str]], *, per_category: int, seed: int) -> list[dict[str, str]]:
    by_cat: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cat[row["audit_category"]].append(row)

    sampled: list[dict[str, str]] = []
    for offset, category in enumerate(AUDIT_CATEGORY_ORDER):
        cat_rows = list(by_cat.get(category) or [])
        if not cat_rows:
            continue
        rng = random.Random(seed + offset)
        single = [r for r in cat_rows if int(r["raw_image_count"] or "0") <= 1]
        multi = [r for r in cat_rows if int(r["raw_image_count"] or "0") > 1]
        rng.shuffle(single)
        rng.shuffle(multi)

        target_single = min(len(single), per_category // 2)
        target_multi = min(len(multi), per_category - target_single)
        chosen = single[:target_single] + multi[:target_multi]

        remaining = per_category - len(chosen)
        if remaining > 0:
            pool = [r for r in (single[target_single:] + multi[target_multi:]) if r not in chosen]
            chosen.extend(pool[:remaining])

        sampled.extend(sorted(chosen, key=lambda r: r["issue_id"]))
    return sampled


def _materialize_sample(
    sample_rows: list[dict[str, str]],
    *,
    bundle_root: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    sample_md_dir = out_dir / "sample_ordinances"
    sample_md_dir.mkdir(parents=True, exist_ok=True)
    for row in sample_rows:
        rel = row["parsed_relpath"]
        src = bundle_root / rel
        if not src.is_file():
            raise SystemExit(f"Missing parsed ordinance markdown: {src}")
        text = src.read_text(encoding="utf-8", errors="replace")
        dst = sample_md_dir / f"{row['issue_id']}.md"
        dst.write_text(text, encoding="utf-8")
        materialized.append(
            {
                **row,
                "parsed_abs_path": str(src),
                "sample_md_path": str(dst),
                "ordinance_chars": len(text),
                "ordinal_page_bucket": "single_page" if int(row["raw_image_count"] or "0") <= 1 else "multi_page",
            }
        )
    return materialized


def _fill_prompt(template: str, *, issue_id: str, ordinance_text: str) -> str:
    out = template
    out = out.replace("{{ISSUE_ID}}", issue_id)
    out = out.replace("{{ORDINANCE_TEXT}}", ordinance_text)
    return out


def _make_request(*, custom_id: str, prompt_text: str, request_model: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": request_model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}],
            "text": _json_schema_text_config(),
            "reasoning": {"effort": "medium"},
            "stream": False,
        },
    }


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


def _load_result_predictions(result_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with result_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            custom_id = _norm(obj.get("custom_id"))
            body = obj.get("response", {}).get("body", {})
            text = _extract_openai_output_text(body if isinstance(body, dict) else {})
            try:
                parsed = json.loads(text)
            except Exception as exc:
                raise SystemExit(f"Could not parse JSON response for {custom_id}: {exc}\n{text[:1000]}") from exc
            if not isinstance(parsed, dict):
                raise SystemExit(f"Parsed response for {custom_id} is not an object")
            out[custom_id] = parsed
    return out


def _label_rank(label: str) -> int:
    return {
        "complete_or_nearly_complete": 0,
        "uncertain": 1,
        "likely_incomplete": 2,
        "definitely_incomplete": 3,
    }.get(_norm(label), -1)


def _gold_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_rows = [r for r in rows if _norm(r.get("manual_completeness_label"))]
    if not gold_rows:
        return {}

    exact = 0
    within_one = 0
    false_complete = 0
    false_incomplete = 0
    abs_errors: list[int] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for row in gold_rows:
        pred = _norm(row.get("completeness_label"))
        gold = _norm(row.get("manual_completeness_label"))
        pred_rank = _label_rank(pred)
        gold_rank = _label_rank(gold)
        if pred == gold:
            exact += 1
        if pred_rank >= 0 and gold_rank >= 0:
            err = abs(pred_rank - gold_rank)
            abs_errors.append(err)
            if err <= 1:
                within_one += 1
        if gold_rank >= 2 and pred == "complete_or_nearly_complete":
            false_complete += 1
        if gold == "complete_or_nearly_complete" and pred_rank >= 2:
            false_incomplete += 1
        confusion[gold][pred] += 1

    n = len(gold_rows)
    return {
        "n": n,
        "exact_accuracy": exact / n,
        "within_one_rate": within_one / n,
        "mean_abs_rank_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else math.nan,
        "false_complete_rate": false_complete / n,
        "false_incomplete_rate": false_incomplete / n,
        "confusion": {gold: dict(counter) for gold, counter in sorted(confusion.items())},
    }


def _summarize_prompt(
    rows: list[dict[str, Any]],
    *,
    prompt_name: str,
) -> dict[str, Any]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[row["audit_category"]].append(row)

    category_stats: dict[str, Any] = {}
    for cat in AUDIT_CATEGORY_ORDER:
        bucket = by_cat.get(cat, [])
        label_counts = Counter(_norm(r.get("completeness_label")) for r in bucket)
        definitely = sum(_norm(r.get("completeness_label")) == "definitely_incomplete" for r in bucket)
        likely_or_def = sum(_label_rank(_norm(r.get("completeness_label"))) >= 2 for r in bucket)
        avg_share = [
            float(r["estimated_missing_share_0_to_1"])
            for r in bucket
            if isinstance(r.get("estimated_missing_share_0_to_1"), (int, float))
        ]
        category_stats[cat] = {
            "n": len(bucket),
            "label_counts": dict(label_counts),
            "definitely_incomplete_rate": (definitely / len(bucket)) if bucket else math.nan,
            "likely_or_definitely_incomplete_rate": (likely_or_def / len(bucket)) if bucket else math.nan,
            "avg_estimated_missing_share": (sum(avg_share) / len(avg_share)) if avg_share else math.nan,
        }

    score = (
        2.0 * category_stats["high_confidence_missing_other_pages"]["likely_or_definitely_incomplete_rate"]
        + 1.0 * category_stats["high_confidence_missing_other_pages"]["definitely_incomplete_rate"]
        + 1.0 * category_stats["incomplete_text_cause_unclear"]["likely_or_definitely_incomplete_rate"]
        - 1.5 * category_stats["no_direct_gap_clue"]["likely_or_definitely_incomplete_rate"]
        - 0.5 * category_stats["no_direct_gap_clue"]["definitely_incomplete_rate"]
    )
    out = {
        "prompt_name": prompt_name,
        "category_stats": category_stats,
        "separation_score": score,
    }
    gold_metrics = _gold_metrics(rows)
    if gold_metrics:
        out["gold_metrics"] = gold_metrics
    return out


def _run_gateway(
    *,
    request_dir: Path,
    output_dir: Path,
    gateway_runner: Path,
    gateway_model: str,
    max_concurrency: int,
    timeout: float,
    max_retries: int,
    skip_existing: bool,
    gateway_url: str,
    gateway_port: int,
    reuse_existing_gateway: bool,
) -> None:
    cmd = [
        sys.executable,
        str(gateway_runner),
        "--request-dir",
        str(request_dir),
        "--output-dir",
        str(output_dir),
        "--model",
        gateway_model,
        "--max-concurrency",
        str(max_concurrency),
        "--timeout",
        str(timeout),
        "--max-retries",
        str(max_retries),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    if gateway_url:
        cmd.extend(["--gateway-url", gateway_url])
    elif gateway_port:
        cmd.extend(["--port", str(gateway_port)])
        if reuse_existing_gateway:
            cmd.append("--reuse-existing-gateway")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    audit_csv = Path(args.audit_csv).expanduser().resolve()
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    gold_csv = Path(args.gold_csv).expanduser().resolve() if args.gold_csv else None
    issue_id_file = Path(args.issue_id_file).expanduser().resolve() if args.issue_id_file else None
    gateway_runner = Path(args.gateway_runner).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_paths = [Path(p).expanduser().resolve() for p in (args.prompt_path or [])]
    if not prompt_paths:
        prompt_paths = [
            repo_root / "prompts" / "full_ordinance_completeness_eval_v1_direct_signals.txt",
            repo_root / "prompts" / "full_ordinance_completeness_eval_v2_structure_and_coverage.txt",
            repo_root / "prompts" / "full_ordinance_completeness_eval_v3_calibrated_abstention.txt",
        ]

    for path in prompt_paths:
        if not path.is_file():
            raise SystemExit(f"Missing prompt path: {path}")
    if not audit_csv.is_file():
        raise SystemExit(f"Missing audit csv: {audit_csv}")
    if gold_csv and not gold_csv.is_file():
        raise SystemExit(f"Missing gold csv: {gold_csv}")
    if issue_id_file and not issue_id_file.is_file():
        raise SystemExit(f"Missing issue id file: {issue_id_file}")
    if not bundle_root.is_dir():
        raise SystemExit(f"Missing bundle root: {bundle_root}")
    if not gateway_runner.is_file():
        raise SystemExit(f"Missing gateway runner: {gateway_runner}")

    audit_rows = _load_audit_rows(audit_csv)
    if issue_id_file:
        wanted = {line.strip() for line in issue_id_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        audit_rows = [row for row in audit_rows if row["issue_id"] in wanted]
        missing_wanted = sorted(wanted - {row["issue_id"] for row in audit_rows})
        if missing_wanted:
            raise SystemExit(f"Issue ids not found in audit csv: {missing_wanted[:20]}")
    gold_by_issue = _load_gold_rows(gold_csv) if gold_csv else {}
    sampled_rows = _sample_category(audit_rows, per_category=int(args.sample_per_category), seed=int(args.seed))
    if not sampled_rows:
        raise SystemExit("No sampled rows selected")
    sample_rows = _materialize_sample(sampled_rows, bundle_root=bundle_root, out_dir=out_dir)
    for row in sample_rows:
        gold = gold_by_issue.get(row["issue_id"], {})
        row["manual_completeness_label"] = gold.get("manual_completeness_label", "")
        row["manual_artifact_assessment"] = gold.get("manual_artifact_assessment", "")
        row["manual_review_notes"] = gold.get("manual_review_notes", "")
    _write_csv(
        out_dir / "challenge_set.csv",
        sample_rows,
        fieldnames=[
            "issue_id",
            "issue_date",
            "newspaper_slug",
            "audit_category",
            "raw_image_count",
            "parsed_relpath",
            "parsed_abs_path",
            "sample_md_path",
            "ordinance_chars",
            "ordinal_page_bucket",
            "missing_text_placeholder_count",
            "page_gap_placeholder_count",
            "continued_from_edge_count",
            "continued_on_edge_count",
            "manual_completeness_label",
            "manual_artifact_assessment",
            "manual_review_notes",
        ],
    )

    all_prompt_summaries: list[dict[str, Any]] = []
    prompt_result_rows: list[dict[str, Any]] = []

    for prompt_path in prompt_paths:
        prompt_name = prompt_path.stem
        template = prompt_path.read_text(encoding="utf-8")
        prompt_out_dir = out_dir / prompt_name
        request_dir = prompt_out_dir / "requests"
        result_dir = prompt_out_dir / "results"
        request_dir.mkdir(parents=True, exist_ok=True)

        req_rows: list[dict[str, Any]] = []
        build_rows: list[dict[str, Any]] = []
        for row in sample_rows:
            md_path = Path(row["sample_md_path"])
            ordinance_text = md_path.read_text(encoding="utf-8", errors="replace")
            prompt_text = _fill_prompt(template, issue_id=row["issue_id"], ordinance_text=ordinance_text)
            req_rows.append(_make_request(custom_id=row["issue_id"], prompt_text=prompt_text, request_model=str(args.request_model)))
            build_rows.append(
                {
                    "issue_id": row["issue_id"],
                    "audit_category": row["audit_category"],
                    "raw_image_count": row["raw_image_count"],
                    "ordinance_chars": row["ordinance_chars"],
                    "prompt_chars": len(prompt_text),
                    "prompt_path": str(prompt_path),
                }
            )
        _write_jsonl(request_dir / "openai_requests_shard000.jsonl", req_rows)
        _write_csv(
            prompt_out_dir / "build_manifest.csv",
            build_rows,
            fieldnames=["issue_id", "audit_category", "raw_image_count", "ordinance_chars", "prompt_chars", "prompt_path"],
        )

        _run_gateway(
            request_dir=request_dir,
            output_dir=result_dir,
            gateway_runner=gateway_runner,
            gateway_model=str(args.gateway_model),
            max_concurrency=int(args.max_concurrency),
            timeout=float(args.timeout),
            max_retries=int(args.max_retries),
            skip_existing=bool(args.skip_existing),
            gateway_url=str(args.gateway_url or ""),
            gateway_port=int(args.gateway_port),
            reuse_existing_gateway=bool(args.reuse_existing_gateway),
        )

        result_path = result_dir / "openai_results_shard000.jsonl"
        if not result_path.is_file():
            raise SystemExit(f"Missing results file for {prompt_name}: {result_path}")
        predictions = _load_result_predictions(result_path)

        merged_rows: list[dict[str, Any]] = []
        for row in sample_rows:
            pred = predictions.get(row["issue_id"])
            if pred is None:
                raise SystemExit(f"Missing prediction for issue_id={row['issue_id']} prompt={prompt_name}")
            merged = {
                "prompt_name": prompt_name,
                "issue_id": row["issue_id"],
                "audit_category": row["audit_category"],
                "raw_image_count": row["raw_image_count"],
                "ordinance_chars": row["ordinance_chars"],
                "completeness_label": _norm(pred.get("completeness_label")),
                "estimated_missing_share_0_to_1": pred.get("estimated_missing_share_0_to_1"),
                "missing_share_band": _norm(pred.get("missing_share_band")),
                "confidence_0_to_1": pred.get("confidence_0_to_1"),
                "evidence_signals": "|".join(_norm(x) for x in (pred.get("evidence_signals") or [])),
                "evidence_quotes": json.dumps(pred.get("evidence_quotes") or [], ensure_ascii=False),
                "short_explanation": _norm(pred.get("short_explanation")),
                "sample_md_path": row["sample_md_path"],
                "manual_completeness_label": row.get("manual_completeness_label", ""),
                "manual_artifact_assessment": row.get("manual_artifact_assessment", ""),
                "manual_review_notes": row.get("manual_review_notes", ""),
            }
            merged_rows.append(merged)
            prompt_result_rows.append(merged)

        _write_csv(
            prompt_out_dir / "predictions.csv",
            merged_rows,
            fieldnames=list(merged_rows[0].keys()),
        )
        prompt_summary = _summarize_prompt(merged_rows, prompt_name=prompt_name)
        (prompt_out_dir / "summary.json").write_text(json.dumps(prompt_summary, indent=2) + "\n", encoding="utf-8")
        all_prompt_summaries.append(prompt_summary)

    _write_csv(
        out_dir / "all_predictions.csv",
        prompt_result_rows,
        fieldnames=list(prompt_result_rows[0].keys()),
    )
    if any("gold_metrics" in summary for summary in all_prompt_summaries):
        all_prompt_summaries = sorted(
            all_prompt_summaries,
            key=lambda x: (
                x.get("gold_metrics", {}).get("exact_accuracy", -1.0),
                x.get("gold_metrics", {}).get("within_one_rate", -1.0),
                -x.get("gold_metrics", {}).get("mean_abs_rank_error", math.inf),
                x["separation_score"],
            ),
            reverse=True,
        )
    else:
        all_prompt_summaries = sorted(all_prompt_summaries, key=lambda x: x["separation_score"], reverse=True)
    (out_dir / "prompt_summaries.json").write_text(json.dumps(all_prompt_summaries, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Full-Ordinance Completeness Prompt Comparison",
        "",
        "This is a prompt-iteration experiment over the current parsed full-ordinance markdown artifacts.",
        "",
        "## Challenge Set",
        "",
        f"- Audit source: `{audit_csv}`",
        f"- Bundle root: `{bundle_root}`",
        f"- Sample size: `{len(sample_rows)}`",
        f"- Sample per category: `{int(args.sample_per_category)}`",
        f"- Gateway model: `{args.gateway_model}`",
        f"- Request model: `{args.request_model}`",
        "",
        "Category counts in sample:",
        "",
    ]
    if gold_csv:
        lines.insert(7, f"- Gold sample: `{gold_csv}`")
    sample_counts = Counter(r["audit_category"] for r in sample_rows)
    for cat in AUDIT_CATEGORY_ORDER:
        lines.append(f"- `{cat}`: `{sample_counts.get(cat, 0)}`")

    lines.extend(["", "## Prompt Ranking", ""])
    for summary in all_prompt_summaries:
        gold_metrics = summary.get("gold_metrics") or {}
        if gold_metrics:
            lines.append(
                f"- `{summary['prompt_name']}`: gold_exact_accuracy=`{gold_metrics['exact_accuracy']:.3f}`, gold_within_one_rate=`{gold_metrics['within_one_rate']:.3f}`, gold_mean_abs_rank_error=`{gold_metrics['mean_abs_rank_error']:.3f}`, separation_score=`{summary['separation_score']:.3f}`"
            )
        else:
            lines.append(f"- `{summary['prompt_name']}`: separation_score=`{summary['separation_score']:.3f}`")

    lines.extend(["", "## By Prompt", ""])
    for summary in all_prompt_summaries:
        lines.append(f"### {summary['prompt_name']}")
        lines.append("")
        gold_metrics = summary.get("gold_metrics") or {}
        if gold_metrics:
            lines.append(
                f"- gold_exact_accuracy=`{gold_metrics['exact_accuracy']:.3f}`, gold_within_one_rate=`{gold_metrics['within_one_rate']:.3f}`, gold_mean_abs_rank_error=`{gold_metrics['mean_abs_rank_error']:.3f}`, false_complete_rate=`{gold_metrics['false_complete_rate']:.3f}`, false_incomplete_rate=`{gold_metrics['false_incomplete_rate']:.3f}`"
            )
            lines.append(f"- gold_confusion={json.dumps(gold_metrics['confusion'], ensure_ascii=False)}")
        for cat in AUDIT_CATEGORY_ORDER:
            stats = summary["category_stats"][cat]
            lines.append(
                f"- `{cat}`: n=`{stats['n']}`, likely_or_definitely_incomplete_rate=`{stats['likely_or_definitely_incomplete_rate']:.3f}`, definitely_incomplete_rate=`{stats['definitely_incomplete_rate']:.3f}`, avg_estimated_missing_share=`{stats['avg_estimated_missing_share']:.3f}`"
            )
            lines.append(f"  label_counts={json.dumps(stats['label_counts'], ensure_ascii=False)}")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "sample_size": len(sample_rows), "prompt_count": len(prompt_paths)}, indent=2))


if __name__ == "__main__":
    main()
