#!/usr/bin/env python3
"""
Run a small issue-level ordinance extraction pilot comparing OCR-only vs OCR+all-issue-images.

This runner is intentionally post-gate and issue-level:
  - read a curated issue manifest grounded in the audited set
  - read the full OCR transcript for each issue
  - parse every page_id from the transcript in order
  - optionally fetch every original PNG for that issue from Torch
  - build OpenAI-Batch-shaped request JSONL for OCR-only and OCR+all-images variants
  - execute both variants through the local vendor/agent-gateway
  - parse the freeform SOURCE / ORDINANCE TEXT / NOTES contract
  - emit comparison artifacts for manual review
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
import subprocess
import shlex
from pathlib import Path
from typing import Any

from fullnewspaper_v3_common import extract_openai_output_text, iter_jsonl, run_gateway_requests


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{k: _norm(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Compare issue-level ordinance extraction using OCR only versus OCR plus all issue page images.")
    ap.add_argument(
        "--cases-csv",
        default=str(repo_root / "artifacts/scratch" / "page_image_ocr_pilot_20260305" / "issue_level_cases.csv"),
    )
    ap.add_argument(
        "--issue-txt-dir",
        default=str(repo_root / "artifacts/scratch" / "stage2_prompt_eval_full_inputs_20260305" / "issue_txt"),
    )
    ap.add_argument(
        "--ocr-only-prompt-path",
        default=str(repo_root / "prompts" / "transcription_v12_issue_ocr_only_postgate.txt"),
    )
    ap.add_argument(
        "--ocr-image-prompt-path",
        default=str(repo_root / "prompts" / "transcription_v13_issue_multimage_postgate.txt"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(repo_root / "artifacts/scratch" / "issue_image_ocr_pilot_20260306" / "run_001"),
    )
    ap.add_argument("--env-file", default=str(repo_root / ".env"))
    ap.add_argument("--openai-model", default="gpt-5.4")
    ap.add_argument("--gateway-model", default="openai:gpt-5.4")
    ap.add_argument("--max-concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--reasoning-effort", default="", help="Optional reasoning.effort passed through the gateway request body")
    ap.add_argument("--max-output-tokens", type=int, default=0, help="Optional max_output_tokens passed through the gateway request body")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fetch-images", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--torch-host", default="torch")
    ap.add_argument("--torch-png-root", default="/scratch/sxr203/newspaper-downloads/dedupe-webp/unique_png")
    ap.add_argument("--gateway-runner", default=str(repo_root / "scripts" / "run_openai_requests_via_gateway.py"))
    ap.add_argument("--gateway-pythonpath", default=str(repo_root / "vendor/agent-gateway" / "src"))
    return ap.parse_args()


def _extract_issue_pages(issue_transcript: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"(?ms)^=== PAGE START ===\npage_id=(?P<page_id>[^\n]+)\npage_num=(?P<page_num>[^\n]*)\ntext_source=(?P<text_source>[^\n]*)\npage_text:\n(?P<page_text>.*?)(?=^=== PAGE START ===|\Z)"
    )
    pages: list[dict[str, str]] = []
    for m in pattern.finditer(issue_transcript):
        pages.append(
            {
                "page_id": _norm(m.group("page_id")),
                "page_num": _norm(m.group("page_num")),
                "text_source": _norm(m.group("text_source")),
                "page_text": _norm(m.group("page_text")),
            }
        )
    if not pages:
        raise SystemExit("No pages found in issue transcript")
    return pages


def _page_image_order_block(pages: list[dict[str, str]]) -> str:
    lines = []
    for idx, page in enumerate(pages, start=1):
        lines.append(
            f"{idx}. page_id={page['page_id']} page_num={page['page_num']} text_source={page['text_source']}"
        )
    return "\n".join(lines)


def _fill_issue_prompt(template: str, *, source_name: str, issue_transcript: str, pages: list[dict[str, str]]) -> str:
    required = ("{SOURCE_NAME}", "{ISSUE_TRANSCRIPT}", "{PAGE_IMAGE_ORDER}")
    for token in required:
        if token not in template:
            raise SystemExit(f"Prompt template missing placeholder {token}")
    return (
        template.replace("{SOURCE_NAME}", source_name)
        .replace("{ISSUE_TRANSCRIPT}", issue_transcript)
        .replace("{PAGE_IMAGE_ORDER}", _page_image_order_block(pages))
    )


def _image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type or 'image/png'};base64,{payload}"


def _build_request_line(
    *,
    custom_id: str,
    openai_model: str,
    prompt: str,
    image_paths: list[Path],
    pages: list[dict[str, str]],
    reasoning_effort: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for idx, (page, image_path) in enumerate(zip(pages, image_paths, strict=True), start=1):
        content.append(
            {
                "type": "input_text",
                "text": f"Attached page image {idx} of {len(image_paths)} corresponds to page_id={page['page_id']} page_num={page['page_num']}.",
            }
        )
        content.append({"type": "input_image", "image_url": _image_data_url(image_path)})
    body: dict[str, Any] = {
        "model": str(openai_model),
        "input": [{"role": "user", "content": content}],
        "stream": False,
    }
    effort = _norm(reasoning_effort).lower()
    if effort:
        body["reasoning"] = {"effort": effort}
    if int(max_output_tokens) > 0:
        body["max_output_tokens"] = int(max_output_tokens)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_contract(text: str) -> dict[str, Any]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(
        r"^\s*SOURCE:\s*(?P<source>.*?)\n\s*=== ORDINANCE TEXT ===\s*\n(?P<ord>.*?)\n\s*--- NOTES ---\s*\n(?P<notes>.*)\Z",
        raw,
        flags=re.S,
    )
    if not m:
        return {"contract_ok": 0, "source_line": "", "ordinance_text": "", "notes_text": ""}
    return {
        "contract_ok": 1,
        "source_line": _norm(m.group("source")),
        "ordinance_text": _norm(m.group("ord")),
        "notes_text": _norm(m.group("notes")),
    }


def _fetch_image(torch_host: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.is_file():
        return
    subprocess.run(["scp", f"{torch_host}:{remote_path}", str(local_path)], check=True)


def _resolve_remote_image_path(torch_host: str, torch_png_root: str, page_id: str) -> str:
    root = _norm(torch_png_root).rstrip("/")
    name = f"{page_id}.png"
    direct = f"{root}/{name}"
    quoted_root = shlex.quote(root)
    quoted_name = shlex.quote(name)
    quoted_direct = shlex.quote(direct)
    probe = (
        f"if [ -f {quoted_direct} ]; then "
        f"printf '%s\\n' {quoted_direct}; "
        f"else find {quoted_root} -name {quoted_name} -print -quit; fi"
    )
    proc = subprocess.run(
        ["ssh", str(torch_host), probe],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = _norm(proc.stdout)
    if not remote:
        raise SystemExit(f"Could not resolve remote image path for {page_id} under {torch_png_root}")
    return remote


def _render_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        expected = bool(int(row["expected_positive"]))
        predicted = bool(int(row["predicted_positive"]))
        if expected and predicted:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif expected and not predicted:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "cases": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
    }


def _write_report(path: Path, *, cases: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Issue Image OCR Pilot")
    lines.append("")
    lines.append("This compares issue-level ordinance extraction using OCR only versus OCR plus all issue page images in transcript order.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for variant in ("ocr_only", "ocr_plus_images"):
        stats = summary[variant]
        lines.append(
            f"- `{variant}`: tp={stats['tp']} fp={stats['fp']} fn={stats['fn']} tn={stats['tn']} precision={stats['precision']} recall={stats['recall']}"
        )
    lines.append("")
    lines.append("## Case Comparison")
    lines.append("")
    for row in cases:
        lines.append(f"### {row['issue_id']}")
        lines.append("")
        lines.append(f"- Pages in issue transcript: `{row['issue_page_count']}`")
        lines.append(f"- Images attached for OCR+images: `{row['issue_image_count']}`")
        lines.append(f"- Expected positive: `{row['expected_positive']}`")
        lines.append(f"- Manual label: `{row['manual_label']}`")
        lines.append(f"- Manual note: {row['manual_note']}")
        lines.append(f"- OCR-only predicted positive: `{row['ocr_only_predicted_positive']}`")
        lines.append(f"- OCR+images predicted positive: `{row['ocr_plus_images_predicted_positive']}`")
        lines.append(f"- OCR-only notes: {row['ocr_only_notes_text'] or '(empty)'}")
        lines.append(f"- OCR+images notes: {row['ocr_plus_images_notes_text'] or '(empty)'}")
        lines.append(f"- OCR-only ordinance chars: `{row['ocr_only_ordinance_chars']}`")
        lines.append(f"- OCR+images ordinance chars: `{row['ocr_plus_images_ordinance_chars']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    cases_csv = Path(args.cases_csv).expanduser().resolve()
    issue_txt_dir = Path(args.issue_txt_dir).expanduser().resolve()
    ocr_only_prompt_path = Path(args.ocr_only_prompt_path).expanduser().resolve()
    ocr_image_prompt_path = Path(args.ocr_image_prompt_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    env_file = Path(args.env_file).expanduser().resolve()
    gateway_runner = Path(args.gateway_runner).expanduser().resolve()
    gateway_pythonpath = Path(args.gateway_pythonpath).expanduser().resolve()

    if not cases_csv.is_file():
        raise SystemExit(f"Missing cases csv: {cases_csv}")
    if not issue_txt_dir.is_dir():
        raise SystemExit(f"Missing issue transcript dir: {issue_txt_dir}")
    if not ocr_only_prompt_path.is_file():
        raise SystemExit(f"Missing OCR-only prompt: {ocr_only_prompt_path}")
    if not ocr_image_prompt_path.is_file():
        raise SystemExit(f"Missing OCR+image prompt: {ocr_image_prompt_path}")
    if not env_file.is_file():
        raise SystemExit(f"Missing env file: {env_file}")

    ocr_only_template = ocr_only_prompt_path.read_text(encoding="utf-8")
    ocr_image_template = ocr_image_prompt_path.read_text(encoding="utf-8")
    cases = _read_csv(cases_csv)
    if not cases:
        raise SystemExit("No pilot cases found")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    requests_root = output_dir / "requests"
    results_root = output_dir / "results"
    rendered_root = output_dir / "rendered_outputs"
    issue_text_root = output_dir / "issue_ocr"

    manifest_rows: list[dict[str, Any]] = []
    request_rows: dict[str, list[dict[str, Any]]] = {"ocr_only": [], "ocr_plus_images": []}

    for case in cases:
        issue_id = case["issue_id"]
        transcript_path = issue_txt_dir / f"{issue_id}.txt"
        if not transcript_path.is_file():
            raise SystemExit(f"Missing issue transcript: {transcript_path}")
        issue_transcript = transcript_path.read_text(encoding="utf-8", errors="replace")
        pages = _extract_issue_pages(issue_transcript)
        _render_text(issue_text_root / f"{issue_id}.txt", issue_transcript)

        local_image_paths: list[Path] = []
        for page in pages:
            page_id = page["page_id"]
            remote_path = _resolve_remote_image_path(str(args.torch_host), str(args.torch_png_root), page_id)
            local_path = images_dir / issue_id / f"{page_id}.png"
            if bool(args.fetch_images):
                _fetch_image(str(args.torch_host), remote_path, local_path)
            if not local_path.is_file():
                raise SystemExit(f"Missing local image for {page_id}: {local_path}")
            local_image_paths.append(local_path)

        manifest_rows.append(
            {
                "issue_id": issue_id,
                "issue_page_count": len(pages),
                "issue_image_count": len(local_image_paths),
                "ocr_chars": len(issue_transcript),
                "image_bytes_total": sum(p.stat().st_size for p in local_image_paths),
                "expected_positive": case["expected_positive"],
                "manual_label": case["manual_label"],
                "manual_note": case["manual_note"],
                "page_ids": "|".join(page["page_id"] for page in pages),
            }
        )

        ocr_only_prompt = _fill_issue_prompt(
            ocr_only_template,
            source_name=issue_id,
            issue_transcript=issue_transcript,
            pages=pages,
        )
        ocr_image_prompt = _fill_issue_prompt(
            ocr_image_template,
            source_name=issue_id,
            issue_transcript=issue_transcript,
            pages=pages,
        )
        request_rows["ocr_only"].append(
            _build_request_line(
                custom_id=f"{issue_id}__ocr_only",
                openai_model=str(args.openai_model),
                prompt=ocr_only_prompt,
                image_paths=[],
                pages=[],
                reasoning_effort=str(args.reasoning_effort),
                max_output_tokens=int(args.max_output_tokens),
            )
        )
        request_rows["ocr_plus_images"].append(
            _build_request_line(
                custom_id=f"{issue_id}__ocr_plus_images",
                openai_model=str(args.openai_model),
                prompt=ocr_image_prompt,
                image_paths=local_image_paths,
                pages=pages,
                reasoning_effort=str(args.reasoning_effort),
                max_output_tokens=int(args.max_output_tokens),
            )
        )

    _write_csv(
        output_dir / "build_manifest.csv",
        manifest_rows,
        fieldnames=[
            "issue_id",
            "issue_page_count",
            "issue_image_count",
            "ocr_chars",
            "image_bytes_total",
            "expected_positive",
            "manual_label",
            "manual_note",
            "page_ids",
        ],
    )

    for variant, rows in request_rows.items():
        _write_jsonl(requests_root / variant / "openai_requests_shard000.jsonl", rows)
        run_gateway_requests(
            request_dir=requests_root / variant,
            output_dir=results_root / variant,
            runner_path=gateway_runner,
            model=str(args.gateway_model),
            gateway_pythonpath=gateway_pythonpath,
            timeout_s=float(args.timeout),
            concurrency=int(args.max_concurrency),
            gov_env_path=env_file,
            skip_existing=bool(args.skip_existing),
            dry_run=False,
        )

    prediction_rows: list[dict[str, Any]] = []
    per_variant_rows: dict[str, list[dict[str, Any]]] = {"ocr_only": [], "ocr_plus_images": []}
    result_maps: dict[str, dict[str, dict[str, Any]]] = {}

    for variant in ("ocr_only", "ocr_plus_images"):
        result_map: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(results_root / variant / "openai_results_shard000.jsonl"):
            custom_id = _norm(row.get("custom_id"))
            body = ((row.get("response") or {}).get("body") or {})
            text = extract_openai_output_text(body)
            parsed = _parse_contract(text)
            result_map[custom_id] = {
                "raw_text": text,
                **parsed,
                "ordinance_chars": len(parsed["ordinance_text"]),
                "predicted_positive": int(bool(parsed["ordinance_text"])),
            }
        result_maps[variant] = result_map

    manifest_map = {row["issue_id"]: row for row in manifest_rows}
    for case in cases:
        issue_id = case["issue_id"]
        manifest = manifest_map[issue_id]
        row: dict[str, Any] = {
            "issue_id": issue_id,
            "issue_page_count": manifest["issue_page_count"],
            "issue_image_count": manifest["issue_image_count"],
            "expected_positive": int(case["expected_positive"]),
            "manual_label": case["manual_label"],
            "manual_note": case["manual_note"],
        }
        for variant in ("ocr_only", "ocr_plus_images"):
            custom_id = f"{issue_id}__{variant}"
            parsed = result_maps[variant].get(custom_id)
            if parsed is None:
                raise SystemExit(f"Missing result for {custom_id}")
            _render_text(rendered_root / variant / f"{issue_id}.txt", parsed["raw_text"])
            row[f"{variant}_contract_ok"] = parsed["contract_ok"]
            row[f"{variant}_predicted_positive"] = parsed["predicted_positive"]
            row[f"{variant}_ordinance_chars"] = parsed["ordinance_chars"]
            row[f"{variant}_notes_text"] = parsed["notes_text"]
            row[f"{variant}_source_line"] = parsed["source_line"]
            per_variant_rows[variant].append(
                {
                    "issue_id": issue_id,
                    "expected_positive": int(case["expected_positive"]),
                    "predicted_positive": parsed["predicted_positive"],
                    "ordinance_chars": parsed["ordinance_chars"],
                    "contract_ok": parsed["contract_ok"],
                    "manual_label": case["manual_label"],
                }
            )
        prediction_rows.append(row)

    _write_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        fieldnames=[
            "issue_id",
            "issue_page_count",
            "issue_image_count",
            "expected_positive",
            "manual_label",
            "manual_note",
            "ocr_only_contract_ok",
            "ocr_only_predicted_positive",
            "ocr_only_ordinance_chars",
            "ocr_only_notes_text",
            "ocr_only_source_line",
            "ocr_plus_images_contract_ok",
            "ocr_plus_images_predicted_positive",
            "ocr_plus_images_ordinance_chars",
            "ocr_plus_images_notes_text",
            "ocr_plus_images_source_line",
        ],
    )

    summary = {
        "ocr_only": _summarize_variant(per_variant_rows["ocr_only"]),
        "ocr_plus_images": _summarize_variant(per_variant_rows["ocr_plus_images"]),
        "images_dir": str(images_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(output_dir / "report.md", cases=prediction_rows, summary=summary)

    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
