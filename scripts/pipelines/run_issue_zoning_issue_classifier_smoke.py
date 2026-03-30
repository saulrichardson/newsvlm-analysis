#!/usr/bin/env python3
"""
End-to-end smoke test for the issue-level zoning classifier workflow.

It verifies, using the keys in this repo's `.env`:
  1) request export (Paddle + VLM fallbacks; issue-level aggregation)
  2) live execution via vendor/agent-gateway (OpenAI + Gemini)
  3) asynchronous execution via OpenAI Batch API (submit + poll + download)
  4) rehydration into parsed outputs + per-issue files

This script intentionally uses a *tiny synthetic dataset* so it is safe/cheap.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _run(cmd: list[str], *, cwd: Path) -> str:
    _eprint("[run] " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-1200:]}")
    return p.stdout


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _free_local_port() -> int:
    """Pick an ephemeral local TCP port that is free *right now* (best-effort)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _make_synthetic_inputs(out_root: Path) -> dict[str, Path]:
    """
    Construct a minimal issue with 4 pages, each sourced from a different transcript backend:
      p1: Paddle markdown JSONL
      p2: per-page *.vlm.json
      p3: OpenAI per-box openai_results_shard*.jsonl
      p4: Gemini per-box gemini_results_shard*.jsonl
    """
    data_dir = out_root / "synthetic_inputs"
    unique_png = data_dir / "unique_png"
    vlm_root = data_dir / "vlm_pages"
    unique_png.mkdir(parents=True, exist_ok=True)
    vlm_root.mkdir(parents=True, exist_ok=True)

    # Canonical page ids.
    issue_slug = "abilene-reporter-news"
    issue_date = "1945-11-28"
    mon = "nov"
    dd = "28"
    yyyy = "1945"
    pages = [
        f"{issue_slug}-{mon}-{dd}-{yyyy}-p-1",
        f"{issue_slug}-{mon}-{dd}-{yyyy}-p-2",
        f"{issue_slug}-{mon}-{dd}-{yyyy}-p-3",
        f"{issue_slug}-{mon}-{dd}-{yyyy}-p-4",
    ]
    for pid in pages:
        (unique_png / f"{pid}.png").write_bytes(b"")  # exporter only needs filenames

    # Prompt: keep it tiny and schema-friendly.
    prompt_path = data_dir / "prompt_smoke.txt"
    _write_text(
        prompt_path,
        "\n".join(
            [
                "Return exactly ONE valid JSON object and nothing else.",
                'Keys required: "ok" (boolean), "issue_id" (string).',
                "Use the ISSUE_ID value shown below.",
                "",
                "Output example: {\"ok\":true,\"issue_id\":\"...\"}",
            ]
        )
        + "\n",
    )

    # Paddle markdown JSONL: page 1 only.
    paddle_jsonl = data_dir / "paddle_vl15_markdown.jsonl"
    _write_jsonl(
        paddle_jsonl,
        [
            {
                "slug": pages[0],
                "md_path": str(data_dir / f"{pages[0]}.md"),
                "status": "ok",
                "text_md": "PADDLE PAGE 1: ZONING ORDINANCE — BE IT ORDAINED. SECTION 1. This ordinance shall ...",
            }
        ],
    )

    # Per-page VLM file (page 2).
    (vlm_root / f"{pages[1]}.vlm.json").write_text(
        json.dumps(
            {
                "page_id": pages[1],
                "boxes": [
                    {
                        "id": 1,
                        "status": "ok",
                        "transcript": "VLM PAGE 2: Notice is hereby given (procedural).",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # OpenAI per-box results JSONL (page 3).
    openai_box_jsonl = data_dir / "openai_results_shard000.jsonl"
    _write_jsonl(
        openai_box_jsonl,
        [
            {
                "id": "smoke_openai_box_1",
                "custom_id": f"{pages[2]}:1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "status": "completed",
                        "output": [
                            {
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": json.dumps(
                                            {"status": "ok", "transcript": "OPENAI BOX: SECTION 2. Zoning district A shall ..."},
                                            ensure_ascii=False,
                                        ),
                                    }
                                ]
                            }
                        ],
                    },
                },
                "error": None,
            }
        ],
    )

    # Gemini per-box results JSONL (page 4).
    gemini_box_jsonl = data_dir / "gemini_results_shard000.jsonl"
    gemini_text = json.dumps({"status": "ok", "transcript": "GEMINI BOX: ARTICLE I. Definitions. Zoning ..." }, ensure_ascii=False)
    _write_jsonl(
        gemini_box_jsonl,
        [
            {
                "key": f"{pages[3]}:1",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        # Sometimes fenced in real runs; keep plain here.
                                        "text": gemini_text
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        ],
    )

    return {
        "data_dir": data_dir,
        "unique_png": unique_png,
        "vlm_root": vlm_root,
        "paddle_jsonl": paddle_jsonl,
        "openai_box_jsonl": openai_box_jsonl,
        "gemini_box_jsonl": gemini_box_jsonl,
        "prompt_path": prompt_path,
        "issue_id": f"{issue_slug}__{issue_date}",
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Smoke test issue-level zoning classifier workflows (gateway + OpenAI Batch).")
    ap.add_argument(
        "--output-dir",
        default="",
        help="Run directory (default: artifacts/runs/issue_zoning_issue_classifier_smoke_<timestamp>)",
    )
    ap.add_argument("--openai-model", default="gpt-5-mini", help="OpenAI model for Batch request bodies")
    ap.add_argument("--gateway-openai-model", default="openai:gpt-5-mini", help="Gateway model for live OpenAI smoke")
    ap.add_argument("--gateway-gemini-model", default="gemini:gemini-2.5-flash", help="Gateway model for live Gemini smoke")
    ap.add_argument("--gateway-timeout", type=float, default=90.0)
    ap.add_argument("--gateway-concurrency", type=int, default=1)
    ap.add_argument("--batch-poll-seconds", type=float, default=15.0)
    ap.add_argument("--batch-max-polls", type=int, default=6)
    ap.add_argument("--skip-openai-batch", action="store_true", help="Skip the OpenAI Batch portion (gateway only).")
    ap.add_argument(
        "--require-openai-batch-completion",
        action="store_true",
        help="Fail if the OpenAI Batch job does not complete within the poll window.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else (REPO_ROOT / "reports" / "runs" / f"issue_zoning_issue_classifier_smoke_{ts}")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _eprint(f"run_dir={run_dir}")

    inputs = _make_synthetic_inputs(run_dir)

    # Export requests (Mode B: page aggregation).
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_issue_zoning_issue_classifier_batch_requests.py"),
            "--output-dir",
            str(run_dir),
            "--prompt-path",
            str(inputs["prompt_path"]),
            "--openai-model",
            str(args.openai_model),
            "--unique-png-root",
            str(inputs["unique_png"]),
            "--paddle-vl15-jsonl",
            str(inputs["paddle_jsonl"]),
            "--vlm-page-roots",
            str(inputs["vlm_root"]),
            "--openai-box-results-jsonl",
            str(inputs["openai_box_jsonl"]),
            "--gemini-box-results-jsonl",
            str(inputs["gemini_box_jsonl"]),
            "--missing-page-policy",
            "require",
            "--issues-per-shard",
            "10",
        ],
        cwd=REPO_ROOT,
    )

    request_dir = run_dir / "requests"
    if not (request_dir / "openai_requests_shard000.jsonl").is_file():
        raise RuntimeError("Exporter did not produce requests/openai_requests_shard000.jsonl")

    # Live gateway run: OpenAI
    gw_openai_results = run_dir / "gateway_openai_results"
    gw_openai_port = _free_local_port()
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_openai_requests_via_gateway.py"),
            "--request-dir",
            str(request_dir),
            "--output-dir",
            str(gw_openai_results),
            "--model",
            str(args.gateway_openai_model),
            "--max-concurrency",
            str(int(args.gateway_concurrency)),
            "--timeout",
            str(float(args.gateway_timeout)),
            "--port",
            str(gw_openai_port),
        ],
        cwd=REPO_ROOT,
    )
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "rehydrate_issue_zoning_issue_classifier_results.py"),
            "--request-dir",
            str(request_dir),
            "--results-dir",
            str(gw_openai_results),
            "--output-dir",
            str(run_dir / "gateway_openai_outputs"),
        ],
        cwd=REPO_ROOT,
    )

    # Live gateway run: Gemini
    gw_gemini_results = run_dir / "gateway_gemini_results"
    gw_gemini_port = _free_local_port()
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_openai_requests_via_gateway.py"),
            "--request-dir",
            str(request_dir),
            "--output-dir",
            str(gw_gemini_results),
            "--model",
            str(args.gateway_gemini_model),
            "--max-concurrency",
            str(int(args.gateway_concurrency)),
            "--timeout",
            str(float(args.gateway_timeout)),
            "--port",
            str(gw_gemini_port),
        ],
        cwd=REPO_ROOT,
    )
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "rehydrate_issue_zoning_issue_classifier_results.py"),
            "--request-dir",
            str(request_dir),
            "--results-dir",
            str(gw_gemini_results),
            "--output-dir",
            str(run_dir / "gateway_gemini_outputs"),
        ],
        cwd=REPO_ROOT,
    )

    # OpenAI Batch API run (async) + download.
    if args.skip_openai_batch:
        _eprint("skip_openai_batch=true")
        return

    record_path = run_dir / "openai_batch_submission" / f"batch_jobs_{ts}.jsonl"
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "openai_batch_submit_curl.py"),
            "--request-dir",
            str(request_dir),
            "--record-path",
            str(record_path),
            "--env-file",
            str(REPO_ROOT / ".env"),
            "--endpoint",
            "/v1/responses",
            "--completion-window",
            "24h",
            "--run-label",
            f"issue_zoning_smoke_{ts}",
            "--model-label",
            str(args.openai_model),
        ],
        cwd=REPO_ROOT,
    )

    completed_dir = run_dir / "openai_batch_submission" / "completed_outputs"
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "openai_batch_monitor_curl.py"),
            "--record-path",
            str(record_path),
            "--env-file",
            str(REPO_ROOT / ".env"),
            "--output-dir",
            str(completed_dir),
            "--poll-seconds",
            str(float(args.batch_poll_seconds)),
            "--max-polls",
            str(int(args.batch_max_polls)),
            "--stop-when-final",
        ],
        cwd=REPO_ROOT,
    )

    outputs = sorted(completed_dir.glob("*_output.jsonl"))
    if not outputs:
        msg = (
            "OpenAI Batch job did not reach a completed state within the smoke-test poll window.\n"
            f"- record_path: {record_path}\n"
            f"- latest_status_snapshot: {run_dir / 'openai_batch_submission' / 'latest_status_snapshot.json'}\n"
            "Re-run the monitor step later with a larger --batch-max-polls, for example:\n"
            f"  python scripts/platform/openai_batch_monitor_curl.py --record-path {record_path} --env-file {REPO_ROOT / '.env'} --stop-when-final\n"
        )
        if args.require_openai_batch_completion:
            raise RuntimeError(msg)
        _eprint(msg)
        _eprint("smoke_ok=true (batch_pending=true)")
        return

    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "rehydrate_issue_zoning_issue_classifier_results.py"),
            "--request-dir",
            str(request_dir),
            "--results-dir",
            str(completed_dir),
            "--output-dir",
            str(run_dir / "openai_batch_outputs"),
        ],
        cwd=REPO_ROOT,
    )

    _eprint("smoke_ok=true")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        _eprint(f"fatal: {exc}")
        sys.exit(1)
