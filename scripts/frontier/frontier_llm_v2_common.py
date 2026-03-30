#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier.llm_v2 import (
    DEFAULT_BASELINE_RUN_ROOT,
    DEFAULT_PROMPT_DIR,
    ensure_llm_v2_run_layout,
    initialize_llm_v2_manifest,
    llm_v2_layout_from_run_root,
)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", default="", help="Existing v2 run root. If omitted, a new run can be created.")
    parser.add_argument("--run-id", default="", help="Optional run id when creating a new run.")
    parser.add_argument("--base-dir", default="", help="Optional base dir for new runs.")
    parser.add_argument("--baseline-run-root", default="", help="Override baseline frontier run root.")
    parser.add_argument("--prompt-dir", default="", help="Override frontier_llm_v2 prompt directory.")
    parser.add_argument("--pilot-size", type=int, default=64, help="Pilot sample size recorded in the manifest.")
    parser.add_argument(
        "--execution-mode",
        default="batch_and_live_pilot",
        help="Execution mode label stored in the run manifest.",
    )


def resolve_layout_and_manifest(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, object]]:
    if str(getattr(args, "run_root", "")).strip():
        layout = llm_v2_layout_from_run_root(args.run_root)
    else:
        layout = ensure_llm_v2_run_layout(base_dir=args.base_dir or None, run_id=args.run_id or None)

    manifest_path = layout["manifests"] / "run_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return layout, payload

    payload = initialize_llm_v2_manifest(
        layout=layout,
        baseline_run_root=args.baseline_run_root or DEFAULT_BASELINE_RUN_ROOT,
        prompt_dir=args.prompt_dir or DEFAULT_PROMPT_DIR,
        pilot_size=int(getattr(args, "pilot_size", 64) or 64),
        execution_mode=str(getattr(args, "execution_mode", "") or "batch_and_live_pilot"),
    )
    return layout, payload
