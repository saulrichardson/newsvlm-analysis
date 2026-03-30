#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier.legal_complexity import (
    DEFAULT_PILOT_RUN_ROOT,
    DEFAULT_SOURCE_RUN_ROOT,
    LegalComplexityPaths,
    build_fullsample_legal_complexity,
    ensure_legal_complexity_layout,
    initialize_legal_complexity_manifest,
    legal_complexity_layout_from_run_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full-sample legal complexity pipeline.")
    parser.add_argument("--run-root", default="", help="Existing legal-complexity run root.")
    parser.add_argument("--run-id", default="", help="Optional run id when creating a new run.")
    parser.add_argument("--base-dir", default="", help="Optional base dir for new runs.")
    parser.add_argument(
        "--source-run-root",
        default=str(DEFAULT_SOURCE_RUN_ROOT),
        help="Full full-code source run root with corpus/structure artifacts.",
    )
    parser.add_argument(
        "--pilot-run-root",
        default=str(DEFAULT_PILOT_RUN_ROOT),
        help="Optional LLM-v2 pilot run root for bridge analyses.",
    )
    args = parser.parse_args()

    if str(args.run_root).strip():
        layout = legal_complexity_layout_from_run_root(args.run_root)
    else:
        layout = ensure_legal_complexity_layout(base_dir=args.base_dir or None, run_id=args.run_id or None)

    initialize_legal_complexity_manifest(
        layout=layout,
        paths=LegalComplexityPaths(
            source_run_root=str(Path(args.source_run_root).expanduser()),
            pilot_run_root=str(Path(args.pilot_run_root).expanduser()),
        ),
    )
    summary = build_fullsample_legal_complexity(
        run_root=layout["run_root"],
        source_run_root=args.source_run_root,
        pilot_run_root=args.pilot_run_root,
    )
    print(f"completed_legal_complexity_fullsample run_root={layout['run_root']} doc_count={summary['doc_count']}")


if __name__ == "__main__":
    main()
