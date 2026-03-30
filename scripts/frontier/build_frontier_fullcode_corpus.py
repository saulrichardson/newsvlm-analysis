#!/usr/bin/env python3
from __future__ import annotations

import argparse

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier import build_fullcode_corpus, write_stage_summary

from frontier_fullcode_lab_common import add_run_args, resolve_layout_and_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical full-code frontier corpus.")
    add_run_args(parser, include_sources=True)
    parser.add_argument("--max-issues", type=int, default=0, help="Optional max issues for a smoke-scale build.")
    args = parser.parse_args()

    layout, paths = resolve_layout_and_manifest(args, include_sources=True)
    summary = build_fullcode_corpus(run_root=layout["run_root"], paths=paths, max_issues=args.max_issues or None)
    write_stage_summary(layout, "corpus", summary)
    print(f"built_frontier_corpus run_root={layout['run_root']} issues={summary['issue_count']} dedup_groups={summary['dedup_group_count']}")


if __name__ == "__main__":
    main()
