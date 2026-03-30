#!/usr/bin/env python3
from __future__ import annotations

import argparse

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier import build_review_packets, write_stage_summary

from frontier_fullcode_lab_common import add_run_args, resolve_layout_and_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frontier review packets for jurisdiction, sections, clauses, and clusters.")
    add_run_args(parser, include_sources=False)
    args = parser.parse_args()
    layout, _ = resolve_layout_and_manifest(args, include_sources=False)
    summary = build_review_packets(run_root=layout["run_root"])
    write_stage_summary(layout, "review", summary)
    print(
        f"built_frontier_review_packets run_root={layout['run_root']} "
        f"jurisdiction={summary['jurisdiction_review_count']} clauses={summary['clause_review_count']}"
    )


if __name__ == "__main__":
    main()
