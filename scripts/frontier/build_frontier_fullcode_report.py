#!/usr/bin/env python3
from __future__ import annotations

import argparse

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier import build_report_bundle, write_stage_summary

from frontier_fullcode_lab_common import add_run_args, resolve_layout_and_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static frontier full-code report bundle.")
    add_run_args(parser, include_sources=False)
    args = parser.parse_args()
    layout, _ = resolve_layout_and_manifest(args, include_sources=False)
    summary = build_report_bundle(run_root=layout["run_root"])
    write_stage_summary(layout, "report", summary)
    print(f"built_frontier_report run_root={layout['run_root']} report={summary['report_path']}")


if __name__ == "__main__":
    main()
