#!/usr/bin/env python3
from __future__ import annotations

import argparse

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier import build_semantics_outputs, write_stage_summary

from frontier_fullcode_lab_common import add_run_args, resolve_layout_and_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frontier semantic features and hard-feature observations.")
    add_run_args(parser, include_sources=False)
    args = parser.parse_args()
    layout, _ = resolve_layout_and_manifest(args, include_sources=False)
    summary = build_semantics_outputs(run_root=layout["run_root"])
    write_stage_summary(layout, "semantics", summary)
    print(f"built_frontier_semantics run_root={layout['run_root']} clauses={summary['clause_rows']} features={summary['feature_rows']}")


if __name__ == "__main__":
    main()
