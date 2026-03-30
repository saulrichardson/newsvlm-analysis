#!/usr/bin/env python3
from __future__ import annotations

import argparse

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier import build_geo_outputs, write_stage_summary

from frontier_fullcode_lab_common import add_run_args, resolve_layout_and_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frontier geography outputs and mismatch audit.")
    add_run_args(parser, include_sources=False)
    args = parser.parse_args()
    layout, _ = resolve_layout_and_manifest(args, include_sources=False)
    summary = build_geo_outputs(run_root=layout["run_root"])
    write_stage_summary(layout, "geo", summary)
    print(
        f"built_frontier_geo run_root={layout['run_root']} "
        f"match={summary['match_count']} mismatch={summary['mismatch_count']} missing={summary['missing_jurisdiction_count']}"
    )


if __name__ == "__main__":
    main()
