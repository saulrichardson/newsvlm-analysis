#!/usr/bin/env python3
from __future__ import annotations

import argparse

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier import run_cluster_experiments, write_stage_summary

from frontier_fullcode_lab_common import add_run_args, resolve_layout_and_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frontier cluster experiments across ordinance, section, and clause representations.")
    add_run_args(parser, include_sources=False)
    args = parser.parse_args()
    layout, _ = resolve_layout_and_manifest(args, include_sources=False)
    summary = run_cluster_experiments(run_root=layout["run_root"])
    write_stage_summary(layout, "clusters", summary)
    print(f"built_frontier_clusters run_root={layout['run_root']} experiments={summary['experiment_count']}")


if __name__ == "__main__":
    main()
