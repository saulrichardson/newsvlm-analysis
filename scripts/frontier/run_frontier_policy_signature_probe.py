#!/usr/bin/env python3
from __future__ import annotations

import argparse

import frontier_bootstrap  # noqa: F401

from frontier_fullcode_lab_common import add_run_args, resolve_layout_and_manifest
from newsvlm_analysis.frontier import build_policy_signature_outputs, write_stage_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured ordinance policy-signature typologies.")
    add_run_args(parser, include_sources=False)
    parser.add_argument("--cluster-count", type=int, default=8, help="Number of policy-signature clusters.")
    args = parser.parse_args()

    layout, _ = resolve_layout_and_manifest(args, include_sources=False)
    summary = build_policy_signature_outputs(run_root=layout["run_root"], cluster_count=args.cluster_count)
    write_stage_summary(layout, "policy_signature", summary)
    print(
        f"built_frontier_policy_signature_probe run_root={layout['run_root']} "
        f"items={summary['item_count']} k={summary['cluster_count']}"
    )


if __name__ == "__main__":
    main()
