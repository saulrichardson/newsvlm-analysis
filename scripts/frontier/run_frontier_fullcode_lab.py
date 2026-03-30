#!/usr/bin/env python3
from __future__ import annotations

import argparse

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier import (
    build_embedding_outputs,
    build_fullcode_corpus,
    build_geo_outputs,
    build_policy_signature_outputs,
    build_report_bundle,
    build_review_packets,
    build_semantics_outputs,
    build_story_outputs,
    build_structure_outputs,
    run_cluster_experiments,
    write_stage_summary,
)

from frontier_fullcode_lab_common import add_run_args, resolve_layout_and_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Frontier Full-Code Research Lab pipeline.")
    add_run_args(parser, include_sources=True)
    parser.add_argument("--max-issues", type=int, default=0, help="Optional max issues for a smoke-scale run.")
    args = parser.parse_args()

    layout, paths = resolve_layout_and_manifest(args, include_sources=True)
    stage_summaries = {
        "corpus": build_fullcode_corpus(run_root=layout["run_root"], paths=paths, max_issues=args.max_issues or None),
    }
    write_stage_summary(layout, "corpus", stage_summaries["corpus"])
    stage_summaries["geo"] = build_geo_outputs(run_root=layout["run_root"])
    write_stage_summary(layout, "geo", stage_summaries["geo"])
    stage_summaries["structure"] = build_structure_outputs(run_root=layout["run_root"])
    write_stage_summary(layout, "structure", stage_summaries["structure"])
    stage_summaries["semantics"] = build_semantics_outputs(run_root=layout["run_root"])
    write_stage_summary(layout, "semantics", stage_summaries["semantics"])
    stage_summaries["embeddings"] = build_embedding_outputs(run_root=layout["run_root"])
    write_stage_summary(layout, "embeddings", stage_summaries["embeddings"])
    stage_summaries["clusters"] = run_cluster_experiments(run_root=layout["run_root"])
    write_stage_summary(layout, "clusters", stage_summaries["clusters"])
    stage_summaries["policy_signature"] = build_policy_signature_outputs(run_root=layout["run_root"])
    write_stage_summary(layout, "policy_signature", stage_summaries["policy_signature"])
    stage_summaries["review"] = build_review_packets(run_root=layout["run_root"])
    write_stage_summary(layout, "review", stage_summaries["review"])
    stage_summaries["story"] = build_story_outputs(run_root=layout["run_root"])
    write_stage_summary(layout, "story", stage_summaries["story"])
    stage_summaries["report"] = build_report_bundle(run_root=layout["run_root"])
    write_stage_summary(layout, "report", stage_summaries["report"])
    print(f"completed_frontier_fullcode_lab run_root={layout['run_root']}")


if __name__ == "__main__":
    main()
