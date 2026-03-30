#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import frontier_bootstrap  # noqa: F401

from newsvlm_analysis.frontier.llm_v2 import (
    apply_llm_cluster_labels,
    build_cluster_label_request_sources,
    build_llm_v2_embeddings,
    build_llm_v2_research_outputs,
    build_llm_v2_review_packets,
    build_llm_v2_semantics,
    build_llm_v2_story_and_report,
    build_policy_signature_outputs,
    evaluate_llm_v2_review_quality,
    export_llm_v2_requests,
    prepare_llm_v2_documents,
    rehydrate_llm_v2_results,
    run_frontier_llm_v2_live_pilot,
    run_llm_v2_clusters,
)

from frontier_llm_v2_common import add_run_args, resolve_layout_and_manifest


ALL_EXPORTABLE_STAGES = (
    "section_segmentation",
    "document_purpose",
    "section_motive",
    "clause_extraction",
    "clause_enrichment",
    "section_embedding",
    "section_cluster_label",
    "section_local_cluster_label",
    "clause_cluster_label",
)


def _parse_stage_list(raw: str) -> list[str]:
    value = str(raw or "").strip().lower()
    if not value:
        raise SystemExit("Stage list cannot be empty.")
    if value == "all":
        return list(ALL_EXPORTABLE_STAGES)
    if value == "semantic":
        return [
            "section_segmentation",
            "document_purpose",
            "section_motive",
            "clause_extraction",
            "clause_enrichment",
            "section_embedding",
        ]
    if value == "cluster-labels":
        return ["section_cluster_label", "section_local_cluster_label", "clause_cluster_label"]
    stages = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [stage for stage in stages if stage not in ALL_EXPORTABLE_STAGES]
    if invalid:
        raise SystemExit(f"Unsupported stages: {invalid}. Valid stages: {list(ALL_EXPORTABLE_STAGES)}")
    return stages


def _postprocess_sample(
    *,
    run_root: Path,
    sample: str,
    with_cluster_label_sources: bool,
    with_policy_signatures: bool,
    with_review: bool,
    with_quality_gates: bool,
    with_story_report: bool,
) -> dict[str, object]:
    summaries: dict[str, object] = {}
    summaries["semantics"] = build_llm_v2_semantics(run_root=run_root, sample=sample)
    summaries["embeddings"] = build_llm_v2_embeddings(run_root=run_root, sample=sample)
    summaries["clusters"] = run_llm_v2_clusters(run_root=run_root, sample=sample)
    if with_cluster_label_sources:
        summaries["cluster_label_request_sources"] = build_cluster_label_request_sources(run_root=run_root)
    if with_policy_signatures:
        summaries["policy_signatures"] = build_policy_signature_outputs(run_root=run_root, cluster_count=6, sample=sample)
    if with_review:
        summaries["review"] = build_llm_v2_review_packets(run_root=run_root, sample=sample)
    if with_quality_gates:
        summaries["quality_gates"] = evaluate_llm_v2_review_quality(run_root=run_root)
    if with_story_report:
        summaries["story_report"] = build_llm_v2_story_and_report(run_root=run_root, sample=sample)
    return summaries


def _load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def _run_cluster_label_live(
    *,
    run_root: Path,
    env_file: Path,
    max_concurrency: int,
    timeout: float,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    stages = ("section_cluster_label", "section_local_cluster_label", "clause_cluster_label")
    summaries: dict[str, object] = {}
    for stage in stages:
        export_llm_v2_requests(run_root=run_root, sample="pilot", stage=stage)
        subprocess.run(
            [
                "python",
                str(repo_root / "scripts" / "run_openai_responses_requests_sync.py"),
                "--request-dir",
                str(run_root / "requests" / "pilot" / stage),
                "--output-dir",
                str(run_root / "results" / "pilot" / stage),
                "--env-file",
                str(env_file),
                "--max-concurrency",
                str(max_concurrency),
                "--timeout",
                str(timeout),
            ],
            check=True,
            cwd=str(repo_root),
            env={**os.environ, **_load_env_file(env_file)},
        )
        summaries[stage] = rehydrate_llm_v2_results(run_root=run_root, sample="pilot", stage=stage, allow_errors=False)
    summaries["applied"] = apply_llm_cluster_labels(run_root=run_root)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Frontier LLM v2 pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Create or load a Frontier LLM v2 run root and manifest.")
    add_run_args(init_p)

    prep_p = sub.add_parser("prepare-docs", help="Freeze the baseline-linked v2 document sample and pilot split.")
    add_run_args(prep_p)
    prep_p.add_argument(
        "--pilot-region-quotas-json",
        default="",
        help='Optional JSON object overriding pilot region quotas, e.g. {"northeast":1,"midwest":1,"south":1,"west":1}.',
    )

    export_p = sub.add_parser("export-requests", help="Export batch-style requests for one or more stages.")
    add_run_args(export_p)
    export_p.add_argument("--sample", choices=["pilot", "full"], required=True)
    export_p.add_argument("--stages", required=True, help='Comma-separated stages, or one of: "semantic", "cluster-labels", "all".')
    export_p.add_argument("--requests-per-shard", type=int, default=1000)
    export_p.add_argument("--max-bytes-per-shard", type=int, default=180_000_000)
    export_p.add_argument("--source-filter-path", default="", help="Optional parquet/csv/json/jsonl file listing source IDs to include.")
    export_p.add_argument("--source-filter-column", default="", help="Column to read from --source-filter-path and match against the stage source.")
    export_p.add_argument("--append", action="store_true", help="Append new request shards after existing ones instead of replacing the stage directory.")

    rehydrate_p = sub.add_parser("rehydrate-results", help="Rehydrate result shards for one or more stages.")
    add_run_args(rehydrate_p)
    rehydrate_p.add_argument("--sample", choices=["pilot", "full"], required=True)
    rehydrate_p.add_argument("--stages", required=True, help='Comma-separated stages, or one of: "semantic", "cluster-labels", "all".')
    rehydrate_p.add_argument("--allow-errors", action="store_true")
    rehydrate_p.add_argument("--allow-partial-results", action="store_true")

    post_p = sub.add_parser("postprocess", help="Build downstream semantic, embedding, clustering, review, and report artifacts from rehydrated outputs.")
    add_run_args(post_p)
    post_p.add_argument("--sample", choices=["pilot", "full"], required=True)
    post_p.add_argument("--skip-cluster-label-sources", action="store_true")
    post_p.add_argument("--skip-policy-signatures", action="store_true")
    post_p.add_argument("--skip-review", action="store_true")
    post_p.add_argument("--skip-quality-gates", action="store_true")
    post_p.add_argument("--skip-story-report", action="store_true")

    label_src_p = sub.add_parser("build-cluster-label-sources", help="Build request-source tables for cluster labeling.")
    add_run_args(label_src_p)

    apply_labels_p = sub.add_parser("apply-cluster-labels", help="Apply rehydrated LLM cluster labels to assignments.")
    add_run_args(apply_labels_p)

    review_p = sub.add_parser("build-review", help="Build review packets for a given sample.")
    add_run_args(review_p)
    review_p.add_argument("--sample", choices=["pilot", "full"], required=True)

    research_p = sub.add_parser("build-research", help="Build research-oriented analysis outputs for a given sample.")
    add_run_args(research_p)
    research_p.add_argument("--sample", choices=["pilot", "full"], required=True)

    gates_p = sub.add_parser("quality-gates", help="Evaluate pilot quality-gate status from review packets.")
    add_run_args(gates_p)

    report_p = sub.add_parser("build-story-report", help="Build story outputs, report bundle, and baseline comparison appendix.")
    add_run_args(report_p)
    report_p.add_argument("--sample", choices=["pilot", "full"], default="", help="Optional explicit sample target.")

    live_p = sub.add_parser("live-pilot", help="Run the pilot live through gateway/OpenAI and postprocess it.")
    add_run_args(live_p)
    live_p.add_argument("--env-file", default=".env", help="Path to .env with OpenAI key(s).")
    live_p.add_argument("--gateway-model", default="openai:gpt-5-mini")
    live_p.add_argument("--gateway-port", type=int, default=8000)
    live_p.add_argument("--max-concurrency", type=int, default=4)
    live_p.add_argument("--timeout", type=float, default=600.0)
    live_p.add_argument("--skip-cluster-labels", action="store_true", help="Skip the LLM cluster-label live pass.")

    args = parser.parse_args()
    layout, manifest = resolve_layout_and_manifest(args)
    _ = manifest

    if args.command == "init":
        print(json.dumps({"run_root": str(layout["run_root"]), "manifest": manifest}, indent=2))
        return

    if args.command == "prepare-docs":
        quotas = None
        if str(args.pilot_region_quotas_json).strip():
            quotas = json.loads(args.pilot_region_quotas_json)
            if not isinstance(quotas, dict):
                raise SystemExit("--pilot-region-quotas-json must decode to an object.")
        summary = prepare_llm_v2_documents(
            run_root=layout["run_root"],
            pilot_size=int(args.pilot_size),
            pilot_region_quotas=quotas,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "export-requests":
        summaries = {}
        for stage in _parse_stage_list(args.stages):
            summaries[stage] = export_llm_v2_requests(
                run_root=layout["run_root"],
                sample=args.sample,
                stage=stage,
                requests_per_shard=int(args.requests_per_shard),
                max_bytes_per_shard=int(args.max_bytes_per_shard),
                source_filter_path=Path(args.source_filter_path).expanduser().resolve() if str(args.source_filter_path).strip() else None,
                source_filter_column=str(args.source_filter_column).strip() or None,
                append=bool(args.append),
            )
        print(json.dumps(summaries, indent=2))
        return

    if args.command == "rehydrate-results":
        summaries = {}
        for stage in _parse_stage_list(args.stages):
            summaries[stage] = rehydrate_llm_v2_results(
                run_root=layout["run_root"],
                sample=args.sample,
                stage=stage,
                allow_errors=bool(args.allow_errors),
                allow_partial_results=bool(args.allow_partial_results),
            )
        print(json.dumps(summaries, indent=2))
        return

    if args.command == "postprocess":
        summaries = _postprocess_sample(
            run_root=layout["run_root"],
            sample=args.sample,
            with_cluster_label_sources=not args.skip_cluster_label_sources,
            with_policy_signatures=not args.skip_policy_signatures,
            with_review=not args.skip_review,
            with_quality_gates=not args.skip_quality_gates,
            with_story_report=not args.skip_story_report,
        )
        print(json.dumps(summaries, indent=2))
        return

    if args.command == "build-cluster-label-sources":
        print(json.dumps(build_cluster_label_request_sources(run_root=layout["run_root"]), indent=2))
        return

    if args.command == "apply-cluster-labels":
        print(json.dumps(apply_llm_cluster_labels(run_root=layout["run_root"]), indent=2))
        return

    if args.command == "build-review":
        print(json.dumps(build_llm_v2_review_packets(run_root=layout["run_root"], sample=args.sample), indent=2))
        return

    if args.command == "build-research":
        print(json.dumps(build_llm_v2_research_outputs(run_root=layout["run_root"], sample=args.sample), indent=2))
        return

    if args.command == "quality-gates":
        print(json.dumps(evaluate_llm_v2_review_quality(run_root=layout["run_root"]), indent=2))
        return

    if args.command == "build-story-report":
        sample = str(args.sample).strip() or None
        print(json.dumps(build_llm_v2_story_and_report(run_root=layout["run_root"], sample=sample), indent=2))
        return

    if args.command == "live-pilot":
        if not (layout["run_root"] / "corpus" / "pilot_sample.parquet").is_file():
            prepare_llm_v2_documents(run_root=layout["run_root"], pilot_size=int(args.pilot_size))
        env_file = Path(args.env_file).expanduser().resolve()
        summaries: dict[str, object] = {}
        summaries["live_semantic_stages"] = run_frontier_llm_v2_live_pilot(
            run_root=layout["run_root"],
            gateway_model=args.gateway_model,
            max_concurrency=int(args.max_concurrency),
            timeout=float(args.timeout),
            source_env_file=env_file,
            gateway_port=int(args.gateway_port),
        )
        summaries["postprocess_prelabels"] = _postprocess_sample(
            run_root=layout["run_root"],
            sample="pilot",
            with_cluster_label_sources=True,
            with_policy_signatures=True,
            with_review=False,
            with_quality_gates=False,
            with_story_report=False,
        )
        if not args.skip_cluster_labels:
            summaries["cluster_labels"] = _run_cluster_label_live(
                run_root=layout["run_root"],
                env_file=env_file,
                max_concurrency=int(args.max_concurrency),
                timeout=float(args.timeout),
            )
        summaries["postprocess_final"] = _postprocess_sample(
            run_root=layout["run_root"],
            sample="pilot",
            with_cluster_label_sources=False,
            with_policy_signatures=True,
            with_review=True,
            with_quality_gates=True,
            with_story_report=True,
        )
        print(json.dumps(summaries, indent=2))
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
