#!/usr/bin/env python3
"""
Orchestrate PI v1 zoning measurement pipeline.

Execution order:
  1) Core scaling run (prototype_zoning_panel_analysis.py)
  2) Amendment delta module
  3) Section export + optional section/purpose LLM pipelines + similarity module
  4) Covariate panel build
  5) Merge text + covariates
  6) Adoption/content models
  7) PI-facing LaTeX report package
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    where = f" (cwd={cwd})" if cwd else ""
    print(f"[run]{where} " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=(str(cwd) if cwd else None), env=env)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run end-to-end PI v1 pipeline.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/pi_v1_run",
        help="Run output directory.",
    )
    ap.add_argument("--panel-count", type=int, default=60)
    ap.add_argument("--panel-issues", type=int, default=10)
    ap.add_argument("--min-panel-issues", type=int, default=8)
    ap.add_argument("--selection-mode", default="stratified", choices=["top", "stratified"])
    ap.add_argument("--region-targets", default="Midwest:21,South:14,Northeast:13,West:12")
    ap.add_argument("--urbanicity-minimums", default="urban:10,suburban:20,rural:25")
    ap.add_argument("--llm-model", default="gemini:gemini-2.5-flash")
    ap.add_argument("--llm-concurrency", type=int, default=3)
    ap.add_argument("--llm-timeout", type=float, default=180.0)
    ap.add_argument("--gov-env-path", default=".env")
    ap.add_argument(
        "--gateway-runner",
        default=str(Path(__file__).resolve().parents[1] / "scripts" / "run_openai_requests_via_gateway.py"),
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default=str(Path(__file__).resolve().parents[1] / "agent-gateway" / "src"),
    )
    ap.add_argument(
        "--run-section-llm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run old-ocr section segmentation and purpose extraction batch pipelines via gateway.",
    )
    ap.add_argument(
        "--compile-report-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile final LaTeX report with latexmk when available.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    old_ocr_exp = Path("/Users/saulrichardson/projects/newspapers/old-ocr/experimental").resolve()
    old_ocr_src = Path("/Users/saulrichardson/projects/newspapers/old-ocr/src").resolve()

    # 1) Core scaling + contamination + open-coding bridge + numeric tables.
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "prototype_zoning_panel_analysis.py"),
            "--output-dir",
            str(run_dir),
            "--panel-count",
            str(int(args.panel_count)),
            "--panel-issues",
            str(int(args.panel_issues)),
            "--min-panel-issues",
            str(int(args.min_panel_issues)),
            "--selection-mode",
            str(args.selection_mode),
            "--region-targets",
            str(args.region_targets),
            "--urbanicity-minimums",
            str(args.urbanicity_minimums),
            "--run-llm",
            "--llm-model",
            str(args.llm_model),
            "--llm-concurrency",
            str(int(args.llm_concurrency)),
            "--llm-timeout",
            str(float(args.llm_timeout)),
            "--gov-env-path",
            str(args.gov_env_path),
            "--gateway-runner",
            str(args.gateway_runner),
            "--gateway-pythonpath",
            str(args.gateway_pythonpath),
        ],
        cwd=repo_root,
    )

    # 2) Amendment module.
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_amendment_delta_measure.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(run_dir / "amendments"),
        ],
        cwd=repo_root,
    )

    # 3) Section export.
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "export_initial_full_docs_for_section_analysis.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(run_dir / "section_analysis"),
            "--use-earliest-per-city",
        ],
        cwd=repo_root,
    )

    # Optional: run old-ocr section + purpose pipelines.
    if bool(args.run_section_llm):
        env = dict(os.environ)
        env["PYTHONPATH"] = (
            str(old_ocr_src)
            if not env.get("PYTHONPATH")
            else f"{str(old_ocr_src)}:{env['PYTHONPATH']}"
        )
        section_reqs = run_dir / "section_analysis" / "section_requests"
        section_res = run_dir / "section_analysis" / "section_results"
        section_out = run_dir / "section_analysis" / "sections_out"
        purpose_reqs = run_dir / "section_analysis" / "purpose_requests"
        purpose_res = run_dir / "section_analysis" / "purpose_results"
        purpose_out = run_dir / "section_analysis" / "purpose_out"

        _run(
            [
                sys.executable,
                str(old_ocr_exp / "scripts" / "export_regulatory_section_segmentation_batch_requests.py"),
                "--input-jsonl",
                str(run_dir / "section_analysis" / "docs.jsonl"),
                "--output-dir",
                str(section_reqs),
                "--openai-model",
                "gpt-5-nano",
            ],
            cwd=old_ocr_exp,
            env=env,
        )
        _run(
            [
                sys.executable,
                str(args.gateway_runner),
                "--request-dir",
                str(section_reqs),
                "--output-dir",
                str(section_res),
                "--model",
                str(args.llm_model),
                "--max-concurrency",
                str(int(args.llm_concurrency)),
                "--timeout",
                str(float(args.llm_timeout)),
            ],
            cwd=old_ocr_exp,
            env=env,
        )
        _run(
            [
                sys.executable,
                str(old_ocr_exp / "scripts" / "rehydrate_regulatory_section_segmentation_openai_batch_results.py"),
                "--request-dir",
                str(section_reqs),
                "--results-dir",
                str(section_res),
                "--input-jsonl",
                str(run_dir / "section_analysis" / "docs.jsonl"),
                "--output-dir",
                str(section_out),
                "--allow-partial-results",
            ],
            cwd=old_ocr_exp,
            env=env,
        )
        _run(
            [
                sys.executable,
                str(old_ocr_exp / "scripts" / "export_regulatory_document_purpose_batch_requests.py"),
                "--input-jsonl",
                str(run_dir / "section_analysis" / "docs.jsonl"),
                "--output-dir",
                str(purpose_reqs),
                "--openai-model",
                "gpt-5-nano",
            ],
            cwd=old_ocr_exp,
            env=env,
        )
        _run(
            [
                sys.executable,
                str(args.gateway_runner),
                "--request-dir",
                str(purpose_reqs),
                "--output-dir",
                str(purpose_res),
                "--model",
                str(args.llm_model),
                "--max-concurrency",
                str(int(args.llm_concurrency)),
                "--timeout",
                str(float(args.llm_timeout)),
            ],
            cwd=old_ocr_exp,
            env=env,
        )
        _run(
            [
                sys.executable,
                str(old_ocr_exp / "scripts" / "rehydrate_regulatory_document_purpose_openai_batch_results.py"),
                "--request-dir",
                str(purpose_reqs),
                "--results-dir",
                str(purpose_res),
                "--output-dir",
                str(purpose_out),
                "--allow-partial-results",
            ],
            cwd=old_ocr_exp,
            env=env,
        )

    # 3b) Section similarity step (expects section/purpose outputs; no-op if files missing).
    sec_dir = run_dir / "section_analysis"
    section_candidates = [
        sec_dir / "sections_out_with_retry_and_fallback.jsonl",
        sec_dir / "sections_out_with_retry.jsonl",
        sec_dir / "sections_out" / "sections.jsonl",
    ]
    sections_jsonl = next((p for p in section_candidates if p.is_file()), section_candidates[-1])
    purposes_jsonl = run_dir / "section_analysis" / "purpose_out" / "doc_purposes.jsonl"
    if sections_jsonl.is_file() and purposes_jsonl.is_file():
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "compute_section_purpose_similarity.py"),
                "--docs-manifest",
                str(run_dir / "section_analysis" / "docs_manifest.csv"),
                "--sections-jsonl",
                str(sections_jsonl),
                "--purposes-jsonl",
                str(purposes_jsonl),
                "--output-dir",
                str(run_dir / "section_analysis"),
            ],
            cwd=repo_root,
        )
    else:
        print(
            "[warn] skipping section similarity: missing sections or purposes output. "
            "Use --run-section-llm to produce them.",
            flush=True,
        )

    # 4) Covariates.
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_external_covariate_panel.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(run_dir / "covariates"),
        ],
        cwd=repo_root,
    )

    # 5) Merge panel.
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "merge_text_and_covariates_panel.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(run_dir / "analysis_panel"),
        ],
        cwd=repo_root,
    )

    # 6) Models.
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "estimate_adoption_and_content_models.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(run_dir / "models"),
        ],
        cwd=repo_root,
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate_adoption_model_options.py"),
            "--run-dir",
            str(run_dir),
            "--output-csv",
            str(run_dir / "models" / "adoption_option_grid_ml_groupcv_estimable.csv"),
        ],
        cwd=repo_root,
    )

    # 7) PI report.
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_pi_v1_latex_report.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(run_dir / "reports" / "pi_v1_analysis"),
            ("--compile-pdf" if bool(args.compile_report_pdf) else "--no-compile-pdf"),
        ],
        cwd=repo_root,
    )

    print(f"Done. run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
