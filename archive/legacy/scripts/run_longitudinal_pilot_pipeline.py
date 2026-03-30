#!/usr/bin/env python3
"""
Run the panel-level longitudinal pilot pipeline only.

Stages:
  1) run_panel_longitudinal_pilot.py
  2) build_longitudinal_pilot_latex_report.py

This pipeline intentionally excludes adoption-covariate regression stages.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    where = f" (cwd={cwd})" if cwd else ""
    print(f"[run]{where} " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=(str(cwd) if cwd else None))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run longitudinal pilot extraction + report pipeline.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60",
        help="Base run directory containing pi_v1 artifacts.",
    )
    ap.add_argument("--panel-count", type=int, default=20)
    ap.add_argument("--selection-mode", default="quality_top", choices=["quality_top"])
    ap.add_argument("--quality-rule", default="contamination_parse", choices=["contamination_parse"])
    ap.add_argument("--model", default="gemini:gemini-2.5-flash")
    ap.add_argument("--max-panel-chars", type=int, default=700000)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument(
        "--gateway-runner",
        default=str(Path(__file__).resolve().parents[1] / "scripts" / "run_openai_requests_via_gateway.py"),
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default=str(Path(__file__).resolve().parents[1] / "agent-gateway" / "src"),
    )
    ap.add_argument("--gov-env-path", default=".env")
    ap.add_argument(
        "--compile-report-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile longitudinal report PDF with latexmk when available.",
    )
    ap.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip requests that already have result rows.",
    )
    ap.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prepare requests but do not execute live gateway calls.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).expanduser().resolve()

    extraction_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_panel_longitudinal_pilot.py"),
        "--run-dir",
        str(run_dir),
        "--panel-count",
        str(int(args.panel_count)),
        "--selection-mode",
        str(args.selection_mode),
        "--quality-rule",
        str(args.quality_rule),
        "--model",
        str(args.model),
        "--max-panel-chars",
        str(int(args.max_panel_chars)),
        "--concurrency",
        str(int(args.concurrency)),
        "--timeout",
        str(float(args.timeout)),
        "--gateway-runner",
        str(args.gateway_runner),
        "--gateway-pythonpath",
        str(args.gateway_pythonpath),
        "--gov-env-path",
        str(args.gov_env_path),
        "--skip-existing" if bool(args.skip_existing) else "--no-skip-existing",
    ]
    if bool(args.dry_run):
        extraction_cmd.append("--dry-run")

    _run(extraction_cmd, cwd=repo_root)

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_longitudinal_pilot_latex_report.py"),
            "--run-dir",
            str(run_dir),
            "--input-dir",
            str(run_dir / "longitudinal"),
            "--output-dir",
            str(run_dir / "reports" / "longitudinal_pilot20"),
            "--compile-pdf" if bool(args.compile_report_pdf) else "--no-compile-pdf",
        ],
        cwd=repo_root,
    )

    print(
        "Done. "
        f"run_dir={run_dir} "
        f"longitudinal_dir={run_dir / 'longitudinal'} "
        f"report_dir={run_dir / 'reports' / 'longitudinal_pilot20'}"
    )


if __name__ == "__main__":
    main()
