#!/usr/bin/env python3
"""
Run v2 longitudinal pilot pipeline on existing pilot20 outputs.

Stages:
  1) build_longitudinal_v2_metrics.py
  2) build_longitudinal_v2_deep_dive.py
  3) build_longitudinal_v2_latex_report.py
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
    ap = argparse.ArgumentParser(description="Run longitudinal v2 metrics + report pipeline.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60",
        help="Base run directory.",
    )
    ap.add_argument("--compile-report-pdf", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).expanduser().resolve()

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_longitudinal_v2_metrics.py"),
            "--run-dir",
            str(run_dir),
        ],
        cwd=repo_root,
    )

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_longitudinal_v2_deep_dive.py"),
            "--run-dir",
            str(run_dir),
        ],
        cwd=repo_root,
    )

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_longitudinal_v2_latex_report.py"),
            "--run-dir",
            str(run_dir),
            "--compile-pdf" if bool(args.compile_report_pdf) else "--no-compile-pdf",
        ],
        cwd=repo_root,
    )

    print(
        "Done. "
        f"run_dir={run_dir} "
        f"v2_dir={run_dir / 'longitudinal_v2'} "
        f"report_dir={run_dir / 'reports' / 'longitudinal_pilot20_v2'}"
    )


if __name__ == "__main__":
    main()
