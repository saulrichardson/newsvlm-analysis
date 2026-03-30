#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from build_frontier_results_atlas_tex import build_results_atlas


def build_master_results(
    *,
    frontier_run_root: Path,
    complexity_run_root: Path,
    output_path: Path | None = None,
) -> Path:
    frontier_run_root = frontier_run_root.resolve()
    reports_dir = frontier_run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    update_slug = f"update_{date.today().isoformat()}"
    output_path = (output_path or (reports_dir / f"{update_slug}.tex")).resolve()

    temp_path = output_path.with_name(output_path.stem + "__tmp.tex")
    build_results_atlas(
        frontier_run_root=frontier_run_root,
        complexity_run_root=complexity_run_root.resolve(),
        output_path=temp_path,
    )

    tex = temp_path.read_text(encoding="utf-8")
    tex = tex.replace(
        "{\\LARGE Historical Ordinance Research Results}\\\\[0.4em]\n{\\large Methods, Results, and Takeaways Across the Current Research Surfaces}\\\\[0.6em]\n{\\normalsize Generated from the current analysis and legal-complexity run roots}",
        "{\\LARGE Historical Ordinance Research Results}\\\\[0.4em]\n{\\large Methods, Results, and Takeaways Across the Current Research Surfaces}\\\\[0.6em]\n{\\normalsize Generated from the current analysis and legal-complexity run roots}",
    )
    tex = tex.replace(
        "\\end{center}\n\n\\section*{Overview}",
        "\\end{center}\n\n\\section{Overview}",
    )
    tex = tex.replace("\\section*{Full-Sample Section Architecture Results}", "\\section{Section Architecture}")
    tex = tex.replace("\\section*{Full-Sample Document Purposes}", "\\section{Document Purposes}")
    tex = tex.replace("\\section*{Full-Sample Legal Complexity Backbone}", "\\section{Legal Complexity}")
    tex = tex.replace("\\section*{Bridge from Full-Sample Complexity to Pilot Semantics}", "\\section{Complexity-to-Semantics Bridge}")
    tex = tex.replace("\\section*{Pilot Clause and Governance Frontier}", "\\section{Clause and Governance Layer}")
    tex = tex.replace("\\section*{Clause and Governance Frontier}", "\\section{Clause and Governance Layer}")
    tex = tex.replace("\\section*{Main Takeaways}", "\\section{Main Takeaways}")
    tex = tex.replace("\\section*{Recommended Paper Spine}", "\\section{Recommended Paper Spine}")

    output_path.write_text(tex, encoding="utf-8")
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the single canonical master results LaTeX document.")
    parser.add_argument(
        "--frontier-run-root",
        default="artifacts/runs/frontier_llm_v2/pilot64_20260310_a",
        help="Run root for the frontier_llm_v2 research surface.",
    )
    parser.add_argument(
        "--complexity-run-root",
        default="artifacts/runs/legal_complexity_fullsample_v1/full728_20260311_a",
        help="Run root for the legal-complexity full-sample package.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional explicit output path for the generated .tex file.",
    )
    args = parser.parse_args()

    output = build_master_results(
        frontier_run_root=Path(args.frontier_run_root),
        complexity_run_root=Path(args.complexity_run_root),
        output_path=Path(args.output_path) if args.output_path else None,
    )
    print(output)


if __name__ == "__main__":
    main()
