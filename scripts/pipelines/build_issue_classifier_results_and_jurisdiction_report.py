#!/usr/bin/env python3
"""
Build a minimal LaTeX note with a single hierarchy table for the current
issue-classifier outputs.

The table is formatted in Python and uses float placement [H].
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path


TOTAL_CLASSIFIED = 63_794
PRINTED_ZONING_LEGAL_TEXT = 17_089
NO_PRINTED_ZONING_LEGAL_TEXT = 46_705

JURISDICTION_ROWS: list[tuple[str, int, str]] = [
    ("Exact city and state match", 18_968, "29.7"),
    ("Same state, other jurisdiction", 14_121, "22.1"),
    ("Classifier jurisdiction missing", 30_524, "47.8"),
    ("Different state", 93, "0.1"),
    ("Publication metadata missing", 88, "0.1"),
]

CROSS_JURISDICTION_ROWS: list[tuple[str, str, int]] = [
    ("Brewster, NY", "Town of Southeast, NY", 186),
    ("Bellmore, NY", "Town of Hempstead, NY", 146),
    ("Cedar Rapids, IA", "Coralville, IA", 145),
    ("Doylestown, PA", "Doylestown Township, PA", 141),
    ("Doylestown, PA", "Warrington Township, PA", 129),
    ("Doylestown, PA", "Warminster Township, PA", 121),
    ("Bethpage, NY", "Town of Hempstead, NY", 117),
    ("Cedar Rapids, IA", "Linn County, IA", 115),
    ("Bakersfield, CA", "Kern County, CA", 115),
    ("Doylestown, PA", "Buckingham Township, PA", 103),
    ("Arlington Heights, IL", "Mount Prospect, IL", 97),
    ("Altoona, PA", "Hollidaysburg, PA", 95),
    ("Lynbrook, NY", "Town of Hempstead, NY", 93),
    ("Doylestown, PA", "Warwick Township, PA", 90),
    ("Bryan, TX", "College Station, TX", 89),
    ("Annapolis, MD", "Anne Arundel County, MD", 88),
]

HIERARCHY_ROWS: list[tuple[int, str, int, str]] = [
    (0, "All classified records", 63_794, "100.0"),
    (1, "Printed zoning legal text not present", 46_705, "73.2"),
    (1, "Printed zoning legal text present", 17_089, "26.8"),
    (2, "Operative", 10_791, "63.1"),
    (3, "Map or rezoning order", 5_210, "48.3"),
    (3, "Amendment with substantial text", 2_533, "23.5"),
    (3, "Amendment with targeted text", 871, "8.1"),
    (3, "Full-issue code publication", 710, "6.6"),
    (3, "Limited-scope zoning ordinance", 674, "6.2"),
    (3, "Variance or special-use order", 340, "3.2"),
    (3, "Code publication excerpt or installment", 308, "2.9"),
    (3, "Uncertain", 145, "1.3"),
    (2, "Proposed", 5_546, "32.5"),
    (3, "Amendment with substantial text", 2_064, "37.2"),
    (3, "Amendment with targeted text", 1_568, "28.3"),
    (3, "Map or rezoning order", 1_109, "20.0"),
    (3, "Code publication excerpt or installment", 518, "9.3"),
    (3, "Limited-scope zoning ordinance", 174, "3.1"),
    (3, "Uncertain", 99, "1.8"),
    (3, "Full-issue code publication", 10, "0.2"),
    (3, "Variance or special-use order", 4, "0.1"),
    (2, "Unclear", 750, "4.4"),
    (3, "Code publication excerpt or installment", 306, "40.8"),
    (3, "Map or rezoning order", 163, "21.7"),
    (3, "Amendment with substantial text", 112, "14.9"),
    (3, "Amendment with targeted text", 53, "7.1"),
    (3, "Limited-scope zoning ordinance", 51, "6.8"),
    (3, "Uncertain", 39, "5.2"),
    (3, "Variance or special-use order", 13, "1.7"),
    (3, "Full-issue code publication", 13, "1.7"),
    (2, "Other or malformed operativity", 2, "0.0"),
]


def _latex_escape(text: object) -> str:
    s = str(text)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


def _safe_label(label: str) -> str:
    out = re.sub(r"[^A-Za-z0-9:._-]+", "_", label).strip("_")
    return out or "tab:table"


def _indent_label(level: int, text: str) -> str:
    base = _latex_escape(text)
    if level <= 0:
        return rf"\textbf{{{base}}}"
    if level == 1:
        return base
    em = 1.25 * (level - 1)
    return rf"\hspace*{{{em:.2f}em}}{base}"


def _table_block() -> str:
    rows: list[str] = []
    for level, label, count, share in HIERARCHY_ROWS:
        rows.append(f"{_indent_label(level, label)} & {count:,} & {share} \\\\")
    parts = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\caption{Hierarchy of printed zoning legal text, operativity, and specific legal-document class.}",
        rf"\label{{{_safe_label('tab:classification_hierarchy')}}}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrr}",
        r"\toprule",
        r"Hierarchy branch & Records & Share of parent (\%) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\vspace{0.25em}",
        r"\par\small\textit{Top-level rows use all classified records as the parent. Indented rows use the immediately preceding higher branch as the parent.}",
        r"\end{table}",
    ]
    return "\n".join(parts)


def _jurisdiction_summary_table() -> str:
    rows = [f"{_latex_escape(label)} & {count:,} & {share} \\\\" for label, count, share in JURISDICTION_ROWS]
    parts = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\caption{Comparison of publication location and extracted ordinance jurisdiction.}",
        rf"\label{{{_safe_label('tab:jurisdiction_summary')}}}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrr}",
        r"\toprule",
        r"Comparison category & Records & Share of classified records (\%) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ]
    return "\n".join(parts)


def _cross_jurisdiction_examples_table() -> str:
    rows = [
        f"{_latex_escape(publication)} & {_latex_escape(ordinance)} & {count:,} \\\\"
        for publication, ordinance, count in CROSS_JURISDICTION_ROWS
    ]
    parts = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\caption{Examples of recurrent same-state cross-jurisdiction publication patterns.}",
        rf"\label{{{_safe_label('tab:cross_jurisdiction_examples')}}}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.24\textwidth}>{\raggedright\arraybackslash}Xr}",
        r"\toprule",
        r"Publication location & Ordinance jurisdiction & Count \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ]
    return "\n".join(parts)


def _build_document(report_date: str) -> str:
    doc: list[str] = []
    doc.append("% Generated by scripts/pipelines/build_issue_classifier_results_and_jurisdiction_report.py")
    doc.append(r"\documentclass[10pt]{article}")
    doc.append(r"\usepackage[margin=0.85in]{geometry}")
    doc.append(r"\usepackage{array}")
    doc.append(r"\usepackage{booktabs}")
    doc.append(r"\usepackage{float}")
    doc.append(r"\usepackage{tabularx}")
    doc.append(r"\begin{document}")
    doc.append(
        "Table~\\ref{tab:classification_hierarchy} shows the classification hierarchy in a single layout. "
        "The first split is whether printed zoning legal text is present. "
        "Within the printed-zoning-legal-text branch, the table then splits records by operativity and finally by the specific legal-document class."
    )
    doc.append(_table_block())
    doc.append(
        "Table~\\ref{tab:jurisdiction_summary} shows how often the extracted ordinance jurisdiction matches the newspaper's publication location. "
        "Table~\\ref{tab:cross_jurisdiction_examples} lists recurring same-state examples where newspapers publish ordinances or notices for a nearby municipality, township, or county."
    )
    doc.append(_jurisdiction_summary_table())
    doc.append(_cross_jurisdiction_examples_table())
    doc.append(r"\end{document}")
    return "\n\n".join(doc) + "\n"


def _compile_tex(tex_path: Path) -> None:
    latexmk = shutil.which("latexmk")
    if not latexmk:
        raise SystemExit("latexmk not found on PATH")
    subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_out = repo_root / "reports" / "issue_classifier_results_and_jurisdiction_20260306.tex"

    ap = argparse.ArgumentParser(description="Build a minimal issue-classifier hierarchy LaTeX note.")
    ap.add_argument("--out-tex", type=Path, default=default_out, help="Path to the output .tex file.")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="Report date to place in the title block.")
    ap.add_argument("--compile", action="store_true", help="Compile the generated .tex file with latexmk.")
    args = ap.parse_args()

    out_tex = args.out_tex.resolve()
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(_build_document(args.date), encoding="utf-8")

    if args.compile:
        _compile_tex(out_tex)


if __name__ == "__main__":
    main()
