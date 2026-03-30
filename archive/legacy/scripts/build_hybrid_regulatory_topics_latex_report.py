#!/usr/bin/env python3
"""
Build a LaTeX writeup bundle that documents the "hybrid regulatory topics" improvement.

This is meant to be artifact-first: it pulls concrete outputs from a hybrid run directory
(sections, motive labels, extracted clauses, clause clusters, instrument labels) and writes:

  reports/hybrid_regulatory_topics_report/
    - provenance.tex
    - figures/*.png
    - tables/*.tex
    - snippets/*.tex

Then you can compile with:
  cd reports/hybrid_regulatory_topics_report && make

No APIs are called.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from os.path import expanduser
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _latex_escape(text: str) -> str:
    """
    Escape text for LaTeX.
    """
    s = str(text or "")
    # Order matters (backslash first).
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


def _detokenize_path(path: str) -> str:
    """
    Wrap a path for LaTeX in \\texttt{\\detokenize{...}} so underscores don't need escaping.
    """
    return f"\\texttt{{\\detokenize{{{path}}}}}"


def _write_tex(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def _resolve_hybrid_inputs(run_dir: Path) -> dict[str, Path | None]:
    """
    Find canonical outputs inside a hybrid run directory.
    """
    candidates = {
        "sections": run_dir / "sections_out" / "sections.jsonl",
        "chunk_motives": run_dir / "motive_out" / "chunk_motives.jsonl",
        "doc_purposes": run_dir / "purpose_out" / "doc_purposes.jsonl",
        "clauses_strict": run_dir / "clause_out" / "clauses.jsonl",
        "clauses_salvage": run_dir / "clause_out_salvage" / "clauses.jsonl",
        "clusters_local_strict": run_dir / "clause_clusters_local" / "clusters.jsonl",
        "clusters_local_salvage": run_dir / "clause_clusters_local_salvage" / "clusters.jsonl",
        "cluster_summary_local": run_dir / "clause_clusters_local" / "cluster_summary.json",
        "cluster_keywords_local": run_dir / "clause_clusters_local" / "cluster_keywords.jsonl",
        "cluster_scatter_local": run_dir / "clause_clusters_local" / "scatter.png",
        "cluster_labels": run_dir / "instrument_label_out" / "cluster_labels.jsonl",
    }

    out: dict[str, Path | None] = {}
    out["sections"] = candidates["sections"] if candidates["sections"].is_file() else None
    out["chunk_motives"] = candidates["chunk_motives"] if candidates["chunk_motives"].is_file() else None
    out["doc_purposes"] = candidates["doc_purposes"] if candidates["doc_purposes"].is_file() else None

    if candidates["clauses_strict"].is_file():
        out["clauses"] = candidates["clauses_strict"]
        out["clauses_mode"] = Path("clause_out/clauses.jsonl")
    elif candidates["clauses_salvage"].is_file():
        out["clauses"] = candidates["clauses_salvage"]
        out["clauses_mode"] = Path("clause_out_salvage/clauses.jsonl")
    else:
        out["clauses"] = None
        out["clauses_mode"] = None

    if candidates["clusters_local_strict"].is_file():
        out["clusters"] = candidates["clusters_local_strict"]
        out["clusters_mode"] = Path("clause_clusters_local/clusters.jsonl")
    elif candidates["clusters_local_salvage"].is_file():
        out["clusters"] = candidates["clusters_local_salvage"]
        out["clusters_mode"] = Path("clause_clusters_local_salvage/clusters.jsonl")
    else:
        out["clusters"] = None
        out["clusters_mode"] = None

    out["cluster_summary"] = candidates["cluster_summary_local"] if candidates["cluster_summary_local"].is_file() else None
    out["cluster_keywords"] = candidates["cluster_keywords_local"] if candidates["cluster_keywords_local"].is_file() else None
    out["cluster_scatter"] = candidates["cluster_scatter_local"] if candidates["cluster_scatter_local"].is_file() else None
    out["cluster_labels"] = candidates["cluster_labels"] if candidates["cluster_labels"].is_file() else None
    return out


def _write_table_tex(
    *,
    path: Path,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[str]],
    colspec: str,
    notes: str | None = None,
) -> None:
    """
    Write a small LaTeX table using booktabs + tabularx.
    """
    hdr = " & ".join(_latex_escape(c) for c in columns) + " \\\\"
    body_lines = []
    for r in rows:
        body_lines.append(" & ".join(_latex_escape(x) for x in r) + " \\\\")

    notes_block = ""
    if notes and str(notes).strip():
        # Avoid using `\\` outside of tabular environments (it triggers "There's no line here to end").
        notes_block = "\n".join(
            [
                "\\vspace{0.25em}",
                "\\par\\small\\textit{" + _latex_escape(str(notes).strip()) + "}",
            ]
        )

    # Labels must not contain LaTeX escapes (they are written into \csname ...\endcsname).
    safe_label = str(label or "").strip()
    safe_label = re.sub(r"[^A-Za-z0-9:._-]+", "_", safe_label)
    safe_label = safe_label.strip("_")
    if not safe_label:
        safe_label = "tab:table"

    tex = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            f"\\caption{{{_latex_escape(caption)}}}",
            f"\\label{{{safe_label}}}",
            f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}",
            "\\toprule",
            hdr,
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabularx}",
            notes_block,
            "\\end{table}",
        ]
    )
    _write_tex(path, tex)


@dataclass(frozen=True)
class ClauseRow:
    clause_id: str
    doc_id: str
    chunk_id: str
    requirement: str
    clause_text: str
    motive: str
    modality: str


def _load_clauses(path: Path) -> dict[str, ClauseRow]:
    out: dict[str, ClauseRow] = {}
    for obj in _iter_jsonl(path):
        cid = obj.get("clause_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        doc_id = str(obj.get("doc_id") or "").strip()
        chunk_id = str(obj.get("chunk_id") or "").strip()
        req = str(obj.get("requirement") or "").strip()
        quote = str(obj.get("clause_text") or "").strip()
        motive = str(obj.get("motive") or "").strip()
        modality = str(obj.get("modality") or "").strip()
        out[cid.strip()] = ClauseRow(
            clause_id=cid.strip(),
            doc_id=doc_id,
            chunk_id=chunk_id,
            requirement=req,
            clause_text=quote,
            motive=motive,
            modality=modality,
        )
    if not out:
        raise SystemExit(f"No clause rows loaded from {path}")
    return out


def _load_cluster_assignments(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for obj in _iter_jsonl(path):
        cid = obj.get("clause_id")
        cluster_id = obj.get("cluster_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        if not isinstance(cluster_id, int):
            continue
        prob_raw = obj.get("cluster_prob", 1.0)
        try:
            prob = float(prob_raw)
        except Exception:
            prob = 1.0
        out.append({"clause_id": cid.strip(), "cluster_id": int(cluster_id), "cluster_prob": prob})
    if not out:
        raise SystemExit(f"No cluster assignments loaded from {path}")
    return out


def _load_cluster_labels(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for obj in _iter_jsonl(path):
        cid = obj.get("cluster_id")
        if not isinstance(cid, int):
            continue
        out[int(cid)] = obj
    return out


def _load_cluster_keywords(path: Path | None) -> dict[int, list[str]]:
    if path is None or not path.is_file():
        return {}
    out: dict[int, list[str]] = {}
    for obj in _iter_jsonl(path):
        cid = obj.get("cluster_id")
        kws = obj.get("keywords")
        if not isinstance(cid, int) or not isinstance(kws, list):
            continue
        out[int(cid)] = [str(x).strip() for x in kws if str(x).strip()]
    return out


def _render_cluster_examples_snippet(
    *,
    out_path: Path,
    clauses_by_id: dict[str, ClauseRow],
    assignments: list[dict[str, Any]],
    cluster_labels: dict[int, dict[str, Any]],
    cluster_keywords: dict[int, list[str]],
    top_k_clusters: int,
    clauses_per_cluster: int,
    cluster_ids: list[int] | None,
) -> None:
    # Group clause_ids by cluster.
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in assignments:
        by_cluster[int(a["cluster_id"])].append(a)

    if cluster_ids:
        # Explicit selection (in user-provided order). Fail loud if none exist.
        clusters_sorted = []
        seen: set[int] = set()
        for cid in cluster_ids:
            if cid in seen:
                continue
            seen.add(cid)
            rows = by_cluster.get(int(cid))
            if not rows:
                continue
            clusters_sorted.append((int(cid), rows))
        if not clusters_sorted:
            raise SystemExit("No requested --example-cluster-ids were found in cluster assignments.")
    else:
        # Default: sort clusters by size desc, ignore noise.
        clusters_sorted = sorted(
            [c for c in by_cluster.items() if c[0] != -1],
            key=lambda kv: (-len(kv[1]), kv[0]),
        )
        clusters_sorted = clusters_sorted[: max(0, int(top_k_clusters))]

    parts: list[str] = []
    parts.append("% Auto-generated cluster examples (verbatim evidence + paraphrase).")
    parts.append("")

    for cluster_id, rows in clusters_sorted:
        inst = cluster_labels.get(cluster_id, {})
        name = str(inst.get("instrument_name") or f"Cluster {cluster_id}").strip()
        desc = str(inst.get("instrument_description") or "").strip()
        kws = ", ".join(cluster_keywords.get(cluster_id, [])[:12])

        parts.append(f"\\subsubsection*{{Cluster {cluster_id}: {_latex_escape(name)}}}")
        if desc:
            parts.append("\\noindent\\textbf{Instrument description:} " + _latex_escape(desc))
            parts.append("")
        if kws:
            parts.append("\\noindent\\textbf{Keywords (TF-IDF):} " + _latex_escape(kws))
            parts.append("")

        # Pick top clauses by probability (desc).
        rows_sorted = sorted(rows, key=lambda r: (-float(r.get("cluster_prob", 1.0)), str(r.get("clause_id"))))
        rows_sorted = rows_sorted[: max(0, int(clauses_per_cluster))]

        parts.append("\\begin{itemize}[leftmargin=*]")
        for a in rows_sorted:
            cid = str(a["clause_id"])
            c = clauses_by_id.get(cid)
            if c is None:
                continue
            req = c.requirement.strip()
            quote = c.clause_text.strip()
            meta = f"doc={c.doc_id} | chunk={c.chunk_id} | motive={c.motive} | modality={c.modality}"

            parts.append("\\item")
            parts.append("\\textbf{Requirement:} " + _latex_escape(req))
            parts.append("")
            parts.append("\\begin{quote}\\footnotesize " + _latex_escape(quote) + "\\end{quote}")
            parts.append("\\noindent\\texttt{" + _latex_escape(meta) + "}")
        parts.append("\\end{itemize}")
        parts.append("")

    _write_tex(out_path, "\n".join(parts))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build LaTeX bundle documenting hybrid regulatory topic improvements.")
    ap.add_argument("--hybrid-run-dir", required=True, help="Hybrid run directory (e.g., tmp_hybrid_pi_real_1).")
    ap.add_argument(
        "--output-dir",
        default="reports/hybrid_regulatory_topics_report",
        help="Directory containing report.tex where tables/figures will be written.",
    )
    ap.add_argument(
        "--baseline-issue-topics-dir",
        default="reports/issue_topics_report",
        help="Directory containing baseline chunk-topic report artifacts (optional).",
    )
    ap.add_argument("--no-baseline", action="store_true", help="Do not include baseline chunk-topic artifacts.")
    ap.add_argument("--top-clusters-table", type=int, default=25, help="How many clusters to show in the cluster index table.")
    ap.add_argument("--example-clusters", type=int, default=3, help="How many clusters to include in example clause snippets.")
    ap.add_argument("--example-clauses-per-cluster", type=int, default=4, help="How many clauses to show per example cluster.")
    ap.add_argument(
        "--example-cluster-ids",
        default="",
        help="Optional comma-separated cluster IDs to include in the example snippet (overrides --example-clusters).",
    )
    ap.add_argument("--max-docs-sections", type=int, default=5, help="How many docs to include in the sections table (0 disables).")
    ap.add_argument("--max-docs-purposes", type=int, default=8, help="How many docs to include in the purposes table (0 disables).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(expanduser(args.hybrid_run_dir)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()

    if not run_dir.is_dir():
        raise SystemExit(f"--hybrid-run-dir is not a directory: {run_dir}")
    if not (out_dir / "report.tex").is_file():
        raise SystemExit(f"--output-dir must point at an existing report bundle with report.tex: {out_dir}")

    t0 = time.time()
    inputs = _resolve_hybrid_inputs(run_dir)
    clauses_path = inputs.get("clauses")
    clusters_path = inputs.get("clusters")

    if not isinstance(clauses_path, Path) or not clauses_path.is_file():
        raise SystemExit(f"Could not find clauses.jsonl under {run_dir} (expected clause_out/ or clause_out_salvage/).")
    if not isinstance(clusters_path, Path) or not clusters_path.is_file():
        raise SystemExit(
            f"Could not find clusters.jsonl under {run_dir} (expected clause_clusters_local/ or clause_clusters_local_salvage/)."
        )

    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    snippets_dir = out_dir / "snippets"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    snippets_dir.mkdir(parents=True, exist_ok=True)

    # --- Provenance ---
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    prov = "\n".join(
        [
            "% Auto-generated provenance.",
            f"\\newcommand{{\\ReportDate}}{{{_latex_escape(stamp)}}}",
            f"\\newcommand{{\\HybridRunDir}}{{{_detokenize_path(str(run_dir))}}}",
            f"\\newcommand{{\\HybridClausesPath}}{{{_detokenize_path(str(clauses_path))}}}",
            f"\\newcommand{{\\HybridClustersPath}}{{{_detokenize_path(str(clusters_path))}}}",
        ]
    )
    _write_tex(out_dir / "provenance.tex", prov)

    # --- Figures (baseline + hybrid) ---
    if inputs.get("cluster_scatter") is not None:
        _copy_if_exists(inputs["cluster_scatter"], figs_dir / "hybrid_clause_umap.png")  # type: ignore[arg-type]

    baseline_dir = Path(expanduser(args.baseline_issue_topics_dir)).resolve()
    include_baseline = (not bool(args.no_baseline)) and baseline_dir.is_dir()
    if include_baseline:
        _copy_if_exists(
            baseline_dir / "figures" / "full_ordinance__umap_scatter.png",
            figs_dir / "baseline_full_ordinance_umap.png",
        )

    # --- Load core hybrid artifacts ---
    clauses_by_id = _load_clauses(clauses_path)
    assignments = _load_cluster_assignments(clusters_path)
    cluster_labels = _load_cluster_labels(inputs.get("cluster_labels"))  # type: ignore[arg-type]
    cluster_keywords = _load_cluster_keywords(inputs.get("cluster_keywords"))  # type: ignore[arg-type]

    # --- Table: sections ---
    if inputs.get("sections") is not None:
        sections_path = inputs["sections"]
        assert isinstance(sections_path, Path)
        by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for obj in _iter_jsonl(sections_path):
            doc_id = str(obj.get("doc_id") or "").strip()
            if not doc_id:
                continue
            by_doc[doc_id].append(obj)
        docs_sorted = sorted(by_doc.keys())
        if int(args.max_docs_sections) > 0:
            docs_sorted = docs_sorted[: int(args.max_docs_sections)]
        rows: list[list[str]] = []
        for doc_id in docs_sorted:
            secs = sorted(by_doc[doc_id], key=lambda o: int(o.get("section_number") or 0))
            for s in secs:
                rows.append(
                    [
                        doc_id,
                        str(s.get("section_number") or ""),
                        str(s.get("title") or ""),
                        str(s.get("text_chars") or ""),
                    ]
                )
        _write_table_tex(
            path=tables_dir / "sections_index.tex",
            caption="LLM-driven section segmentation (excerpt).",
            label="tab:sections",
            columns=["doc_id", "section", "title", "chars"],
            rows=rows,
            colspec="lrXr",
            notes=(
                "This is the key coherence fix: articles/ads/notices are separated from ordinance text before motive labeling + clause extraction."
            ),
        )

    # --- Table: chunk motive counts ---
    if inputs.get("chunk_motives") is not None:
        motive_path = inputs["chunk_motives"]
        assert isinstance(motive_path, Path)
        counts = Counter()
        for obj in _iter_jsonl(motive_path):
            mot = obj.get("primary_motive")
            if isinstance(mot, str) and mot.strip():
                counts[mot.strip()] += 1
        rows = [[k, str(v)] for k, v in counts.most_common()]
        _write_table_tex(
            path=tables_dir / "chunk_motive_counts.tex",
            caption="Chunk/section primary motive counts (hybrid run).",
            label="tab:chunk_motives",
            columns=["motive", "n_chunks"],
            rows=rows,
            colspec="Xr",
            notes="These are section-level motive labels (controlled taxonomy) used to guide clause extraction and allow motive-filtered clustering.",
        )

    # --- Tables: clause counts ---
    motive_counts = Counter()
    modality_counts = Counter()
    for c in clauses_by_id.values():
        if c.motive:
            motive_counts[c.motive] += 1
        if c.modality:
            modality_counts[c.modality] += 1

    _write_table_tex(
        path=tables_dir / "clause_motive_counts.tex",
        caption="Extracted clause counts by motive (hybrid run).",
        label="tab:clause_motives",
        columns=["motive", "n_clauses"],
        rows=[[k, str(v)] for k, v in motive_counts.most_common()],
        colspec="Xr",
        notes="Clauses are the units being embedded/clusted, not entire chunks. This is the main granularity change.",
    )
    _write_table_tex(
        path=tables_dir / "clause_modality_counts.tex",
        caption="Extracted clause counts by modality (hybrid run).",
        label="tab:clause_modalities",
        columns=["modality", "n_clauses"],
        rows=[[k, str(v)] for k, v in modality_counts.most_common()],
        colspec="Xr",
        notes="A sanity check: we expect many 'must'/'must_not' requirements; definition clauses can be filtered if they overwhelm instruments.",
    )

    # --- Table: cluster index (top K) ---
    # Join clause + assignment.
    cluster_to_clause_ids: dict[int, list[str]] = defaultdict(list)
    for a in assignments:
        cluster_to_clause_ids[int(a["cluster_id"])].append(str(a["clause_id"]))

    clusters_sorted = sorted(
        [cid for cid in cluster_to_clause_ids.keys() if cid != -1],
        key=lambda cid: (-len(cluster_to_clause_ids[cid]), cid),
    )
    if int(args.top_clusters_table) > 0:
        clusters_sorted = clusters_sorted[: int(args.top_clusters_table)]

    cluster_rows: list[list[str]] = []
    for cluster_id in clusters_sorted:
        clause_ids = cluster_to_clause_ids[cluster_id]
        docs = set()
        motives = Counter()
        modalities = Counter()
        for cid in clause_ids:
            c = clauses_by_id.get(cid)
            if c is None:
                continue
            if c.doc_id:
                docs.add(c.doc_id)
            if c.motive:
                motives[c.motive] += 1
            if c.modality:
                modalities[c.modality] += 1
        inst = cluster_labels.get(cluster_id, {})
        inst_name = str(inst.get("instrument_name") or f"cluster_{cluster_id}")
        top_mot = "; ".join(f"{k}:{v}" for k, v in motives.most_common(3))
        top_mod = "; ".join(f"{k}:{v}" for k, v in modalities.most_common(2))
        kw = ", ".join(cluster_keywords.get(cluster_id, [])[:10])
        cluster_rows.append([str(cluster_id), str(len(clause_ids)), str(len(docs)), inst_name, top_mot, top_mod, kw])

    _write_table_tex(
        path=tables_dir / "cluster_index.tex",
        caption="Clause-level instrument clusters (top by size).",
        label="tab:clusters",
        columns=["cluster", "n_clauses", "n_docs", "instrument", "top motives", "top modalities", "keywords"],
        rows=cluster_rows,
        colspec="rrrXXXX",
        notes="Each row is a discovered instrument-like cluster over extracted clauses (not chunks).",
    )

    # --- Snippet: cluster examples ---
    _render_cluster_examples_snippet(
        out_path=snippets_dir / "cluster_examples.tex",
        clauses_by_id=clauses_by_id,
        assignments=assignments,
        cluster_labels=cluster_labels,
        cluster_keywords=cluster_keywords,
        top_k_clusters=int(args.example_clusters),
        clauses_per_cluster=int(args.example_clauses_per_cluster),
        cluster_ids=[int(x) for x in args.example_cluster_ids.split(",") if x.strip().lstrip("-").isdigit()]
        if str(args.example_cluster_ids or "").strip()
        else None,
    )

    # --- Table: document purposes ---
    if inputs.get("doc_purposes") is not None:
        purposes_path = inputs["doc_purposes"]
        assert isinstance(purposes_path, Path)
        rows: list[list[str]] = []
        docs_seen = 0
        for obj in _iter_jsonl(purposes_path):
            doc_id = str(obj.get("doc_id") or "").strip()
            purposes = obj.get("purposes") if isinstance(obj.get("purposes"), list) else []
            notes = str(obj.get("notes") or "").strip()
            if not doc_id:
                continue
            # One row per document: summarize motives + short descriptions.
            summary_parts = []
            for p in purposes[:6]:
                if not isinstance(p, dict):
                    continue
                mot = str(p.get("motive") or "").strip()
                desc = str(p.get("description") or "").strip()
                if not mot or not desc:
                    continue
                summary_parts.append(f"{mot}: {desc}")
            summary = " | ".join(summary_parts)
            rows.append([doc_id, summary, notes])
            docs_seen += 1
            if int(args.max_docs_purposes) > 0 and docs_seen >= int(args.max_docs_purposes):
                break

        _write_table_tex(
            path=tables_dir / "doc_purposes.tex",
            caption="Document-level purpose summaries (LLM; excerpt).",
            label="tab:purposes",
            columns=["doc_id", "purposes (motive: description)", "notes"],
            rows=rows,
            colspec="lXX",
            notes="Purpose extraction is useful for document-level framing; clause clustering is the main instrument discovery step.",
        )

    # --- Baseline table (optional) ---
    if include_baseline:
        baseline_csv = baseline_dir / "tables" / "full_ordinance__top_clusters_by_weight.csv"
        if baseline_csv.is_file():
            df = pd.read_csv(baseline_csv)
            keep = df[["cluster_id", "cluster_weight", "cluster_size", "topic_name"]].head(12)
            rows = [
                [
                    str(r["cluster_id"]),
                    f"{float(r['cluster_weight']):.2f}",
                    str(int(r["cluster_size"])),
                    str(r["topic_name"]),
                ]
                for _, r in keep.iterrows()
            ]
            _write_table_tex(
                path=tables_dir / "baseline_top_clusters.tex",
                caption="Baseline chunk-topic clusters (top by weight; full ordinances).",
                label="tab:baseline",
                columns=["cluster", "weight", "size", "topic label"],
                rows=rows,
                colspec="rrrX",
                notes="These clusters are over multi-thousand-character chunks; hybrid clustering instead targets extracted atomic requirements.",
            )

    elapsed = round(time.time() - t0, 2)
    print(f"Done. wrote={out_dir} elapsed_s={elapsed}")


if __name__ == "__main__":
    main()
