#!/usr/bin/env python3
"""
Build a single LaTeX report that consolidates the issue-topic pipeline outputs.

Inputs
  - A corpus manifest JSONL that lists issues and their issue-level label:
      /vast/.../merged_manifest.jsonl (from merge_issue_zoning_manifests.py)
  - One or more per-label run roots produced by the chunk topic pipeline:
      <RUN_ROOT>/
        clusters_chunks/cluster_summary.json
        clusters_chunks/clusters.jsonl
        clusters_chunks/scatter.png
        plots_chunks_labeled/cluster_frequency_year.png
        plots_chunks_labeled/cluster_counts_by_year.csv
        <CLUSTER_LABEL_DIR>/outputs/cluster_labels.jsonl  (default dir: cluster_topic_labels)

Outputs (output_dir)
  - report.tex (single document)
  - figures/*.png (copied from run roots)
  - tables/*.tex (LaTeX tables)
  - tables/*.csv (machine-readable summaries)
  - provenance.json (paths + options)

This script does NOT compile LaTeX (pdflatex/latexmk) — it only writes sources/assets.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunSpec:
    label: str
    root: Path


def _tex_escape(s: str) -> str:
    # Minimal, pragmatic escaping for LaTeX tables/text.
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9]+")


def _tex_label_key(s: str) -> str:
    """
    Produce a LaTeX-safe key for \\label{...}.

    Do NOT use _tex_escape() here: label keys must not contain LaTeX commands
    like \\_.
    """
    cleaned = _LABEL_SAFE_RE.sub("-", s.strip()).strip("-")
    return cleaned or "label"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise SystemExit(f"Missing file to copy: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _percent(numer: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return 100.0 * numer / denom


def _format_int(n: int) -> str:
    return f"{n:,d}"


def _format_float(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}"


def _format_pct(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}\\%"


def _write_tex(path: Path, tex: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not tex.endswith("\n"):
        tex += "\n"
    path.write_text(tex, encoding="utf-8")


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _parse_run_spec(raw: str) -> RunSpec:
    # Format: LABEL:/path/to/run_root
    if ":" not in raw:
        raise SystemExit(f"Invalid --run '{raw}'. Expected LABEL:/path/to/run_root")
    label, path = raw.split(":", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Invalid --run '{raw}': empty label")
    root = Path(path).expanduser().resolve()
    return RunSpec(label=label, root=root)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a LaTeX report for issue-topic clustering runs.")
    ap.add_argument("--corpus-manifest", required=True, help="Path to merged_manifest.jsonl (issue-level labels).")
    ap.add_argument("--run", action="append", required=True, help="Run spec: LABEL:/path/to/run_root (repeatable).")
    ap.add_argument("--output-dir", required=True, help="Directory to write report.tex + figures + tables into.")

    ap.add_argument("--title", default="Zoning Text Topics from Newspaper Issues", help="Report title.")
    ap.add_argument("--top-clusters", type=int, default=15, help="How many clusters to include in top tables.")
    ap.add_argument(
        "--year-plot-dir-name",
        default="plots_chunks_labeled_top50_noother",
        help=(
            "Subdirectory under each run root containing cluster_frequency_year.png. "
            "Default assumes you generated top-50 plots with no 'other' series."
        ),
    )
    ap.add_argument(
        "--cluster-label-dir-name",
        default="cluster_topic_labels",
        help=(
            "Subdirectory under each run root containing cluster topic labels. "
            "Expected layout: <RUN_ROOT>/<dir>/outputs/cluster_labels.jsonl"
        ),
    )

    ap.add_argument(
        "--include-scatter",
        action="store_true",
        help="Generate + include UMAP scatter plots (top clusters labeled; others gray).",
    )
    ap.add_argument(
        "--include-year-frequency-plots",
        action="store_true",
        help="Copy + include per-label cluster-frequency-over-time plots.",
    )
    ap.add_argument(
        "--figure-layout",
        choices=["standard", "fullpage"],
        default="standard",
        help=(
            "How to embed figures in the LaTeX output. "
            "standard=figure environment with caption inside the normal text block. "
            "fullpage=each figure is its own page with small margins (better readability for top-k=50 plots)."
        ),
    )
    ap.add_argument("--emit-year-counts-table", action="store_true", help="Include a longtable of counts by year.")
    return ap.parse_args()


def _load_cluster_labels(path: Path) -> dict[int, dict[str, Any]]:
    labels: dict[int, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        cid = row.get("cluster_id")
        if not isinstance(cid, int):
            raise SystemExit(f"Invalid cluster label row missing integer cluster_id: {row}")
        labels[cid] = row
    return labels


def _load_cluster_sizes(cluster_summary_path: Path) -> dict[int, int]:
    summary = _read_json(cluster_summary_path)
    raw = summary.get("cluster_sizes")
    if not isinstance(raw, dict):
        raise SystemExit(f"cluster_summary.json missing 'cluster_sizes' dict: {cluster_summary_path}")
    out: dict[int, int] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = int(v)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Bad cluster_sizes entry {k}:{v} in {cluster_summary_path}: {e}") from e
    return out


def _load_cluster_label_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _iter_jsonl(path):
        cid = row.get("cluster_id")
        if not isinstance(cid, int):
            raise SystemExit(f"Invalid cluster label row missing integer cluster_id: {row}")
        if cid == -1:
            continue
        rows.append(row)
    return rows


def _summarize_issue_stats_from_clusters_jsonl(clusters_jsonl: Path) -> dict[str, Any]:
    # issue_id -> chunk_count (expected constant per issue)
    chunks_per_issue: dict[str, int] = {}
    chars_per_issue: defaultdict[str, int] = defaultdict(int)

    for row in _iter_jsonl(clusters_jsonl):
        issue_id = row.get("issue_id")
        if not isinstance(issue_id, str) or not issue_id:
            continue
        cc = row.get("chunk_count")
        if isinstance(cc, int) and cc > 0:
            prev = chunks_per_issue.get(issue_id)
            if prev is None:
                chunks_per_issue[issue_id] = cc
            elif prev != cc:
                raise SystemExit(
                    f"Inconsistent chunk_count for issue_id={issue_id}: saw {prev} then {cc} in {clusters_jsonl}"
                )
        ch = row.get("chunk_chars")
        if isinstance(ch, int) and ch >= 0:
            chars_per_issue[issue_id] += ch

    if not chunks_per_issue:
        raise SystemExit(f"No issue_id/chunk_count rows found in {clusters_jsonl} (unexpected)")

    chunks_values = list(chunks_per_issue.values())
    chars_values = [chars_per_issue[iid] for iid in chunks_per_issue.keys()]

    def pct(vals: list[int], q: float) -> float:
        if not vals:
            return 0.0
        vs = sorted(vals)
        k = max(0, min(len(vs) - 1, int(round((len(vs) - 1) * q))))
        return float(vs[k])

    return {
        "issues": len(chunks_values),
        "chunks_per_issue": {
            "min": int(min(chunks_values)),
            "p10": pct(chunks_values, 0.10),
            "median": float(statistics.median(chunks_values)),
            "p90": pct(chunks_values, 0.90),
            "max": int(max(chunks_values)),
            "mean": float(statistics.mean(chunks_values)),
        },
        "chars_per_issue": {
            "min": int(min(chars_values)),
            "p10": pct(chars_values, 0.10),
            "median": float(statistics.median(chars_values)),
            "p90": pct(chars_values, 0.90),
            "max": int(max(chars_values)),
            "mean": float(statistics.mean(chars_values)),
        },
    }


def _load_cluster_ids_and_weights(clusters_jsonl: Path) -> tuple[list[int], dict[int, float]]:
    """
    Read per-chunk cluster assignments and doc-weights from clusters.jsonl.

    Returns:
      - cluster_ids: list[int] aligned with umap_2d.npy row order
      - weights: cluster_id -> sum(doc_weight) across all chunks
    """
    cluster_ids: list[int] = []
    weights: defaultdict[int, float] = defaultdict(float)
    for row in _iter_jsonl(clusters_jsonl):
        cid = row.get("cluster_id")
        if not isinstance(cid, int):
            raise SystemExit(f"clusters.jsonl row missing integer cluster_id: {row}")
        cluster_ids.append(int(cid))
        w = row.get("doc_weight")
        if not isinstance(w, (int, float)):
            raise SystemExit(f"clusters.jsonl row missing numeric doc_weight for cluster_id={cid}: got {w!r}")
        weights[int(cid)] += float(w)
    if not cluster_ids:
        raise SystemExit(f"No rows found in clusters.jsonl: {clusters_jsonl}")
    return cluster_ids, dict(weights)


def _write_hybrid_umap_scatter(
    *,
    label: str,
    umap_2d_npy: Path,
    cluster_ids: list[int],
    weights: dict[int, float],
    cluster_labels: dict[int, dict[str, Any]],
    out_png: Path,
    top_k: int,
    max_points_plot: int = 20000,
) -> None:
    """
    Render a "hybrid" UMAP scatter:
      - Top-K clusters (by cumulative issue-weighted mass) are colored and labeled.
      - All other clusters (including noise=-1) are gray.
    """
    if top_k <= 0:
        raise SystemExit(f"top_k must be > 0 (got {top_k})")
    if not umap_2d_npy.is_file():
        raise SystemExit(f"Missing umap_2d.npy for scatter: {umap_2d_npy}")

    X = np.load(umap_2d_npy)
    if X.ndim != 2 or X.shape[1] != 2:
        raise SystemExit(f"Expected umap_2d.npy shape (n,2); got {X.shape} at {umap_2d_npy}")

    if len(cluster_ids) != int(X.shape[0]):
        raise SystemExit(
            f"UMAP/cluster alignment mismatch for {label}: "
            f"umap_rows={int(X.shape[0])} clusters_rows={len(cluster_ids)} "
            f"(refusing to plot misaligned data)"
        )
    c = np.asarray(cluster_ids, dtype=np.int32)

    # Top clusters by issue-weighted mass (sum(doc_weight)).
    weight_items = [(cid, w) for cid, w in weights.items() if cid != -1]
    weight_items.sort(key=lambda x: x[1], reverse=True)
    top_clusters = [cid for cid, _ in weight_items[:top_k]]
    if not top_clusters:
        raise SystemExit(f"No non-noise clusters found for {label} (unexpected)")

    top_set = set(top_clusters)
    is_top = np.isin(c, np.array(top_clusters, dtype=np.int32))
    top_idx_all = np.flatnonzero(is_top)
    other_idx_all = np.flatnonzero(~is_top)

    # Downsample for readability/plot size, while biasing toward keeping top clusters visible.
    rng = np.random.RandomState(0)
    max_total = int(max_points_plot)
    if max_total > 0 and int(X.shape[0]) > max_total:
        max_top = min(int(len(top_idx_all)), int(max_total * 0.60))
        max_other = min(int(len(other_idx_all)), int(max_total - max_top))
        if max_other < (max_total - max_top) and len(top_idx_all) > max_top:
            max_top = min(int(len(top_idx_all)), int(max_total - max_other))

        top_idx = (
            rng.choice(top_idx_all, size=max_top, replace=False) if len(top_idx_all) > max_top else top_idx_all
        )
        other_idx = (
            rng.choice(other_idx_all, size=max_other, replace=False)
            if len(other_idx_all) > max_other
            else other_idx_all
        )
    else:
        top_idx = top_idx_all
        other_idx = other_idx_all

    cmap = plt.get_cmap("tab20")
    color_by_cluster = {cid: cmap(i % cmap.N) for i, cid in enumerate(top_clusters)}

    plt.figure(figsize=(10, 8))
    if len(other_idx) > 0:
        plt.scatter(
            X[other_idx, 0],
            X[other_idx, 1],
            c="#bdbdbd",
            s=2,
            alpha=0.20,
            linewidths=0,
        )

    for cid in top_clusters:
        idx = top_idx[c[top_idx] == cid]
        if len(idx) == 0:
            continue
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=[color_by_cluster[cid]],
            s=4,
            alpha=0.75,
            linewidths=0,
        )

    # Label top clusters at robust centroids (medians), with light collision-avoidance.
    x_lo, x_hi = np.quantile(X[:, 0], [0.05, 0.95])
    y_lo, y_hi = np.quantile(X[:, 1], [0.05, 0.95])
    span = float(max(x_hi - x_lo, y_hi - y_lo))
    min_sep = 0.03 * span if span > 0 else 0.5

    placed: list[tuple[float, float]] = []
    weight_rank = {cid: w for cid, w in weight_items}
    top_clusters_sorted = sorted(top_clusters, key=lambda cid: float(weight_rank.get(cid, 0.0)), reverse=True)
    for cid in top_clusters_sorted:
        pts = X[c == cid]
        if pts.shape[0] == 0:
            continue
        cx = float(np.median(pts[:, 0]))
        cy = float(np.median(pts[:, 1]))

        px, py = cx, cy
        for t in range(0, 60):
            if t > 0:
                ang = 0.9 * t
                rad = (t / 12.0) * min_sep
                px = cx + rad * float(np.cos(ang))
                py = cy + rad * float(np.sin(ang))
            if all((px - ox) ** 2 + (py - oy) ** 2 >= (min_sep**2) for ox, oy in placed):
                break

        placed.append((px, py))

        topic = cluster_labels.get(cid, {}).get("topic_name")
        if isinstance(topic, str) and topic.strip():
            topic_txt = topic.strip()
            if len(topic_txt) > 60:
                topic_txt = topic_txt[:57] + "..."
            txt = f"{topic_txt} (#{cid})"
        else:
            txt = f"#{cid}"

        plt.scatter([cx], [cy], c="black", s=14, marker="x", linewidths=0.7)
        plt.text(
            px,
            py,
            txt,
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
        )

    plt.title(f"{label}: UMAP scatter (top {len(top_clusters)} clusters labeled; others gray)")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def _corpus_counts_by_label_and_year(corpus_manifest: Path) -> tuple[Counter[str], dict[int, Counter[str]]]:
    by_label: Counter[str] = Counter()
    by_year: dict[int, Counter[str]] = defaultdict(Counter)
    for row in _iter_jsonl(corpus_manifest):
        label = row.get("label")
        if not isinstance(label, str) or not label:
            continue
        by_label[label] += 1
        d = row.get("issue_date")
        if isinstance(d, str) and len(d) >= 4:
            try:
                y = int(d[:4])
            except ValueError:
                continue
            by_year[y][label] += 1
    return by_label, by_year


def _render_dataset_summary_table(run_summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Dataset and clustering summary by issue label.}")
    lines.append("\\label{tab:dataset-summary}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append("Label & Issues & Chunks & Clusters & Noise chunks & Noise \\% & Avg chunks/issue \\\\")
    lines.append("\\midrule")
    for s in run_summaries:
        lines.append(
            f"{_tex_escape(str(s['label']))} & "
            f"{_format_int(int(s['issues']))} & "
            f"{_format_int(int(s['n_chunks']))} & "
            f"{_format_int(int(s['n_clusters']))} & "
            f"{_format_int(int(s['n_noise']))} & "
            f"{_format_pct(float(s['noise_pct']))} & "
            f"{_format_float(float(s['avg_chunks_per_issue']), 2)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _render_cluster_param_table(run_summaries: list[dict[str, Any]]) -> str:
    # Render per-label params (even if identical; keeps report explicit).
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{UMAP and HDBSCAN parameters used for clustering (by label).}")
    lines.append("\\label{tab:clustering-params}")
    lines.append("\\begin{tabular}{lrrrrlrr}")
    lines.append("\\toprule")
    lines.append("Label & $n_{nbr}$ & $min\\_dist$ & $n_{cmp}$ & metric & min\\_cluster & min\\_samples \\\\")
    lines.append("\\midrule")
    for s in run_summaries:
        params = s.get("params") or {}
        umap_p = params.get("umap") if isinstance(params, dict) else {}
        hdb_p = params.get("hdbscan") if isinstance(params, dict) else {}
        lines.append(
            f"{_tex_escape(str(s['label']))} & "
            f"{int(umap_p.get('n_neighbors', 0))} & "
            f"{_format_float(float(umap_p.get('min_dist', 0.0)), 2)} & "
            f"{int(umap_p.get('n_components', 0))} & "
            f"{_tex_escape(str(umap_p.get('metric', '')))} & "
            f"{int(hdb_p.get('min_cluster_size', 0))} & "
            f"{int(hdb_p.get('min_samples', 0))} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _render_issue_length_stats_table(label: str, stats: dict[str, Any]) -> str:
    c = stats["chunks_per_issue"]
    t = stats["chars_per_issue"]
    key = _tex_label_key(label)
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Issue size statistics for {_tex_escape(label)} (from chunk-level metadata).}}")
    lines.append(f"\\label{{tab:{key}-issue-size}}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append("Metric & Min & P10 & Median & P90 & Max & Mean \\\\")
    lines.append("\\midrule")
    lines.append(
        "Chunks per issue & "
        f"{_format_int(int(c['min']))} & "
        f"{_format_float(float(c['p10']), 1)} & "
        f"{_format_float(float(c['median']), 1)} & "
        f"{_format_float(float(c['p90']), 1)} & "
        f"{_format_int(int(c['max']))} & "
        f"{_format_float(float(c['mean']), 2)} \\\\"
    )
    lines.append(
        "Chars per issue & "
        f"{_format_int(int(t['min']))} & "
        f"{_format_float(float(t['p10']), 0)} & "
        f"{_format_float(float(t['median']), 0)} & "
        f"{_format_float(float(t['p90']), 0)} & "
        f"{_format_int(int(t['max']))} & "
        f"{_format_float(float(t['mean']), 0)} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _render_top_clusters_by_size_table(label: str, rows: list[dict[str, Any]]) -> str:
    key = _tex_label_key(label)
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Top clusters by chunk count for {_tex_escape(label)}.}}")
    lines.append(f"\\label{{tab:{key}-top-clusters-size}}")
    lines.append("\\begin{tabular}{r p{0.28\\textwidth} r p{0.46\\textwidth}}")
    lines.append("\\toprule")
    lines.append("Cluster & Topic name & Size & Keywords (abridged) \\\\")
    lines.append("\\midrule")
    for r in rows:
        kw = r.get("keywords", [])
        if isinstance(kw, list):
            kw_txt = "; ".join(str(x) for x in kw[:8])
        else:
            kw_txt = str(kw)
        lines.append(
            f"{int(r['cluster_id'])} & "
            f"{_tex_escape(str(r.get('topic_name', '')))} & "
            f"{_format_int(int(r['cluster_size']))} & "
            f"{_tex_escape(kw_txt)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _render_top_clusters_by_weight_table(label: str, rows: list[dict[str, Any]]) -> str:
    key = _tex_label_key(label)
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Top clusters by issue-weighted chunk share ($\\sum$doc\\_weight) for {_tex_escape(label)}.}}"
    )
    lines.append(f"\\label{{tab:{key}-top-clusters-weight}}")
    lines.append("\\begin{tabular}{r l r r}")
    lines.append("\\toprule")
    lines.append("Cluster & Topic name & Weight & Size \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(
            f"{int(r['cluster_id'])} & "
            f"{_tex_escape(str(r.get('topic_name', '')))} & "
            f"{_format_float(float(r['cluster_weight']), 2)} & "
            f"{_format_int(int(r['cluster_size']))} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _render_year_counts_longtable(by_year: dict[int, Counter[str]], labels: list[str]) -> str:
    years = sorted(by_year)
    cols = "l" + "r" * len(labels)
    header = "Year" + "".join([f" & {_tex_escape(l)}" for l in labels]) + " \\\\"

    lines: list[str] = []
    lines.append("\\begin{longtable}{%s}" % cols)
    lines.append(
        "\\caption{Issue counts by year and label (corpus manifest).}"
        "\\label{tab:issue-counts-by-year}\\\\"
    )
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    lines.append("\\endfirsthead")

    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    lines.append("\\endhead")

    lines.append("\\midrule")
    lines.append("\\multicolumn{%d}{r}{\\emph{Continued on next page}}\\\\" % (1 + len(labels)))
    lines.append("\\endfoot")

    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")

    for y in years:
        row = [str(y)]
        for lab in labels:
            row.append(str(int(by_year[y].get(lab, 0))))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\end{longtable}")
    return "\n".join(lines)


def _render_all_clusters_longtable(label: str, rows: list[dict[str, Any]]) -> str:
    """
    Appendix table: one row per cluster (all non-noise clusters).
    We keep it compact (no full descriptions) and push full details into CSV.
    """
    key = _tex_label_key(label)
    # Prefer sorting by weight if present; fall back to size.
    def sort_key(r: dict[str, Any]) -> tuple[float, int]:
        w = r.get("cluster_weight_total")
        s = r.get("cluster_size_chunks")
        wf = float(w) if isinstance(w, (int, float)) else 0.0
        sf = int(s) if isinstance(s, int) else 0
        return (wf, sf)

    rows_sorted = sorted(rows, key=sort_key, reverse=True)

    lines: list[str] = []
    lines.append("\\begin{longtable}{r p{0.28\\textwidth} r r p{0.42\\textwidth}}")
    lines.append(
        f"\\caption{{All discovered clusters for {_tex_escape(label)} (sorted by weight).}}"
        f"\\label{{tab:{key}-all-clusters}}\\\\"
    )
    lines.append("\\toprule")
    lines.append("Cluster & Topic name & Size & Weight & Keywords \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")

    lines.append("\\toprule")
    lines.append("Cluster & Topic name & Size & Weight & Keywords \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")

    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{r}{\\emph{Continued on next page}}\\\\")
    lines.append("\\endfoot")

    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")

    for r in rows_sorted:
        cid = int(r.get("cluster_id"))
        name = str(r.get("topic_name") or "")
        size = int(r.get("cluster_size_chunks") or 0)
        weight = float(r.get("cluster_weight_total") or 0.0)
        kw = r.get("keywords") or []
        if isinstance(kw, list):
            kw_txt = "; ".join(str(x) for x in kw[:12])
        else:
            kw_txt = str(kw)
        lines.append(
            f"{cid} & "
            f"{_tex_escape(name)} & "
            f"{_format_int(size)} & "
            f"{_format_float(weight, 2)} & "
            f"{_tex_escape(kw_txt)} \\\\"
        )
    lines.append("\\end{longtable}")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()

    corpus_manifest = Path(args.corpus_manifest).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    by_label, by_year = _corpus_counts_by_label_and_year(corpus_manifest)

    run_specs = [_parse_run_spec(r) for r in args.run]
    run_specs.sort(key=lambda r: r.label)
    all_labels = [r.label for r in run_specs]

    run_summaries: list[dict[str, Any]] = []
    label_dir_name = str(args.cluster_label_dir_name)

    for run in run_specs:
        root = run.root
        cluster_summary_path = root / "clusters_chunks" / "cluster_summary.json"
        clusters_jsonl_path = root / "clusters_chunks" / "clusters.jsonl"
        umap_2d_path = root / "clusters_chunks" / "umap_2d.npy"
        labels_jsonl_path = root / label_dir_name / "outputs" / "cluster_labels.jsonl"

        summary = _read_json(cluster_summary_path)
        n_chunks = int(summary["n_chunks"])
        n_noise = int(summary["n_noise"])
        n_clusters = int(summary["n_clusters"])

        issues = int(by_label.get(run.label, 0))
        if issues <= 0:
            raise SystemExit(
                f"Corpus manifest has 0 issues for label '{run.label}', but a run root was provided: {root}"
            )

        run_summaries.append(
            {
                "label": run.label,
                "issues": issues,
                "n_chunks": n_chunks,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": _percent(n_noise, n_chunks),
                "avg_chunks_per_issue": float(n_chunks) / float(issues) if issues else 0.0,
                "params": summary.get("params"),
            }
        )

        # Copy figures (optional).
        if args.include_year_frequency_plots:
            _copy_file(
                root / str(args.year_plot_dir_name) / "cluster_frequency_year.png",
                figures_dir / f"{run.label}__cluster_frequency_year.png",
            )

        # Per-issue stats (table).
        issue_stats = _summarize_issue_stats_from_clusters_jsonl(clusters_jsonl_path)
        _write_tex(tables_dir / f"{run.label}__issue_size_stats.tex", _render_issue_length_stats_table(run.label, issue_stats))

        # Top clusters tables.
        label_rows = _load_cluster_labels(labels_jsonl_path)
        sizes = _load_cluster_sizes(cluster_summary_path)
        cluster_ids, weights = _load_cluster_ids_and_weights(clusters_jsonl_path)

        # UMAP scatter plot (hybrid: top-K labeled, others gray).
        if args.include_scatter:
            _write_hybrid_umap_scatter(
                label=run.label,
                umap_2d_npy=umap_2d_path,
                cluster_ids=cluster_ids,
                weights=weights,
                cluster_labels=label_rows,
                out_png=figures_dir / f"{run.label}__umap_scatter.png",
                top_k=int(args.top_clusters),
            )

        top_by_size: list[dict[str, Any]] = []
        for cid, sz in sorted(((c, s) for c, s in sizes.items() if c != -1), key=lambda x: x[1], reverse=True)[
            : int(args.top_clusters)
        ]:
            lr = label_rows.get(cid)
            if lr is None:
                raise SystemExit(f"Missing label for cluster_id={cid} in {labels_jsonl_path}")
            top_by_size.append(
                {
                    "cluster_id": int(cid),
                    "cluster_size": int(sz),
                    "topic_name": lr.get("topic_name", ""),
                    "keywords": lr.get("keywords", []),
                }
            )
        _write_tex(
            tables_dir / f"{run.label}__top_clusters_by_size.tex",
            _render_top_clusters_by_size_table(run.label, top_by_size),
        )
        _write_csv_rows(
            tables_dir / f"{run.label}__top_clusters_by_size.csv",
            ["cluster_id", "cluster_size", "topic_name", "keywords"],
            top_by_size,
        )

        top_by_weight: list[dict[str, Any]] = []
        items = [(cid, w) for cid, w in weights.items() if cid != -1]
        items.sort(key=lambda x: x[1], reverse=True)
        for cid, w in items[: int(args.top_clusters)]:
            lr = label_rows.get(cid)
            if lr is None:
                raise SystemExit(f"Missing label for cluster_id={cid} in {labels_jsonl_path}")
            top_by_weight.append(
                {
                    "cluster_id": int(cid),
                    "cluster_weight": float(w),
                    "cluster_size": int(sizes.get(cid, 0)),
                    "topic_name": lr.get("topic_name", ""),
                    "keywords": lr.get("keywords", []),
                }
            )
        _write_tex(
            tables_dir / f"{run.label}__top_clusters_by_weight.tex",
            _render_top_clusters_by_weight_table(run.label, top_by_weight),
        )
        _write_csv_rows(
            tables_dir / f"{run.label}__top_clusters_by_weight.csv",
            ["cluster_id", "cluster_weight", "cluster_size", "topic_name", "keywords"],
            top_by_weight,
        )

        # Appendix artifacts: ALL clusters (non-noise).
        all_rows = _load_cluster_label_rows(labels_jsonl_path)
        _write_tex(
            tables_dir / f"{run.label}__all_clusters.tex",
            _render_all_clusters_longtable(run.label, all_rows),
        )
        _write_csv_rows(
            tables_dir / f"{run.label}__all_clusters.csv",
            [
                "cluster_id",
                "topic_name",
                "topic_description",
                "keywords",
                "cluster_size_chunks",
                "cluster_weight_total",
                "classifier_model",
                "final_prompt_sha256",
                "final_prompt_path",
            ],
            all_rows,
        )

    # Shared tables.
    _write_tex(tables_dir / "dataset_summary.tex", _render_dataset_summary_table(run_summaries))
    _write_tex(tables_dir / "clustering_params.tex", _render_cluster_param_table(run_summaries))

    if args.emit_year_counts_table:
        _write_tex(tables_dir / "issue_counts_by_year.tex", _render_year_counts_longtable(by_year, all_labels))

    # Main LaTeX doc.
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    doc: list[str] = []
    doc.append("\\documentclass[11pt]{article}")
    doc.append("\\usepackage[margin=1in]{geometry}")
    doc.append("\\usepackage{graphicx}")
    doc.append("\\usepackage{booktabs}")
    doc.append("\\usepackage{longtable}")
    doc.append("\\usepackage{float}")
    doc.append("\\usepackage{hyperref}")
    doc.append("\\usepackage{subcaption}")
    doc.append("\\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}")
    doc.append("\\title{%s}" % _tex_escape(str(args.title)))
    doc.append("\\date{%s}" % _tex_escape(now))
    doc.append("\\begin{document}")
    doc.append("\\maketitle")

    doc.append("\\section{Overview}")
    doc.append(
        "This report consolidates the outputs of a topic-discovery pipeline applied to newspaper zoning texts. "
        "Issues are pre-classified into three categories (\\texttt{full\\_ordinance}, "
        "\\texttt{amendment\\_substantial}, and \\texttt{amendment\\_targeted}). "
        "Within each category, we split each issue into roughly 3,000-character chunks, embed each chunk "
        "with OpenAI \\texttt{text-embedding-3-large}, cluster with UMAP$\\rightarrow$HDBSCAN, "
        "and label each discovered cluster with OpenAI (Batch)."
    )
    doc.append("")
    doc.append(
        "\\textbf{Time-series figure convention.} "
        "For readability, the cluster frequency-over-time figures show only the top 50 clusters "
        "(by cumulative issue-weighted mass across all years) and intentionally exclude the remaining clusters. "
        "We do \\emph{not} aggregate the excluded clusters into an \\emph{other} band in the plot. "
        "All discovered (non-noise) clusters are listed in the appendix."
    )

    doc.append("\\section{Dataset summary}")
    doc.append("\\input{tables/dataset_summary.tex}")
    doc.append("\\input{tables/clustering_params.tex}")

    if args.emit_year_counts_table:
        doc.append("\\subsection{Issue counts by year}")
        doc.append("\\input{tables/issue_counts_by_year.tex}")

    for run in run_specs:
        doc.append("\\section{%s}" % _tex_escape(run.label))
        doc.append("\\input{tables/%s__issue_size_stats.tex}" % _tex_escape(run.label))
        doc.append("\\input{tables/%s__top_clusters_by_size.tex}" % _tex_escape(run.label))
        doc.append("\\input{tables/%s__top_clusters_by_weight.tex}" % _tex_escape(run.label))

        figs: list[tuple[str, str]] = []
        if args.include_year_frequency_plots:
            figs.append(("Top clusters over time (issue-weighted)", f"figures/{run.label}__cluster_frequency_year.png"))
        if args.include_scatter:
            figs.append(("UMAP scatter (top clusters labeled; others gray)", f"figures/{run.label}__umap_scatter.png"))

        if figs:
            fig_layout = str(args.figure_layout)
            if fig_layout == "standard":
                # Put figures on their own pages for easy viewing.
                doc.append("\\clearpage")
                doc.append("\\subsection{Figures}")
                for cap, rel in figs:
                    doc.append("\\begin{figure}[H]")
                    doc.append("\\centering")
                    doc.append(
                        "\\includegraphics[width=0.98\\textwidth,height=0.9\\textheight,keepaspectratio]{%s}"
                        % _tex_escape(rel)
                    )
                    doc.append("\\caption{%s: %s}" % (_tex_escape(run.label), _tex_escape(cap)))
                    doc.append("\\end{figure}")
                    doc.append("\\clearpage")
            elif fig_layout == "fullpage":
                # Make figures maximally readable:
                # - each figure becomes its own page
                # - small margins
                # - no caption inside the figure page (caption costs too much real estate)
                #
                # The section label already provides context; open the underlying PNG for pixel-perfect zoom.
                for i, (_, rel) in enumerate(figs):
                    if i == 0:
                        doc.append("\\clearpage")
                    doc.append("\\newgeometry{margin=0.25in}")
                    doc.append("\\thispagestyle{empty}")
                    doc.append("\\begin{center}")
                    doc.append(
                        "\\includegraphics[width=\\linewidth,height=0.98\\textheight,keepaspectratio]{%s}"
                        % _tex_escape(rel)
                    )
                    doc.append("\\end{center}")
                    doc.append("\\restoregeometry")
                    doc.append("\\clearpage")
            else:
                raise SystemExit(f"Unknown --figure-layout: {fig_layout}")

    doc.append("\\section{Notes}")
    doc.append("\\begin{itemize}")
    doc.append("\\item Cluster IDs are only meaningful within a run (not aligned across labels).")
    doc.append("\\item HDBSCAN assigns many chunks to noise (cluster $-1$); noise rates are reported in Table~\\ref{tab:dataset-summary}.")
    doc.append(
        "\\item The `weight' statistic in top-cluster tables is $\\sum$doc\\_weight across chunks, "
        "where doc\\_weight = 1/(\\# chunks in issue)."
    )
    doc.append("\\end{itemize}")

    # Appendix: all clusters per label.
    doc.append("\\clearpage")
    doc.append("\\appendix")
    doc.append("\\section{Appendix: All clusters}")
    doc.append(
        "This appendix lists every discovered (non-noise) cluster for each label, "
        "sorted by cluster weight."
    )
    for run in run_specs:
        doc.append("\\subsection{%s}" % _tex_escape(run.label))
        doc.append("\\input{tables/%s__all_clusters.tex}" % _tex_escape(run.label))
        doc.append("\\clearpage")
    doc.append("\\end{document}")
    _write_tex(out_dir / "report.tex", "\n".join(doc))

    provenance = {
        "created_at": now,
        "corpus_manifest": str(corpus_manifest),
        "runs": [{"label": r.label, "root": str(r.root)} for r in run_specs],
        "top_clusters": int(args.top_clusters),
        "year_plot_dir_name": str(args.year_plot_dir_name),
        "cluster_label_dir_name": label_dir_name,
        "figure_layout": str(args.figure_layout),
        "include_scatter": bool(args.include_scatter),
        "include_year_frequency_plots": bool(args.include_year_frequency_plots),
        "emit_year_counts_table": bool(args.emit_year_counts_table),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Done. wrote={out_dir} runs={len(run_specs)}")


if __name__ == "__main__":
    main()
