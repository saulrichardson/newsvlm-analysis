from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import write_json


_SECTION_CLUSTER_LABEL_OVERRIDES: dict[int, str] = {
    23: "Residential District Uses and Standards",
    24: "Supplementary Regulations and Exceptions",
}


def _plot_geo_status(audit: pd.DataFrame, out_path: Path) -> None:
    counts = audit["jurisdiction_match_status"].value_counts().reindex(["match", "mismatch", "missing_jurisdiction"]).fillna(0)
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color=["#2f855a", "#dd6b20", "#718096"])
    plt.title("Publication vs Governing Jurisdiction")
    plt.ylabel("Issue count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_heatmap(df: pd.DataFrame, *, index_col: str, column_col: str, value_col: str, title: str, colorbar_label: str, out_path: Path) -> None:
    pivot = df.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc="sum", fill_value=0.0)
    if pivot.empty:
        return
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(8, max(4, 0.45 * len(pivot.index))))
    plt.imshow(pivot.values, aspect="auto", cmap="Blues")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_duplicate_histogram(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 5))
    df["issue_count"].plot(kind="hist", bins=min(20, max(5, int(df["issue_count"].max()))), color="#4a5568")
    plt.title("Duplicate Group Size Distribution")
    plt.xlabel("Issues per dedup group")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _clean_cluster_label(value: object) -> str:
    text = str(value or "").strip().replace("_", " ")
    return text or "cluster"


def _apply_cluster_label_overrides(plot_df: pd.DataFrame, title: str) -> pd.DataFrame:
    lowered = str(title or "").strip().lower()
    if not lowered.startswith("section_semantic_hdbscan"):
        return plot_df
    if "cluster_id" not in plot_df.columns or "cluster_label" not in plot_df.columns:
        return plot_df

    updated = plot_df.copy()
    cluster_ids = pd.to_numeric(updated["cluster_id"], errors="coerce")
    raw_labels = updated["cluster_label"].astype(str).str.strip()
    placeholder_mask = raw_labels.str.fullmatch(r"cluster\s+\d+", case=False) | raw_labels.eq("") | raw_labels.eq("nan")

    for cluster_id, label in _SECTION_CLUSTER_LABEL_OVERRIDES.items():
        mask = cluster_ids.eq(cluster_id) & placeholder_mask
        if mask.any():
            updated.loc[mask, "cluster_label"] = label
    return updated


def _max_embedded_labels(cluster_count: int) -> int:
    if cluster_count <= 8:
        return cluster_count
    if cluster_count <= 16:
        return 10
    if cluster_count <= 30:
        return 10
    if cluster_count <= 50:
        return 12
    return 12


def _label_limit_for_title(title: str, cluster_count: int) -> int:
    lowered = str(title or "").strip().lower()
    base = _max_embedded_labels(cluster_count)
    if lowered.startswith("clause_instrument_hdbscan"):
        return max(base, 14)
    return base


def _select_labeled_centroids(centroids: pd.DataFrame, *, limit: int, title: str) -> pd.DataFrame:
    if centroids.empty or limit <= 0:
        return centroids.head(0)

    lowered = str(title or "").strip().lower()
    x_span = max(float(centroids["umap_x"].max() - centroids["umap_x"].min()), 1e-6)
    y_span = max(float(centroids["umap_y"].max() - centroids["umap_y"].min()), 1e-6)
    min_sep = 0.10
    if lowered.startswith("clause_instrument_hdbscan"):
        min_sep = 0.11
    elif lowered.startswith("section_semantic_hdbscan"):
        min_sep = 0.11

    selected_rows: list[int] = []
    selected_points: list[tuple[float, float]] = []
    for idx, row in centroids.iterrows():
        x = float(row["umap_x"])
        y = float(row["umap_y"])
        too_close = False
        for sx, sy in selected_points:
            dx = (x - sx) / x_span
            dy = (y - sy) / y_span
            if (dx * dx + dy * dy) ** 0.5 < min_sep:
                too_close = True
                break
        if too_close:
            continue
        selected_rows.append(idx)
        selected_points.append((x, y))
        if len(selected_rows) >= limit:
            break
    return centroids.loc[selected_rows].reset_index(drop=True)


def _cluster_centroids(assignments: pd.DataFrame) -> pd.DataFrame:
    non_noise = assignments.loc[assignments["cluster_id"].astype(int) >= 0].copy()
    if non_noise.empty:
        return pd.DataFrame(columns=["cluster_id", "umap_x", "umap_y", "cluster_size", "cluster_label"])

    rows: list[dict[str, object]] = []
    for cluster_id, group in non_noise.groupby("cluster_id", dropna=False):
        label_series = group["cluster_label"].astype(str).str.strip()
        label = next((x for x in label_series.tolist() if x), "cluster")
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "umap_x": float(group["umap_x"].median()),
                "umap_y": float(group["umap_y"].median()),
                "cluster_size": int(len(group)),
                "cluster_label": label,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)


def _label_text_for_plot(*, title: str, cluster_label: object, cluster_size: object) -> str:
    lowered = str(title or "").strip().lower()
    clean = _clean_cluster_label(cluster_label)
    if lowered.startswith("clause_instrument_hdbscan"):
        return textwrap.fill(clean, width=26)
    return f"{clean}\n(n={int(cluster_size)})"


def _plot_umap(assignments: pd.DataFrame, out_path: Path, title: str) -> None:
    if assignments.empty or "umap_x" not in assignments.columns or "umap_y" not in assignments.columns:
        return
    fig, ax = plt.subplots(figsize=(11.5, 8.0))

    plot_df = _apply_cluster_label_overrides(assignments, title)
    plot_df = plot_df[np.isfinite(plot_df["umap_x"]) & np.isfinite(plot_df["umap_y"])].copy()
    if plot_df.empty:
        plt.close(fig)
        return

    noise = plot_df.loc[plot_df["cluster_id"].astype(int) < 0]
    non_noise = plot_df.loc[plot_df["cluster_id"].astype(int) >= 0]

    if not noise.empty:
        ax.scatter(
            noise["umap_x"],
            noise["umap_y"],
            color="#cbd5e0",
            s=12,
            alpha=0.45,
            linewidths=0.0,
            label="noise",
            zorder=1,
        )

    if not non_noise.empty:
        cluster_ids = sorted(int(x) for x in non_noise["cluster_id"].unique().tolist())
        cmap = plt.get_cmap("tab20", max(1, len(cluster_ids)))
        for idx, cluster_id in enumerate(cluster_ids):
            group = non_noise.loc[non_noise["cluster_id"].astype(int) == cluster_id]
            ax.scatter(
                group["umap_x"],
                group["umap_y"],
                color=cmap(idx),
                s=16,
                alpha=0.75,
                linewidths=0.0,
                zorder=2,
            )

    centroids = _cluster_centroids(plot_df)
    label_limit = _label_limit_for_title(title, len(centroids))
    labeled = _select_labeled_centroids(centroids, limit=label_limit, title=title)
    offsets = [
        (16, 16),
        (16, -16),
        (-16, 16),
        (-16, -16),
        (22, 0),
        (-22, 0),
        (0, 22),
        (0, -22),
        (28, 14),
        (28, -14),
        (-28, 14),
        (-28, -14),
        (14, 28),
        (14, -28),
        (-14, 28),
        (-14, -28),
        (34, 0),
        (-34, 0),
        (0, 34),
        (0, -34),
    ]
    placed_bboxes = []

    fig.canvas.draw()

    for _, row in labeled.iterrows():
        x = float(row["umap_x"])
        y = float(row["umap_y"])
        label_text = _label_text_for_plot(
            title=title,
            cluster_label=row["cluster_label"],
            cluster_size=row["cluster_size"],
        )
        ax.scatter([x], [y], color="black", s=24, marker="x", linewidths=0.8, zorder=4)

        placed = False
        renderer = fig.canvas.get_renderer()
        for dx, dy in offsets:
            ann = ax.annotate(
                label_text,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.0,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#1a202c", "alpha": 0.94, "linewidth": 0.5},
                arrowprops={"arrowstyle": "-", "color": "#1a202c", "linewidth": 0.45, "alpha": 0.60},
                zorder=5,
            )
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.03, 1.12)
            if any(bbox.overlaps(other) for other in placed_bboxes):
                ann.remove()
                continue
            placed_bboxes.append(bbox)
            placed = True
            break

        if not placed and not str(title or "").strip().lower().startswith("clause_instrument_hdbscan"):
            ax.annotate(
                label_text,
                xy=(x, y),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=6.6,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#1a202c", "alpha": 0.88, "linewidth": 0.45},
                zorder=5,
            )

    label_note = "all non-noise clusters labeled" if len(centroids) <= label_limit else f"largest {label_limit} clusters labeled"
    ax.set_title(f"{title}\n{label_note}", fontsize=15)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.12, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _resolve_llm_v2_sample(run_root: Path, sample: Literal["pilot", "full"] | None) -> Literal["pilot", "full"]:
    if sample is not None:
        target = str(sample).strip().lower()
        if target not in {"pilot", "full"}:
            raise SystemExit(f"Unsupported report sample: {sample}")
        return target  # type: ignore[return-value]
    available = sorted(path.stem.removeprefix("research_summary__") for path in (run_root / "research").glob("research_summary__*.json"))
    if len(available) == 1 and available[0] in {"pilot", "full"}:
        return available[0]  # type: ignore[return-value]
    if not available:
        raise SystemExit(f"No research_summary__*.json files found under {run_root / 'research'}")
    raise SystemExit(
        "Multiple research summary files found. Pass an explicit sample so the report does not mix pilot/full surfaces: "
        + ", ".join(available)
    )


def build_report_bundle(*, run_root: Path, sample: Literal["pilot", "full"] | None = None) -> dict[str, object]:
    reports_dir = run_root / "reports"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    audit = pd.read_parquet(run_root / "geo" / "jurisdiction_audit.parquet")
    motive_by_region = pd.read_parquet(run_root / "story" / "motive_shares_by_region.parquet")
    instrument_by_region = pd.read_parquet(run_root / "story" / "instrument_shares_by_region.parquet")
    substantive_instrument_by_region = pd.read_parquet(run_root / "story" / "substantive_instrument_shares_by_region.parquet")
    duplicates = pd.read_parquet(run_root / "story" / "duplicate_publication_patterns.parquet")
    mismatch = pd.read_parquet(run_root / "story" / "high_mismatch_publications.parquet")
    outliers = pd.read_parquet(run_root / "story" / "outlier_ordinances.parquet")
    regional_raw = pd.read_parquet(run_root / "story" / "regional_counts_raw.parquet")
    regional_dedup = pd.read_parquet(run_root / "story" / "regional_counts_dedup.parquet")
    review_summary_path = run_root / "review" / "review_summary.json"
    review_summary = json.loads(review_summary_path.read_text(encoding="utf-8")) if review_summary_path.is_file() else {}
    semantics_summary = json.loads((run_root / "semantics" / "semantics_summary.json").read_text(encoding="utf-8"))
    story_summary = json.loads((run_root / "story" / "story_summary.json").read_text(encoding="utf-8"))
    llm_sample = _resolve_llm_v2_sample(run_root, sample) if (run_root / "research").is_dir() else None
    research_summary_path = (run_root / "research" / f"research_summary__{llm_sample}.json") if llm_sample else None
    if research_summary_path is not None and not research_summary_path.is_file():
        raise SystemExit(f"Missing requested research summary: {research_summary_path}")
    research_summary = json.loads(research_summary_path.read_text(encoding="utf-8")) if research_summary_path is not None else None
    candidate_storylines_path = (run_root / "research" / f"candidate_storylines__{llm_sample}.md") if llm_sample else None
    if candidate_storylines_path is not None and not candidate_storylines_path.is_file():
        candidate_storylines_path = None

    policy_cluster_path = run_root / "story" / "policy_signature_clusters.parquet"
    policy_region_path = run_root / "story" / "policy_signature_shares_by_region.parquet"
    policy_clusters = pd.read_parquet(policy_cluster_path) if policy_cluster_path.is_file() else pd.DataFrame()
    policy_region = pd.read_parquet(policy_region_path) if policy_region_path.is_file() else pd.DataFrame()

    _plot_geo_status(audit, figures_dir / "jurisdiction_match_status.png")
    _plot_heatmap(
        motive_by_region,
        index_col="primary_motive",
        column_col="jurisdiction_region",
        value_col="share_within_region",
        title="Primary Motive Share by Region",
        colorbar_label="Share within region",
        out_path=figures_dir / "motive_by_region_heatmap.png",
    )
    _plot_heatmap(
        substantive_instrument_by_region,
        index_col="instrument_type",
        column_col="jurisdiction_region",
        value_col="share_within_region",
        title="Substantive Instrument Presence by Region",
        colorbar_label="Share of ordinances within region",
        out_path=figures_dir / "instrument_by_region_heatmap.png",
    )
    _plot_duplicate_histogram(duplicates, figures_dir / "duplicate_group_sizes.png")
    if not policy_region.empty:
        _plot_heatmap(
            policy_region,
            index_col="cluster_label",
            column_col="jurisdiction_region",
            value_col="share_within_region",
            title="Policy Signature Share by Region",
            colorbar_label="Share of ordinances within region",
            out_path=figures_dir / "policy_signature_by_region_heatmap.png",
        )

    exp_index = pd.read_parquet(run_root / "clusters" / "experiment_index.parquet")
    for exp in exp_index.itertuples(index=False):
        exp_name = str(getattr(exp, "experiment_name", ""))
        assign_path = run_root / "clusters" / exp_name / "assignments.parquet"
        umap_path = run_root / "embeddings" / str(getattr(exp, "representation_level", "")) / "umap_2d.parquet"
        if assign_path.is_file() and umap_path.is_file():
            assign_df = pd.read_parquet(assign_path)
            umap_df = pd.read_parquet(umap_path)
            merged = assign_df.merge(umap_df, left_on="item_id", right_on="item_id", how="left")
            _plot_umap(merged, figures_dir / f"{exp_name}_umap.png", f"{exp_name} UMAP")

    regional_raw.to_csv(tables_dir / "regional_counts_raw.csv", index=False)
    regional_dedup.to_csv(tables_dir / "regional_counts_dedup.csv", index=False)
    motive_by_region.to_csv(tables_dir / "motive_shares_by_region.csv", index=False)
    instrument_by_region.to_csv(tables_dir / "instrument_shares_by_region.csv", index=False)
    substantive_instrument_by_region.to_csv(tables_dir / "substantive_instrument_shares_by_region.csv", index=False)
    mismatch.head(50).to_csv(tables_dir / "high_mismatch_publications_top50.csv", index=False)
    outliers.head(50).to_csv(tables_dir / "outlier_ordinances_top50.csv", index=False)
    duplicates.head(100).to_csv(tables_dir / "duplicate_publication_patterns_top100.csv", index=False)
    if not policy_clusters.empty:
        policy_clusters.to_csv(tables_dir / "policy_signature_clusters.csv", index=False)
    if not policy_region.empty:
        policy_region.to_csv(tables_dir / "policy_signature_shares_by_region.csv", index=False)

    is_llm_v2 = research_summary is not None
    title = "# Frontier LLM v2 Pilot Report" if is_llm_v2 else "# Frontier Full-Code Research Lab v1"

    lines = [
        title,
        "",
        "## Summary",
        "",
        f"- Corpus issues: {len(audit)}",
        f"- Jurisdiction matches: {(audit['jurisdiction_match_status'] == 'match').sum()}",
        f"- Jurisdiction mismatches: {(audit['jurisdiction_match_status'] == 'mismatch').sum()}",
        f"- Missing jurisdiction: {(audit['jurisdiction_match_status'] == 'missing_jurisdiction').sum()}",
        f"- Clause rows: {semantics_summary.get('clause_rows', 0)}",
        f"- Operative clause rows: {semantics_summary.get('operative_clause_rows', 0)}",
        f"- Operative hard-feature rows: {semantics_summary.get('operative_feature_rows', 0)}",
        f"- Policy signature clusters: {story_summary.get('policy_signature_cluster_count', 0)}",
        f"- Reportable policy signature clusters: {story_summary.get('reportable_policy_signature_cluster_count', 0)}",
        "",
        "## Report Chapters",
        "",
    ]
    if is_llm_v2:
        lines.extend(
            [
                "### LLM-Native Pilot Status",
                "",
                f"- OpenAI section embedding winner: {research_summary.get('embedding_surface_winner', 'n/a')}",
                f"- Section semantic clusters: {research_summary.get('section_cluster_count', 0)}",
                f"- Enriched-clause documents covered: {research_summary.get('enriched_clause_doc_count', 0)} / {research_summary.get('doc_count', 0)}",
                f"- Enriched-clause rows: {research_summary.get('enriched_clause_row_count', 0)}",
                f"- Clause clusters covered: {research_summary.get('clause_cluster_count', 0)}",
                f"- Regulatory style families: {research_summary.get('regulatory_style_family_count', 0)}",
                f"- Complexity/alignment raw correlation: {float(research_summary.get('complexity_alignment_correlation', 0.0)):.3f}",
                f"- Complexity/alignment adjusted slope: {float(research_summary.get('complexity_alignment_adjusted_coef', 0.0)):.3f}",
                "",
                "### Corpus and Jurisdiction Audit",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "### Corpus and Jurisdiction Audit",
                "",
            ]
        )
    lines.extend(
        [
            f"![Jurisdiction Match Status]({(figures_dir / 'jurisdiction_match_status.png').relative_to(reports_dir)})",
            "",
            "### Geography Coverage and Regional Differences",
            "",
            f"- Regional raw rows: {len(regional_raw)}",
            f"- Regional deduplicated rows: {len(regional_dedup)}",
            "",
            "### Motive and Operative Instrument Distributions",
            "",
            f"![Motive Heatmap]({(figures_dir / 'motive_by_region_heatmap.png').relative_to(reports_dir)})",
            "",
            f"![Substantive Instrument Heatmap]({(figures_dir / 'instrument_by_region_heatmap.png').relative_to(reports_dir)})",
            "",
            "### Structured Ordinance Typologies",
            "",
        ]
    )
    if not policy_region.empty:
        lines.extend(
            [
                f"![Policy Signature Heatmap]({(figures_dir / 'policy_signature_by_region_heatmap.png').relative_to(reports_dir)})",
                "",
            ]
        )
    if is_llm_v2:
        lines.extend(
            [
                "### LLM-Native Research Frontiers",
                "",
                f"- Architecture families: {research_summary.get('architecture_family_count', 0)}",
                f"- Clause-side governance documents: {research_summary.get('governance_style_doc_count', 0)}",
                f"- Instrument diversity documents: {research_summary.get('instrument_diversity_doc_count', 0)}",
                f"- Regulatory style silhouette: {float(research_summary.get('regulatory_style_family_silhouette_score', 0.0)):.3f}",
                f"- Policy-signature silhouette: {float(research_summary.get('policy_signature_silhouette_score', 0.0)):.3f}",
                "",
                "### Coherence and Complexity",
                "",
                f"- Mean purpose/section alignment share: {float(research_summary.get('mean_alignment_share', 0.0)):.3f}",
                f"- Mean non-`other` alignment share: {float(research_summary.get('mean_non_other_alignment_share', 0.0)):.3f}",
                (
                    f"- Most coherence-aligned regulatory style family: `{str(research_summary.get('regulatory_style_most_coherent_family', ''))}`"
                    if str(research_summary.get("regulatory_style_most_coherent_family", "")).strip()
                    else "- Most coherence-aligned regulatory style family: unavailable"
                ),
                (
                    f"- Most fragmented regulatory style family: `{str(research_summary.get('regulatory_style_most_fragmented_family', ''))}`"
                    if str(research_summary.get("regulatory_style_most_fragmented_family", "")).strip()
                    else "- Most fragmented regulatory style family: unavailable"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "### Publication-vs-Jurisdiction Mismatch Patterns",
            "",
            f"- High mismatch publications identified: {len(mismatch)}",
            "",
            "### Outliers and Duplicate Publication Patterns",
            "",
            f"![Duplicate Group Sizes]({(figures_dir / 'duplicate_group_sizes.png').relative_to(reports_dir)})",
            "",
            "### Review Infrastructure",
            "",
            f"- Jurisdiction review rows: {review_summary.get('jurisdiction_review_count', 0)}",
            f"- Section review rows: {review_summary.get('section_review_count', 0)}",
            f"- Clause review rows: {review_summary.get('clause_review_count', 0)}",
            f"- Cluster review packets: {review_summary.get('cluster_packet_count', 0)}",
            "",
            "## Outputs",
            "",
            "- Tables are under `tables/`.",
            "- Figures are under `figures/`.",
            "- Story packets are under `../story/packets/`.",
            "- Review packets are under `../review/`.",
            "- Research outputs are under `../research/`.",
            (
                f"- Candidate story note: `../research/{candidate_storylines_path.name}`."
                if candidate_storylines_path is not None
                else "- Candidate story note: unavailable."
            ),
            "",
        ]
    )
    (reports_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = {
        "run_root": str(run_root),
        "tables": sorted(p.name for p in tables_dir.glob("*.csv")),
        "figures": sorted(p.name for p in figures_dir.glob("*.png")),
    }
    write_json(reports_dir / "provenance.json", provenance)
    return {"report_path": str(reports_dir / "report.md"), "figure_count": len(list(figures_dir.glob("*.png")))}
