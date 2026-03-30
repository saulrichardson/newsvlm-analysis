#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import frontier_bootstrap  # noqa: F401


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _surface(
    *,
    name: str,
    status: str,
    rationale: str,
    artifact: str,
    caveat: str = "",
) -> dict[str, str]:
    row = {
        "surface": name,
        "status": status,
        "artifact": artifact,
        "rationale": rationale,
    }
    if caveat:
        row["caveat"] = caveat
    return row


def build_coverage_audit(*, run_root: Path) -> dict[str, Any]:
    research_dir = run_root / "research"
    reports_dir = run_root / "reports"
    coverage_dir = research_dir / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)

    pilot = _load_json(research_dir / "research_summary__pilot.json")
    full = _load_json(research_dir / "research_summary__full.json")

    surfaces: list[dict[str, str]] = []

    surfaces.append(
        _surface(
            name="Corpus and PID6 linkage",
            status="stable",
            artifact=str(run_root / "corpus" / "full_sample.parquet"),
            rationale="The frozen full linked sample exists and the full section-side v2 surface is built on it.",
        )
    )
    surfaces.append(
        _surface(
            name="Full-sample section structure and motives",
            status="stable",
            artifact=str(run_root / "semantics" / "section_motives__full.parquet"),
            rationale=f"All {int(full.get('doc_count', 0))} full documents are covered with section motives.",
        )
    )
    surfaces.append(
        _surface(
            name="Full-sample section embeddings and architecture clusters",
            status="stable",
            artifact=str(run_root / "clusters" / "section_semantic_hdbscan" / "cluster_labels.parquet"),
            rationale=f"Full architecture surface exists with {int(full.get('section_cluster_count', 0))} non-noise section families.",
        )
    )
    surfaces.append(
        _surface(
            name="Full-sample regional and jurisdiction-type architecture story",
            status="stable",
            artifact=str(research_dir / "research_memo__full.md"),
            rationale="Vintage-adjusted section-cluster contrasts, neighbor retrieval, and region/level effects are built on the full sample.",
        )
    )
    surfaces.append(
        _surface(
            name="Full-sample complexity vs coherence frontier",
            status="stable",
            artifact=str(research_dir / "coherence" / "complexity_alignment_models__full.parquet"),
            rationale=f"Adjusted complexity coefficient is {float(full.get('complexity_alignment_adjusted_coef', 0.0)):.3f} with p={float(full.get('complexity_alignment_adjusted_pvalue', 1.0)):.3f}, so the full-sample residual frontier is real enough to use.",
        )
    )
    surfaces.append(
        _surface(
            name="Embedding comparison",
            status="stable",
            artifact=str(research_dir / "embedding_comparison" / "surface_comparison__full.parquet"),
            rationale=f"The full-sample comparison is built, with current winner `{str(full.get('embedding_surface_winner', ''))}`.",
            caveat="The winner differs between checkpoints, so the writeup should present the comparison as empirical rather than assumed.",
        )
    )

    pilot_doc_count = int(pilot.get("doc_count", 0))
    pilot_clause_docs = int(pilot.get("enriched_clause_doc_count", 0))
    pilot_clause_cov = float(pilot.get("enriched_clause_doc_coverage_rate", 0.0))
    surfaces.append(
        _surface(
            name="Pilot enriched-clause semantic layer",
            status="emerging" if pilot_clause_docs < pilot_doc_count else "stable",
            artifact=str(run_root / "semantics" / "enriched_clauses__pilot.parquet"),
            rationale=f"Pilot clause enrichment now covers {pilot_clause_docs}/{pilot_doc_count} pilot ordinances ({pilot_clause_cov:.1%}).",
            caveat="Useful for governance style and clause families, but still pilot-only and not full national coverage.",
        )
    )
    surfaces.append(
        _surface(
            name="Pilot governance-style families",
            status="emerging",
            artifact=str(research_dir / "style_families" / "regulatory_style_family_summary__pilot.parquet"),
            rationale=f"Pilot governance style has {int(pilot.get('regulatory_style_family_count', 0))} families and silhouette {float(pilot.get('regulatory_style_family_silhouette_score', 0.0)):.3f}.",
            caveat="Strong enough to interpret, but still tied to pilot clause coverage.",
        )
    )
    surfaces.append(
        _surface(
            name="Pilot raw-clause policy signatures",
            status="partial",
            artifact=str(run_root / "clusters" / "policy_signature_kmeans_probe" / "summary.json"),
            rationale=f"Raw clause-share typology has silhouette {float(pilot.get('policy_signature_silhouette_score', 0.0)):.3f}.",
            caveat="Geometrically tighter than alternatives, but still degenerates toward one dominant family.",
        )
    )
    surfaces.append(
        _surface(
            name="Pilot section-normalized policy signatures",
            status="partial",
            artifact=str(run_root / "clusters" / "policy_signature_section_normalized_clause_share_probe" / "summary.json"),
            rationale=f"Section-normalized clause-share typology has {int(pilot.get('policy_signature_section_normalized_clause_share_reportable_cluster_count', 0))} reportable clusters.",
            caveat="Less tight geometrically, but more robust to clause-list inflation and more useful as a document-level synthesis layer.",
        )
    )
    surfaces.append(
        _surface(
            name="Full-sample clause/governance surface",
            status="deferred",
            artifact=str(run_root / "semantics" / "enriched_clauses__full.parquet"),
            rationale="No full enriched-clause surface exists yet, so full national clause/governance results are intentionally deferred.",
            caveat="Do not overclaim full-sample governance or policy-signature results until a true full enriched-clause run exists.",
        )
    )
    surfaces.append(
        _surface(
            name="Longitudinal / within-place change",
            status="deferred",
            artifact=str(run_root / "corpus" / "full_sample.parquet"),
            rationale="Current design is cross-sectional and vintage-adjusted rather than within-place longitudinal.",
            caveat="Repeated amendments and true jurisdiction sequences are not yet integrated.",
        )
    )

    stable = sum(row["status"] == "stable" for row in surfaces)
    emerging = sum(row["status"] == "emerging" for row in surfaces)
    partial = sum(row["status"] == "partial" for row in surfaces)
    deferred = sum(row["status"] == "deferred" for row in surfaces)

    summary = {
        "stable_surface_count": stable,
        "emerging_surface_count": emerging,
        "partial_surface_count": partial,
        "deferred_surface_count": deferred,
        "recommended_backbone": [
            "Full-sample section-level legal architecture by region, jurisdiction type, and vintage-adjusted comparison",
            "Full-sample coherence vs fragmentation frontier",
        ],
        "recommended_second_pillar": [
            "Pilot governance-style families from enriched clauses",
            "Pilot clause-family and document-typology comparisons, with explicit pilot-only caveats",
        ],
    }

    payload = {"summary": summary, "surfaces": surfaces}
    (coverage_dir / "research_surface_coverage_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Frontier LLM v2 Coverage Audit",
        "",
        "## Summary",
        "",
        f"- stable surfaces: {stable}",
        f"- emerging surfaces: {emerging}",
        f"- partial surfaces: {partial}",
        f"- deferred surfaces: {deferred}",
        "",
        "## Recommended Backbone",
        "",
    ]
    for item in summary["recommended_backbone"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended Second Pillar", ""])
    for item in summary["recommended_second_pillar"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Surface Status", ""])
    for row in surfaces:
        lines.extend(
            [
                f"### {row['surface']}",
                "",
                f"- status: `{row['status']}`",
                f"- artifact: `{row['artifact']}`",
                f"- rationale: {row['rationale']}",
            ]
        )
        if row.get("caveat"):
            lines.append(f"- caveat: {row['caveat']}")
        lines.append("")
    (coverage_dir / "research_surface_coverage_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stable-vs-emerging coverage audit for Frontier LLM v2.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = build_coverage_audit(run_root=Path(args.run_root).expanduser().resolve())
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
