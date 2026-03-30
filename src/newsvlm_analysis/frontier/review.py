from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import clean_optional_str, write_json


def _stratified_sample(df: pd.DataFrame, group_col: str, target_total: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    group_count = max(1, df[group_col].nunique(dropna=False))
    per_group = max(1, target_total // group_count)
    sampled_parts: list[pd.DataFrame] = []
    for _, group in df.groupby(group_col, dropna=False):
        sampled_parts.append(group.sample(min(len(group), per_group), random_state=0))
    if not sampled_parts:
        return []
    return pd.concat(sampled_parts, ignore_index=True).to_dict(orient="records")


def _init_packet(packet_dir: Path, *, metadata: dict[str, Any], review_columns: list[str]) -> None:
    packet_dir.mkdir(parents=True, exist_ok=True)
    write_json(packet_dir / "metadata.json", metadata)
    (packet_dir / "review_events.jsonl").write_text("", encoding="utf-8")
    with (packet_dir / "review_snapshot.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_columns)
        writer.writeheader()


def _write_issue_packets(packet_dir: Path, rows: list[dict[str, Any]], key: str) -> None:
    item_dir = packet_dir / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    with (packet_dir / "index.jsonl").open("w", encoding="utf-8") as index_handle:
        for row in rows:
            index_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            item_path = item_dir / f"{row[key]}.json"
            item_path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame.from_records(rows).to_csv(packet_dir / "review_sheet.csv", index=False)


def build_review_packets(*, run_root: Path) -> dict[str, Any]:
    review_dir = run_root / "review"

    audit_df = pd.read_parquet(run_root / "geo" / "jurisdiction_audit.parquet")
    sections_df = pd.read_parquet(run_root / "structure" / "sections.parquet")
    operative_clause_path = run_root / "semantics" / "operative_clause_semantics.parquet"
    clause_df = pd.read_parquet(operative_clause_path if operative_clause_path.is_file() else run_root / "semantics" / "clause_semantics.parquet")
    experiment_index = pd.read_parquet(run_root / "clusters" / "experiment_index.parquet")

    jurisdiction_packet = review_dir / "jurisdiction_review"
    _init_packet(
        jurisdiction_packet,
        metadata={"packet_type": "jurisdiction_review", "default_sample_size": 100},
        review_columns=["issue_id", "ordinance_doc_id", "jurisdiction_match_status", "manual_status", "notes"],
    )
    jur_rows = _stratified_sample(audit_df, "jurisdiction_match_status", 100)
    _write_issue_packets(jurisdiction_packet, jur_rows, "issue_id")

    section_packet = review_dir / "section_boundary_review"
    _init_packet(
        section_packet,
        metadata={"packet_type": "section_boundary_review", "default_sample_size": 150},
        review_columns=["section_id", "ordinance_doc_id", "section_kind", "boundary_ok", "notes"],
    )
    sec_rows = _stratified_sample(sections_df, "section_kind", 150)
    _write_issue_packets(section_packet, sec_rows, "section_id")

    clause_packet = review_dir / "clause_review"
    _init_packet(
        clause_packet,
        metadata={"packet_type": "clause_review", "default_sample_size": 200},
        review_columns=["clause_id", "ordinance_doc_id", "modality", "atomic_ok", "instrument_ok", "notes"],
    )
    if not clause_df.empty:
        group_col = "instrument_type" if "instrument_type" in clause_df.columns else "modality"
        clause_rows = _stratified_sample(clause_df, group_col, 200)
    else:
        clause_rows = []
    _write_issue_packets(clause_packet, clause_rows, "clause_id")

    cluster_packet_root = review_dir / "cluster_packets"
    cluster_packet_root.mkdir(parents=True, exist_ok=True)
    cluster_packet_count = 0
    for exp in experiment_index.itertuples(index=False):
        exp_name = clean_optional_str(getattr(exp, "experiment_name", ""))
        labels_path = run_root / "clusters" / exp_name / "cluster_labels.parquet"
        exemplars_path = run_root / "clusters" / exp_name / "cluster_exemplars.parquet"
        if not labels_path.is_file() or not exemplars_path.is_file():
            continue
        labels_df = pd.read_parquet(labels_path)
        exemplars_df = pd.read_parquet(exemplars_path)
        exp_dir = cluster_packet_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        for row in labels_df.itertuples(index=False):
            if int(getattr(row, "cluster_id", -1)) < 0 or int(getattr(row, "cluster_size", 0)) < 5:
                continue
            cluster_id = int(getattr(row, "cluster_id"))
            examples = exemplars_df[exemplars_df["cluster_id"] == cluster_id].head(5)
            lines = [
                f"# {exp_name} cluster {cluster_id}",
                "",
                f"- cluster_label: {clean_optional_str(getattr(row, 'cluster_label', ''))}",
                f"- label_taxonomy: {clean_optional_str(getattr(row, 'label_taxonomy', ''))}",
                f"- cluster_size: {int(getattr(row, 'cluster_size', 0))}",
                f"- keywords: {clean_optional_str(getattr(row, 'keywords', ''))}",
                "",
                "## Exemplars",
                "",
            ]
            for ex in examples.itertuples(index=False):
                lines.append(f"### {clean_optional_str(getattr(ex, 'item_id', ''))}")
                text = clean_optional_str(getattr(ex, "text_for_embedding", ""))
                lines.append("")
                lines.append("```text")
                lines.append(text[:2000])
                lines.append("```")
                lines.append("")
            (exp_dir / f"cluster_{cluster_id:04d}.md").write_text("\n".join(lines), encoding="utf-8")
            cluster_packet_count += 1

    write_json(
        review_dir / "review_summary.json",
        {
            "jurisdiction_review_count": len(jur_rows),
            "section_review_count": len(sec_rows),
            "clause_review_count": len(clause_rows),
            "cluster_packet_count": cluster_packet_count,
        },
    )
    return {
        "jurisdiction_review_count": len(jur_rows),
        "section_review_count": len(sec_rows),
        "clause_review_count": len(clause_rows),
        "cluster_packet_count": cluster_packet_count,
    }
