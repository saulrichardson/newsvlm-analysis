#!/usr/bin/env python3
"""
Build a human-readable inspection packet for clause-level instrument clusters.

Why this exists
---------------
The PI feedback asks for more granular "regulatory topics/questions" than the
issue/chunk-level clustering yields. The hybrid pipeline extracts atomic clauses,
clusters them, and (optionally) labels clusters into instruments.

This script makes that output easy to *read* and *audit* by producing a Markdown
report (and per-cluster packets) grounded in the exact artifacts on disk.

Inputs
------
- clauses.jsonl (from rehydrate_regulatory_clause_extraction_openai_batch_results.py)
- clusters.jsonl (from cluster_regulatory_clauses_local.py OR cluster_clause_embeddings.py)
- cluster_keywords.jsonl (from cluster_regulatory_clauses_local.py) [optional but recommended]
- cluster_labels.jsonl (from rehydrate_regulatory_instrument_cluster_labels_openai_batch_results.py) [optional]
- sections.jsonl (from rehydrate_regulatory_section_segmentation_openai_batch_results.py) [optional]

Outputs (output_dir)
--------------------
- report.md
- clusters_index.csv
- cluster_packets/cluster_<id>__<safe_name>.md

No APIs are called.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from os.path import expanduser
from pathlib import Path
from typing import Any, Iterable


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


def _safe_slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:100] or "x"


def _preview(s: str, *, max_chars: int) -> str:
    t = (s or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 3)].rstrip() + "..."


def _load_cluster_keywords(path: Path | None) -> dict[int, list[str]]:
    if path is None or not path.is_file():
        return {}
    out: dict[int, list[str]] = {}
    for obj in _iter_jsonl(path):
        cid = obj.get("cluster_id")
        kws = obj.get("keywords")
        if not isinstance(cid, int):
            continue
        if not isinstance(kws, list):
            continue
        out[int(cid)] = [str(x).strip() for x in kws if str(x).strip()]
    return out


def _load_cluster_labels(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for obj in _iter_jsonl(path):
        if obj.get("error") is not None:
            continue
        cid = obj.get("cluster_id")
        if not isinstance(cid, int):
            continue
        out[int(cid)] = obj
    return out


def _load_sections_title_map(path: Path | None) -> dict[str, str]:
    """
    Map chunk_id -> section title when available.
    """
    if path is None or not path.is_file():
        return {}
    out: dict[str, str] = {}
    for obj in _iter_jsonl(path):
        chunk_id = obj.get("chunk_id")
        title = obj.get("title")
        if isinstance(chunk_id, str) and chunk_id.strip() and isinstance(title, str) and title.strip():
            out[chunk_id.strip()] = title.strip()
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build an inspection report for clause-level instrument clusters.")
    ap.add_argument("--clauses-jsonl", required=True, help="Path to clauses.jsonl")
    ap.add_argument("--clusters-jsonl", required=True, help="Path to clusters.jsonl (per-clause cluster assignment)")
    ap.add_argument("--output-dir", required=True, help="Directory to write report + packets")

    ap.add_argument("--cluster-keywords-jsonl", default="", help="Optional cluster_keywords.jsonl (local TF-IDF run).")
    ap.add_argument("--cluster-labels-jsonl", default="", help="Optional cluster_labels.jsonl (LLM instrument labels).")
    ap.add_argument("--sections-jsonl", default="", help="Optional sections.jsonl (chunk_id -> title).")

    ap.add_argument("--top-clauses", type=int, default=12, help="How many example clauses to include per cluster.")
    ap.add_argument("--max-clause-text", type=int, default=260, help="Max characters of clause_text to show.")
    ap.add_argument("--max-requirement-text", type=int, default=320, help="Max characters of requirement to show.")
    ap.add_argument("--include-noise", action="store_true", help="Include cluster_id=-1 (noise) in outputs.")
    ap.add_argument("--min-prob", type=float, default=0.0, help="Drop assignments with cluster_prob < this value.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    packets_dir = out_dir / "cluster_packets"
    packets_dir.mkdir(parents=True, exist_ok=True)

    clauses_path = Path(expanduser(args.clauses_jsonl)).resolve()
    clusters_path = Path(expanduser(args.clusters_jsonl)).resolve()

    kw_path = Path(expanduser(args.cluster_keywords_jsonl)).resolve() if str(args.cluster_keywords_jsonl).strip() else None
    labels_path = Path(expanduser(args.cluster_labels_jsonl)).resolve() if str(args.cluster_labels_jsonl).strip() else None
    sections_path = Path(expanduser(args.sections_jsonl)).resolve() if str(args.sections_jsonl).strip() else None

    # Load clauses keyed by clause_id for joining.
    clauses_by_id: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(clauses_path):
        cid = row.get("clause_id")
        if isinstance(cid, str) and cid.strip():
            clauses_by_id[cid.strip()] = row

    if not clauses_by_id:
        raise SystemExit(f"No clause rows loaded from {clauses_path}")

    keywords_by_cluster = _load_cluster_keywords(kw_path)
    labels_by_cluster = _load_cluster_labels(labels_path)
    section_title_by_chunk = _load_sections_title_map(sections_path)

    # Group clause assignments by cluster_id.
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    min_prob = float(args.min_prob)
    if min_prob < 0.0 or min_prob > 1.0:
        raise SystemExit("--min-prob must be in [0,1]")

    missing_clause_rows = 0
    for asn in _iter_jsonl(clusters_path):
        clause_id = asn.get("clause_id")
        if not isinstance(clause_id, str) or not clause_id.strip():
            continue
        clause_id = clause_id.strip()

        cl_row = clauses_by_id.get(clause_id)
        if cl_row is None:
            missing_clause_rows += 1
            continue

        cluster_id = asn.get("cluster_id")
        if not isinstance(cluster_id, int):
            continue
        if int(cluster_id) == -1 and not bool(args.include_noise):
            continue

        prob = asn.get("cluster_prob", 1.0)
        try:
            prob_f = float(prob)
        except Exception:
            prob_f = 1.0
        if prob_f < min_prob:
            continue

        # Merge fields needed for reporting.
        merged = {
            **{k: v for k, v in asn.items()},
            **{k: v for k, v in cl_row.items()},
            "cluster_prob": prob_f,
        }
        grouped[int(cluster_id)].append(merged)

    if not grouped:
        raise SystemExit("No cluster assignments after filtering (check inputs / --include-noise / --min-prob).")

    # Sort clusters by size desc, then id.
    clusters_sorted = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    stamp = datetime.now(timezone.utc).isoformat()
    report_path = out_dir / "report.md"
    index_path = out_dir / "clusters_index.csv"

    # Write index CSV.
    with index_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cluster_id",
                "n_clauses",
                "n_docs",
                "instrument_name",
                "top_motives",
                "top_modalities",
                "keywords",
            ]
        )
        for cluster_id, rows in clusters_sorted:
            doc_ids = {str(r.get("doc_id") or "").strip() for r in rows if str(r.get("doc_id") or "").strip()}
            motives = Counter(str(r.get("motive") or "").strip() for r in rows if str(r.get("motive") or "").strip())
            mods = Counter(str(r.get("modality") or "").strip() for r in rows if str(r.get("modality") or "").strip())
            inst = labels_by_cluster.get(cluster_id, {}).get("instrument_name") or f"cluster_{cluster_id}"
            kw = ", ".join(keywords_by_cluster.get(cluster_id, [])[:12])
            w.writerow(
                [
                    cluster_id,
                    len(rows),
                    len(doc_ids),
                    str(inst),
                    "; ".join([f"{k}:{v}" for k, v in motives.most_common(5)]),
                    "; ".join([f"{k}:{v}" for k, v in mods.most_common(5)]),
                    kw,
                ]
            )

    # Write markdown report + per-cluster packets.
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Regulatory instrument cluster report\n\n")
        f.write(f"Generated: {stamp}\n\n")
        f.write("## Inputs\n\n")
        f.write(f"- clauses: `{clauses_path}`\n")
        f.write(f"- clusters: `{clusters_path}`\n")
        if kw_path:
            f.write(f"- cluster_keywords: `{kw_path}`\n")
        if labels_path:
            f.write(f"- cluster_labels: `{labels_path}`\n")
        if sections_path:
            f.write(f"- sections: `{sections_path}`\n")
        if missing_clause_rows:
            f.write(f"\nNote: {missing_clause_rows} cluster assignments had missing clause_id rows.\n")
        f.write("\n")

        n_total = sum(len(v) for v in grouped.values())
        f.write("## Summary\n\n")
        f.write(f"- clusters: {len(grouped)}\n")
        f.write(f"- clause assignments (after filters): {n_total}\n")
        f.write(f"- include_noise: {bool(args.include_noise)}\n")
        f.write(f"- min_prob: {min_prob}\n")
        f.write("\n")

        f.write("## Cluster index\n\n")
        f.write(f"CSV: `{index_path}`\n\n")

        top_n = int(args.top_clauses)
        if top_n <= 0:
            raise SystemExit("--top-clauses must be > 0")
        max_clause_text = int(args.max_clause_text)
        max_req_text = int(args.max_requirement_text)
        if max_clause_text <= 0 or max_req_text <= 0:
            raise SystemExit("--max-clause-text and --max-requirement-text must be > 0")

        for cluster_id, rows in clusters_sorted:
            label = labels_by_cluster.get(cluster_id) or {}
            inst_name = str(label.get("instrument_name") or f"cluster_{cluster_id}")
            inst_desc = str(label.get("instrument_description") or "").strip()
            kw = keywords_by_cluster.get(cluster_id) or []

            motives = Counter(str(r.get("motive") or "").strip() for r in rows if str(r.get("motive") or "").strip())
            mods = Counter(str(r.get("modality") or "").strip() for r in rows if str(r.get("modality") or "").strip())
            doc_ids = {str(r.get("doc_id") or "").strip() for r in rows if str(r.get("doc_id") or "").strip()}

            f.write(f"## Cluster {cluster_id}: {inst_name}\n\n")
            f.write(f"- size: {len(rows)} clauses across {len(doc_ids)} docs\n")
            if kw:
                f.write(f"- keywords: {', '.join(kw)}\n")
            if motives:
                f.write(f"- motives: {', '.join([f'{k}({v})' for k, v in motives.most_common(6)])}\n")
            if mods:
                f.write(f"- modalities: {', '.join([f'{k}({v})' for k, v in mods.most_common(6)])}\n")
            if inst_desc:
                f.write("\n**Instrument description**\n\n")
                f.write(inst_desc + "\n\n")

            # Representative examples: prefer highest-probability within cluster.
            rows_sorted = sorted(rows, key=lambda r: float(r.get("cluster_prob") or 0.0), reverse=True)
            examples = rows_sorted[:top_n]

            packet_name = f"cluster_{cluster_id:03d}__{_safe_slug(inst_name)}.md"
            packet_path = packets_dir / packet_name

            f.write(f"Packet: `{packet_path}`\n\n")

            # Write packet.
            with packet_path.open("w", encoding="utf-8") as pf:
                pf.write(f"# Cluster {cluster_id}: {inst_name}\n\n")
                pf.write(f"Generated: {stamp}\n\n")
                pf.write(f"- size: {len(rows)} clauses\n")
                pf.write(f"- docs: {len(doc_ids)}\n")
                if kw:
                    pf.write(f"- keywords: {', '.join(kw)}\n")
                if motives:
                    pf.write(f"- motives: {', '.join([f'{k}({v})' for k, v in motives.most_common(10)])}\n")
                if mods:
                    pf.write(f"- modalities: {', '.join([f'{k}({v})' for k, v in mods.most_common(10)])}\n")
                if inst_desc:
                    pf.write("\n## Instrument description\n\n")
                    pf.write(inst_desc + "\n\n")

                pf.write("## Example clauses\n\n")
                for ex in examples:
                    chunk_id = str(ex.get("chunk_id") or "").strip()
                    sec_title = section_title_by_chunk.get(chunk_id, "") if chunk_id else ""
                    meta_line = " | ".join(
                        part
                        for part in [
                            f"prob={float(ex.get('cluster_prob') or 0.0):.3f}",
                            f"motive={str(ex.get('motive') or '').strip()}",
                            f"modality={str(ex.get('modality') or '').strip()}",
                            f"doc_id={str(ex.get('doc_id') or '').strip()}",
                            f"issue_date={str(ex.get('issue_date') or '').strip()}",
                            f"chunk_id={chunk_id}" if chunk_id else "",
                            f"section={sec_title}" if sec_title else "",
                        ]
                        if part
                    )
                    pf.write(f"- {meta_line}\n\n")
                    req = _preview(str(ex.get('requirement') or ''), max_chars=max_req_text)
                    quote = _preview(str(ex.get('clause_text') or ''), max_chars=max_clause_text)
                    pf.write(f"  requirement: {req}\n\n")
                    pf.write(f"  quote: {quote}\n\n")

    print(f"Done. wrote={out_dir} clusters={len(grouped)} packets={len(clusters_sorted)}")


if __name__ == "__main__":
    main()

