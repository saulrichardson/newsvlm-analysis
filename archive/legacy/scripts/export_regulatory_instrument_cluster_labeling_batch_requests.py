#!/usr/bin/env python3
"""
Export OpenAI Batch request shards for labeling clause clusters into regulatory instruments.

Inputs:
  - clusters.jsonl (from scripts/cluster_regulatory_clauses_local.py or scripts/cluster_clause_embeddings.py)
  - clauses.jsonl  (from scripts/rehydrate_regulatory_clause_extraction_openai_batch_results.py)
  - optional cluster_keywords.jsonl (from the local clustering script)

Output (output_dir):
  - openai_requests_shardNNN.jsonl  (POST /v1/responses)
  - mapping_shardNNN.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from os.path import expanduser
from pathlib import Path
from typing import Any

from newsvlm.regulatory_motive_schema import instrument_cluster_label_openai_text_config


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_prompt_text(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"Prompt file not found: {path}")
    t = path.read_text(encoding="utf-8")
    if not t.strip():
        raise SystemExit(f"Prompt file is empty: {path}")
    return t.strip()


def _iter_jsonl(path: Path) -> Any:
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


def _load_cluster_keywords(path: Path) -> dict[int, list[str]]:
    if not path.is_file():
        return {}
    out: dict[int, list[str]] = {}
    for obj in _iter_jsonl(path):
        cid = obj.get("cluster_id")
        kws = obj.get("keywords")
        if isinstance(cid, int) and isinstance(kws, list):
            out[int(cid)] = [str(x).strip() for x in kws if str(x).strip()]
    return out


def _build_prompt(
    *,
    base_prompt: str,
    cluster_id: int,
    cluster_keywords: list[str] | None,
    examples: list[dict[str, Any]],
) -> str:
    lines: list[str] = [base_prompt.strip(), "", f"CLUSTER_ID: {int(cluster_id)}"]
    if cluster_keywords:
        lines.append("CLUSTER_KEYWORDS (auto): " + ", ".join(cluster_keywords[:18]))
    lines.append("")
    lines.append("Representative clauses:")
    for i, ex in enumerate(examples, start=1):
        lines.append("")
        lines.append(
            f"[CLAUSE {i}] clause_id={ex.get('clause_id')} motive={ex.get('motive')} modality={ex.get('modality')} instrument_hint={ex.get('instrument_hint')}"
        )
        lines.append("requirement: " + str(ex.get("requirement") or "").strip())
        lines.append("clause_text: " + str(ex.get("clause_text") or "").strip())
    lines.append("")
    lines.append("Return the JSON now.")
    return "\n".join(lines).strip()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export OpenAI Batch request shards for clause-cluster labeling.")
    ap.add_argument("--clusters-jsonl", required=True, help="Path to clusters.jsonl (clause_id -> cluster_id)")
    ap.add_argument("--clauses-jsonl", required=True, help="Path to clauses.jsonl (clause_id -> text)")
    ap.add_argument("--output-dir", required=True, help="Directory to write request shards into.")
    ap.add_argument(
        "--cluster-keywords-jsonl",
        default="",
        help="Optional: cluster_keywords.jsonl from local TF-IDF clustering to include extra signal.",
    )
    ap.add_argument(
        "--prompt-path",
        default="prompts/regulatory_instrument_cluster_labeler_prompt_text.txt",
        help="Prompt text file.",
    )
    ap.add_argument("--openai-model", default="gpt-5-nano", help="OpenAI model name (no provider prefix).")
    ap.add_argument("--max-examples-per-cluster", type=int, default=10, help="How many clause examples to show per cluster.")
    ap.add_argument("--min-cluster-size", type=int, default=3, help="Only label clusters with at least this many clauses.")
    ap.add_argument("--max-clusters", type=int, default=0, help="Optional cap on number of clusters to label (0=all).")
    ap.add_argument("--requests-per-shard", type=int, default=5000)
    ap.add_argument("--max-bytes-per-shard", type=int, default=180_000_000)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    clusters_path = Path(expanduser(args.clusters_jsonl)).resolve()
    clauses_path = Path(expanduser(args.clauses_jsonl)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(expanduser(args.prompt_path)).resolve()
    prompt_text = _load_prompt_text(prompt_path)
    prompt_sha = _sha256(prompt_text)

    cluster_kw_path = Path(expanduser(args.cluster_keywords_jsonl)).resolve() if str(args.cluster_keywords_jsonl).strip() else None
    keywords_by_cluster = _load_cluster_keywords(cluster_kw_path) if cluster_kw_path is not None else {}

    max_examples = int(args.max_examples_per_cluster)
    if max_examples <= 0:
        raise SystemExit("--max-examples-per-cluster must be > 0")
    min_cluster_size = int(args.min_cluster_size)
    if min_cluster_size <= 0:
        raise SystemExit("--min-cluster-size must be > 0")
    max_clusters = int(args.max_clusters)
    if max_clusters < 0:
        raise SystemExit("--max-clusters must be >= 0")

    req_limit = int(args.requests_per_shard)
    if req_limit <= 0:
        raise SystemExit("--requests-per-shard must be > 0")
    max_bytes = int(args.max_bytes_per_shard)
    if max_bytes < 0:
        raise SystemExit("--max-bytes-per-shard must be >= 0")

    # clause_id -> metadata (keep only what we need for the prompt)
    clause_by_id: dict[str, dict[str, Any]] = {}
    for obj in _iter_jsonl(clauses_path):
        cid = obj.get("clause_id")
        if isinstance(cid, str) and cid.strip():
            clause_by_id[cid.strip()] = {
                "clause_id": cid.strip(),
                "requirement": obj.get("requirement"),
                "clause_text": obj.get("clause_text"),
                "motive": obj.get("motive"),
                "modality": obj.get("modality"),
                "instrument_hint": obj.get("instrument_hint"),
                "doc_id": obj.get("doc_id"),
                "chunk_id": obj.get("chunk_id"),
            }
    if not clause_by_id:
        raise SystemExit(f"No clauses loaded from {clauses_path}")

    # cluster_id -> clause rows (clause_id + prob)
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for obj in _iter_jsonl(clusters_path):
        cid = obj.get("clause_id")
        lab = obj.get("cluster_id")
        prob = obj.get("cluster_prob")
        if not isinstance(cid, str) or not cid.strip():
            continue
        if not isinstance(lab, int):
            continue
        if int(lab) == -1:
            continue
        p = float(prob) if isinstance(prob, (int, float)) else 0.0
        by_cluster[int(lab)].append({"clause_id": cid.strip(), "cluster_prob": p})
    if not by_cluster:
        raise SystemExit(f"No non-noise clusters found in {clusters_path}")

    # Pick clusters to label.
    cluster_ids = [cid for cid, rows in sorted(by_cluster.items()) if len(rows) >= min_cluster_size]
    if not cluster_ids:
        raise SystemExit(f"No clusters meet --min-cluster-size={min_cluster_size}")
    if max_clusters > 0:
        cluster_ids = cluster_ids[:max_clusters]

    text_cfg = instrument_cluster_label_openai_text_config()

    shard_idx = 0
    requests_in_shard = 0
    bytes_in_shard = 0
    openai_path = out_dir / f"openai_requests_shard{shard_idx:03d}.jsonl"
    mapping_path = out_dir / f"mapping_shard{shard_idx:03d}.jsonl"
    openai_f = openai_path.open("w", encoding="utf-8")
    mapping_f = mapping_path.open("w", encoding="utf-8")

    def rotate_if_needed(next_line: str) -> None:
        nonlocal shard_idx, requests_in_shard, bytes_in_shard, openai_f, mapping_f, openai_path, mapping_path
        if requests_in_shard <= 0:
            return
        if requests_in_shard < req_limit:
            if max_bytes <= 0:
                return
            if bytes_in_shard + len(next_line.encode("utf-8")) <= max_bytes:
                return
        openai_f.close()
        mapping_f.close()
        shard_idx += 1
        requests_in_shard = 0
        bytes_in_shard = 0
        openai_path = out_dir / f"openai_requests_shard{shard_idx:03d}.jsonl"
        mapping_path = out_dir / f"mapping_shard{shard_idx:03d}.jsonl"
        openai_f = openai_path.open("w", encoding="utf-8")
        mapping_f = mapping_path.open("w", encoding="utf-8")

    clusters_written = 0
    for cid in cluster_ids:
        rows = by_cluster[cid]
        rows = sorted(rows, key=lambda r: (-float(r.get("cluster_prob") or 0.0), str(r.get("clause_id"))))

        # Diversify examples by doc_id when possible.
        chosen: list[dict[str, Any]] = []
        seen_docs: set[str] = set()
        for r in rows:
            if len(chosen) >= max_examples:
                break
            clause_id = str(r["clause_id"])
            meta = clause_by_id.get(clause_id)
            if meta is None:
                continue
            did = str(meta.get("doc_id") or "")
            if did and did in seen_docs and len(seen_docs) < max_examples:
                continue
            if did:
                seen_docs.add(did)
            chosen.append(meta)
        # Fill remaining slots without doc_id constraint.
        if len(chosen) < max_examples:
            chosen_ids = {c["clause_id"] for c in chosen}
            for r in rows:
                if len(chosen) >= max_examples:
                    break
                clause_id = str(r["clause_id"])
                if clause_id in chosen_ids:
                    continue
                meta = clause_by_id.get(clause_id)
                if meta is None:
                    continue
                chosen.append(meta)
                chosen_ids.add(clause_id)

        # Helpful: count instrument_hints in chosen examples.
        hint_counts = Counter(str(c.get("instrument_hint") or "").strip() for c in chosen)
        hint_counts.pop("", None)
        hint_top = [h for h, _ in hint_counts.most_common(8)]

        kw = keywords_by_cluster.get(int(cid))
        prompt = _build_prompt(
            base_prompt=prompt_text,
            cluster_id=int(cid),
            cluster_keywords=kw,
            examples=chosen,
        )
        if hint_top:
            prompt += "\n\nInstrument_hint counts (auto): " + ", ".join(hint_top)
        final_sha = _sha256(prompt)

        custom_id = f"clause_cluster::{int(cid)}"
        body: dict[str, Any] = {
            "model": str(args.openai_model),
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "text": text_cfg,
        }
        openai_line = {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}
        openai_s = json.dumps(openai_line, ensure_ascii=False) + "\n"

        rotate_if_needed(openai_s)

        mapping_line = {
            "id": custom_id,
            "custom_id": custom_id,
            "cluster_id": int(cid),
            "cluster_size_clauses": int(len(rows)),
            "cluster_keywords": kw or [],
            "example_clause_ids": [c.get("clause_id") for c in chosen],
            "prompt_path": str(prompt_path),
            "prompt_text_sha256": prompt_sha,
            "final_prompt_sha256": final_sha,
            "openai_model": str(args.openai_model),
            "exported_at": time.time(),
        }

        openai_f.write(openai_s)
        mapping_f.write(json.dumps(mapping_line, ensure_ascii=False) + "\n")
        requests_in_shard += 1
        bytes_in_shard += len(openai_s.encode("utf-8"))
        clusters_written += 1

    openai_f.close()
    mapping_f.close()

    print(f"Done. clusters_written={clusters_written} shards={shard_idx + 1} out={out_dir}")


if __name__ == "__main__":
    main()
