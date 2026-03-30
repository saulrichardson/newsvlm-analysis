#!/usr/bin/env python3
"""
Export OpenAI Batch request shards for embeddings over extracted regulatory clauses.

This enables the PI-requested final step:
  - clause-level embedding
  - clustering / instrument discovery

Input:
  - clauses.jsonl from scripts/rehydrate_regulatory_clause_extraction_openai_batch_results.py

Outputs (output_dir):
  - openai_requests_shardNNN.jsonl  (POST /v1/embeddings; custom_id = clause_id)
  - mapping_shardNNN.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from os.path import expanduser
from pathlib import Path
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export OpenAI Batch request shards for clause embeddings.")
    ap.add_argument("--clauses-jsonl", required=True, help="Path to clauses.jsonl")
    ap.add_argument("--output-dir", required=True, help="Directory to write request shards into.")
    ap.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model name.")
    ap.add_argument(
        "--text-field",
        default="requirement",
        help="Which clause field to embed (default: requirement).",
    )
    ap.add_argument("--requests-per-shard", type=int, default=20000, help="How many requests per shard.")
    ap.add_argument("--max-bytes-per-shard", type=int, default=180_000_000, help="Shard byte cap (0 disables).")
    ap.add_argument("--max-clauses", type=int, default=0, help="Optional: export only first N clauses (0=all).")
    ap.add_argument("--max-text-chars", type=int, default=0, help="Optional: truncate embedded text (0 disables).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    in_path = Path(expanduser(args.clauses_jsonl)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    text_field = str(args.text_field).strip()
    if not text_field:
        raise SystemExit("--text-field must be non-empty")

    req_limit = int(args.requests_per_shard)
    if req_limit <= 0:
        raise SystemExit("--requests-per-shard must be > 0")
    max_bytes = int(args.max_bytes_per_shard)
    if max_bytes < 0:
        raise SystemExit("--max-bytes-per-shard must be >= 0")
    max_clauses = int(args.max_clauses)
    if max_clauses < 0:
        raise SystemExit("--max-clauses must be >= 0")
    max_text_chars = int(args.max_text_chars)
    if max_text_chars < 0:
        raise SystemExit("--max-text-chars must be >= 0")

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

    clauses_written = 0
    for obj in _iter_jsonl(in_path):
        if max_clauses and clauses_written >= max_clauses:
            break
        clause_id = obj.get("clause_id")
        if not isinstance(clause_id, str) or not clause_id.strip():
            continue
        clause_id = clause_id.strip()
        txt = obj.get(text_field)
        if not isinstance(txt, str) or not txt.strip():
            continue
        t = txt.strip()
        if max_text_chars > 0 and len(t) > max_text_chars:
            t = t[:max_text_chars].rstrip()

        body = {"model": str(args.embedding_model), "input": t}
        req_line = {"custom_id": clause_id, "method": "POST", "url": "/v1/embeddings", "body": body}
        req_s = json.dumps(req_line, ensure_ascii=False) + "\n"

        rotate_if_needed(req_s)

        mapping = {
            "custom_id": clause_id,
            "clause_id": clause_id,
            "embedding_model": str(args.embedding_model),
            "text_field": text_field,
            "text_chars": int(len(t)),
            "text_sha256": _sha256(t),
            "exported_at": time.time(),
        }
        # Carry provenance from clause row, excluding large text fields.
        for k, v in obj.items():
            if k in {"clause_text", "requirement"}:
                continue
            if k not in mapping:
                mapping[k] = v

        openai_f.write(req_s)
        mapping_f.write(json.dumps(mapping, ensure_ascii=False) + "\n")
        requests_in_shard += 1
        bytes_in_shard += len(req_s.encode("utf-8"))
        clauses_written += 1

    openai_f.close()
    mapping_f.close()

    print(f"Done. clauses_written={clauses_written} shards={shard_idx + 1} out={out_dir}")


if __name__ == "__main__":
    main()

