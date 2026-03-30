#!/usr/bin/env python3
"""
Rehydrate OpenAI Batch embedding outputs into CHUNK-level embeddings.

Why this exists
  - export_issue_topic_embedding_batch_requests.py emits one embeddings request per
    issue-chunk (custom_id = "<issue_id>::chunkNNN").
  - For multi-topic discovery, we want to cluster *chunks* (tweet-topics style),
    not just a single averaged vector per issue.

Inputs
  - request_dir:
      - mapping_shardNNN.jsonl (from export_issue_topic_embedding_batch_requests.py)
      - submitted_batches.jsonl (optional; provenance only)
  - results_dir:
      - openai_results_shardNNN.jsonl (from download_openai_batch_results.py)

Outputs (output_dir)
  - chunk_ids.txt: chunk_id (custom_id) per row (aligned with the embeddings matrix rows)
  - chunk_embeddings.npy: float32 matrix [n_chunks, dim]
  - chunk_metadata.jsonl: one line per chunk_id with issue/date/label and chunk stats

This script does not call any APIs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChunkMapping:
    custom_id: str
    issue_id: str
    issue_date: str
    slug: str
    classification_label: str
    chunk_index: int
    chunk_count: int
    chunk_chars: int
    chunk_sha256: str
    issue_zoning_output_path: str
    embedding_model: str


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _parse_shard_index(name: str, *, prefix: str, suffix: str) -> int:
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise ValueError(f"Unexpected filename: {name}")
    mid = name[len(prefix) : -len(suffix)]
    if not mid.isdigit():
        raise ValueError(f"Unexpected shard token in filename: {name}")
    return int(mid)


def _discover_shards(request_dir: Path, results_dir: Path) -> list[int]:
    mapping = {}
    results = {}
    for p in request_dir.glob("mapping_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="mapping_shard", suffix=".jsonl")
        mapping[s] = p
    for p in results_dir.glob("openai_results_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="openai_results_shard", suffix=".jsonl")
        results[s] = p

    if not mapping:
        raise SystemExit(f"No mapping_shard*.jsonl found in {request_dir}")
    if not results:
        raise SystemExit(f"No openai_results_shard*.jsonl found in {results_dir}")

    missing = sorted(set(mapping) - set(results))
    if missing:
        raise SystemExit(f"Missing results shards for mapping shards: {missing} (results_dir={results_dir})")

    return sorted(mapping)


def _count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                n += 1
    return n


def _load_mapping(path: Path) -> list[ChunkMapping]:
    out: list[ChunkMapping] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                ChunkMapping(
                    custom_id=str(obj.get("custom_id") or obj.get("id") or ""),
                    issue_id=str(obj.get("issue_id") or ""),
                    issue_date=str(obj.get("issue_date") or ""),
                    slug=str(obj.get("slug") or ""),
                    classification_label=str(obj.get("classification_label") or ""),
                    chunk_index=int(obj.get("chunk_index") or 0),
                    chunk_count=int(obj.get("chunk_count") or 0),
                    chunk_chars=int(obj.get("chunk_chars") or 0),
                    chunk_sha256=str(obj.get("chunk_sha256") or ""),
                    issue_zoning_output_path=str(obj.get("issue_zoning_output_path") or ""),
                    embedding_model=str(obj.get("embedding_model") or ""),
                )
            )

    bad = [
        m
        for m in out
        if (
            not m.custom_id
            or not m.issue_id
            or not m.issue_date
            or not m.slug
            or not m.issue_zoning_output_path
            or not m.embedding_model
        )
    ]
    if bad:
        raise SystemExit(f"Malformed mapping rows in {path}: {len(bad)} rows missing required fields")
    return out


def _extract_embedding_from_response_body(body: dict[str, Any]) -> list[float] | None:
    # Embeddings endpoint response: {"data":[{"embedding":[...]}], ...}
    data = body.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            emb = first.get("embedding")
            if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
                return [float(x) for x in emb]
    return None


_BATCH_REQ_ID_RE = re.compile(r"^batch_req_", re.IGNORECASE)


def _load_results_embeddings(path: Path) -> dict[str, list[float]]:
    """Return custom_id -> embedding vector for a results shard."""
    out: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id.strip():
                continue

            if obj.get("error") is not None:
                raise SystemExit(f"Per-request error in results {path} for custom_id={custom_id}: {obj.get('error')}")

            resp = obj.get("response") or {}
            if not isinstance(resp, dict):
                raise SystemExit(f"Malformed response wrapper in {path} for custom_id={custom_id}")
            body = resp.get("body") or {}
            if not isinstance(body, dict):
                raise SystemExit(f"Missing response.body in {path} for custom_id={custom_id}")

            emb = _extract_embedding_from_response_body(body)
            if emb is None:
                rid = obj.get("id")
                if isinstance(rid, str) and _BATCH_REQ_ID_RE.match(rid):
                    raise SystemExit(f"Could not extract embedding vector from response.body for custom_id={custom_id}")
                raise SystemExit(f"Could not extract embedding vector from results row for custom_id={custom_id}")

            out[custom_id] = emb
    if not out:
        raise SystemExit(f"No embeddings found in results shard: {path}")
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rehydrate OpenAI batch embeddings into chunk-level vectors.")
    ap.add_argument("--request-dir", required=True, help="Directory containing mapping_shardNNN.jsonl")
    ap.add_argument("--results-dir", required=True, help="Directory containing openai_results_shardNNN.jsonl")
    ap.add_argument("--output-dir", required=True, help="Directory to write chunk_embeddings.npy + metadata")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir is not a directory: {results_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    shard_indices = _discover_shards(request_dir, results_dir)

    # Count chunks first so we can allocate a single dense matrix.
    n_chunks = 0
    for shard in shard_indices:
        mapping_path = request_dir / f"mapping_shard{shard:03d}.jsonl"
        n_chunks += _count_nonempty_lines(mapping_path)
    if n_chunks <= 0:
        raise SystemExit(f"No chunks found in mapping files under {request_dir}")

    chunk_ids_path = out_dir / "chunk_ids.txt"
    meta_path = out_dir / "chunk_metadata.jsonl"

    row_idx = 0
    embedding_dim: int | None = None
    mat: np.ndarray | None = None

    with chunk_ids_path.open("w", encoding="utf-8") as ids_f, meta_path.open("w", encoding="utf-8") as meta_f:
        for shard in shard_indices:
            mapping_path = request_dir / f"mapping_shard{shard:03d}.jsonl"
            results_path = results_dir / f"openai_results_shard{shard:03d}.jsonl"

            mappings = _load_mapping(mapping_path)
            emb_by_custom_id = _load_results_embeddings(results_path)

            for m in mappings:
                emb = emb_by_custom_id.get(m.custom_id)
                if emb is None:
                    raise SystemExit(f"Missing embedding result for custom_id={m.custom_id} (shard {shard:03d})")

                if embedding_dim is None:
                    embedding_dim = len(emb)
                    if embedding_dim <= 0:
                        raise SystemExit(f"Invalid embedding dimension for custom_id={m.custom_id}: {embedding_dim}")
                    mat = np.zeros((n_chunks, embedding_dim), dtype=np.float32)
                elif len(emb) != embedding_dim:
                    raise SystemExit(
                        f"Inconsistent embedding dimensions: expected {embedding_dim} got {len(emb)} for custom_id={m.custom_id}"
                    )

                assert mat is not None  # for mypy; allocated on first row
                if row_idx >= n_chunks:
                    raise SystemExit(f"Internal error: row_idx={row_idx} exceeds allocated n_chunks={n_chunks}")

                mat[row_idx, :] = np.asarray(emb, dtype=np.float32)
                ids_f.write(m.custom_id + "\n")

                doc_weight = 1.0 / float(m.chunk_count) if int(m.chunk_count) > 0 else 1.0
                meta_f.write(
                    json.dumps(
                        {
                            "chunk_id": m.custom_id,
                            "issue_id": m.issue_id,
                            "issue_date": m.issue_date,
                            "slug": m.slug,
                            "classification_label": m.classification_label,
                            "issue_zoning_output_path": m.issue_zoning_output_path,
                            "chunk_index": int(m.chunk_index),
                            "chunk_count": int(m.chunk_count),
                            "chunk_chars": int(m.chunk_chars),
                            "chunk_sha256": m.chunk_sha256,
                            "embedding_model": m.embedding_model,
                            "doc_weight": float(doc_weight),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                row_idx += 1

    if embedding_dim is None or mat is None:
        raise SystemExit("No embeddings were processed (unexpected)")
    if row_idx != n_chunks:
        raise SystemExit(f"Row count mismatch: expected n_chunks={n_chunks} wrote={row_idx}")

    np.save(out_dir / "chunk_embeddings.npy", mat)
    (out_dir / "rehydrate_summary.json").write_text(
        json.dumps(
            {
                "n_chunks": int(n_chunks),
                "embedding_dim": int(embedding_dim),
                "request_dir": str(request_dir),
                "results_dir": str(results_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _eprint(f"Done. chunks={n_chunks} dim={embedding_dim} wrote={out_dir}")


if __name__ == "__main__":
    main()

