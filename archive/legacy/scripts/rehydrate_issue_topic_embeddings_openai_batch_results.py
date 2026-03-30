#!/usr/bin/env python3
"""
Rehydrate OpenAI Batch embedding outputs into per-issue embeddings.

Inputs
  - request_dir:
      - mapping_shardNNN.jsonl (from export_issue_topic_embedding_batch_requests.py)
      - submitted_batches.jsonl (optional; for provenance only)
  - results_dir:
      - openai_results_shardNNN.jsonl (from download_openai_batch_results.py)

Outputs (output_dir)
  - issue_ids.txt: issue_id per row (matches embedding matrix row order)
  - issue_embeddings.npy: float32 matrix [n_issues, dim]
  - issue_metadata.jsonl: issue-level metadata used for plotting (label/date/source)

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
                    embedding_model=str(obj.get("embedding_model") or ""),
                )
            )

    bad = [m for m in out if (not m.custom_id or not m.issue_id or not m.issue_date or not m.slug)]
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
                # Some OpenAI batch formats include an 'id' field only; but our exporter always sets custom_id.
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
                # If this happens, the result isn't an embeddings response.
                # Fail loudly so we don't silently write garbage.
                rid = obj.get("id")
                if isinstance(rid, str) and _BATCH_REQ_ID_RE.match(rid):
                    raise SystemExit(f"Could not extract embedding vector from response.body for custom_id={custom_id}")
                raise SystemExit(f"Could not extract embedding vector from results row for custom_id={custom_id}")

            out[custom_id] = emb
    if not out:
        raise SystemExit(f"No embeddings found in results shard: {path}")
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rehydrate OpenAI batch embeddings into per-issue vectors.")
    ap.add_argument("--request-dir", required=True, help="Directory containing mapping_shardNNN.jsonl")
    ap.add_argument("--results-dir", required=True, help="Directory containing openai_results_shardNNN.jsonl")
    ap.add_argument("--output-dir", required=True, help="Directory to write issue_embeddings.npy + metadata")
    ap.add_argument(
        "--aggregation",
        choices=["mean", "weighted_mean"],
        default="weighted_mean",
        help="How to aggregate multiple chunk embeddings into a single issue embedding.",
    )
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

    # Incrementally aggregate chunk embeddings by issue_id.
    issue_sum: dict[str, np.ndarray] = {}
    issue_weight: dict[str, float] = {}
    issue_meta: dict[str, dict[str, Any]] = {}
    embedding_dim: int | None = None

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
            elif len(emb) != embedding_dim:
                raise SystemExit(
                    f"Inconsistent embedding dimensions: expected {embedding_dim} got {len(emb)} for custom_id={m.custom_id}"
                )

            w = float(m.chunk_chars if args.aggregation == "weighted_mean" else 1.0)
            if w <= 0:
                w = 1.0
            v = np.asarray(emb, dtype=np.float32)

            acc = issue_sum.get(m.issue_id)
            if acc is None:
                issue_sum[m.issue_id] = v * w
                issue_weight[m.issue_id] = w
            else:
                issue_sum[m.issue_id] = acc + (v * w)
                issue_weight[m.issue_id] = issue_weight[m.issue_id] + w

            meta = issue_meta.get(m.issue_id)
            if meta is None:
                issue_meta[m.issue_id] = {
                    "issue_id": m.issue_id,
                    "issue_date": m.issue_date,
                    "slug": m.slug,
                    "classification_label": m.classification_label,
                    "embedding_model": m.embedding_model,
                    "aggregation": str(args.aggregation),
                    "chunks_seen": 1,
                }
            else:
                meta["chunks_seen"] = int(meta.get("chunks_seen", 0)) + 1

    if embedding_dim is None:
        raise SystemExit("No embeddings were processed (unexpected)")

    # Finalize into a dense matrix with deterministic row order.
    issue_ids = sorted(issue_sum.keys())
    mat = np.zeros((len(issue_ids), embedding_dim), dtype=np.float32)

    for i, issue_id in enumerate(issue_ids):
        w = issue_weight.get(issue_id, 0.0)
        if w <= 0:
            raise SystemExit(f"Non-positive aggregation weight for issue_id={issue_id}")
        mat[i, :] = issue_sum[issue_id] / float(w)

    (out_dir / "issue_ids.txt").write_text("\n".join(issue_ids) + "\n", encoding="utf-8")
    np.save(out_dir / "issue_embeddings.npy", mat)

    meta_path = out_dir / "issue_metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for issue_id in issue_ids:
            f.write(json.dumps(issue_meta[issue_id], ensure_ascii=False) + "\n")

    print(f"Done. issues={len(issue_ids)} dim={embedding_dim} wrote={out_dir}")


if __name__ == "__main__":
    main()

