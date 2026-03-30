#!/usr/bin/env python3
"""
Rehydrate OpenAI Batch embedding outputs into clause-level embeddings.

Inputs
  - request_dir:
      - mapping_shardNNN.jsonl (from scripts/export_clause_embedding_batch_requests.py)
  - results_dir:
      - openai_results_shardNNN.jsonl (from download_openai_batch_results.py)

Outputs (output_dir)
  - clause_ids.txt
  - clause_embeddings.npy
  - clause_metadata.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClauseMapping:
    custom_id: str
    embedding_model: str
    meta: dict[str, Any]


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


def _load_mapping(path: Path) -> list[ClauseMapping]:
    out: list[ClauseMapping] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id") or obj.get("id")
            model = obj.get("embedding_model") or ""
            if not isinstance(cid, str) or not cid.strip():
                raise SystemExit(f"Malformed mapping row in {path}: missing custom_id")
            if not isinstance(model, str) or not model.strip():
                raise SystemExit(f"Malformed mapping row in {path}: missing embedding_model (custom_id={cid})")
            out.append(ClauseMapping(custom_id=cid.strip(), embedding_model=model.strip(), meta=obj))
    if not out:
        raise SystemExit(f"No mapping rows found in {path}")
    return out


def _extract_embedding_from_response_body(body: dict[str, Any]) -> list[float] | None:
    data = body.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            emb = first.get("embedding")
            if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
                return [float(x) for x in emb]
    return None


def _load_results_embeddings(path: Path) -> dict[str, list[float]]:
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
                raise SystemExit(f"Could not extract embedding vector from results row for custom_id={custom_id}")
            out[custom_id.strip()] = emb
    if not out:
        raise SystemExit(f"No embeddings found in results shard: {path}")
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rehydrate OpenAI batch embeddings into clause-level vectors.")
    ap.add_argument("--request-dir", required=True, help="Directory containing mapping_shardNNN.jsonl")
    ap.add_argument("--results-dir", required=True, help="Directory containing openai_results_shardNNN.jsonl")
    ap.add_argument("--output-dir", required=True, help="Directory to write clause_embeddings.npy + metadata")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_indices = _discover_shards(request_dir, results_dir)

    # Count rows for allocation.
    n_rows = 0
    for shard in shard_indices:
        n_rows += _count_nonempty_lines(request_dir / f"mapping_shard{shard:03d}.jsonl")
    if n_rows <= 0:
        raise SystemExit(f"No rows found in mapping files under {request_dir}")

    ids_path = out_dir / "clause_ids.txt"
    meta_path = out_dir / "clause_metadata.jsonl"

    row_idx = 0
    emb_dim: int | None = None
    mat: np.ndarray | None = None

    with ids_path.open("w", encoding="utf-8") as ids_f, meta_path.open("w", encoding="utf-8") as meta_f:
        for shard in shard_indices:
            mapping_path = request_dir / f"mapping_shard{shard:03d}.jsonl"
            results_path = results_dir / f"openai_results_shard{shard:03d}.jsonl"

            mappings = _load_mapping(mapping_path)
            emb_by_id = _load_results_embeddings(results_path)

            for m in mappings:
                emb = emb_by_id.get(m.custom_id)
                if emb is None:
                    raise SystemExit(f"Missing embedding result for custom_id={m.custom_id} (shard {shard:03d})")
                if emb_dim is None:
                    emb_dim = len(emb)
                    mat = np.zeros((n_rows, emb_dim), dtype=np.float32)
                if mat is None or emb_dim is None:
                    raise SystemExit("Internal error: embedding matrix not allocated")
                if len(emb) != emb_dim:
                    raise SystemExit(f"Inconsistent embedding dim for {m.custom_id}: got {len(emb)} expected {emb_dim}")

                mat[row_idx] = np.asarray(emb, dtype=np.float32)
                ids_f.write(m.custom_id + "\n")
                meta_f.write(json.dumps(m.meta, ensure_ascii=False) + "\n")
                row_idx += 1

    if mat is None or emb_dim is None:
        raise SystemExit("No embeddings written (unexpected).")
    if row_idx != n_rows:
        raise SystemExit(f"Row mismatch: expected {n_rows} wrote {row_idx}")

    np.save(out_dir / "clause_embeddings.npy", mat)
    print(f"Done. wrote={out_dir} rows={n_rows} dim={emb_dim}")


if __name__ == "__main__":
    main()

