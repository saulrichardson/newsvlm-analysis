#!/usr/bin/env python3
"""
Extract chunk text from OpenAI embedding request shards into a chunks.jsonl.

Why this exists
---------------
On Greene/VAST, the issue-topic pipeline already produces:
  - requests/openai_requests_shard*.jsonl  (POST /v1/embeddings; body.input contains chunk text)
  - requests/mapping_shard*.jsonl         (chunk metadata)

The hybrid PI workflow (motive labeling → clause extraction) needs chunk text.
Instead of rebuilding issue texts, we can reuse the embedding request shards as the
source-of-truth for chunk text.
"""

from __future__ import annotations

import argparse
import json
import re
from os.path import expanduser
from pathlib import Path
from typing import Any


_CHUNK_ID_RE = re.compile(r"^(?P<doc_id>.+?)::chunk(?P<chunk_idx>\d{3,})$")


def _iter_jsonl(path: Path) -> Any:
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
    ap = argparse.ArgumentParser(description="Extract chunks.jsonl from OpenAI embedding request shards.")
    ap.add_argument("--request-dir", required=True, help="Directory containing openai_requests_shard*.jsonl")
    ap.add_argument(
        "--mapping-dir",
        default="",
        help="Optional: directory containing mapping_shard*.jsonl to enrich metadata (usually same as request-dir).",
    )
    ap.add_argument("--output", required=True, help="Output chunks.jsonl path.")
    ap.add_argument("--max-chunks", type=int, default=0, help="Optional: extract only first N chunks (0=all).")
    ap.add_argument("--max-text-chars", type=int, default=0, help="Optional: truncate text (0 disables).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    req_dir = Path(expanduser(args.request_dir)).resolve()
    out_path = Path(expanduser(args.output)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not req_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {req_dir}")

    max_chunks = int(args.max_chunks)
    if max_chunks < 0:
        raise SystemExit("--max-chunks must be >= 0")
    max_text_chars = int(args.max_text_chars)
    if max_text_chars < 0:
        raise SystemExit("--max-text-chars must be >= 0")

    # Optional mapping enrichment.
    mapping_dir = Path(expanduser(str(args.mapping_dir))).resolve() if str(args.mapping_dir).strip() else None
    mapping_by_id: dict[str, dict[str, Any]] = {}
    if mapping_dir is not None:
        if not mapping_dir.is_dir():
            raise SystemExit(f"--mapping-dir is not a directory: {mapping_dir}")
        for mp in sorted(mapping_dir.glob("mapping_shard*.jsonl")):
            for obj in _iter_jsonl(mp):
                cid = obj.get("custom_id") or obj.get("id")
                if isinstance(cid, str) and cid.strip():
                    mapping_by_id[cid.strip()] = obj

    req_paths = sorted(req_dir.glob("openai_requests_shard*.jsonl"))
    if not req_paths:
        raise SystemExit(f"No openai_requests_shard*.jsonl found in {req_dir}")

    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for rp in req_paths:
            for obj in _iter_jsonl(rp):
                if max_chunks and written >= max_chunks:
                    break

                url = obj.get("url")
                if url != "/v1/embeddings":
                    # Skip non-embedding requests.
                    continue

                custom_id = obj.get("custom_id")
                body = obj.get("body") or {}
                if not isinstance(custom_id, str) or not custom_id.strip():
                    skipped += 1
                    continue
                if not isinstance(body, dict):
                    skipped += 1
                    continue
                text = body.get("input")
                if not isinstance(text, str) or not text.strip():
                    skipped += 1
                    continue

                chunk_id = custom_id.strip()
                m = _CHUNK_ID_RE.match(chunk_id)
                if m:
                    doc_id = m.group("doc_id")
                    chunk_index = int(m.group("chunk_idx"))
                else:
                    doc_id = chunk_id
                    chunk_index = 0

                t = text.strip()
                if max_text_chars > 0 and len(t) > max_text_chars:
                    t = t[:max_text_chars].rstrip()

                row: dict[str, Any] = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": int(chunk_index),
                    "text": t,
                    "text_preview": t[:240],
                    "chunk_chars": int(len(t)),
                }

                meta = mapping_by_id.get(chunk_id)
                if isinstance(meta, dict):
                    # Carry useful fields from the embedding mapping.
                    for k in [
                        "issue_id",
                        "issue_date",
                        "slug",
                        "classification_label",
                        "chunk_count",
                        "chunk_chars",
                        "chunk_sha256",
                        "run_root",
                        "issue_zoning_output_path",
                        "embedding_model",
                    ]:
                        if k in meta and k not in row:
                            row[k] = meta.get(k)

                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            if max_chunks and written >= max_chunks:
                break

    print(f"Done. wrote={out_path} chunks={written} skipped={skipped}")


if __name__ == "__main__":
    main()

