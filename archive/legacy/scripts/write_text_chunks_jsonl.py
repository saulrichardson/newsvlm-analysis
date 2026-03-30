#!/usr/bin/env python3
"""
Write a chunk-level JSONL from an input JSONL of documents with a text field.

Why this exists
---------------
The repo has multiple pipelines that operate at different "document levels"
(page → issue → issue-chunks → clauses).

This helper creates a standard chunk table that downstream LLM steps can consume:
  - chunk motive labeling
  - clause extraction
  - embeddings/clustering over clauses

It intentionally keeps chunking logic *mechanical* (paragraph-aware + max chars);
semantic segmentation (sections/subsections) is handled separately by LLM prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _chunk_text(*, text: str, chunk_size_chars: int) -> list[str]:
    if chunk_size_chars <= 0:
        t = (text or "").strip()
        return [t] if t else []

    clean = (text or "").strip()
    if not clean:
        return []
    if len(clean) <= chunk_size_chars:
        return [clean]

    # Chunk on paragraph boundaries first, then fall back to hard splits for
    # pathological long paragraphs (OCR boxes can be enormous).
    paras = [p.strip() for p in clean.split("\n\n") if p.strip()]

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append("\n\n".join(buf).strip())
        buf = []
        buf_len = 0

    for para in paras:
        if len(para) > chunk_size_chars:
            flush()
            start = 0
            while start < len(para):
                end = min(len(para), start + chunk_size_chars)
                part = para[start:end].strip()
                if part:
                    chunks.append(part)
                start = end
            continue

        sep = 2 if buf else 0
        if buf_len + sep + len(para) <= chunk_size_chars:
            buf.append(para)
            buf_len += sep + len(para)
        else:
            flush()
            buf.append(para)
            buf_len = len(para)

    flush()
    return [c for c in chunks if c.strip()]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Chunk an input JSONL of documents into a chunk-level JSONL.")
    ap.add_argument("--input", required=True, help="Input JSONL path (one JSON object per line).")
    ap.add_argument("--output", required=True, help="Output JSONL path (one chunk per line).")
    ap.add_argument(
        "--id-field",
        default="doc_id",
        help="Field name in input JSON object that contains the document id (default: doc_id).",
    )
    ap.add_argument(
        "--text-field",
        default="text",
        help="Field name in input JSON object that contains the document text (default: text).",
    )
    ap.add_argument(
        "--chunk-size-chars",
        type=int,
        default=8000,
        help="Max chars per chunk. 0 disables chunking and emits one chunk per doc.",
    )
    ap.add_argument(
        "--copy-other-fields",
        action="store_true",
        help="If set, copy all other input fields (except text-field) into each chunk row.",
    )
    ap.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional: process at most this many docs (0 = all).",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="Drop docs with fewer than this many text chars after strip (default: 1).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.is_file():
        raise SystemExit(f"--input not found: {in_path}")

    id_field = str(args.id_field).strip()
    text_field = str(args.text_field).strip()
    if not id_field:
        raise SystemExit("--id-field must be non-empty")
    if not text_field:
        raise SystemExit("--text-field must be non-empty")

    chunk_size_chars = int(args.chunk_size_chars)
    if chunk_size_chars < 0:
        raise SystemExit("--chunk-size-chars must be >= 0")

    max_docs = int(args.max_docs)
    min_chars = int(args.min_chars)
    if max_docs < 0:
        raise SystemExit("--max-docs must be >= 0")
    if min_chars < 0:
        raise SystemExit("--min-chars must be >= 0")

    docs_seen = 0
    docs_written = 0
    chunks_written = 0
    skipped_missing = 0
    skipped_short = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for raw in f_in:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            docs_seen += 1
            if max_docs and docs_seen > max_docs:
                break

            doc_id = obj.get(id_field)
            if not isinstance(doc_id, str) or not doc_id.strip():
                skipped_missing += 1
                continue
            doc_id = doc_id.strip()

            text = obj.get(text_field)
            if not isinstance(text, str):
                skipped_missing += 1
                continue
            text = text.strip()
            if len(text) < min_chars:
                skipped_short += 1
                continue

            chunks = _chunk_text(text=text, chunk_size_chars=chunk_size_chars)
            if not chunks:
                skipped_short += 1
                continue

            docs_written += 1
            chunk_count = len(chunks)

            # Optional field copy.
            copied: dict[str, Any] = {}
            if bool(args.copy_other_fields):
                for k, v in obj.items():
                    if k in {text_field}:
                        continue
                    copied[k] = v

            for i, chunk_text in enumerate(chunks):
                row = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}::chunk{i:03d}",
                    "chunk_index": int(i),
                    "chunk_count": int(chunk_count),
                    "chunk_chars": int(len(chunk_text)),
                    "text": chunk_text,
                    "text_preview": chunk_text[:240],
                    **copied,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                chunks_written += 1

    print(
        "Done. "
        f"docs_seen={docs_seen} "
        f"docs_written={docs_written} "
        f"chunks_written={chunks_written} "
        f"skipped_missing={skipped_missing} "
        f"skipped_short={skipped_short} "
        f"out={out_path}"
    )


if __name__ == "__main__":
    main()

