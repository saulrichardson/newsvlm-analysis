#!/usr/bin/env python3
"""
Export OpenAI embedding request shards for ISSUE-chunk documents built from per-page *.vlm.json outputs.

Why this exists
  - scripts/export_issue_topic_embedding_batch_requests.py is the "real" pipeline entrypoint,
    but it expects an issue-zoning manifest + issue-level JSON outputs that point at VAST paths.
  - For local iteration, we often just have a directory of per-page *.vlm.json files.
  - This script builds issue documents directly from those per-page outputs and emits the same
    embedding-request format (POST /v1/embeddings) used by the Batch pipeline.

Inputs
  - --pages: glob or "@file" list of per-page *.vlm.json outputs

Outputs (output_dir)
  - openai_requests_shardNNN.jsonl
  - mapping_shardNNN.jsonl
  - inputs/* (for provenance)

This script does NOT submit any API requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from glob import glob
from os.path import expanduser
from pathlib import Path
from typing import Any

from newsvlm.issue_zoning_classifier import load_page_result, page_text_from_boxes, parse_page_id


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_page_paths(spec: str) -> list[Path]:
    """Collect per-page .vlm.json paths from a glob or @file list."""
    spec = spec.strip()
    if spec.startswith("@"):
        list_path = Path(expanduser(spec[1:]))
        if not list_path.is_file():
            raise SystemExit(f"Pages file not found: {list_path}")
        paths: list[Path] = []
        for raw in list_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(expanduser(line))
            if not p.is_file():
                raise SystemExit(f"Page JSON listed but not found: {p}")
            paths.append(p)
        return sorted(paths)

    expanded = expanduser(spec)
    return sorted(Path(p) for p in glob(expanded))


def _chunk_text(*, text: str, chunk_size_chars: int) -> list[str]:
    """
    Chunk on paragraph boundaries first (\\n\\n) to avoid splitting in the middle
    of an OCR box transcript.
    """
    if chunk_size_chars < 0:
        raise ValueError("chunk_size_chars must be >= 0")

    clean = (text or "").strip()
    if not clean:
        return []
    if chunk_size_chars == 0 or len(clean) <= chunk_size_chars:
        return [clean]

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
                chunk = para[start:end].strip()
                if chunk:
                    chunks.append(chunk)
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


@dataclass(frozen=True)
class PageDoc:
    page_id: str
    issue_id: str
    issue_date: str
    slug: str
    page_num: int
    page_path: Path
    text: str

    @property
    def text_sha256(self) -> str:
        return _sha256(self.text)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export OpenAI embedding request shards from per-page *.vlm.json files.")
    ap.add_argument(
        "--pages",
        required=True,
        help='Glob for per-page *.vlm.json files (absolute OK) or "@file" listing those paths',
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write request shards into")

    ap.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model name")
    ap.add_argument(
        "--chunk-size-chars",
        type=int,
        default=16000,
        help="Chunk size for issue text (0 disables chunking).",
    )
    ap.add_argument(
        "--max-page-chars",
        type=int,
        default=0,
        help="Optional: truncate each page transcript to this many chars before building the issue text (0 disables).",
    )
    ap.add_argument("--max-issues", type=int, default=0, help="Optional: cap number of issues exported (0 disables).")
    ap.add_argument("--skip-bad-pages", action="store_true", help="Skip pages that fail to load/parse instead of exiting.")

    ap.add_argument("--requests-per-shard", type=int, default=5000, help="Maximum requests per output shard file")
    ap.add_argument(
        "--max-bytes-per-shard",
        type=int,
        default=180_000_000,
        help="Max bytes per OpenAI request shard file (mapping is not capped). Set 0 to disable.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    page_paths = collect_page_paths(args.pages)
    if not page_paths:
        raise SystemExit(f"No pages matched: {args.pages}")

    chunk_size_chars = int(args.chunk_size_chars)
    if chunk_size_chars < 0:
        raise SystemExit("--chunk-size-chars must be >= 0")
    max_page_chars = int(args.max_page_chars)
    if max_page_chars < 0:
        raise SystemExit("--max-page-chars must be >= 0")
    max_issues = int(args.max_issues) if int(args.max_issues) > 0 else None

    request_limit = int(args.requests_per_shard)
    if request_limit <= 0:
        raise SystemExit("--requests-per-shard must be > 0")
    max_bytes = int(args.max_bytes_per_shard)
    if max_bytes < 0:
        raise SystemExit("--max-bytes-per-shard must be >= 0")

    # Load + group by issue.
    pages_by_issue: dict[str, list[PageDoc]] = {}
    for p in page_paths:
        try:
            page = load_page_result(p)
        except Exception as exc:  # noqa: BLE001
            if bool(args.skip_bad_pages):
                print(f"Skipping bad page JSON: {p} ({exc})", file=sys.stderr)
                continue
            raise SystemExit(f"Failed to load page JSON: {p} ({exc})") from exc

        pid = parse_page_id(page.page_id)
        if pid is None:
            msg = f"Unparseable page_id: {page.page_id} (path={p})"
            if bool(args.skip_bad_pages):
                print(f"Skipping: {msg}", file=sys.stderr)
                continue
            raise SystemExit(msg)

        text = page_text_from_boxes(page)
        if max_page_chars > 0 and len(text) > max_page_chars:
            text = text[:max_page_chars].rstrip()

        doc = PageDoc(
            page_id=pid.page_id,
            issue_id=pid.issue_id,
            issue_date=pid.issue_date,
            slug=pid.slug,
            page_num=pid.page_num,
            page_path=p,
            text=text,
        )
        pages_by_issue.setdefault(pid.issue_id, []).append(doc)

    issue_ids = sorted(pages_by_issue.keys())
    if max_issues is not None:
        issue_ids = issue_ids[:max_issues]

    # Record inputs/config for reproducibility.
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "pages_spec.txt").write_text(str(args.pages) + "\n", encoding="utf-8")
    (inputs_dir / "export_config.json").write_text(
        json.dumps(
            {
                "embedding_model": str(args.embedding_model),
                "chunk_size_chars": int(chunk_size_chars),
                "max_page_chars": int(max_page_chars),
                "max_issues": max_issues,
                "requests_per_shard": int(request_limit),
                "max_bytes_per_shard": int(max_bytes),
                "skip_bad_pages": bool(args.skip_bad_pages),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Shard writers.
    shard_idx = 0
    requests_in_shard = 0
    bytes_in_shard = 0

    openai_path = out_dir / f"openai_requests_shard{shard_idx:03d}.jsonl"
    mapping_path = out_dir / f"mapping_shard{shard_idx:03d}.jsonl"
    openai_f = openai_path.open("w", encoding="utf-8")
    mapping_f = mapping_path.open("w", encoding="utf-8")

    def rotate_if_needed(next_openai_line: str) -> None:
        nonlocal shard_idx, requests_in_shard, bytes_in_shard, openai_f, mapping_f, openai_path, mapping_path
        if requests_in_shard <= 0:
            return
        if requests_in_shard < request_limit:
            if max_bytes <= 0:
                return
            if bytes_in_shard + len(next_openai_line.encode("utf-8")) <= max_bytes:
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

    # Export loop.
    t0 = time.perf_counter()
    issues_written = 0
    chunks_written = 0

    for issue_id in issue_ids:
        pages = sorted(pages_by_issue[issue_id], key=lambda d: d.page_num)
        if not pages:
            continue

        issue_date = pages[0].issue_date
        slug = pages[0].slug
        if any(p.issue_date != issue_date or p.slug != slug for p in pages):
            raise SystemExit(f"Issue metadata mismatch within issue_id={issue_id} (unexpected)")

        used_pages: list[dict[str, Any]] = []
        parts: list[str] = []
        for p in pages:
            parts.append(f"=== PAGE {p.page_num} {p.page_id} ===")
            parts.append(p.text.strip())
            used_pages.append(
                {
                    "page_id": p.page_id,
                    "page_num": p.page_num,
                    "page_path": str(p.page_path),
                    "text_chars": len(p.text),
                    "text_sha256": p.text_sha256,
                }
            )

        issue_text = "\n\n".join(x for x in parts if x.strip()).strip()
        chunks = _chunk_text(text=issue_text, chunk_size_chars=chunk_size_chars)
        if not chunks:
            continue

        issues_written += 1

        for chunk_idx, chunk_text in enumerate(chunks):
            custom_id = f"{issue_id}::chunk{chunk_idx:03d}"
            chunk_sha = _sha256(chunk_text)

            body = {"model": str(args.embedding_model), "input": chunk_text}
            openai_line = {"custom_id": custom_id, "method": "POST", "url": "/v1/embeddings", "body": body}
            openai_line_s = json.dumps(openai_line, ensure_ascii=False) + "\n"

            rotate_if_needed(openai_line_s)

            mapping_line = {
                "id": custom_id,
                "custom_id": custom_id,
                "issue_id": issue_id,
                "issue_date": issue_date,
                "slug": slug,
                "used_pages": used_pages,
                "chunk_index": int(chunk_idx),
                "chunk_count": int(len(chunks)),
                "chunk_chars": int(len(chunk_text)),
                "chunk_sha256": chunk_sha,
                "chunk_text_preview": chunk_text[:240],
                "embedding_model": str(args.embedding_model),
                "exported_at": time.time(),
            }

            openai_f.write(openai_line_s)
            mapping_f.write(json.dumps(mapping_line, ensure_ascii=False) + "\n")
            requests_in_shard += 1
            bytes_in_shard += len(openai_line_s.encode("utf-8"))
            chunks_written += 1

    openai_f.close()
    mapping_f.close()

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"Done. issues_written={issues_written} chunks_written={chunks_written} shards={shard_idx + 1} elapsed_s={elapsed}")


if __name__ == "__main__":
    main()
