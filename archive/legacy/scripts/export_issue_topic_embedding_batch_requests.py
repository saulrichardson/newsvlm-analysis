#!/usr/bin/env python3
"""
Export OpenAI Batch request shards for ISSUE-level topic embeddings.

Goal
  - Input: issue-level zoning classification outputs (*.issue_zoning.json) via manifest.jsonl.
  - Filter to a subset of issues (by label, source manifest, etc.).
  - Build an "issue document" by concatenating per-page transcript text (from *.vlm.json).
  - Chunk at the *issue* level (multiple chunks per issue) to produce richer embeddings.
  - Output: sharded OpenAI Batch request JSONLs (POST /v1/embeddings) + mapping JSONLs.

This script ONLY exports request shards. It does not submit or download anything.

Typical usage (Greene):
  python scripts/export_issue_topic_embedding_batch_requests.py \
    --issue-zoning-manifest /vast/.../issue_zoning_.../outputs/manifest.jsonl \
    --labels full_ordinance \
    --output-dir /vast/.../issue_topics_full_ordinance/embeddings/requests \
    --embedding-model text-embedding-3-small \
    --chunk-size-chars 16000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from glob import glob
from os.path import expanduser
from pathlib import Path
from typing import Any, Iterable, Literal

from newsvlm.issue_zoning_classifier import load_page_result, page_text_from_boxes


PageSelection = Literal["pages_with_zoning", "all_pages"]


@dataclass(frozen=True)
class IssueRow:
    issue_id: str
    issue_date: str
    slug: str
    label: str
    output_path: Path


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collect_manifest_rows(path: Path) -> list[IssueRow]:
    if not path.is_file():
        raise SystemExit(f"--issue-zoning-manifest not found: {path}")

    out: list[IssueRow] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error") is not None:
                continue
            issue_id = obj.get("issue_id")
            issue_date = obj.get("issue_date")
            slug = obj.get("slug")
            label = obj.get("label")
            output_path = obj.get("output_path")
            if not all(isinstance(v, str) and v.strip() for v in [issue_id, issue_date, slug, label, output_path]):
                raise SystemExit(f"Malformed manifest row (missing required fields): {obj}")
            out.append(
                IssueRow(
                    issue_id=str(issue_id).strip(),
                    issue_date=str(issue_date).strip(),
                    slug=str(slug).strip(),
                    label=str(label).strip(),
                    output_path=Path(str(output_path)).expanduser(),
                )
            )
    if not out:
        raise SystemExit(f"No successful issues found in manifest: {path}")
    return out


def _load_issue_output(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Issue output JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_stem(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(s))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "x"


def _issue_text_from_pages(
    *,
    issue_id: str,
    issue_json: dict[str, Any],
    page_selection: PageSelection,
    max_page_chars: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Return (issue_text, used_pages metadata)."""
    source_pages = issue_json.get("source_pages")
    if not isinstance(source_pages, list) or not source_pages:
        raise SystemExit(f"{issue_id}: missing/empty source_pages in issue output JSON")

    cls = issue_json.get("classification") or {}
    pages_with_zoning = cls.get("pages_with_zoning") or []

    # Determine which pages to include.
    include_ids: set[str] | None
    if page_selection == "all_pages":
        include_ids = None
    elif page_selection == "pages_with_zoning":
        if not isinstance(pages_with_zoning, list):
            raise SystemExit(f"{issue_id}: pages_with_zoning is not a list")
        include_ids = set()
        for item in pages_with_zoning:
            if not isinstance(item, dict):
                continue
            pid = item.get("page_id")
            if isinstance(pid, str) and pid.strip():
                include_ids.add(pid.strip())
        if not include_ids:
            raise SystemExit(f"{issue_id}: no pages_with_zoning page_ids found (refusing to embed empty selection)")
    else:
        raise ValueError(f"Unknown page_selection: {page_selection}")

    used_pages: list[dict[str, Any]] = []
    parts: list[str] = []

    for sp in source_pages:
        if not isinstance(sp, dict):
            raise SystemExit(f"{issue_id}: malformed source_pages entry: {sp!r}")
        page_id = sp.get("page_id")
        page_num = sp.get("page_num")
        page_path = sp.get("page_path")
        if not isinstance(page_id, str) or not page_id.strip():
            raise SystemExit(f"{issue_id}: source_pages entry missing page_id: {sp}")
        if include_ids is not None and page_id.strip() not in include_ids:
            continue
        if not isinstance(page_path, str) or not page_path.strip():
            raise SystemExit(f"{issue_id}: source_pages entry missing page_path: {sp}")

        page_path_p = Path(page_path)
        page = load_page_result(page_path_p)
        text = page_text_from_boxes(page)
        if max_page_chars is not None and len(text) > max_page_chars:
            text = text[:max_page_chars].rstrip()

        parts.append(f"=== PAGE {page_num} {page_id} ===".strip())
        parts.append(text.strip())

        used_pages.append(
            {
                "page_id": page_id.strip(),
                "page_num": page_num,
                "page_path": str(page_path_p),
                "text_chars": len(text),
                "text_sha256": _sha256(text),
            }
        )

    issue_text = "\n\n".join(p for p in parts if p.strip()).strip()
    if not issue_text:
        raise SystemExit(f"{issue_id}: issue_text is empty after page selection ({page_selection})")
    if not any(int(p.get("text_chars") or 0) > 0 for p in used_pages):
        raise SystemExit(f"{issue_id}: selected pages produced 0 transcript chars (refusing to embed empty content)")

    return issue_text, used_pages


def _chunk_text(*, text: str, chunk_size_chars: int) -> list[str]:
    if chunk_size_chars <= 0:
        return [text.strip()]

    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= chunk_size_chars:
        return [clean]

    # Chunk on paragraph boundaries first to avoid splitting in the middle of a box transcript.
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
    out = [c for c in chunks if c.strip()]
    if not out:
        raise SystemExit("Chunking produced no non-empty chunks (unexpected)")
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export OpenAI Batch request shards for issue-level topic embeddings.")
    ap.add_argument("--issue-zoning-manifest", required=True, help="Path to outputs/manifest.jsonl from issue zoning run")
    ap.add_argument("--output-dir", required=True, help="Directory to write request shards into")
    ap.add_argument(
        "--labels",
        required=True,
        help="Comma-separated labels to include (e.g. full_ordinance,amendment_substantial)",
    )
    ap.add_argument(
        "--page-selection",
        choices=["pages_with_zoning", "all_pages"],
        default="pages_with_zoning",
        help="Which pages to embed for each issue.",
    )
    ap.add_argument(
        "--max-page-chars",
        type=int,
        default=0,
        help="Optional: truncate each page transcript to this many chars before building the issue text (0 disables).",
    )
    ap.add_argument(
        "--chunk-size-chars",
        type=int,
        default=16000,
        help="Chunk size for issue text (0 disables chunking).",
    )
    ap.add_argument("--max-issues", type=int, default=0, help="Stop after exporting this many issues (0 disables).")
    ap.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model name for /v1/embeddings batch body",
    )
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

    manifest_path = Path(expanduser(args.issue_zoning_manifest)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {p.strip() for p in str(args.labels).split(",") if p.strip()}
    if not labels:
        raise SystemExit("--labels must include at least one label")

    page_selection: PageSelection = str(args.page_selection)  # type: ignore[assignment]

    max_issues = int(args.max_issues) if int(args.max_issues) > 0 else None
    max_page_chars = int(args.max_page_chars) if int(args.max_page_chars) > 0 else None
    chunk_size_chars = int(args.chunk_size_chars) if int(args.chunk_size_chars) >= 0 else None
    if chunk_size_chars is None:
        raise SystemExit("--chunk-size-chars must be >= 0")

    request_limit = int(args.requests_per_shard)
    if request_limit <= 0:
        raise SystemExit("--requests-per-shard must be > 0")
    max_bytes = int(args.max_bytes_per_shard)
    if max_bytes < 0:
        raise SystemExit("--max-bytes-per-shard must be >= 0")

    rows = _collect_manifest_rows(manifest_path)
    rows = [r for r in rows if r.label in labels]
    if not rows:
        raise SystemExit(f"No rows matched labels={sorted(labels)} in manifest: {manifest_path}")
    rows = sorted(rows, key=lambda r: (r.issue_date, r.issue_id))
    if max_issues is not None:
        rows = rows[:max_issues]

    # Inputs + config for reproducibility.
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "issue_zoning_manifest_path.txt").write_text(str(manifest_path) + "\n", encoding="utf-8")
    (inputs_dir / "issue_zoning_manifest_sha256.txt").write_text(_sha256(manifest_path.read_text()) + "\n", encoding="utf-8")
    (inputs_dir / "labels.txt").write_text("\n".join(sorted(labels)) + "\n", encoding="utf-8")
    (inputs_dir / "export_config.json").write_text(
        json.dumps(
            {
                "page_selection": page_selection,
                "max_page_chars": max_page_chars,
                "chunk_size_chars": chunk_size_chars,
                "embedding_model": str(args.embedding_model),
                "requests_per_shard": request_limit,
                "max_bytes_per_shard": max_bytes,
                "max_issues": max_issues,
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
    issues_skipped_empty = 0
    skipped_rows: list[dict[str, Any]] = []

    for r in rows:
        issue = _load_issue_output(r.output_path)

        issue_text, used_pages = _issue_text_from_pages(
            issue_id=r.issue_id,
            issue_json=issue,
            page_selection=page_selection,
            max_page_chars=max_page_chars,
        )
        chunks = _chunk_text(text=issue_text, chunk_size_chars=chunk_size_chars)
        if not chunks:
            issues_skipped_empty += 1
            skipped_rows.append({"issue_id": r.issue_id, "reason": "no_text_after_chunking"})
            continue

        issues_written += 1

        for chunk_idx_local, chunk_text in enumerate(chunks):
            custom_id = f"{r.issue_id}::chunk{chunk_idx_local:03d}"
            chunk_sha = _sha256(chunk_text)

            body = {"model": str(args.embedding_model), "input": chunk_text}
            openai_line = {"custom_id": custom_id, "method": "POST", "url": "/v1/embeddings", "body": body}
            openai_line_s = json.dumps(openai_line, ensure_ascii=False) + "\n"

            rotate_if_needed(openai_line_s)

            mapping_line = {
                "id": custom_id,
                "custom_id": custom_id,
                "issue_id": r.issue_id,
                "issue_date": r.issue_date,
                "slug": r.slug,
                "classification_label": r.label,
                "issue_zoning_output_path": str(r.output_path),
                "page_selection": page_selection,
                "used_pages": used_pages,
                "chunk_index": chunk_idx_local,
                "chunk_count": len(chunks),
                "chunk_chars": len(chunk_text),
                "chunk_sha256": chunk_sha,
                "chunk_text_preview": chunk_text[:240],
                "embedding_model": str(args.embedding_model),
            }

            openai_f.write(openai_line_s)
            mapping_f.write(json.dumps(mapping_line, ensure_ascii=False) + "\n")

            requests_in_shard += 1
            chunks_written += 1
            bytes_in_shard += len(openai_line_s.encode("utf-8"))

    openai_f.close()
    mapping_f.close()

    if skipped_rows:
        (inputs_dir / "skipped_issues.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in skipped_rows) + "\n",
            encoding="utf-8",
        )

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"Done. issues_written={issues_written} chunks_written={chunks_written} skipped_empty={issues_skipped_empty} elapsed_s={elapsed}")


if __name__ == "__main__":
    main()
