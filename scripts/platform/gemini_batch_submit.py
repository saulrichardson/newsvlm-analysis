#!/usr/bin/env python3
"""
Submit Gemini Batch API jobs for shard JSONL request files using curl/REST.

Expected request files (under --request-dir):
  - gemini_requests_shardNNN.jsonl   (each line: {"key": "...", "request": {...}})
  - mapping_shardNNN.jsonl           (provenance; used for line-count validation)

Outputs:
  - submitted_gemini_batches.jsonl   (append-only, resumable submission record)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _resolve_gemini_key(env_file: Path | None) -> str:
    for name in ("GEMINI_KEY", "GEMINI_API_KEY"):
        val = os.environ.get(name)
        if val:
            return str(val)
    if env_file is not None:
        file_env = _parse_env_file(env_file)
        for name in ("GEMINI_KEY", "GEMINI_API_KEY"):
            val = file_env.get(name)
            if val:
                return val
    raise SystemExit("Missing Gemini key. Set GEMINI_KEY in env or pass --env-file with GEMINI_KEY=...")


def _parse_shard_index(filename: str, *, prefix: str, suffix: str) -> int:
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        raise ValueError(f"Unexpected filename: {filename}")
    mid = filename[len(prefix) : -len(suffix)]
    if not mid.isdigit():
        raise ValueError(f"Unexpected shard token in filename: {filename}")
    return int(mid)


def _line_count(path: Path) -> int:
    out = subprocess.check_output(["wc", "-l", str(path)], text=True).strip()
    return int(out.split()[0])


def _load_existing_shards(record_path: Path) -> set[int]:
    if not record_path.is_file():
        return set()
    seen: set[int] = set()
    for raw in record_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("provider") != "gemini":
            continue
        shard = obj.get("shard")
        if isinstance(shard, int):
            seen.add(shard)
    return seen


def _append_record(record_path: Path, obj: dict[str, Any]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _run_json(cmd: list[str]) -> dict[str, Any]:
    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[:800]}")
    try:
        return json.loads(p.stdout)
    except Exception as exc:
        raise RuntimeError(f"Expected JSON response, got parse error: {exc}\n{p.stdout[:1000]}") from exc


def _write_key_header_file(*, key: str, env_file: Path | None, record_path: Path) -> Path:
    base_dir = env_file.parent if env_file is not None else record_path.parent
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        base_dir = record_path.parent
    fd, tmp_path = tempfile.mkstemp(prefix="gemini_headers_", suffix=".txt", dir=str(base_dir))
    path = Path(tmp_path)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"x-goog-api-key: {key}\n")
    os.chmod(path, 0o600)
    return path


def _start_resumable_upload(*, req_path: Path, display_name: str, header_path: Path) -> str:
    with tempfile.NamedTemporaryFile(prefix="gemini_upload_headers_", suffix=".txt", delete=False) as tmp:
        header_dump_path = Path(tmp.name)
    try:
        payload = {"file": {"display_name": display_name}}
        cmd = [
            "curl",
            "-sS",
            "-D",
            str(header_dump_path),
            "https://generativelanguage.googleapis.com/upload/v1beta/files",
            "-H",
            f"@{str(header_path)}",
            "-H",
            "X-Goog-Upload-Protocol: resumable",
            "-H",
            "X-Goog-Upload-Command: start",
            "-H",
            f"X-Goog-Upload-Header-Content-Length: {req_path.stat().st_size}",
            "-H",
            "X-Goog-Upload-Header-Content-Type: application/jsonl",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload, ensure_ascii=False),
        ]
        p = _run(cmd)
        if p.returncode != 0:
            raise RuntimeError(f"Gemini upload start failed ({p.returncode}) for {req_path.name}\n{p.stderr[:800]}")
        headers_text = header_dump_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?im)^x-goog-upload-url:\s*(\S+)\s*$", headers_text)
        if not match:
            raise RuntimeError(f"Could not find x-goog-upload-url in response headers for {req_path.name}\n{headers_text[:1000]}")
        return match.group(1).strip()
    finally:
        try:
            header_dump_path.unlink()
        except Exception:
            pass


def _finalize_upload(*, upload_url: str, req_path: Path) -> dict[str, Any]:
    return _run_json(
        [
            "curl",
            "-sS",
            upload_url,
            "-H",
            f"Content-Length: {req_path.stat().st_size}",
            "-H",
            "X-Goog-Upload-Offset: 0",
            "-H",
            "X-Goog-Upload-Command: upload, finalize",
            "--data-binary",
            f"@{str(req_path)}",
        ]
    )


def _create_batch(*, model: str, uploaded_file_name: str, display_name: str, header_path: Path) -> dict[str, Any]:
    model_name = str(model).strip()
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    payload = {
        "batch": {
            "display_name": display_name,
            "input_config": {"file_name": uploaded_file_name},
        }
    }
    return _run_json(
        [
            "curl",
            "-sS",
            f"https://generativelanguage.googleapis.com/v1beta/{model_name}:batchGenerateContent",
            "-H",
            f"@{str(header_path)}",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload, ensure_ascii=False),
        ]
    )


def _extract_batch_state(batch_obj: dict[str, Any]) -> str:
    for key in ("state",):
        val = batch_obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    metadata = batch_obj.get("metadata")
    if isinstance(metadata, dict):
        val = metadata.get("state")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Submit Gemini Batch jobs for gemini_requests_shard*.jsonl.")
    ap.add_argument("--request-dir", required=True, help="Directory containing gemini_requests_shard*.jsonl + mapping_shard*.jsonl")
    ap.add_argument(
        "--record-path",
        default="",
        help="Append-only submission record path (default: <request-dir>/submitted_gemini_batches.jsonl)",
    )
    ap.add_argument("--env-file", default="", help="Optional .env file with GEMINI_KEY (env vars win).")
    ap.add_argument("--model", default="models/gemini-2.5-pro", help="Gemini model id for Batch create.")
    ap.add_argument("--display-name-prefix", default="issue-classifier", help="Gemini display name prefix.")
    ap.add_argument("--sleep-seconds", type=float, default=1.0, help="Pause between submissions.")
    ap.add_argument("--dry-run", action="store_true", help="Do not call APIs; print planned submissions.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser().resolve()
    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")

    record_path = (
        Path(args.record_path).expanduser().resolve()
        if str(args.record_path).strip()
        else (request_dir / "submitted_gemini_batches.jsonl")
    )
    env_file = Path(args.env_file).expanduser().resolve() if str(args.env_file).strip() else None
    key = _resolve_gemini_key(env_file)
    header_path = _write_key_header_file(key=key, env_file=env_file, record_path=record_path)

    mapping_paths: dict[int, Path] = {}
    gemini_paths: dict[int, Path] = {}
    for p in request_dir.glob("mapping_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="mapping_shard", suffix=".jsonl")
        mapping_paths[s] = p
    for p in request_dir.glob("gemini_requests_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="gemini_requests_shard", suffix=".jsonl")
        gemini_paths[s] = p

    shard_indices = sorted(set(mapping_paths) & set(gemini_paths))
    if not shard_indices:
        raise SystemExit(f"No shard pairs found in {request_dir} (expected mapping_shard*.jsonl + gemini_requests_shard*.jsonl)")

    existing = _load_existing_shards(record_path)

    submitted = 0
    skipped = 0
    try:
        for s in shard_indices:
            if s in existing:
                skipped += 1
                print(f"skip_already_submitted shard={s:03d}")
                continue

            mapping_path = mapping_paths[s]
            req_path = gemini_paths[s]
            mapping_lines = _line_count(mapping_path)
            req_lines = _line_count(req_path)
            if mapping_lines != req_lines:
                raise SystemExit(
                    f"Shard {s:03d} mismatch: mapping={mapping_lines} gemini={req_lines} "
                    f"({mapping_path.name} vs {req_path.name})"
                )

            display = f"{str(args.display_name_prefix).strip()}-shard{s:03d}"
            print(f"submit_start shard={s:03d} lines={mapping_lines} file={req_path.name} model={args.model} display_name={display}")

            if args.dry_run:
                print(f"dry_run shard={s:03d}")
                continue

            t0 = time.time()
            upload_url = _start_resumable_upload(req_path=req_path, display_name=display, header_path=header_path)
            upload_resp = _finalize_upload(upload_url=upload_url, req_path=req_path)
            file_obj = upload_resp.get("file") if isinstance(upload_resp.get("file"), dict) else upload_resp
            uploaded_file_name = str((file_obj or {}).get("name") or "").strip()
            if not uploaded_file_name:
                raise RuntimeError(f"Upload did not return file name for shard={s:03d}: {json.dumps(upload_resp)[:1000]}")

            batch = _create_batch(
                model=str(args.model),
                uploaded_file_name=uploaded_file_name,
                display_name=display,
                header_path=header_path,
            )
            elapsed = round(time.time() - t0, 2)
            batch_name = str(batch.get("name") or "").strip()
            batch_state = _extract_batch_state(batch)
            if not batch_name:
                raise RuntimeError(f"Batch create did not return name for shard={s:03d}: {json.dumps(batch)[:1000]}")

            rec: dict[str, Any] = {
                "provider": "gemini",
                "shard": int(s),
                "display_name": display,
                "request_dir": str(request_dir),
                "mapping_path": str(mapping_path),
                "mapping_lines": int(mapping_lines),
                "requests_path": str(req_path),
                "model": str(args.model),
                "created_at": int(time.time()),
                "elapsed_seconds": float(elapsed),
                "uploaded_file_name": uploaded_file_name,
                "batch_name": batch_name,
                "batch_state": batch_state,
                "batch": batch,
            }
            _append_record(record_path, rec)
            submitted += 1
            print(f"submit_ok shard={s:03d} batch_name={batch_name} state={batch_state}")
            time.sleep(max(0.0, float(args.sleep_seconds)))
    finally:
        try:
            header_path.unlink()
        except Exception:
            pass

    print(json.dumps({"record_path": str(record_path), "total_shards": len(shard_indices), "submitted": submitted, "skipped": skipped}))


if __name__ == "__main__":
    main()
