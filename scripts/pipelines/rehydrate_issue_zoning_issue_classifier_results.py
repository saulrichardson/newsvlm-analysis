#!/usr/bin/env python3
"""
Rehydrate issue-level zoning classifier results into durable on-disk artifacts.

This is the “parser + audit trail” step after you run:
  - OpenAI Batch API (downloaded *_output.jsonl / *_error.jsonl), OR
  - scripts/platform/run_openai_requests_via_gateway.py (openai_results_shard*.jsonl / openai_errors_shard*.jsonl)

Inputs:
  - mapping_shardNNN.jsonl (from scripts/pipelines/export_issue_zoning_issue_classifier_batch_requests.py)
  - result JSONLs (OpenAI-batch-shaped):
      - openai_results_shardNNN.jsonl (gateway) OR *_output.jsonl (OpenAI Batch download)
      - openai_errors_shardNNN.jsonl  (gateway) OR *_error.jsonl  (OpenAI Batch download; optional)

Outputs (under --output-dir):
  - issue_zoning_parsed_outputs.jsonl : one row per issue_id with parsed JSON + provenance
  - manifest.jsonl                    : one row per issue_id with status + output_path
  - *.issue_zoning.json               : (optional) one file per issue_id

This script does NOT call any APIs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator


REQUIRED_TOP_LEVEL_KEYS: set[str] = {
    "zoning_presence",
    "legal_document",
    "ordinance_jurisdiction",
    "code_components_present",
    "quality_flags",
    "rationale",
}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_issue_filename(issue_id: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(issue_id))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "issue"


def _repair_truncated_json(candidate: str) -> str:
    s = _norm(candidate)
    if not s:
        return s
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    fixed = s
    if in_str:
        fixed += '"'
    if stack:
        fixed += "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    return fixed


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    s = _norm(text)
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)

    candidates: list[dict[str, Any]] = []

    def _add(obj: Any) -> None:
        if isinstance(obj, dict) and REQUIRED_TOP_LEVEL_KEYS.issubset(set(obj.keys())):
            candidates.append(obj)

    try:
        _add(json.loads(s))
    except Exception:
        pass

    first = s.find("{")
    if first != -1:
        candidate = s[first:]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        repaired = _repair_truncated_json(candidate)
        try:
            _add(json.loads(repaired))
        except Exception:
            pass

    dec = json.JSONDecoder()
    start = first
    while start != -1:
        try:
            obj, _ = dec.raw_decode(s[start:])
            _add(obj)
        except Exception:
            pass
        start = s.find("{", start + 1)

    if not candidates:
        return None
    return sorted(candidates, key=lambda x: len(json.dumps(x, ensure_ascii=False)), reverse=True)[0]


def _extract_openai_output_text(body: dict[str, Any]) -> str:
    out = body.get("output")
    if not isinstance(out, list):
        return ""
    parts: list[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "output_text":
                parts.append(str(c.get("text") or ""))
    return "".join(parts)


def _iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _discover_results_files(results_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    Support both:
      - gateway runner: openai_results_shard*.jsonl / openai_errors_shard*.jsonl
      - OpenAI Batch download: *_output.jsonl / *_error.jsonl
    """
    shard_results = sorted(results_dir.glob("openai_results_shard*.jsonl"))
    shard_errors = sorted(results_dir.glob("openai_errors_shard*.jsonl"))
    if shard_results or shard_errors:
        return shard_results, shard_errors

    batch_results = sorted(results_dir.glob("*_output.jsonl"))
    batch_errors = sorted(results_dir.glob("*_error.jsonl"))
    return batch_results, batch_errors


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rehydrate issue-level zoning classifier outputs into parsed artifacts.")
    ap.add_argument("--request-dir", required=True, help="Directory containing mapping_shard*.jsonl")
    ap.add_argument("--results-dir", required=True, help="Directory containing result JSONLs")
    ap.add_argument("--output-dir", required=True, help="Directory to write parsed outputs into")
    ap.add_argument(
        "--write-per-issue-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to write <issue_id>.issue_zoning.json files under --output-dir",
    )
    ap.add_argument(
        "--allow-partial-results",
        action="store_true",
        help="Do not abort if some custom_ids have no result line; mark them missing instead.",
    )
    ap.add_argument(
        "--allow-errors",
        action="store_true",
        help="Exit 0 even if some issues have missing/invalid outputs.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir is not a directory: {results_dir}")

    mapping_paths = sorted(request_dir.glob("mapping_shard*.jsonl"))
    if not mapping_paths:
        raise SystemExit(f"No mapping_shard*.jsonl found in {request_dir}")

    mapping_by_cid: dict[str, dict[str, Any]] = {}
    for mp in mapping_paths:
        for obj in _iter_jsonl_rows(mp):
            cid = obj.get("custom_id") or obj.get("id") or obj.get("issue_id")
            if not isinstance(cid, str) or not cid.strip():
                raise SystemExit(f"Bad mapping line in {mp}: missing custom_id")
            cid = cid.strip()
            if cid in mapping_by_cid:
                raise SystemExit(f"Duplicate custom_id across mapping shards: {cid}")
            mapping_by_cid[cid] = obj

    results_files, errors_files = _discover_results_files(results_dir)
    if not results_files and not errors_files:
        raise SystemExit(f"No results found in {results_dir} (expected openai_results_shard*.jsonl or *_output.jsonl)")

    results_by_cid: dict[str, dict[str, Any]] = {}
    errors_by_cid: dict[str, dict[str, Any]] = {}

    for rp in results_files:
        for obj in _iter_jsonl_rows(rp):
            cid = obj.get("custom_id")
            if isinstance(cid, str) and cid.strip():
                results_by_cid[cid.strip()] = obj

    for ep in errors_files:
        for obj in _iter_jsonl_rows(ep):
            cid = obj.get("custom_id")
            if isinstance(cid, str) and cid.strip():
                errors_by_cid[cid.strip()] = obj

    parsed_path = out_dir / "issue_zoning_parsed_outputs.jsonl"
    manifest_path = out_dir / "manifest.jsonl"
    errors_path = out_dir / "rehydration_errors.jsonl"
    prov_path = out_dir / "rehydration_provenance.json"

    totals = {
        "issues_total": len(mapping_by_cid),
        "ok": 0,
        "missing": 0,
        "error_line": 0,
        "empty_output": 0,
        "json_parse_fail": 0,
        "status_code_fail": 0,
    }

    ts = dt.datetime.now().isoformat(timespec="seconds")
    prov_path.write_text(
        json.dumps(
            {
                "created_at": ts,
                "request_dir": str(request_dir),
                "results_dir": str(results_dir),
                "results_files": [str(p) for p in results_files],
                "errors_files": [str(p) for p in errors_files],
                "mapping_files": [str(p) for p in mapping_paths],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Stream writes so we don't hold everything in memory for big runs.
    with (
        parsed_path.open("w", encoding="utf-8") as parsed_f,
        manifest_path.open("w", encoding="utf-8") as man_f,
        errors_path.open("w", encoding="utf-8") as err_f,
    ):
        for cid in sorted(mapping_by_cid.keys()):
            m = mapping_by_cid[cid]
            issue_id = _norm(m.get("issue_id") or cid)
            slug = _norm(m.get("slug") or (issue_id.split("__", 1)[0] if "__" in issue_id else ""))
            issue_date = _norm(m.get("issue_date") or (issue_id.split("__", 1)[1] if "__" in issue_id else ""))

            out_file = out_dir / f"{_safe_issue_filename(issue_id)}.issue_zoning.json"
            output_path = str(out_file) if bool(args.write_per_issue_files) else ""

            result_obj = results_by_cid.get(cid)
            error_obj = errors_by_cid.get(cid)

            status = "unknown"
            parsed_json: dict[str, Any] | None = None
            output_text = ""
            output_text_sha = ""
            status_code = None
            model_name = ""
            error_msg = ""

            if result_obj is None and error_obj is None:
                totals["missing"] += 1
                status = "missing_result"
                if not args.allow_partial_results:
                    error_msg = "missing result line"
            elif error_obj is not None and result_obj is None:
                totals["error_line"] += 1
                status = "error_line"
                resp = (error_obj.get("response") or {}) if isinstance(error_obj.get("response"), dict) else {}
                body = (resp.get("body") or {}) if isinstance(resp.get("body"), dict) else {}
                err = (body.get("error") or {}) if isinstance(body.get("error"), dict) else {}
                error_msg = _norm(err.get("message") or body.get("message") or error_obj.get("error") or "unknown error")
            else:
                resp = (result_obj.get("response") or {}) if isinstance(result_obj.get("response"), dict) else {}
                status_code = resp.get("status_code")
                body = (resp.get("body") or {}) if isinstance(resp.get("body"), dict) else {}
                model_name = _norm(body.get("model") or "")
                output_text = _extract_openai_output_text(body)
                output_text_sha = _sha256(output_text) if output_text else ""

                try:
                    if status_code is not None and int(status_code) != 200:
                        totals["status_code_fail"] += 1
                        status = "bad_status_code"
                        error_msg = _norm(body.get("error") or "")
                    elif not output_text.strip():
                        totals["empty_output"] += 1
                        status = "empty_output_text"
                    else:
                        parsed_json = _parse_json_from_text(output_text)
                        if parsed_json is None:
                            totals["json_parse_fail"] += 1
                            status = "json_parse_fail"
                        else:
                            totals["ok"] += 1
                            status = "ok"
                except Exception as exc:  # noqa: BLE001
                    totals["json_parse_fail"] += 1
                    status = "exception"
                    error_msg = str(exc)

            row = {
                "custom_id": cid,
                "issue_id": issue_id,
                "slug": slug,
                "issue_date": issue_date,
                "status": status,
                "classifier_model": model_name,
                "output_text_sha256": output_text_sha,
                "parsed": parsed_json,
                "mapping": m,
                "error": error_msg,
                "rehydrated_at": ts,
            }
            parsed_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            man_row = {
                "custom_id": cid,
                "issue_id": issue_id,
                "slug": slug,
                "issue_date": issue_date,
                "status": status,
                "output_path": output_path,
                "classifier_model": model_name,
                "output_text_sha256": output_text_sha,
                "rehydrated_at": ts,
            }
            man_f.write(json.dumps(man_row, ensure_ascii=False) + "\n")

            if status != "ok":
                err_f.write(json.dumps({"custom_id": cid, "issue_id": issue_id, "status": status, "error": error_msg}, ensure_ascii=False) + "\n")
                continue

            if bool(args.write_per_issue_files):
                issue_obj = {
                    "issue_id": issue_id,
                    "slug": slug,
                    "issue_date": issue_date,
                    "classification": parsed_json,
                    "classifier_model": model_name,
                    "output_text_sha256": output_text_sha,
                    "mapping": m,
                    "rehydrated_at": ts,
                }
                out_file.write_text(json.dumps(issue_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _eprint(json.dumps(totals, ensure_ascii=False))

    bad = totals["missing"] + totals["error_line"] + totals["empty_output"] + totals["json_parse_fail"] + totals["status_code_fail"]
    if bad and not args.allow_errors:
        raise SystemExit(f"Rehydration completed with {bad} non-ok issues (see {errors_path}). Pass --allow-errors to exit 0.")


if __name__ == "__main__":
    main()
