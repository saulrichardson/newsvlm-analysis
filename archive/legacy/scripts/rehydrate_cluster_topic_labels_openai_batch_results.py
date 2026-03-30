#!/usr/bin/env python3
"""
Rehydrate OpenAI Batch outputs for CLUSTER topic labeling into on-disk label artifacts.

Inputs:
  - mapping_shardNNN.jsonl (from scripts/export_cluster_topic_labeling_batch_requests.py)
  - openai_results_shardNNN.jsonl (downloaded OpenAI Batch output)
  - openai_errors_shardNNN.jsonl  (downloaded OpenAI Batch error output; may be empty)

Output:
  - <output_dir>/cluster_labels.jsonl   (one line per cluster_id)
  - <output_dir>/manifest.jsonl        (one line per cluster_id; includes output_path)
  - <output_dir>/clusters/cluster_<id>.cluster_topic_label.json (optional per-cluster files)

This script does NOT call any APIs; it only reshapes existing artifacts and validates JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from os.path import expanduser
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from newsvlm.pipeline import _coerce_json


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _extract_output_text(body: dict) -> str:
    """Extract concatenated output_text blocks from an OpenAI Responses API body."""
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


_CUSTOM_ID_RE = re.compile(r"^cluster::(-?\d+)$")


class ClusterTopicLabel(BaseModel):
    cluster_id: int
    topic_name: str = Field(min_length=1)
    topic_description: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    # v2 (optional): structured zoning mechanics tags produced by the LLM.
    zoning_relevance: str | None = None
    mechanics: "ClusterMechanics | None" = None

    @field_validator("keywords")
    @classmethod
    def _validate_keywords(cls, v: list[str]) -> list[str]:
        clean = [str(x).strip() for x in v if str(x).strip()]
        if len(clean) < 3:
            raise ValueError("keywords must include at least 3 non-empty strings")
        # Deduplicate while preserving order.
        seen: set[str] = set()
        uniq: list[str] = []
        for k in clean:
            lk = k.lower()
            if lk in seen:
                continue
            seen.add(lk)
            uniq.append(k)
        return uniq


class ClusterMechanicsEvidence(BaseModel):
    tag: str = Field(min_length=1)
    example_numbers: list[int] = Field(min_length=1)


class ClusterMechanics(BaseModel):
    action_tags: list[str] = Field(default_factory=list)
    dimension_tags: list[str] = Field(default_factory=list)
    decision_body_tags: list[str] = Field(default_factory=list)
    instrument_tags: list[str] = Field(default_factory=list)
    district_tokens: list[str] = Field(default_factory=list)
    mechanism_phrases: list[str] = Field(default_factory=list)
    evidence: list[ClusterMechanicsEvidence] = Field(default_factory=list)


ClusterTopicLabel.model_rebuild()


@dataclass(frozen=True)
class MappingRow:
    custom_id: str
    cluster_id: int
    cluster_size_chunks: int
    cluster_weight_total: float
    examples: list[dict[str, Any]]
    prompt_path: str
    final_prompt_sha256: str
    openai_model: str
    exported_at: float | None


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rehydrate OpenAI cluster-topic-label batch results.")
    ap.add_argument("--request-dir", required=True, help="Directory containing mapping_shardNNN.jsonl files")
    ap.add_argument("--results-dir", required=True, help="Directory containing openai_results_shardNNN.jsonl files")
    ap.add_argument("--output-dir", required=True, help="Directory to write cluster label outputs")
    ap.add_argument(
        "--allow-partial-results",
        action="store_true",
        help="Allow missing result shards (will emit errors for missing clusters instead of aborting).",
    )
    ap.add_argument(
        "--allow-errors",
        action="store_true",
        help="Exit 0 even if some clusters failed rehydration/validation.",
    )
    return ap.parse_args()


def _discover_shards(request_dir: Path) -> list[int]:
    shards: set[int] = set()
    for p in request_dir.glob("mapping_shard*.jsonl"):
        token = p.name.removeprefix("mapping_shard").removesuffix(".jsonl")
        if token.isdigit():
            shards.add(int(token))
    return sorted(shards)


def _load_jsonl_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("custom_id")
            if isinstance(cid, str) and cid.strip():
                out[cid.strip()] = obj
    return out


def _safe_cluster_filename(cluster_id: int) -> str:
    return f"cluster_{int(cluster_id):04d}.cluster_topic_label.json"


def main() -> None:
    args = _parse_args()
    request_dir = Path(expanduser(args.request_dir)).resolve()
    results_dir = Path(expanduser(args.results_dir)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir is not a directory: {results_dir}")

    shards = _discover_shards(request_dir)
    if not shards:
        raise SystemExit(f"No mapping_shard*.jsonl found under {request_dir}")

    # Load mapping for all clusters.
    mapping_by_custom_id: dict[str, MappingRow] = {}
    for shard in shards:
        mp = request_dir / f"mapping_shard{shard:03d}.jsonl"
        if not mp.is_file():
            raise SystemExit(f"Missing mapping file for shard {shard:03d}: {mp}")
        with mp.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                custom_id = obj.get("custom_id") or obj.get("id")
                if not isinstance(custom_id, str) or not custom_id.strip():
                    raise SystemExit(f"Bad mapping line in {mp}: missing custom_id")
                custom_id = custom_id.strip()
                if custom_id in mapping_by_custom_id:
                    raise SystemExit(f"Duplicate custom_id across mapping shards: {custom_id}")
                mapping_by_custom_id[custom_id] = MappingRow(
                    custom_id=custom_id,
                    cluster_id=int(obj.get("cluster_id")),
                    cluster_size_chunks=int(obj.get("cluster_size_chunks") or 0),
                    cluster_weight_total=float(obj.get("cluster_weight_total") or 0.0),
                    examples=list(obj.get("examples") or []),
                    prompt_path=str(obj.get("prompt_path") or ""),
                    final_prompt_sha256=str(obj.get("final_prompt_sha256") or ""),
                    openai_model=str(obj.get("openai_model") or ""),
                    exported_at=float(obj.get("exported_at")) if isinstance(obj.get("exported_at"), (int, float)) else None,
                )

    if not mapping_by_custom_id:
        raise SystemExit("No mapping rows loaded; nothing to do.")

    # Load results/errors for all shards.
    results_by_cid: dict[str, dict[str, Any]] = {}
    errors_by_cid: dict[str, dict[str, Any]] = {}
    missing_result_files: list[str] = []

    for shard in shards:
        rp = results_dir / f"openai_results_shard{shard:03d}.jsonl"
        ep = results_dir / f"openai_errors_shard{shard:03d}.jsonl"
        if not rp.is_file():
            missing_result_files.append(str(rp))
            continue
        r = _load_jsonl_by_custom_id(rp)
        for cid, row in r.items():
            if cid in results_by_cid:
                raise SystemExit(f"Duplicate custom_id across result shards: {cid}")
            results_by_cid[cid] = row
        e = _load_jsonl_by_custom_id(ep) if ep.is_file() else {}
        for cid, row in e.items():
            if cid in errors_by_cid:
                raise SystemExit(f"Duplicate custom_id across error shards: {cid}")
            errors_by_cid[cid] = row

    if missing_result_files and not bool(args.allow_partial_results):
        preview = "\n".join(missing_result_files[:20])
        more = "" if len(missing_result_files) <= 20 else f"\n(+{len(missing_result_files) - 20} more)"
        raise SystemExit(f"Missing result shard files under {results_dir}:\n{preview}{more}")

    labels_path = out_dir / "cluster_labels.jsonl"
    manifest_path = out_dir / "manifest.jsonl"
    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    err = 0
    t0 = time.perf_counter()
    now_iso = datetime.now(timezone.utc).isoformat()

    with labels_path.open("w", encoding="utf-8") as labels_f, manifest_path.open("w", encoding="utf-8") as man_f:
        for custom_id in sorted(mapping_by_custom_id.keys()):
            m = mapping_by_custom_id[custom_id]
            out_path = clusters_dir / _safe_cluster_filename(m.cluster_id)

            row = results_by_cid.get(custom_id) or errors_by_cid.get(custom_id)
            parsed: ClusterTopicLabel | None = None
            error_obj: dict[str, Any] | None = None

            started_at = (
                datetime.fromtimestamp(float(m.exported_at), tz=timezone.utc).isoformat()
                if m.exported_at
                else now_iso
            )

            if row is None:
                error_obj = {"message": "missing OpenAI batch output for custom_id", "custom_id": custom_id}
            else:
                resp = row.get("response") or {}
                status_code = resp.get("status_code")
                body = resp.get("body") or {}
                if int(status_code or 0) != 200:
                    error_obj = {
                        "message": "openai status_code != 200",
                        "status_code": status_code,
                        "response_error": resp.get("error"),
                        "body": body,
                    }
                else:
                    text = _extract_output_text(body)
                    if not isinstance(text, str) or not text.strip():
                        error_obj = {"message": "empty output_text from OpenAI response body"}
                    else:
                        try:
                            obj = _coerce_json(text)
                            parsed = ClusterTopicLabel.model_validate(obj)

                            # Validate cluster_id matches our request.
                            mo = _CUSTOM_ID_RE.match(custom_id)
                            if not mo:
                                raise ValueError(f"Unexpected custom_id format: {custom_id}")
                            want = int(mo.group(1))
                            if int(parsed.cluster_id) != int(want) or int(parsed.cluster_id) != int(m.cluster_id):
                                raise ValueError(
                                    f"cluster_id mismatch: custom_id={custom_id} mapping={m.cluster_id} parsed={parsed.cluster_id}"
                                )
                        except Exception as exc:  # noqa: BLE001
                            error_obj = {"message": "failed to parse/validate JSON", "exception": str(exc), "raw_text": text[:2000]}

            finished_at = datetime.now(timezone.utc).isoformat()
            duration_ms = int(1000 * (time.perf_counter() - t0))

            if parsed is None:
                err += 1
                out_obj = {
                    "cluster_id": int(m.cluster_id),
                    "topic_name": None,
                    "topic_description": None,
                    "keywords": None,
                    "error": error_obj,
                    "classifier_model": f"openai:{m.openai_model}" if m.openai_model else "openai:unknown",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_ms": duration_ms,
                    "cluster_size_chunks": int(m.cluster_size_chunks),
                    "cluster_weight_total": float(m.cluster_weight_total),
                    "examples": m.examples,
                    "final_prompt_path": m.prompt_path,
                    "final_prompt_sha256": m.final_prompt_sha256,
                }
            else:
                ok += 1
                out_obj = {
                    # Use mapping cluster_id as the source of truth for on-disk identity.
                    # We already validate parsed.cluster_id matches mapping/custom_id; this makes
                    # the output robust against any downstream consumer assuming filename/JSON align.
                    "cluster_id": int(m.cluster_id),
                    "topic_name": parsed.topic_name,
                    "topic_description": parsed.topic_description,
                    "keywords": parsed.keywords,
                    "zoning_relevance": parsed.zoning_relevance,
                    "mechanics": parsed.mechanics.model_dump() if parsed.mechanics else None,
                    "error": None,
                    "classifier_model": f"openai:{m.openai_model}" if m.openai_model else "openai:unknown",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_ms": duration_ms,
                    "cluster_size_chunks": int(m.cluster_size_chunks),
                    "cluster_weight_total": float(m.cluster_weight_total),
                    "examples": m.examples,
                    "final_prompt_path": m.prompt_path,
                    "final_prompt_sha256": m.final_prompt_sha256,
                }

            out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            man_row = {
                "custom_id": custom_id,
                "cluster_id": int(m.cluster_id),
                "output_path": str(out_path),
                "topic_name": out_obj.get("topic_name"),
                "error": out_obj.get("error"),
            }
            man_f.write(json.dumps(man_row, ensure_ascii=False) + "\n")
            labels_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    elapsed = round(time.perf_counter() - t0, 2)
    msg = f"Done. clusters_ok={ok} clusters_error={err} wrote={out_dir} elapsed_s={elapsed}"
    if err and not bool(args.allow_errors):
        raise SystemExit(msg)
    _eprint(msg)


if __name__ == "__main__":
    main()
