#!/usr/bin/env python3
"""
Rehydrate OpenAI Batch outputs for clause-cluster labeling into instrument labels.

Inputs:
  - mapping_shardNNN.jsonl (from scripts/export_regulatory_instrument_cluster_labeling_batch_requests.py)
  - openai_results_shardNNN.jsonl
  - openai_errors_shardNNN.jsonl (optional)

Outputs (output_dir):
  - cluster_labels.jsonl
  - invalid_rows.jsonl
  - manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from os.path import expanduser
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from newsvlm.pipeline import _coerce_json
from newsvlm.regulatory_motive_schema import REGULATORY_MOTIVES


def _extract_output_text(body: dict[str, Any]) -> str:
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


class InstrumentClusterLabel(BaseModel):
    cluster_id: int
    instrument_name: str = Field(min_length=1, max_length=80)
    instrument_description: str = Field(min_length=1, max_length=900)
    keywords: list[str] = Field(min_length=1)
    motive_hints: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("keywords")
    @classmethod
    def _validate_keywords(cls, v: list[str]) -> list[str]:
        clean = [str(x).strip() for x in (v or []) if str(x).strip()]
        if len(clean) < 3:
            raise ValueError("keywords must include at least 3 non-empty strings")
        # Deduplicate while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for k in clean:
            lk = k.lower()
            if lk in seen:
                continue
            seen.add(lk)
            out.append(k)
        return out[:25]

    @field_validator("motive_hints")
    @classmethod
    def _validate_motive_hints(cls, v: list[str]) -> list[str]:
        clean = []
        for x in v or []:
            s = str(x).strip()
            if not s:
                continue
            if s not in set(REGULATORY_MOTIVES):
                continue
            clean.append(s)
        # Deduplicate
        seen: set[str] = set()
        out: list[str] = []
        for s in clean:
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out[:4]


@dataclass(frozen=True)
class MappingRow:
    custom_id: str
    cluster_id: int
    meta: dict[str, Any]


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
    ap = argparse.ArgumentParser(description="Rehydrate clause-cluster instrument labels from OpenAI Batch results.")
    ap.add_argument("--request-dir", required=True, help="Directory containing mapping_shardNNN.jsonl")
    ap.add_argument("--results-dir", required=True, help="Directory containing openai_results_shardNNN.jsonl")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs")
    ap.add_argument("--allow-partial-results", action="store_true")
    ap.add_argument("--allow-errors", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    request_dir = Path(expanduser(args.request_dir)).resolve()
    results_dir = Path(expanduser(args.results_dir)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = _discover_shards(request_dir)
    if not shards:
        raise SystemExit(f"No mapping_shard*.jsonl found under {request_dir}")

    mapping_by_custom_id: dict[str, MappingRow] = {}
    for shard in shards:
        mp = request_dir / f"mapping_shard{shard:03d}.jsonl"
        if not mp.is_file():
            raise SystemExit(f"Missing mapping file: {mp}")
        for obj in _iter_jsonl(mp):
            cid = obj.get("custom_id") or obj.get("id")
            cluster_id = obj.get("cluster_id")
            if not isinstance(cid, str) or not cid.strip():
                raise SystemExit(f"Bad mapping row in {mp}: missing custom_id")
            if not isinstance(cluster_id, int):
                raise SystemExit(f"Bad mapping row in {mp}: missing cluster_id")
            cid = cid.strip()
            if cid in mapping_by_custom_id:
                raise SystemExit(f"Duplicate custom_id across mapping shards: {cid}")
            mapping_by_custom_id[cid] = MappingRow(custom_id=cid, cluster_id=int(cluster_id), meta=obj)

    results_by_cid: dict[str, dict[str, Any]] = {}
    errors_by_cid: dict[str, dict[str, Any]] = {}
    for shard in shards:
        rp = results_dir / f"openai_results_shard{shard:03d}.jsonl"
        ep = results_dir / f"openai_errors_shard{shard:03d}.jsonl"
        if not rp.is_file():
            if bool(args.allow_partial_results):
                continue
            raise SystemExit(f"Missing results file: {rp}")
        results_by_cid.update(_load_jsonl_by_custom_id(rp))
        errors_by_cid.update(_load_jsonl_by_custom_id(ep))

    out_labels = out_dir / "cluster_labels.jsonl"
    out_invalid = out_dir / "invalid_rows.jsonl"
    out_manifest = out_dir / "manifest.jsonl"
    stamp = datetime.now(timezone.utc).isoformat()

    ok = 0
    bad = 0

    with out_labels.open("w", encoding="utf-8") as f_ok, out_invalid.open("w", encoding="utf-8") as f_bad, out_manifest.open(
        "w", encoding="utf-8"
    ) as f_m:
        for cid, m in mapping_by_custom_id.items():
            res = results_by_cid.get(cid)
            err = errors_by_cid.get(cid)
            if res is None:
                bad += 1
                f_bad.write(
                    json.dumps({"custom_id": cid, "cluster_id": m.cluster_id, "error": "missing_result", "has_error_line": bool(err)}, ensure_ascii=False)
                    + "\n"
                )
                continue

            resp = res.get("response") or {}
            body = (resp.get("body") or {}) if isinstance(resp, dict) else {}
            text = _extract_output_text(body) if isinstance(body, dict) else ""
            if not text.strip():
                bad += 1
                f_bad.write(json.dumps({"custom_id": cid, "cluster_id": m.cluster_id, "error": "empty_output_text"}, ensure_ascii=False) + "\n")
                continue

            try:
                parsed = _coerce_json(text)
                lab = InstrumentClusterLabel.model_validate(parsed)
                if int(lab.cluster_id) != int(m.cluster_id):
                    raise ValueError(f"cluster_id mismatch: model={lab.cluster_id} expected={m.cluster_id}")
            except Exception as exc:  # noqa: BLE001
                bad += 1
                f_bad.write(
                    json.dumps(
                        {"custom_id": cid, "cluster_id": m.cluster_id, "error": "validation_failed", "message": str(exc), "raw_text": text[:4000]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            out_row = {
                "cluster_id": int(lab.cluster_id),
                "instrument_name": lab.instrument_name,
                "instrument_description": lab.instrument_description,
                "keywords": lab.keywords,
                "motive_hints": lab.motive_hints,
                "notes": lab.notes,
                "custom_id": cid,
                "rehydrated_at": stamp,
            }
            f_ok.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_m.write(json.dumps({"cluster_id": int(lab.cluster_id), "output_path": str(out_labels), "custom_id": cid, "rehydrated_at": stamp}, ensure_ascii=False) + "\n")
            ok += 1

    if bad:
        msg = f"Rehydration had failures: ok={ok} bad={bad} out={out_dir}"
        if not bool(args.allow_errors):
            raise SystemExit(msg)
        print(msg, file=sys.stderr)

    print(f"Done. ok={ok} bad={bad} out={out_dir}")


if __name__ == "__main__":
    main()
