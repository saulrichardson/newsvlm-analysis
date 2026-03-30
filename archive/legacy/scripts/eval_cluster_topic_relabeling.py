#!/usr/bin/env python3
"""
Evaluate improved cluster topic labels by relabeling a sampled subset of *existing* clusters.

This script is intentionally "LLM-only" labeling:
  - It reuses the already-exported cluster examples from mapping_shard*.jsonl (no TF‑IDF keywords).
  - It does NOT inject heuristic mechanics tags.
  - It uses an OpenAI model via agent-gateway to generate more mechanism-oriented topic labels.

Inputs (local copies of Greene artifacts)
  <input_root>/<run_name>/
    - mapping_shard000.jsonl           (copied from cluster_topic_labels/requests/)
    - cluster_labels_old.jsonl         (copy of prior cluster_topic_labels/outputs/cluster_labels.jsonl)

Outputs
  <output_root>/<run_name>/
    - requests/ openai_requests_shard000.jsonl + mapping_shard000.jsonl
    - results/  openai_results_shard000.jsonl + openai_errors_shard000.jsonl
    - outputs/  cluster_labels.jsonl + per-cluster jsons
    - report.md (old vs new comparison)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from newsvlm_analysis.cluster_topic_label_schema import SCHEMA_VERSIONS, openai_text_config


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            out.append(json.loads(line))
    if not out:
        raise SystemExit(f"Empty JSONL: {path}")
    return out


def _load_mapping_by_cluster_id(mapping_path: Path) -> dict[int, dict[str, Any]]:
    rows = _load_jsonl(mapping_path)
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        cid = r.get("cluster_id")
        if cid is None:
            raise SystemExit(f"Bad mapping row in {mapping_path}: missing cluster_id")
        cid_i = int(cid)
        if cid_i in out:
            raise SystemExit(f"Duplicate cluster_id in mapping: {cid_i}")
        out[cid_i] = r
    return out


def _load_labels_by_cluster_id(labels_path: Path) -> dict[int, dict[str, Any]]:
    rows = _load_jsonl(labels_path)
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        cid = r.get("cluster_id")
        if cid is None:
            raise SystemExit(f"Bad labels row in {labels_path}: missing cluster_id")
        cid_i = int(cid)
        if cid_i in out:
            raise SystemExit(f"Duplicate cluster_id in labels: {cid_i}")
        out[cid_i] = r
    return out


_MECH_CUE_WORDS = {
    # dimensional standards
    "setback",
    "setbacks",
    "yard",
    "yards",
    "lot",
    "lots",
    "lot area",
    "minimum lot",
    "height",
    "density",
    "dwelling",
    "units",
    "acre",
    "parking",
    "loading",
    "sign",
    "signs",
    "billboard",
    "billboards",
    # procedural instruments
    "variance",
    "appeal",
    "appeals",
    "conditional",
    "special exception",
    "permit",
    "permits",
    "public hearing",
    "hearing",
    "notice",
    "rezoning",
    "amendment",
    "amendments",
    "map",
    # zoning structures
    "district",
    "districts",
    "nonconforming",
    "subdivision",
    "plat",
    "pud",
    "planned unit",
    "site plan",
    # common zoning subtopics
    "floodplain",
    "airport",
    "adult",
    "alcohol",
}


def _looks_weak_zoning_label(topic_name: str) -> bool:
    n = (topic_name or "").strip().lower()
    n = re.sub(r"\s+", " ", n)
    if "zoning" not in n:
        return False
    for cue in _MECH_CUE_WORDS:
        if cue in n:
            return False
    return True


def _build_cluster_prompt(*, prompt_text: str, cluster_id: int, examples: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(prompt_text.strip())
    lines.append("")
    lines.append(f"Cluster id: {cluster_id}")
    lines.append("")
    lines.append("Examples (each is an excerpt from a chunk in this cluster):")
    for i, ex in enumerate(examples, start=1):
        lines.append("")
        lines.append(f"[EXAMPLE {i}] issue_id={ex.get('issue_id')} chunk_id={ex.get('chunk_id')}")
        lines.append(str(ex.get("excerpt") or "").strip())
    lines.append("")
    lines.append("Return the JSON now.")
    return "\n".join(lines).strip()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class _RunSpec:
    name: str
    mapping_path: Path
    old_labels_path: Path


def _discover_runs(input_root: Path) -> list[_RunSpec]:
    runs: list[_RunSpec] = []
    if not input_root.is_dir():
        raise SystemExit(f"--input-root is not a directory: {input_root}")
    for p in sorted(input_root.iterdir()):
        if not p.is_dir():
            continue
        mapping = p / "mapping_shard000.jsonl"
        old_labels = p / "cluster_labels_old.jsonl"
        if mapping.is_file() and old_labels.is_file():
            runs.append(_RunSpec(name=p.name, mapping_path=mapping, old_labels_path=old_labels))
    if not runs:
        raise SystemExit(
            f"No runs discovered under {input_root} "
            "(expected subdirs with mapping_shard000.jsonl + cluster_labels_old.jsonl)"
        )
    return runs


def _select_cluster_ids(
    *,
    mapping_by_cid: dict[int, dict[str, Any]],
    old_by_cid: dict[int, dict[str, Any]],
    sample_n: int,
    selection_mode: str,
) -> list[int]:
    cids = sorted(set(mapping_by_cid.keys()) & set(old_by_cid.keys()))
    if not cids:
        raise SystemExit("No overlapping cluster_ids between mapping and old labels")

    if selection_mode != "weak_zoning":
        raise SystemExit(f"Unknown selection mode: {selection_mode}")

    weak: list[int] = []
    for cid in cids:
        name = str(old_by_cid[cid].get("topic_name") or "")
        if _looks_weak_zoning_label(name):
            weak.append(cid)

    # Prefer larger clusters so the evaluation focuses on substantive topics.
    weak_sorted = sorted(
        weak,
        key=lambda cid: (
            -int(old_by_cid[cid].get("cluster_size_chunks") or 0),
            -float(old_by_cid[cid].get("cluster_weight_total") or 0.0),
            cid,
        ),
    )
    selected = weak_sorted[:sample_n]

    if len(selected) < sample_n:
        # Backfill with any zoning-labeled clusters if weak zoning is scarce.
        zoning = [
            cid
            for cid in cids
            if cid not in set(selected) and "zoning" in str(old_by_cid[cid].get("topic_name") or "").lower()
        ]
        selected.extend(zoning[: max(0, sample_n - len(selected))])

    return selected


def _render_excerpt_preview(excerpt: str, max_chars: int = 420) -> str:
    s = (excerpt or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_chars:
        return s[: max_chars - 1] + "…"
    return s


def _write_report(
    *,
    out_path: Path,
    run_name: str,
    prompt_path: Path,
    gateway_model: str,
    selected: list[int],
    old_by_cid: dict[int, dict[str, Any]],
    new_by_cid: dict[int, dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(selected)
    weak_old = sum(1 for cid in selected if _looks_weak_zoning_label(str(old_by_cid[cid].get("topic_name") or "")))
    weak_new = sum(1 for cid in selected if _looks_weak_zoning_label(str(new_by_cid[cid].get("topic_name") or "")))

    # Optional v2 mechanics summary.
    action_tags: set[str] = set()
    dimension_tags: set[str] = set()
    decision_body_tags: set[str] = set()
    instrument_tags: set[str] = set()
    district_tokens: set[str] = set()
    zoning_relevance_counts: dict[str, int] = {}
    for cid in selected:
        row = new_by_cid.get(cid) or {}
        zr = str(row.get("zoning_relevance") or "").strip()
        if zr:
            zoning_relevance_counts[zr] = zoning_relevance_counts.get(zr, 0) + 1
        mech = row.get("mechanics") or {}
        if isinstance(mech, dict):
            action_tags.update([str(x) for x in (mech.get("action_tags") or []) if str(x).strip()])
            dimension_tags.update([str(x) for x in (mech.get("dimension_tags") or []) if str(x).strip()])
            decision_body_tags.update([str(x) for x in (mech.get("decision_body_tags") or []) if str(x).strip()])
            instrument_tags.update([str(x) for x in (mech.get("instrument_tags") or []) if str(x).strip()])
            district_tokens.update([str(x) for x in (mech.get("district_tokens") or []) if str(x).strip()])

    lines: list[str] = []
    lines.append(f"# Cluster topic relabeling eval — {run_name}")
    lines.append("")
    lines.append(f"- Prompt: `{prompt_path}`")
    lines.append(f"- Gateway model: `{gateway_model}`")
    lines.append(f"- Clusters relabeled: {total}")
    lines.append(f"- Weak-zoning labels (old → new): {weak_old} → {weak_new}")
    if zoning_relevance_counts:
        zr_bits = ", ".join(f"{k}={v}" for k, v in sorted(zoning_relevance_counts.items()))
        lines.append(f"- zoning_relevance counts: {zr_bits}")
    if action_tags or dimension_tags or decision_body_tags or instrument_tags:
        lines.append(
            "- mechanics tag coverage: "
            f"actions={len(action_tags)} dimensions={len(dimension_tags)} bodies={len(decision_body_tags)} "
            f"instruments={len(instrument_tags)} district_tokens={len(district_tokens)}"
        )
    lines.append("")

    for cid in selected:
        old = old_by_cid.get(cid, {})
        new = new_by_cid.get(cid, {})
        lines.append(f"## Cluster {cid}")
        lines.append("")
        lines.append(f"- size_chunks: {old.get('cluster_size_chunks')}  weight_total: {old.get('cluster_weight_total')}")
        lines.append(f"- old: **{old.get('topic_name')}**")
        lines.append(f"- new: **{new.get('topic_name')}**")
        lines.append("")
        if old.get("topic_description"):
            lines.append(f"Old: {old.get('topic_description')}")
        if new.get("topic_description"):
            lines.append(f"New: {new.get('topic_description')}")
        lines.append("")

        zr = new.get("zoning_relevance")
        mech = new.get("mechanics")
        if zr is not None or mech is not None:
            lines.append(f"- zoning_relevance: {zr}")
            if isinstance(mech, dict):
                for key in ["action_tags", "dimension_tags", "decision_body_tags", "instrument_tags", "district_tokens"]:
                    vals = mech.get(key) if isinstance(mech.get(key), list) else []
                    if vals:
                        lines.append(f"- {key}: " + ", ".join(str(v) for v in vals))
                phrases = mech.get("mechanism_phrases") if isinstance(mech.get("mechanism_phrases"), list) else []
                if phrases:
                    show = "; ".join(str(p) for p in phrases[:10])
                    lines.append(f"- mechanism_phrases: {show}")
            lines.append("")

        examples = list(new.get("examples") or old.get("examples") or [])
        if examples:
            lines.append("Example excerpts:")
            for ex in examples[:2]:
                preview = _render_excerpt_preview(str(ex.get("excerpt") or ""))
                lines.append(f"- {ex.get('issue_id')} {ex.get('chunk_id')}: {preview}")
            lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_cmd(cmd: list[str]) -> None:
    _eprint("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Relabel a sampled subset of existing cluster topics and write a comparison report."
    )
    ap.add_argument(
        "--input-root",
        required=True,
        help="Directory containing per-run subdirs with mapping_shard000.jsonl + cluster_labels_old.jsonl",
    )
    ap.add_argument(
        "--output-root",
        required=True,
        help="Directory to write relabeling requests/results/outputs + report.md into",
    )
    ap.add_argument(
        "--prompt-path",
        default="prompts/cluster_topic_labeler_prompt_text.txt",
        help="Prompt text file for the relabeler",
    )
    ap.add_argument(
        "--schema-version",
        choices=list(SCHEMA_VERSIONS),
        default="v1",
        help="Cluster label JSON schema version (v2 adds mechanics tags).",
    )
    ap.add_argument(
        "--openai-model",
        default="gpt-5-mini",
        help="Model name to record in request mapping (gateway uses --gateway-model).",
    )
    ap.add_argument(
        "--gateway-model",
        default="openai:gpt-5-mini",
        help="Gateway model to use (provider:model)",
    )
    ap.add_argument("--sample-per-run", type=int, default=10, help="How many clusters to relabel per run")
    ap.add_argument(
        "--selection-mode",
        choices=["weak_zoning"],
        default="weak_zoning",
        help="How to pick clusters for evaluation",
    )
    ap.add_argument(
        "--max-examples",
        type=int,
        default=6,
        help="Max excerpts to include per cluster prompt (truncates existing mapping examples)",
    )
    ap.add_argument("--max-concurrency", type=int, default=4, help="Concurrency for gateway runner")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt_path).expanduser().resolve()
    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_path}")
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if not prompt_text.strip():
        raise SystemExit(f"Prompt file is empty: {prompt_path}")

    openai_model = str(args.openai_model)
    gateway_model = str(args.gateway_model)
    schema_version = str(args.schema_version)

    sample_n = int(args.sample_per_run)
    if sample_n <= 0:
        raise SystemExit("--sample-per-run must be > 0")
    max_examples = int(args.max_examples)
    if max_examples <= 0:
        raise SystemExit("--max-examples must be > 0")

    runs = _discover_runs(input_root)
    _eprint(f"Discovered {len(runs)} run(s): {[r.name for r in runs]}")

    for run in runs:
        _eprint(f"\n=== Relabeling run: {run.name} ===")
        mapping_by_cid = _load_mapping_by_cluster_id(run.mapping_path)
        old_by_cid = _load_labels_by_cluster_id(run.old_labels_path)

        selected = _select_cluster_ids(
            mapping_by_cid=mapping_by_cid,
            old_by_cid=old_by_cid,
            sample_n=sample_n,
            selection_mode=str(args.selection_mode),
        )
        if not selected:
            raise SystemExit(f"No clusters selected for run {run.name}")

        out_run = output_root / run.name
        req_dir = out_run / "requests"
        res_dir = out_run / "results"
        out_dir = out_run / "outputs"
        req_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Export a fresh set of batch-like requests using the existing mapping examples.
        requests: list[dict[str, Any]] = []
        mappings: list[dict[str, Any]] = []
        text_cfg = openai_text_config("json_schema", schema_version=schema_version)
        if text_cfg is None:
            raise SystemExit("Internal error: missing OpenAI text config")

        for cid in selected:
            m = mapping_by_cid[cid]
            examples = list(m.get("examples") or [])[:max_examples]
            prompt = _build_cluster_prompt(prompt_text=prompt_text, cluster_id=cid, examples=examples)
            prompt_sha = _sha256(prompt)

            custom_id = f"cluster::{cid}"
            body: dict[str, Any] = {
                "model": openai_model,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                "text": text_cfg,
            }
            requests.append({"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body})

            mappings.append(
                {
                    "id": custom_id,
                    "custom_id": custom_id,
                    "cluster_id": int(cid),
                    "cluster_size_chunks": int(m.get("cluster_size_chunks") or 0),
                    "cluster_weight_total": float(m.get("cluster_weight_total") or 0.0),
                    "examples": examples,
                    "prompt_path": str(prompt_path),
                    "final_prompt_sha256": prompt_sha,
                    "openai_model": openai_model,
                    "schema_version": schema_version,
                    "exported_at": time.time(),
                    # Keep provenance so the report can point back to Greene sources.
                    "source_run_root": str(m.get("run_root") or ""),
                    "source_prompt_path": str(m.get("prompt_path") or ""),
                    "source_prompt_sha256": str(m.get("prompt_text_sha256") or ""),
                }
            )

        _write_jsonl(req_dir / "openai_requests_shard000.jsonl", requests)
        _write_jsonl(req_dir / "mapping_shard000.jsonl", mappings)

        # Run requests through the local agent-gateway.
        _run_cmd(
            [
                sys.executable,
                "scripts/run_openai_requests_via_gateway.py",
                "--request-dir",
                str(req_dir),
                "--output-dir",
                str(res_dir),
                "--model",
                gateway_model,
                "--max-concurrency",
                str(int(args.max_concurrency)),
            ]
        )

        # Rehydrate and validate into cluster_labels.jsonl.
        _run_cmd(
            [
                sys.executable,
                "scripts/rehydrate_cluster_topic_labels_openai_batch_results.py",
                "--request-dir",
                str(req_dir),
                "--results-dir",
                str(res_dir),
                "--output-dir",
                str(out_dir),
            ]
        )

        # Write an easy-to-skim comparison report.
        new_by_cid = _load_labels_by_cluster_id(out_dir / "cluster_labels.jsonl")
        report_path = out_run / "report.md"
        _write_report(
            out_path=report_path,
            run_name=run.name,
            prompt_path=prompt_path,
            gateway_model=gateway_model,
            selected=selected,
            old_by_cid=old_by_cid,
            new_by_cid=new_by_cid,
        )

        _eprint(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
