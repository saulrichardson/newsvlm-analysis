#!/usr/bin/env python3
"""
Run an end-to-end single-city zoning pipeline:
  1) Issue-type discovery (ontology A/B)
  2) Zoning-text extraction (multiple methods)
  3) Longitudinal panel analysis (multiple prompt variants)
  4) Consolidated summaries + manual review packet

This script is intentionally compositional: it calls existing project scripts,
then materializes per-method single-city run directories for longitudinal runs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
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
                rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run single-city end-to-end zoning pipeline.")
    ap.add_argument(
        "--source-run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Source run dir with panels/<city_key>/issue_texts.jsonl",
    )
    ap.add_argument("--city-key", default="levittown__pa")
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output root (default: <source-run-dir>/single_city_pipeline/<city_key>_<timestamp>)",
    )
    ap.add_argument("--ontology-model", default="gemini:gemini-2.5-flash")
    ap.add_argument("--extraction-model", default="openai:gpt-5-mini")
    ap.add_argument("--longitudinal-model", default="gemini:gemini-2.5-flash")
    ap.add_argument(
        "--longitudinal-max-panel-chars",
        type=int,
        default=850000,
        help="Hard cap passed to run_panel_longitudinal_llm_iteration.py (--max-panel-chars).",
    )
    ap.add_argument(
        "--longitudinal-model-context-tokens",
        type=int,
        default=1_000_000,
        help="Context token budget hint for run_panel_longitudinal_llm_iteration.py.",
    )
    ap.add_argument(
        "--longitudinal-context-utilization",
        type=float,
        default=0.70,
        help="Context utilization ratio for run_panel_longitudinal_llm_iteration.py.",
    )
    ap.add_argument(
        "--longitudinal-chars-per-token",
        type=float,
        default=4.0,
        help="Chars-per-token conversion used by run_panel_longitudinal_llm_iteration.py.",
    )
    ap.add_argument(
        "--extract-methods",
        default="llm_only,llm_hybrid",
        help="Comma-separated extraction methods for scripts/extract_zoning_text_from_panels.py",
    )
    ap.add_argument(
        "--longitudinal-variants",
        default="compact_closed_v2,compact_numeric_quote_v1,panel_lean_numeric_quote_v1,timeline_evidence_v1,hybrid_open_v1",
        help="Comma-separated variants for scripts/run_panel_longitudinal_llm_iteration.py",
    )
    ap.add_argument("--include-raw-baseline", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--longitudinal-run-winner-on-all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether run_panel_longitudinal_llm_iteration.py should execute the post-selection final pass. "
            "When false, this script uses the winner's sample outputs."
        ),
    )
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--gov-env-path",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/.env",
    )
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _build_city_dataset(selected_df: pd.DataFrame, issue_rows: list[dict[str, Any]], city_key: str) -> pd.DataFrame:
    city_sel = selected_df[selected_df["city_key"].astype(str) == str(city_key)].copy()
    if city_sel.empty:
        raise SystemExit(f"city_key not found in selected_panel_issues.csv: {city_key}")
    issue_map = {str(r.get("issue_id")): r for r in issue_rows if _norm(r.get("issue_id"))}
    out_rows: list[dict[str, Any]] = []
    for r in city_sel.itertuples(index=False):
        iid = _norm(getattr(r, "issue_id", ""))
        ir = issue_map.get(iid)
        if not ir:
            continue
        text = _norm(ir.get("text"))
        out_rows.append(
            {
                "issue_id": iid,
                "city_key": _norm(getattr(r, "city_key", "")),
                "city_name": _norm(getattr(r, "city_name", "")),
                "state_abbr": _norm(getattr(r, "state_abbr", "")).lower(),
                "issue_date": _norm(getattr(r, "issue_date", "")),
                "classification_label": _norm(getattr(r, "classification_label", "")),
                "text": text,
                "text_chars_actual": int(len(text)),
                "manual_bucket": "",
                "diagnostic_split": "single_city",
                "region": _norm(getattr(r, "region", "")),
                "urbanicity_proxy": _norm(getattr(r, "urbanicity_proxy", "")),
            }
        )
    out = pd.DataFrame.from_records(out_rows)
    if out.empty:
        raise SystemExit("No issue rows constructed for city dataset.")
    out = out.drop_duplicates(subset=["issue_id"]).sort_values(["issue_date", "issue_id"]).reset_index(drop=True)
    return out


def _make_single_city_run(
    *,
    city_df: pd.DataFrame,
    issue_rows: list[dict[str, Any]],
    extracted_issue_rows: list[dict[str, Any]] | None,
    ontology_df: pd.DataFrame | None,
    city_key: str,
    out_run_dir: Path,
    label_source: str,
) -> dict[str, Any]:
    issue_text_map = {str(r.get("issue_id")): r for r in issue_rows if _norm(r.get("issue_id"))}
    extract_map: dict[str, dict[str, Any]] = {}
    if extracted_issue_rows is not None:
        extract_map = {str(r.get("issue_id")): r for r in extracted_issue_rows if _norm(r.get("issue_id"))}

    ontology_map: dict[str, dict[str, Any]] = {}
    if ontology_df is not None and not ontology_df.empty:
        for r in ontology_df.itertuples(index=False):
            ontology_map[_norm(getattr(r, "issue_id", ""))] = {
                "b_primary_class": _norm(getattr(r, "b_primary_class", "")),
                "b_doc_class": _norm(getattr(r, "b_doc_class", "")),
                "b_legacy_projection_label": _norm(getattr(r, "b_legacy_projection_label", "")),
            }

    panel_issue_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    empty_extract = 0

    for r in city_df.itertuples(index=False):
        iid = _norm(getattr(r, "issue_id", ""))
        base = issue_text_map.get(iid) or {}
        orig_text = _norm(base.get("text"))
        ext = extract_map.get(iid) or {}
        clean_text = _norm(ext.get("zoning_text")) if extracted_issue_rows is not None else orig_text
        if extracted_issue_rows is not None and not clean_text:
            empty_extract += 1

        cls_orig = _norm(getattr(r, "classification_label", ""))
        cls = cls_orig
        if label_source == "ontology_b" and iid in ontology_map:
            proj = _norm(ontology_map[iid].get("b_legacy_projection_label"))
            if proj:
                cls = proj

        row = {
            "issue_id": iid,
            "city_name": _norm(getattr(r, "city_name", "")),
            "state_abbr": _norm(getattr(r, "state_abbr", "")).lower(),
            "city_key": _norm(getattr(r, "city_key", "")),
            "region": _norm(getattr(r, "region", "")),
            "urbanicity_proxy": _norm(getattr(r, "urbanicity_proxy", "")),
            "issue_date": _norm(getattr(r, "issue_date", "")),
            "classification_label": cls,
            "classification_label_original": cls_orig,
            "classification_source": label_source,
            "ontology_b_primary_class": _norm((ontology_map.get(iid) or {}).get("b_primary_class")),
            "ontology_b_doc_class": _norm((ontology_map.get(iid) or {}).get("b_doc_class")),
            "text": clean_text,
            "text_chars": int(len(clean_text)),
            "original_text_chars": int(len(orig_text)),
            "page_count": int(base.get("page_count") or 0),
            "source_mode": "extracted" if extracted_issue_rows is not None else "raw",
        }
        panel_issue_rows.append(row)
        selected_rows.append(
            {
                "city_key": row["city_key"],
                "city_name": row["city_name"],
                "state_abbr": row["state_abbr"],
                "region": row["region"],
                "urbanicity_proxy": row["urbanicity_proxy"],
                "issue_id": row["issue_id"],
                "issue_date": row["issue_date"],
                "classification_label": row["classification_label"],
                "text_chars": row["text_chars"],
                "page_count": row["page_count"],
            }
        )

    panel_issue_rows = sorted(panel_issue_rows, key=lambda x: (x["issue_date"], x["issue_id"]))
    selected_df = pd.DataFrame.from_records(selected_rows).sort_values(["issue_date", "issue_id"]).reset_index(drop=True)

    panel_dir = out_run_dir / "panels" / city_key
    panel_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(panel_dir / "issue_texts.jsonl", panel_issue_rows)
    _write_csv(out_run_dir / "panels" / "selected_panel_issues.csv", selected_df)

    # Minimal compatibility table for scripts expecting contamination path.
    tdir = out_run_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    contam_stub = pd.DataFrame(
        [
            {
                "city_key": city_key,
                "post_filter_noise_share": None,
                "lexical_noise_share": None,
            }
        ]
    )
    _write_csv(tdir / "contamination_panel_summary.csv", contam_stub)

    stats = {
        "city_key": city_key,
        "issue_count": int(len(panel_issue_rows)),
        "empty_extracted_issue_count": int(empty_extract),
        "empty_extracted_issue_share": float(empty_extract / len(panel_issue_rows)) if panel_issue_rows else 0.0,
        "total_text_chars": int(sum(int(r["text_chars"]) for r in panel_issue_rows)),
        "total_original_text_chars": int(sum(int(r["original_text_chars"]) for r in panel_issue_rows)),
        "text_keep_ratio": (
            float(sum(int(r["text_chars"]) for r in panel_issue_rows))
            / float(max(1, sum(int(r["original_text_chars"]) for r in panel_issue_rows)))
        ),
        "run_dir": str(out_run_dir),
    }
    (out_run_dir / "single_city_run_manifest.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def _collect_evidence_issue_ids(parsed_rows: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    if not parsed_rows:
        return out
    rec = parsed_rows[0] if isinstance(parsed_rows[0], dict) else {}
    pj = rec.get("parsed_json") if isinstance(rec.get("parsed_json"), dict) else {}
    fe = pj.get("first_events") if isinstance(pj.get("first_events"), dict) else {}
    fzs = fe.get("first_zoning_signal") if isinstance(fe.get("first_zoning_signal"), dict) else {}
    ffo = fe.get("first_full_ordinance") if isinstance(fe.get("first_full_ordinance"), dict) else {}
    for iid in (_norm(fzs.get("issue_id")), _norm(ffo.get("issue_id"))):
        if iid:
            out.add(iid)

    for key in ("phase_segments", "amendment_waves"):
        rows = pj.get(key) if isinstance(pj.get(key), list) else []
        for r in rows:
            if not isinstance(r, dict):
                continue
            for iid in r.get("evidence_issue_ids") or []:
                s = _norm(iid)
                if s:
                    out.add(s)

    de = pj.get("district_evolution") if isinstance(pj.get("district_evolution"), dict) else {}
    for iid in de.get("evidence_issue_ids") or []:
        s = _norm(iid)
        if s:
            out.add(s)

    nc = pj.get("numeric_constraints_single_family") if isinstance(pj.get("numeric_constraints_single_family"), dict) else {}
    for met in ("lot_size_sqft", "height_ft", "front_setback_ft", "parking_spaces_per_unit"):
        obj = nc.get(met) if isinstance(nc.get(met), dict) else {}
        for iid in obj.get("evidence_issue_ids") or []:
            s = _norm(iid)
            if s:
                out.add(s)
        for q in obj.get("evidence_quotes") or []:
            if isinstance(q, dict):
                s = _norm(q.get("issue_id"))
                if s:
                    out.add(s)
    return out


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_run = Path(args.source_run_dir).expanduser().resolve()

    if _norm(args.output_dir):
        out_root = Path(args.output_dir).expanduser().resolve()
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = source_run / "single_city_pipeline" / f"{args.city_key}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    city_panel_path = source_run / "panels" / args.city_key / "issue_texts.jsonl"
    selected_path = source_run / "panels" / "selected_panel_issues.csv"
    if not city_panel_path.is_file():
        raise SystemExit(f"Missing city panel file: {city_panel_path}")
    if not selected_path.is_file():
        raise SystemExit(f"Missing selected panel file: {selected_path}")

    issue_rows = _iter_jsonl(city_panel_path)
    selected_df = pd.read_csv(selected_path)
    city_df = _build_city_dataset(selected_df, issue_rows, args.city_key)

    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = data_dir / "single_city_dataset.csv"
    _write_csv(dataset_csv, city_df)

    # 1) Ontology discovery A/B.
    ontology_dir = out_root / "ontology_ab"
    cmd_ontology = [
        sys.executable,
        str(repo_root / "scripts" / "run_zoning_ontology_ab_test.py"),
        "--dataset-csv",
        str(dataset_csv),
        "--output-dir",
        str(ontology_dir),
        "--model",
        str(args.ontology_model),
        "--max-concurrency",
        str(int(args.concurrency)),
        "--timeout",
        str(float(args.timeout)),
        "--gov-env-path",
        str(Path(args.gov_env_path).expanduser().resolve()),
    ]
    if bool(args.skip_existing):
        cmd_ontology.append("--skip-existing")
    if bool(args.dry_run):
        cmd_ontology.append("--dry-run")
    _run(cmd_ontology, cwd=repo_root)

    ontology_merged_path = ontology_dir / "ab_outputs_merged.csv"
    ontology_df = pd.read_csv(ontology_merged_path) if ontology_merged_path.is_file() else pd.DataFrame()

    # 2) Zoning text extraction (multi-method).
    extraction_dir = out_root / "extraction"
    cmd_extract = [
        sys.executable,
        str(repo_root / "scripts" / "extract_zoning_text_from_panels.py"),
        "--run-dir",
        str(source_run),
        "--output-dir",
        str(extraction_dir),
        "--city-keys",
        str(args.city_key),
        "--methods",
        str(args.extract_methods),
        "--experiment",
        "--model",
        str(args.extraction_model),
        "--concurrency",
        str(int(args.concurrency)),
        "--timeout",
        str(float(args.timeout)),
        "--gov-env-path",
        str(Path(args.gov_env_path).expanduser().resolve()),
    ]
    if bool(args.skip_existing):
        cmd_extract.append("--skip-existing")
    if bool(args.dry_run):
        cmd_extract.append("--dry-run")
    _run(cmd_extract, cwd=repo_root)

    methods = [x.strip() for x in _norm(args.extract_methods).split(",") if x.strip()]
    longitudinal_targets: list[tuple[str, Path, dict[str, Any]]] = []

    # Raw baseline target.
    if bool(args.include_raw_baseline):
        raw_run = out_root / "prepared_runs" / "raw_baseline"
        raw_stats = _make_single_city_run(
            city_df=city_df,
            issue_rows=issue_rows,
            extracted_issue_rows=None,
            ontology_df=ontology_df,
            city_key=args.city_key,
            out_run_dir=raw_run,
            label_source="ontology_b",
        )
        longitudinal_targets.append(("raw_baseline", raw_run, raw_stats))

    # Extracted targets.
    for method in methods:
        m_path = extraction_dir / method / "issue_zoning_extract.jsonl"
        if not m_path.is_file():
            continue
        m_rows = _iter_jsonl(m_path)
        run_dir = out_root / "prepared_runs" / method
        stats = _make_single_city_run(
            city_df=city_df,
            issue_rows=issue_rows,
            extracted_issue_rows=m_rows,
            ontology_df=ontology_df,
            city_key=args.city_key,
            out_run_dir=run_dir,
            label_source="ontology_b",
        )
        longitudinal_targets.append((method, run_dir, stats))

    if not longitudinal_targets:
        raise SystemExit("No longitudinal targets were prepared.")

    # 3) Longitudinal experiments per target.
    variants = [x.strip() for x in _norm(args.longitudinal_variants).split(",") if x.strip()]
    if not variants:
        raise SystemExit("No longitudinal variants provided.")

    summary_rows: list[dict[str, Any]] = []
    for target_name, run_dir, prep_stats in longitudinal_targets:
        out_dir = out_root / "longitudinal" / target_name
        cmd_long = [
            sys.executable,
            str(repo_root / "scripts" / "run_panel_longitudinal_llm_iteration.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(out_dir),
            "--variants",
            ",".join(variants),
            "--sample-city-keys",
            str(args.city_key),
            "--sample-city-count",
            "1",
            "--model",
            str(args.longitudinal_model),
            "--max-panel-chars",
            str(int(args.longitudinal_max_panel_chars)),
            "--model-context-tokens",
            str(int(args.longitudinal_model_context_tokens)),
            "--context-utilization",
            str(float(args.longitudinal_context_utilization)),
            "--chars-per-token",
            str(float(args.longitudinal_chars_per_token)),
            "--concurrency",
            str(int(args.concurrency)),
            "--timeout",
            str(float(max(args.timeout, 420.0))),
            "--gov-env-path",
            str(Path(args.gov_env_path).expanduser().resolve()),
        ]
        cmd_long.append("--run-winner-on-all" if bool(args.longitudinal_run_winner_on_all) else "--no-run-winner-on-all")
        if bool(args.skip_existing):
            cmd_long.append("--skip-existing")
        if bool(args.dry_run):
            cmd_long.append("--dry-run")
        _run(cmd_long, cwd=repo_root)

        winner = ""
        winner_path = out_dir / "winner_manifest.json"
        if winner_path.is_file():
            wobj = json.loads(winner_path.read_text(encoding="utf-8"))
            winner = _norm(wobj.get("selected_winner_variant"))

        # Prefer final outputs when requested; otherwise use winner sample outputs.
        city_score = pd.DataFrame()
        sample_city_score_path = out_dir / "experiments" / winner / "sample" / "city_scores.csv"
        final_city_score_path = out_dir / "final" / winner / "city_scores.csv"
        if final_city_score_path.is_file():
            city_score = pd.read_csv(final_city_score_path)
        elif sample_city_score_path.is_file():
            city_score = pd.read_csv(sample_city_score_path)

        first_signal = ""
        first_full = ""
        panel_mix = {}
        evidence_issue_ids: set[str] = set()
        if winner:
            parsed_rows: list[dict[str, Any]] = []
            final_parsed_path = out_dir / "final" / winner / "panel_parsed_outputs.jsonl"
            sample_parsed_path = out_dir / "experiments" / winner / "sample" / "panel_parsed_outputs.jsonl"
            if final_parsed_path.is_file():
                parsed_rows = _iter_jsonl(final_parsed_path)
            # Fallback to sample outputs when final outputs are missing/empty/invalid.
            if not parsed_rows or not any(int(r.get("parse_valid") or 0) == 1 for r in parsed_rows):
                parsed_rows = _iter_jsonl(sample_parsed_path)
            evidence_issue_ids = _collect_evidence_issue_ids(parsed_rows)
            if parsed_rows:
                best = None
                for rr in parsed_rows:
                    if int(rr.get("parse_valid") or 0) == 1 and isinstance(rr.get("parsed_json"), dict):
                        best = rr
                        break
                if best is None:
                    best = parsed_rows[0]
                pj = best.get("parsed_json") if isinstance(best.get("parsed_json"), dict) else {}
                fe = pj.get("first_events") if isinstance(pj.get("first_events"), dict) else {}
                fzs = fe.get("first_zoning_signal") if isinstance(fe.get("first_zoning_signal"), dict) else {}
                ffo = fe.get("first_full_ordinance") if isinstance(fe.get("first_full_ordinance"), dict) else {}
                first_signal = _norm(fzs.get("issue_id"))
                first_full = _norm(ffo.get("issue_id"))
                pm = pj.get("panel_mix_counts") if isinstance(pj.get("panel_mix_counts"), dict) else {}
                panel_mix = {
                    "full_ordinance_issues": int(pm.get("full_ordinance_issues") or 0),
                    "amendment_issues": int(pm.get("amendment_issues") or 0),
                    "mention_only_issues": int(pm.get("mention_only_issues") or 0),
                    "non_zoning_or_uncertain_issues": int(pm.get("non_zoning_or_uncertain_issues") or 0),
                }

        score = float(city_score.iloc[0]["overall_quality"]) if not city_score.empty and "overall_quality" in city_score.columns else None
        summary_rows.append(
            {
                "target": target_name,
                "prepared_run_dir": str(run_dir),
                "longitudinal_dir": str(out_dir),
                "winner_variant": winner,
                "overall_quality": score,
                "first_zoning_signal_issue_id": first_signal,
                "first_full_ordinance_issue_id": first_full,
                "evidence_issue_id_count": int(len(evidence_issue_ids)),
                "text_keep_ratio": float(prep_stats.get("text_keep_ratio") or 0.0),
                **panel_mix,
            }
        )

        # Manual packet per target: evidence-linked issue rows with original/extracted snippets.
        issue_map = {_norm(r.get("issue_id")): r for r in issue_rows if _norm(r.get("issue_id"))}
        extract_rows = None
        if target_name != "raw_baseline":
            extract_rows = _iter_jsonl(extraction_dir / target_name / "issue_zoning_extract.jsonl")
        extract_map = {_norm(r.get("issue_id")): r for r in extract_rows or [] if _norm(r.get("issue_id"))}

        packet_rows: list[dict[str, Any]] = []
        for iid in sorted(evidence_issue_ids):
            src = issue_map.get(iid) or {}
            ext = extract_map.get(iid) or {}
            ot = _norm(src.get("text"))
            et = _norm(ext.get("zoning_text")) if target_name != "raw_baseline" else ot
            packet_rows.append(
                {
                    "issue_id": iid,
                    "issue_date": _norm(src.get("issue_date")),
                    "classification_label": _norm(src.get("classification_label")),
                    "original_text_chars": len(ot),
                    "extracted_text_chars": len(et),
                    "original_excerpt": ot[:3000],
                    "extracted_excerpt": et[:3000],
                }
            )
        packet_df = pd.DataFrame.from_records(packet_rows)
        _write_csv(out_root / "manual_review" / target_name / "evidence_issue_packet.csv", packet_df)

    summary_df = pd.DataFrame.from_records(summary_rows).sort_values(["overall_quality", "target"], ascending=[False, True]).reset_index(drop=True)
    _write_csv(out_root / "pipeline_summary.csv", summary_df)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_run_dir": str(source_run),
        "city_key": str(args.city_key),
        "output_dir": str(out_root),
        "ontology_model": str(args.ontology_model),
        "extraction_model": str(args.extraction_model),
        "longitudinal_model": str(args.longitudinal_model),
        "extract_methods": methods,
        "longitudinal_variants": variants,
        "include_raw_baseline": bool(args.include_raw_baseline),
    }
    (out_root / "pipeline_provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"done_single_city_pipeline city={args.city_key} output_dir={out_root}")


if __name__ == "__main__":
    main()
