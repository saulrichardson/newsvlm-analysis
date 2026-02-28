#!/usr/bin/env python3
"""
Precision-first recovery pilot runner (v4, 8-newspaper target).

Implements:
- Round 0 baseline lock
- Round 1 axis screening
- Round 2 top-3 end-to-end candidates
- Round 3 robustness check for winner (alternate model path)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


FULL_CLASSES = {"full_code_comprehensive", "full_code_recodification", "full_code_partial_enacted"}
AMEND_CLASSES = {"amendment_substantial", "amendment_targeted"}
PROPOSED_CLASSES = {"full_code_proposed", "full_code_partial_proposed", "amendment_proposed"}
NOTICE_CLASSES = {"zoning_notice_adoption", "zoning_notice_procedural"}
LAW_CLASSES = FULL_CLASSES | AMEND_CLASSES | PROPOSED_CLASSES | NOTICE_CLASSES
NONLAW_CLASSES = {"not_verbatim_law_text", "non_zoning", "building_code_non_zoning"}
ENACTED_STATUSES = {"enacted_complete", "enacted_partial_installment"}
PROPOSED_STATUSES = {"proposed_full_text", "proposed_partial_installment"}

DEFAULT_PILOT_NEWSPAPERS = [
    "brownsville-herald",
    "alamogordo-daily-news",
    "bucks-county-courier-times",
    "benton-harbor-news-palladium",
    "brookfield-citizen",
    "bountiful-davis-county-clipper",
    "albion-evening-recorder",
    "cazenovia-republican",
]


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run precision-first ordinance extraction recovery pilot (v4).")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
        help="Source run directory containing panels/<city_key>/issue_texts.jsonl.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output root for this pilot run.",
    )
    ap.add_argument(
        "--newspapers",
        default=",".join(DEFAULT_PILOT_NEWSPAPERS),
        help="Comma-separated newspaper slugs for pilot.",
    )
    ap.add_argument("--model-primary", default="openai:gpt-5")
    ap.add_argument("--model-secondary", default="gemini:gemini-2.5-flash")
    ap.add_argument("--round1-issue-limit", type=int, default=140, help="Issue cap for Round 1 axis-screening runs.")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--max-panel-chars",
        type=int,
        default=0,
        help="Optional hard cap passed to panel longitudinal stage. 0 means no local cap (no clipping). If a positive cap is exceeded, the stage fails.",
    )
    ap.add_argument(
        "--gateway-runner",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/experimental/scripts/run_openai_requests_via_gateway.py",
    )
    ap.add_argument(
        "--gateway-pythonpath",
        default="/Users/saulrichardson/projects/newspapers/old-ocr/newspaper-parsing-local/agent-gateway/src",
    )
    ap.add_argument("--gov-env-path", default=".env")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _score_run(run_root: Path) -> dict[str, Any]:
    cls = pd.read_csv(run_root / "classification" / "issue_labels_v3.csv")
    ev = pd.read_csv(run_root / "events_status" / "ordinance_event_clusters_status_v3.csv")
    asn = pd.read_csv(run_root / "events_status" / "issue_event_assignments_status_v3.csv")
    clean = pd.read_csv(run_root / "cleaning" / "clean_zoning_text_flat.csv")

    for df in (cls, ev, asn, clean):
        for c in ("newspaper_slug", "issue_id", "ordinance_event_id", "event_type", "event_status", "issue_class", "qualification_decision"):
            if c in df.columns:
                df[c] = df[c].astype(str)

    full_ev = ev[ev["event_type"] == "full_ordinance"].copy()
    amend_ev = ev[ev["event_type"] == "amendment"].copy()
    enacted_full = full_ev[full_ev["event_status"].isin(list(ENACTED_STATUSES))].copy()

    # Full precision proxy: proposal leakage + failed qualifications.
    full_bad = full_ev["event_status"].isin(list(PROPOSED_STATUSES))
    if "qualification_decision" in full_ev.columns:
        full_bad = full_bad | (full_ev["qualification_decision"].astype(str) != "pass")
    full_precision_proxy = 1.0 - (float(full_bad.sum()) / max(1.0, float(len(full_ev))))

    # Amendment precision proxy via member-class dominance.
    amend_precision_proxy = math.nan
    if not amend_ev.empty:
        tmp = asn[asn["event_type"] == "amendment"].copy()
        rows = []
        for (slug, eid), g in tmp.groupby(["newspaper_slug", "ordinance_event_id"], dropna=False):
            cls_set = g["issue_class"].astype(str).tolist()
            amend_like = sum(1 for x in cls_set if x in AMEND_CLASSES)
            full_like = sum(1 for x in cls_set if x in FULL_CLASSES)
            proposal_like = sum(1 for x in cls_set if x in PROPOSED_CLASSES)
            good = (amend_like >= max(full_like, proposal_like, 1))
            rows.append({"good": int(good)})
        t = pd.DataFrame.from_records(rows)
        amend_precision_proxy = float(t["good"].mean()) if not t.empty else math.nan
    if not math.isfinite(amend_precision_proxy):
        amend_precision_proxy = 0.0

    # Legal text purity in enacted buckets.
    enacted_members = asn[
        (asn["event_type"].isin(["full_ordinance", "amendment"]))
        & (asn["event_status"].isin(list(ENACTED_STATUSES)))
    ].copy()
    legal_text_purity = 1.0
    if not enacted_members.empty:
        bad = enacted_members["issue_class"].isin(list(NONLAW_CLASSES))
        legal_text_purity = 1.0 - float(bad.sum()) / max(1.0, float(len(enacted_members)))

    # Collision rate in enacted buckets.
    clean_map = {str(r.issue_id): _norm(r.clean_zoning_text) for r in clean.itertuples(index=False)}
    enacted_ev_ids = enacted_members[["newspaper_slug", "ordinance_event_id"]].drop_duplicates()
    hashes: list[dict[str, Any]] = []
    for r in enacted_ev_ids.itertuples(index=False):
        g = enacted_members[
            (enacted_members["newspaper_slug"] == r.newspaper_slug)
            & (enacted_members["ordinance_event_id"] == r.ordinance_event_id)
        ].sort_values(["issue_date", "issue_id"])
        txt = "\n".join([clean_map.get(_norm(x), "") for x in g["issue_id"].astype(str).tolist()]).strip()
        normed = "".join(ch for ch in txt.lower() if ch.isalnum() or ch.isspace())
        normed = " ".join(normed.split())
        if not normed:
            continue
        h = hashlib.sha256(normed.encode("utf-8")).hexdigest()
        bucket = "full" if g["event_type"].iloc[0] == "full_ordinance" else "amendment"
        hashes.append(
            {
                "newspaper_slug": _norm(r.newspaper_slug),
                "ordinance_event_id": _norm(r.ordinance_event_id),
                "bucket": bucket,
                "h": h,
            }
        )
    hdf = pd.DataFrame.from_records(hashes)
    collision_rate = 0.0
    if not hdf.empty:
        grp = hdf.groupby(["newspaper_slug", "bucket", "h"], dropna=False)["ordinance_event_id"].nunique().reset_index(name="n")
        coll = grp[grp["n"] > 1]
        collided_ids = set()
        for r in coll.itertuples(index=False):
            sub = hdf[(hdf["newspaper_slug"] == r.newspaper_slug) & (hdf["bucket"] == r.bucket) & (hdf["h"] == r.h)]
            collided_ids |= set(sub["ordinance_event_id"].astype(str).tolist())
        collision_rate = float(len(collided_ids) / max(1.0, float(hdf["ordinance_event_id"].nunique())))

    score = (
        0.45 * max(0.0, min(1.0, full_precision_proxy))
        + 0.25 * max(0.0, min(1.0, amend_precision_proxy))
        + 0.15 * (1.0 - max(0.0, min(1.0, collision_rate)))
        + 0.15 * max(0.0, min(1.0, legal_text_purity))
    )

    return {
        "run_root": str(run_root),
        "score": float(score),
        "full_precision_proxy": float(full_precision_proxy),
        "amend_precision_proxy": float(amend_precision_proxy),
        "dedup_collision_rate": float(collision_rate),
        "legal_text_purity": float(legal_text_purity),
        "full_event_count": int(len(full_ev)),
        "enacted_full_event_count": int(len(enacted_full)),
        "amend_event_count": int(len(amend_ev)),
        "proposal_as_full_count": int((full_ev["event_status"].isin(list(PROPOSED_STATUSES))).sum()),
    }


def _build_exhaustive_validation_packet(run_root: Path) -> None:
    mv = run_root / "manual_validation"
    mv.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(run_root / "events_status" / "ordinance_event_clusters_status_v3.csv")
    asn = pd.read_csv(run_root / "events_status" / "issue_event_assignments_status_v3.csv")
    cls = pd.read_csv(run_root / "classification" / "issue_labels_v3.csv")
    clean = pd.read_csv(run_root / "cleaning" / "clean_zoning_text_flat.csv")

    for df in (ev, asn, cls, clean):
        for c in ("newspaper_slug", "issue_id", "ordinance_event_id", "event_type", "event_status", "issue_class", "issue_date"):
            if c in df.columns:
                df[c] = df[c].astype(str)

    clean_map = {str(r.issue_id): int(len(_norm(r.clean_zoning_text))) for r in clean.itertuples(index=False)}

    rows: list[dict[str, Any]] = []
    target = ev[ev["event_type"].isin(["full_ordinance", "amendment"])].copy()
    for r in target.itertuples(index=False):
        slug = _norm(r.newspaper_slug)
        eid = _norm(r.ordinance_event_id)
        mem = asn[(asn["newspaper_slug"] == slug) & (asn["ordinance_event_id"] == eid)].sort_values(["issue_date", "issue_id"])
        issue_ids = mem["issue_id"].astype(str).tolist()
        classes = mem["issue_class"].astype(str).tolist() if "issue_class" in mem.columns else []
        rows.append(
            {
                "newspaper_slug": slug,
                "ordinance_event_id": eid,
                "event_type": _norm(r.event_type),
                "event_status": _norm(getattr(r, "event_status", "")),
                "qualification_decision": _norm(getattr(r, "qualification_decision", "")),
                "issue_count": int(len(set(issue_ids))),
                "first_issue_date": _norm(getattr(r, "first_issue_date", "")),
                "last_issue_date": _norm(getattr(r, "last_issue_date", "")),
                "member_issue_ids": "|".join(issue_ids),
                "member_issue_classes": "|".join(classes),
                "member_clean_chars_total": int(sum(clean_map.get(iid, 0) for iid in issue_ids)),
                "review_label_correct": "",
                "review_status_correct": "",
                "review_completeness_correct": "",
                "review_dedup_correct": "",
                "review_clean_text_fidelity": "",
                "review_notes": "",
            }
        )
    out = pd.DataFrame.from_records(rows).sort_values(["event_type", "newspaper_slug", "first_issue_date", "ordinance_event_id"])
    _write_csv(mv / "exhaustive_full_amend_validation_log.csv", out)

    summary = (
        out.groupby(["event_type", "event_status", "qualification_decision"], dropna=False)["ordinance_event_id"]
        .count()
        .reset_index(name="event_count")
        .sort_values(["event_type", "event_status", "qualification_decision"])
    )
    _write_csv(mv / "exhaustive_full_amend_validation_summary.csv", summary)


def _run_pipeline_once(
    *,
    repo_root: Path,
    run_dir: Path,
    output_root: Path,
    newspapers: list[str],
    page_tags_jsonl: Path,
    law_presence_jsonl: Path,
    model_primary: str,
    model_secondary: str,
    cleaning_variant: str,
    strict_narrative_mode: bool,
    classification_variant: str,
    strict_contradiction_mode: bool,
    dedup_variant: str,
    use_collision_recheck: bool,
    run_longitudinal: bool,
    issue_limit: int,
    concurrency: int,
    timeout: float,
    max_panel_chars: int,
    gateway_runner: Path,
    gateway_pythonpath: Path,
    gov_env_path: Path,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    py = sys.executable
    output_root.mkdir(parents=True, exist_ok=True)
    shared = [
        "--run-dir",
        str(run_dir),
        "--newspapers",
        ",".join(newspapers),
        "--concurrency",
        str(int(concurrency)),
        "--timeout",
        str(float(timeout)),
        "--gateway-runner",
        str(gateway_runner),
        "--gateway-pythonpath",
        str(gateway_pythonpath),
        "--gov-env-path",
        str(gov_env_path),
    ]
    if skip_existing:
        shared.append("--skip-existing")
    if dry_run:
        shared.append("--dry-run")

    # S2 anchor cleaning
    clean_cmd = [
        py,
        str(repo_root / "scripts" / "llm_anchor_clean_v3.py"),
        *shared,
        "--page-tags-jsonl",
        str(page_tags_jsonl),
        "--output-dir",
        str(output_root / "cleaning"),
        "--variant",
        str(cleaning_variant),
        "--model",
        str(model_primary),
    ]
    if strict_narrative_mode:
        clean_cmd.append("--strict-narrative-mode")
    if issue_limit > 0:
        clean_cmd.extend(["--limit-issues", str(int(issue_limit))])
    _run(clean_cmd, cwd=repo_root)

    # S3 issue ontology
    cls_cmd = [
        py,
        str(repo_root / "scripts" / "llm_issue_classify_v3.py"),
        *shared,
        "--clean-jsonl",
        str(output_root / "cleaning" / "clean_zoning_text.jsonl"),
        "--law-presence-jsonl",
        str(law_presence_jsonl),
        "--output-dir",
        str(output_root / "classification"),
        "--variant",
        str(classification_variant),
        "--classification-input-mode",
        "raw_only",
        "--model-primary",
        str(model_primary),
        "--model-secondary",
        str(model_secondary),
    ]
    if strict_contradiction_mode:
        cls_cmd.append("--strict-contradiction-mode")
    if issue_limit > 0:
        cls_cmd.extend(["--limit-issues", str(int(issue_limit))])
    _run(cls_cmd, cwd=repo_root)

    # S4 dedup
    dedup_base_out = output_root / "events"
    _run(
        [
            py,
            str(repo_root / "scripts" / "llm_event_cluster_v3.py"),
            "--classification-jsonl",
            str(output_root / "classification" / "issue_labels_v3.jsonl"),
            "--clean-jsonl",
            str(output_root / "cleaning" / "clean_zoning_text.jsonl"),
            "--output-dir",
            str(dedup_base_out),
            "--variant",
            str(dedup_variant),
            "--model",
            str(model_primary),
            "--concurrency",
            str(int(concurrency)),
            "--timeout",
            str(float(timeout)),
            "--gateway-runner",
            str(gateway_runner),
            "--gateway-pythonpath",
            str(gateway_pythonpath),
            "--gov-env-path",
            str(gov_env_path),
            *(["--skip-existing"] if skip_existing else []),
            *(["--dry-run"] if dry_run else []),
        ],
        cwd=repo_root,
    )

    event_clusters_csv = dedup_base_out / "ordinance_event_clusters_v3.csv"
    event_assignments_csv = dedup_base_out / "issue_event_assignments_v3.csv"

    if use_collision_recheck:
        recheck_out = output_root / "events_recheck"
        _run(
            [
                py,
                str(repo_root / "scripts" / "llm_event_collision_recheck_v4.py"),
                "--classification-jsonl",
                str(output_root / "classification" / "issue_labels_v3.jsonl"),
                "--clean-jsonl",
                str(output_root / "cleaning" / "clean_zoning_text.jsonl"),
                "--event-clusters-csv",
                str(event_clusters_csv),
                "--event-assignments-csv",
                str(event_assignments_csv),
                "--output-dir",
                str(recheck_out),
                "--model",
                str(model_primary),
                "--concurrency",
                str(int(concurrency)),
                "--timeout",
                str(float(timeout)),
                "--gateway-runner",
                str(gateway_runner),
                "--gateway-pythonpath",
                str(gateway_pythonpath),
                "--gov-env-path",
                str(gov_env_path),
                *(["--skip-existing"] if skip_existing else []),
                *(["--dry-run"] if dry_run else []),
            ],
            cwd=repo_root,
        )
        event_clusters_csv = recheck_out / "ordinance_event_clusters_v3_rechecked.csv"
        event_assignments_csv = recheck_out / "issue_event_assignments_v3_rechecked.csv"

    # S5 event qualification
    _run(
        [
            py,
            str(repo_root / "scripts" / "llm_event_status_adjudicate_v3.py"),
            "--classification-jsonl",
            str(output_root / "classification" / "issue_labels_v3.jsonl"),
            "--clean-jsonl",
            str(output_root / "cleaning" / "clean_zoning_text.jsonl"),
            "--event-clusters-csv",
            str(event_clusters_csv),
            "--event-assignments-csv",
            str(event_assignments_csv),
            "--output-dir",
            str(output_root / "events_status"),
            "--model",
            str(model_primary),
            "--strict-qualification-mode",
            "--concurrency",
            str(int(concurrency)),
            "--timeout",
            str(float(timeout)),
            "--gateway-runner",
            str(gateway_runner),
            "--gateway-pythonpath",
            str(gateway_pythonpath),
            "--gov-env-path",
            str(gov_env_path),
            *(["--skip-existing"] if skip_existing else []),
            *(["--dry-run"] if dry_run else []),
        ],
        cwd=repo_root,
    )

    # Precision-first consistency gate.
    _run(
        [
            py,
            str(repo_root / "scripts" / "enforce_precision_event_consistency_v1.py"),
            "--pipeline-root",
            str(output_root),
        ],
        cwd=repo_root,
    )

    event_assign_path_for_downstream = output_root / "events_status" / "precision_v1" / "issue_event_assignments_status_v3_precision.csv"
    if not event_assign_path_for_downstream.is_file():
        event_assign_path_for_downstream = output_root / "events_status" / "issue_event_assignments_status_v3.csv"
    event_cluster_path_for_downstream = output_root / "events_status" / "precision_v1" / "ordinance_event_clusters_status_v3_precision.csv"
    if not event_cluster_path_for_downstream.is_file():
        event_cluster_path_for_downstream = output_root / "events_status" / "ordinance_event_clusters_status_v3.csv"

    # Page-level classification index from final issue/event decisions.
    _run(
        [
            py,
            str(repo_root / "scripts" / "build_page_classification_index_v1.py"),
            "--source-run-dir",
            str(run_dir),
            "--pipeline-root",
            str(output_root),
            "--event-assignments-csv",
            str(event_assign_path_for_downstream),
            "--output-dir",
            str(output_root / "page_classification_precision_v1"),
        ],
        cwd=repo_root,
    )

    # Classification invariants/QC checks.
    _run(
        [
            py,
            str(repo_root / "scripts" / "check_classification_hardening_v1.py"),
            "--pipeline-root",
            str(output_root),
            "--issue-assignments-csv",
            str(event_assign_path_for_downstream),
            "--page-classification-csv",
            str(output_root / "page_classification_precision_v1" / "page_classification_index_v1.csv"),
            "--output-dir",
            str(output_root / "qc_precision_v1"),
        ],
        cwd=repo_root,
    )

    # S6 organized export
    _run(
        [
            py,
            str(repo_root / "scripts" / "export_organized_zoning_corpus_v3.py"),
            "--run-root",
            str(output_root),
            "--classification-csv",
            str(output_root / "classification" / "issue_labels_v3.csv"),
            "--clean-csv",
            str(output_root / "cleaning" / "clean_zoning_text_flat.csv"),
            "--event-assignments-csv",
            str(event_assign_path_for_downstream),
            "--event-clusters-csv",
            str(event_cluster_path_for_downstream),
        ],
        cwd=repo_root,
    )

    # S7 longitudinal on qualified legal objects (optional for faster calibration rounds).
    if run_longitudinal:
        _run(
            [
                py,
                str(repo_root / "scripts" / "llm_panel_longitudinal_v3.py"),
                "--run-dir",
                str(run_dir),
                "--classification-jsonl",
                str(output_root / "classification" / "issue_labels_v3.jsonl"),
                "--clean-jsonl",
                str(output_root / "cleaning" / "clean_zoning_text.jsonl"),
                "--event-assignments-csv",
                str(event_assign_path_for_downstream),
                "--output-dir",
                str(output_root / "longitudinal"),
                "--model",
                str(model_primary),
                "--qualified-only",
                "--max-panel-chars",
                str(int(max_panel_chars)),
                "--concurrency",
                str(max(1, int(concurrency) - 1)),
                "--timeout",
                str(float(timeout)),
                "--gateway-runner",
                str(gateway_runner),
                "--gateway-pythonpath",
                str(gateway_pythonpath),
                "--gov-env-path",
                str(gov_env_path),
                *(["--skip-existing"] if skip_existing else []),
                *(["--dry-run"] if dry_run else []),
            ],
            cwd=repo_root,
        )

    _build_exhaustive_validation_packet(output_root)
    metrics = _score_run(output_root)
    (output_root / "pilot_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).expanduser().resolve()
    gateway_runner = Path(args.gateway_runner).expanduser().resolve()
    gateway_pythonpath = Path(args.gateway_pythonpath).expanduser().resolve()
    gov_env_path = Path(args.gov_env_path).expanduser().resolve()

    newspapers = [x.strip().lower() for x in _norm(args.newspapers).split(",") if x.strip()]
    if not newspapers:
        raise SystemExit("No newspapers provided.")

    if _norm(args.output_dir):
        out_root = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = run_dir / f"precision_recovery_pilot_v4_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    shared = [
        "--run-dir",
        str(run_dir),
        "--newspapers",
        ",".join(newspapers),
        "--concurrency",
        str(int(args.concurrency)),
        "--timeout",
        str(float(args.timeout)),
        "--gateway-runner",
        str(gateway_runner),
        "--gateway-pythonpath",
        str(gateway_pythonpath),
        "--gov-env-path",
        str(gov_env_path),
        "--model",
        str(args.model_primary),
    ]
    if bool(args.skip_existing):
        shared.append("--skip-existing")
    if bool(args.dry_run):
        shared.append("--dry-run")

    # Shared S1 inputs (single pass for all rounds).
    shared_dir = out_root / "shared"
    _run(
        [
            py,
            str(repo_root / "scripts" / "llm_page_tag_v3.py"),
            *shared,
            "--output-dir",
            str(shared_dir / "page_tagging"),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            str(repo_root / "scripts" / "llm_law_presence_v4.py"),
            "--run-dir",
            str(run_dir),
            "--newspapers",
            ",".join(newspapers),
            "--output-dir",
            str(shared_dir / "law_presence"),
            "--model",
            str(args.model_primary),
            "--concurrency",
            str(int(args.concurrency)),
            "--timeout",
            str(float(args.timeout)),
            "--gateway-runner",
            str(gateway_runner),
            "--gateway-pythonpath",
            str(gateway_pythonpath),
            "--gov-env-path",
            str(gov_env_path),
            *(["--skip-existing"] if bool(args.skip_existing) else []),
            *(["--dry-run"] if bool(args.dry_run) else []),
        ],
        cwd=repo_root,
    )
    page_tags_jsonl = shared_dir / "page_tagging" / "page_tags.jsonl"
    law_presence_jsonl = shared_dir / "law_presence" / "law_presence_v4.jsonl"

    round_rows: list[dict[str, Any]] = []

    # Round 0 baseline
    baseline_dir = out_root / "round0_baseline"
    _run_pipeline_once(
        repo_root=repo_root,
        run_dir=run_dir,
        output_root=baseline_dir,
        newspapers=newspapers,
        page_tags_jsonl=page_tags_jsonl,
        law_presence_jsonl=law_presence_jsonl,
        model_primary=str(args.model_primary),
        model_secondary=str(args.model_secondary),
        cleaning_variant="single_pass",
        strict_narrative_mode=False,
        classification_variant="single_model",
        strict_contradiction_mode=False,
        dedup_variant="direct_cluster",
        use_collision_recheck=False,
        run_longitudinal=False,
        issue_limit=0,
        concurrency=int(args.concurrency),
        timeout=float(args.timeout),
        max_panel_chars=int(args.max_panel_chars),
        gateway_runner=gateway_runner,
        gateway_pythonpath=gateway_pythonpath,
        gov_env_path=gov_env_path,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )
    m = _score_run(baseline_dir)
    m.update({"round": "round0", "axis": "baseline", "variant_label": "single_pass|single_model|direct_cluster"})
    round_rows.append(m)

    # Round 1 axis screening
    axis_dir = out_root / "round1_axis"
    axis_dir.mkdir(parents=True, exist_ok=True)
    default_cfg = {
        "cleaning_variant": "three_pass",
        "strict_narrative_mode": False,
        "classification_variant": "dual_model_with_llm_adjudicator",
        "strict_contradiction_mode": True,
        "dedup_variant": "pairwise_then_cluster",
        "use_collision_recheck": True,
    }

    cleaning_candidates = [
        {"cleaning_variant": "single_pass", "strict_narrative_mode": False, "label": "single_pass"},
        {"cleaning_variant": "three_pass", "strict_narrative_mode": False, "label": "three_pass"},
        {"cleaning_variant": "three_pass", "strict_narrative_mode": True, "label": "three_pass_strict_narrative"},
    ]
    cleaning_scores = []
    for c in cleaning_candidates:
        cfg = dict(default_cfg)
        cfg.update(c)
        label = str(cfg.pop("label"))
        runp = axis_dir / f"cleaning__{label}"
        _run_pipeline_once(
            repo_root=repo_root,
            run_dir=run_dir,
            output_root=runp,
            newspapers=newspapers,
            page_tags_jsonl=page_tags_jsonl,
            law_presence_jsonl=law_presence_jsonl,
            model_primary=str(args.model_primary),
            model_secondary=str(args.model_secondary),
            run_longitudinal=False,
            issue_limit=int(args.round1_issue_limit),
            concurrency=int(args.concurrency),
            timeout=float(args.timeout),
            max_panel_chars=int(args.max_panel_chars),
            gateway_runner=gateway_runner,
            gateway_pythonpath=gateway_pythonpath,
            gov_env_path=gov_env_path,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
            **cfg,
        )
        ms = _score_run(runp)
        ms.update({"round": "round1", "axis": "cleaning", "variant_label": label, "run_root": str(runp)})
        cleaning_scores.append(ms)
        round_rows.append(ms)
    cleaning_best = sorted(cleaning_scores, key=lambda x: x["score"], reverse=True)[0]
    default_cfg["cleaning_variant"] = "three_pass" if "three_pass" in cleaning_best["variant_label"] else "single_pass"
    default_cfg["strict_narrative_mode"] = "strict_narrative" in cleaning_best["variant_label"]

    cls_candidates = [
        {"classification_variant": "single_model", "strict_contradiction_mode": False, "label": "single_model"},
        {"classification_variant": "dual_model_with_llm_adjudicator", "strict_contradiction_mode": False, "label": "dual_model"},
        {"classification_variant": "dual_model_with_llm_adjudicator", "strict_contradiction_mode": True, "label": "dual_model_strict_contradiction"},
    ]
    cls_scores = []
    for c in cls_candidates:
        cfg = dict(default_cfg)
        cfg.update(c)
        label = str(cfg.pop("label"))
        runp = axis_dir / f"classification__{label}"
        _run_pipeline_once(
            repo_root=repo_root,
            run_dir=run_dir,
            output_root=runp,
            newspapers=newspapers,
            page_tags_jsonl=page_tags_jsonl,
            law_presence_jsonl=law_presence_jsonl,
            model_primary=str(args.model_primary),
            model_secondary=str(args.model_secondary),
            run_longitudinal=False,
            issue_limit=int(args.round1_issue_limit),
            concurrency=int(args.concurrency),
            timeout=float(args.timeout),
            max_panel_chars=int(args.max_panel_chars),
            gateway_runner=gateway_runner,
            gateway_pythonpath=gateway_pythonpath,
            gov_env_path=gov_env_path,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
            **cfg,
        )
        ms = _score_run(runp)
        ms.update({"round": "round1", "axis": "classification", "variant_label": label, "run_root": str(runp)})
        cls_scores.append(ms)
        round_rows.append(ms)
    cls_best = sorted(cls_scores, key=lambda x: x["score"], reverse=True)[0]
    default_cfg["classification_variant"] = "single_model" if "single_model" in cls_best["variant_label"] else "dual_model_with_llm_adjudicator"
    default_cfg["strict_contradiction_mode"] = "strict_contradiction" in cls_best["variant_label"]

    dedup_candidates = [
        {"dedup_variant": "direct_cluster", "use_collision_recheck": False, "label": "direct_cluster"},
        {"dedup_variant": "pairwise_then_cluster", "use_collision_recheck": False, "label": "pairwise_then_cluster"},
        {"dedup_variant": "pairwise_then_cluster", "use_collision_recheck": True, "label": "pairwise_then_cluster_recheck"},
    ]
    dedup_scores = []
    for c in dedup_candidates:
        cfg = dict(default_cfg)
        cfg.update(c)
        label = str(cfg.pop("label"))
        runp = axis_dir / f"dedup__{label}"
        _run_pipeline_once(
            repo_root=repo_root,
            run_dir=run_dir,
            output_root=runp,
            newspapers=newspapers,
            page_tags_jsonl=page_tags_jsonl,
            law_presence_jsonl=law_presence_jsonl,
            model_primary=str(args.model_primary),
            model_secondary=str(args.model_secondary),
            run_longitudinal=False,
            issue_limit=int(args.round1_issue_limit),
            concurrency=int(args.concurrency),
            timeout=float(args.timeout),
            max_panel_chars=int(args.max_panel_chars),
            gateway_runner=gateway_runner,
            gateway_pythonpath=gateway_pythonpath,
            gov_env_path=gov_env_path,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
            **cfg,
        )
        ms = _score_run(runp)
        ms.update({"round": "round1", "axis": "dedup", "variant_label": label, "run_root": str(runp)})
        dedup_scores.append(ms)
        round_rows.append(ms)
    dedup_best = sorted(dedup_scores, key=lambda x: x["score"], reverse=True)[0]
    default_cfg["dedup_variant"] = "direct_cluster" if "direct_cluster" in dedup_best["variant_label"] else "pairwise_then_cluster"
    default_cfg["use_collision_recheck"] = "recheck" in dedup_best["variant_label"]

    # Round 2 top-3 end-to-end candidates.
    round2_dir = out_root / "round2_candidates"
    round2_dir.mkdir(parents=True, exist_ok=True)
    cls_second = sorted(cls_scores, key=lambda x: x["score"], reverse=True)[1] if len(cls_scores) > 1 else cls_best
    dedup_second = sorted(dedup_scores, key=lambda x: x["score"], reverse=True)[1] if len(dedup_scores) > 1 else dedup_best

    cands = [
        ("candidate_1_best", dict(default_cfg)),
        (
            "candidate_2_alt_classification",
            {
                **default_cfg,
                "classification_variant": "single_model" if "single_model" in cls_second["variant_label"] else "dual_model_with_llm_adjudicator",
                "strict_contradiction_mode": "strict_contradiction" in cls_second["variant_label"],
            },
        ),
        (
            "candidate_3_alt_dedup",
            {
                **default_cfg,
                "dedup_variant": "direct_cluster" if "direct_cluster" in dedup_second["variant_label"] else "pairwise_then_cluster",
                "use_collision_recheck": "recheck" in dedup_second["variant_label"],
            },
        ),
    ]
    round2_scores = []
    for name, cfg in cands:
        runp = round2_dir / name
        _run_pipeline_once(
            repo_root=repo_root,
            run_dir=run_dir,
            output_root=runp,
            newspapers=newspapers,
            page_tags_jsonl=page_tags_jsonl,
            law_presence_jsonl=law_presence_jsonl,
            model_primary=str(args.model_primary),
            model_secondary=str(args.model_secondary),
            run_longitudinal=True,
            issue_limit=0,
            concurrency=int(args.concurrency),
            timeout=float(args.timeout),
            max_panel_chars=int(args.max_panel_chars),
            gateway_runner=gateway_runner,
            gateway_pythonpath=gateway_pythonpath,
            gov_env_path=gov_env_path,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
            **cfg,
        )
        ms = _score_run(runp)
        ms.update({"round": "round2", "axis": "candidate", "variant_label": name, "run_root": str(runp)})
        round2_scores.append(ms)
        round_rows.append(ms)
    winner = sorted(round2_scores, key=lambda x: x["score"], reverse=True)[0]

    # Round 3 robustness: alternate model path for winner.
    round3_dir = out_root / "round3_robustness"
    round3_dir.mkdir(parents=True, exist_ok=True)
    winner_cfg = dict(default_cfg)
    for name, cfg in cands:
        if name == winner["variant_label"]:
            winner_cfg = cfg
            break
    robust_run = round3_dir / "winner_alt_model_path"
    _run_pipeline_once(
        repo_root=repo_root,
        run_dir=run_dir,
        output_root=robust_run,
        newspapers=newspapers,
        page_tags_jsonl=page_tags_jsonl,
        law_presence_jsonl=law_presence_jsonl,
        model_primary=str(args.model_secondary),
        model_secondary=str(args.model_primary),
        run_longitudinal=True,
        issue_limit=0,
        concurrency=int(args.concurrency),
        timeout=float(args.timeout),
        max_panel_chars=int(args.max_panel_chars),
        gateway_runner=gateway_runner,
        gateway_pythonpath=gateway_pythonpath,
        gov_env_path=gov_env_path,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
        **winner_cfg,
    )
    ms = _score_run(robust_run)
    ms.update({"round": "round3", "axis": "robustness", "variant_label": "winner_alt_model_path", "run_root": str(robust_run)})
    round_rows.append(ms)

    score_df = pd.DataFrame.from_records(round_rows).sort_values(["round", "axis", "score"], ascending=[True, True, False])
    _write_csv(out_root / "pilot_scoreboard.csv", score_df)

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_root),
        "newspapers": newspapers,
        "round1_issue_limit": int(args.round1_issue_limit),
        "winner_round2": winner,
        "selected_config": winner_cfg,
        "model_primary": str(args.model_primary),
        "model_secondary": str(args.model_secondary),
    }
    (out_root / "pilot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"done_precision_recovery_pilot_v4 output_dir={out_root}")


if __name__ == "__main__":
    main()
