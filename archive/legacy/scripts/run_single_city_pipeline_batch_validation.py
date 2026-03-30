#!/usr/bin/env python3
"""
Run single-city zoning pipeline across a diverse city sample and collect diagnostics.

This script wraps scripts/run_single_newspaper_zoning_pipeline.py and writes:
  - batch_selection.csv
  - batch_run_status.csv
  - batch_target_summary.csv
  - batch_alerts.csv
  - batch_provenance.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _choose_city_row(df: pd.DataFrame, city_key: str) -> dict[str, Any]:
    hit = df[df["city_key"].astype(str) == str(city_key)]
    if hit.empty:
        return {"city_key": city_key, "region": "", "urbanicity_proxy": "", "issue_count": 0}
    row = hit.iloc[0]
    return {
        "city_key": _norm(row.get("city_key")),
        "region": _norm(row.get("region")),
        "urbanicity_proxy": _norm(row.get("urbanicity_proxy")),
        "issue_count": int(row.get("issue_count") or 0),
    }


def _select_diverse(city_stats: pd.DataFrame, city_count: int) -> list[str]:
    """
    Deterministic diverse selection:
      1) per region pick largest, median, smallest issue_count city (if available)
      2) fill remainder by highest issue_count not already selected
    """
    stats = city_stats.copy()
    stats = stats.sort_values(["region", "issue_count", "city_key"], ascending=[True, True, True]).reset_index(drop=True)
    selected: list[str] = []

    for region in sorted(stats["region"].dropna().astype(str).unique().tolist()):
        sub = stats[stats["region"].astype(str) == str(region)].copy().sort_values(
            ["issue_count", "city_key"], ascending=[True, True]
        )
        if sub.empty:
            continue
        picks = [
            _norm(sub.iloc[-1]["city_key"]),  # max
            _norm(sub.iloc[len(sub) // 2]["city_key"]),  # median
            _norm(sub.iloc[0]["city_key"]),  # min
        ]
        for c in picks:
            if c and c not in selected:
                selected.append(c)
            if len(selected) >= city_count:
                return selected[:city_count]

    for row in stats.sort_values(["issue_count", "city_key"], ascending=[False, True]).itertuples(index=False):
        c = _norm(getattr(row, "city_key", ""))
        if c and c not in selected:
            selected.append(c)
        if len(selected) >= city_count:
            break
    return selected[:city_count]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch-validate single-city zoning pipeline across diverse cities.")
    ap.add_argument(
        "--source-run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60_fullnewspaper",
    )
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--cities", default="", help="Comma-separated explicit city_key list. Overrides auto selection.")
    ap.add_argument("--city-count", type=int, default=8)
    ap.add_argument("--ontology-model", default="gemini:gemini-2.5-flash")
    ap.add_argument("--extraction-model", default="openai:gpt-5-mini")
    ap.add_argument("--longitudinal-model", default="gemini:gemini-2.5-flash")
    ap.add_argument("--extract-methods", default="llm_only,llm_hybrid")
    ap.add_argument(
        "--longitudinal-variants",
        default="compact_closed_v2,compact_numeric_quote_v1,panel_lean_numeric_quote_v1,timeline_evidence_v1,hybrid_open_v1",
    )
    ap.add_argument("--longitudinal-max-panel-chars", type=int, default=850000)
    ap.add_argument("--longitudinal-model-context-tokens", type=int, default=1_000_000)
    ap.add_argument("--longitudinal-context-utilization", type=float, default=0.70)
    ap.add_argument("--longitudinal-chars-per-token", type=float, default=4.0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--gov-env-path", default="/Users/saulrichardson/projects/gov-gpt/.env")
    ap.add_argument("--include-raw-baseline", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--longitudinal-run-winner-on-all",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    source_run = Path(args.source_run_dir).expanduser().resolve()
    if not source_run.is_dir():
        raise SystemExit(f"Missing source run dir: {source_run}")

    selected_path = source_run / "panels" / "selected_panel_issues.csv"
    if not selected_path.is_file():
        raise SystemExit(f"Missing selected panel file: {selected_path}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.output_dir).expanduser().resolve()
        if _norm(args.output_dir)
        else source_run / "single_city_pipeline_batch" / f"batch_{ts}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(selected_path)
    city_stats = (
        sel.groupby(["city_key", "region", "urbanicity_proxy"], as_index=False)
        .agg(issue_count=("issue_id", "nunique"))
        .sort_values(["issue_count", "city_key"], ascending=[False, True])
        .reset_index(drop=True)
    )

    if _norm(args.cities):
        cities = [_norm(x) for x in str(args.cities).split(",") if _norm(x)]
    else:
        cities = _select_diverse(city_stats, city_count=max(1, int(args.city_count)))
    if not cities:
        raise SystemExit("No cities selected.")

    selection_rows = [_choose_city_row(city_stats, c) for c in cities]
    selection_df = pd.DataFrame.from_records(selection_rows)
    _write_csv(out_root / "batch_selection.csv", selection_df)

    script_path = Path(__file__).resolve().parent / "run_single_newspaper_zoning_pipeline.py"
    if not script_path.is_file():
        raise SystemExit(f"Missing script: {script_path}")

    status_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for i, city_key in enumerate(cities, start=1):
        meta = _choose_city_row(city_stats, city_key)
        city_out = out_root / city_key
        city_out.mkdir(parents=True, exist_ok=True)
        log_path = city_out / "run.log"

        cmd = [
            "python",
            str(script_path),
            "--source-run-dir",
            str(source_run),
            "--city-key",
            city_key,
            "--output-dir",
            str(city_out),
            "--ontology-model",
            str(args.ontology_model),
            "--extraction-model",
            str(args.extraction_model),
            "--longitudinal-model",
            str(args.longitudinal_model),
            "--extract-methods",
            str(args.extract_methods),
            "--longitudinal-variants",
            str(args.longitudinal_variants),
            "--longitudinal-max-panel-chars",
            str(int(args.longitudinal_max_panel_chars)),
            "--longitudinal-model-context-tokens",
            str(int(args.longitudinal_model_context_tokens)),
            "--longitudinal-context-utilization",
            str(float(args.longitudinal_context_utilization)),
            "--longitudinal-chars-per-token",
            str(float(args.longitudinal_chars_per_token)),
            "--concurrency",
            str(int(args.concurrency)),
            "--timeout",
            str(float(args.timeout)),
            "--gov-env-path",
            str(args.gov_env_path),
        ]
        cmd.append("--skip-existing" if bool(args.skip_existing) else "--no-skip-existing")
        cmd.append("--include-raw-baseline" if bool(args.include_raw_baseline) else "--no-include-raw-baseline")
        cmd.append("--longitudinal-run-winner-on-all" if bool(args.longitudinal_run_winner_on_all) else "--no-longitudinal-run-winner-on-all")
        if args.dry_run:
            cmd.append("--dry-run")

        started = time.time()
        rc = 0
        err_msg = ""
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"[city {i}/{len(cities)}] {city_key}\n")
            logf.write("[cmd] " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
            logf.flush()
            try:
                subprocess.run(cmd, stdout=logf, stderr=logf, check=True, cwd=str(Path(__file__).resolve().parent.parent))
            except subprocess.CalledProcessError as e:
                rc = int(e.returncode or 1)
                err_msg = f"subprocess_failed_rc_{rc}"
            except Exception as e:  # pragma: no cover
                rc = 1
                err_msg = f"exception_{type(e).__name__}"
        elapsed = time.time() - started

        psummary_path = city_out / "pipeline_summary.csv"
        has_summary = psummary_path.is_file()
        if has_summary:
            try:
                psummary = pd.read_csv(psummary_path)
            except Exception:
                psummary = pd.DataFrame()
        else:
            psummary = pd.DataFrame()

        best_row = pd.DataFrame()
        if not psummary.empty and "overall_quality" in psummary.columns:
            try:
                best_row = psummary.sort_values(
                    ["overall_quality", "target"],
                    ascending=[False, True],
                ).head(1)
            except Exception:
                best_row = psummary.head(1)

        best_target = _norm(best_row.iloc[0]["target"]) if not best_row.empty and "target" in best_row.columns else ""
        best_variant = _norm(best_row.iloc[0]["winner_variant"]) if not best_row.empty and "winner_variant" in best_row.columns else ""
        best_quality = float(best_row.iloc[0]["overall_quality"]) if not best_row.empty and "overall_quality" in best_row.columns else float("nan")

        status_rows.append(
            {
                "city_key": city_key,
                "region": meta.get("region", ""),
                "urbanicity_proxy": meta.get("urbanicity_proxy", ""),
                "issue_count": meta.get("issue_count", 0),
                "status": "ok" if rc == 0 else "failed",
                "return_code": rc,
                "elapsed_minutes": round(elapsed / 60.0, 3),
                "has_pipeline_summary": bool(has_summary),
                "best_target": best_target,
                "best_variant": best_variant,
                "best_overall_quality": best_quality,
                "error": err_msg,
                "output_dir": str(city_out),
                "log_path": str(log_path),
            }
        )

        if not psummary.empty:
            psummary = psummary.copy()
            psummary.insert(0, "city_key", city_key)
            psummary.insert(1, "region", meta.get("region", ""))
            psummary.insert(2, "urbanicity_proxy", meta.get("urbanicity_proxy", ""))
            psummary.insert(3, "issue_count", meta.get("issue_count", 0))
            target_rows.extend(psummary.to_dict(orient="records"))

        # Persist incremental progress after each city so partial runs are usable.
        _write_csv(out_root / "batch_run_status.csv", pd.DataFrame.from_records(status_rows))
        _write_csv(out_root / "batch_target_summary.csv", pd.DataFrame.from_records(target_rows))

    status_df = pd.DataFrame.from_records(status_rows)
    target_df = pd.DataFrame.from_records(target_rows)
    _write_csv(out_root / "batch_run_status.csv", status_df)
    _write_csv(out_root / "batch_target_summary.csv", target_df)

    alerts: list[dict[str, Any]] = []
    if not status_df.empty:
        failed = status_df[status_df["status"].astype(str) != "ok"]
        for r in failed.itertuples(index=False):
            alerts.append(
                {
                    "city_key": _norm(getattr(r, "city_key", "")),
                    "severity": "high",
                    "type": "run_failed",
                    "detail": _norm(getattr(r, "error", "")) or f"return_code={getattr(r, 'return_code', '')}",
                }
            )
        lowq = status_df[pd.to_numeric(status_df.get("best_overall_quality"), errors="coerce") < 0.85]
        for r in lowq.itertuples(index=False):
            alerts.append(
                {
                    "city_key": _norm(getattr(r, "city_key", "")),
                    "severity": "medium",
                    "type": "low_quality_winner",
                    "detail": f"best_overall_quality={getattr(r, 'best_overall_quality', '')}",
                }
            )

    if not target_df.empty:
        for r in target_df.itertuples(index=False):
            first_full = _norm(getattr(r, "first_full_ordinance_issue_id", ""))
            if not first_full:
                alerts.append(
                    {
                        "city_key": _norm(getattr(r, "city_key", "")),
                        "severity": "medium",
                        "type": "missing_first_full",
                        "detail": f"target={_norm(getattr(r, 'target', ''))}",
                    }
                )
            try:
                full_cnt = int(getattr(r, "full_ordinance_issues", 0) or 0)
            except Exception:
                full_cnt = 0
            if full_cnt >= 15:
                alerts.append(
                    {
                        "city_key": _norm(getattr(r, "city_key", "")),
                        "severity": "medium",
                        "type": "high_full_ordinance_count",
                        "detail": f"target={_norm(getattr(r, 'target', ''))}, full_ordinance_issues={full_cnt}",
                    }
                )

    alerts_df = pd.DataFrame.from_records(alerts)
    _write_csv(out_root / "batch_alerts.csv", alerts_df)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(source_run),
        "output_dir": str(out_root),
        "cities": cities,
        "args": vars(args),
        "script": str(Path(__file__).resolve()),
    }
    (out_root / "batch_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    print(f"Wrote: {out_root / 'batch_selection.csv'}")
    print(f"Wrote: {out_root / 'batch_run_status.csv'}")
    print(f"Wrote: {out_root / 'batch_target_summary.csv'}")
    print(f"Wrote: {out_root / 'batch_alerts.csv'}")
    print(f"Wrote: {out_root / 'batch_provenance.json'}")
    if not status_df.empty:
        ok = int((status_df["status"].astype(str) == "ok").sum())
        print(f"Completed cities: {ok}/{len(status_df)} ok")


if __name__ == "__main__":
    main()
