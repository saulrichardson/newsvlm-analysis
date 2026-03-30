#!/usr/bin/env python3
"""
Build interpretable v2 longitudinal metrics for the 20-panel pilot.

This script keeps panel-level LLM longitudinal outputs for event/phase/wave structure,
but derives auditable quantitative metrics from issue-level category outputs and
deterministic time bins (issue terciles).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_FIXED_CATEGORIES = [
    "land_use_restrictions",
    "bulk_dimensional_standards",
    "procedural_governance",
    "nonconforming_use",
    "definitions",
    "parking_transport",
    "environmental_protection",
    "historic_preservation",
    "flexible_zoning",
    "inclusionary_affordable_housing",
    "enforcement_penalties",
    "miscellaneous_site_rules",
]

_COMPLEXITY_KEYS = [
    "regulatory_complexity_0_100",
    "procedural_intensity_0_100",
    "dimensional_specificity_0_100",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build longitudinal v2 metrics from pilot20 data.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60",
        help="Base run directory.",
    )
    ap.add_argument(
        "--longitudinal-dir",
        default="",
        help="Longitudinal directory (default: <run-dir>/longitudinal).",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <run-dir>/longitudinal_v2).",
    )
    return ap.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
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
                out.append(obj)
    return out


def _norm_str(x: Any) -> str:
    return str(x or "").strip()


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    if not math.isfinite(v):
        return math.nan
    return v


def _assign_issue_terciles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def _assign(g: pd.DataFrame) -> pd.DataFrame:
        city = str(getattr(g, "name", ""))
        g = g.sort_values(["issue_date", "issue_id"], kind="mergesort").reset_index(drop=True)
        if "city_key" not in g.columns:
            g["city_key"] = city
        n = len(g)
        groups = np.array_split(np.arange(n), 3)
        labels = np.array(["early"] * n, dtype=object)
        if len(groups) >= 2:
            labels[groups[1]] = "mid"
        if len(groups) >= 3:
            labels[groups[2]] = "late"
        g["tercile"] = labels
        g["issue_rank"] = np.arange(1, n + 1)
        g["n_issues_city"] = n
        return g

    return df.groupby("city_key", group_keys=False).apply(_assign, include_groups=False).reset_index(drop=True)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.ndim != 1 or q.ndim != 1 or len(p) != len(q) or len(p) == 0:
        return math.nan
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    ps = p.sum()
    qs = q.sum()
    if ps <= 0 or qs <= 0:
        return math.nan
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(js)


def _extract_snippet(text: str, patterns: list[str], window: int = 170) -> str:
    t = _norm_str(text)
    if not t:
        return ""
    for pat in patterns:
        m = re.search(pat, t, flags=re.I)
        if not m:
            continue
        start = max(0, m.start() - window)
        end = min(len(t), m.end() + window)
        snippet = t[start:end]
        snippet = re.sub(r"\s+", " ", snippet).strip()
        return snippet
    snippet = re.sub(r"\s+", " ", t[: max(220, min(460, len(t)))])
    return snippet.strip()


def _keyword_score(text: str, patterns: list[str]) -> int:
    t = _norm_str(text)
    if not t:
        return 0
    score = 0
    for pat in patterns:
        score += len(re.findall(pat, t, flags=re.I))
    return int(score)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    longitudinal_dir = Path(args.longitudinal_dir).expanduser().resolve() if str(args.longitudinal_dir).strip() else (run_dir / "longitudinal")
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "longitudinal_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = _read_csv(longitudinal_dir / "pilot20_selection_quality.csv")
    if selection.empty:
        raise SystemExit(f"Missing selection file: {longitudinal_dir / 'pilot20_selection_quality.csv'}")

    selected = selection[selection.get("selected_for_pilot", 0) == 1].copy()
    if selected.empty:
        raise SystemExit("No selected pilot cities found in longitudinal selection file.")
    city_keys = sorted(selected["city_key"].astype(str).unique().tolist())

    selected_issues = _read_csv(run_dir / "panels" / "selected_panel_issues.csv")
    if selected_issues.empty:
        raise SystemExit(f"Missing selected issues file: {run_dir / 'panels' / 'selected_panel_issues.csv'}")
    selected_issues = selected_issues[selected_issues["city_key"].astype(str).isin(city_keys)].copy()

    issue_rows = []
    issue_outputs_path = run_dir / "llm_panel_vs_issue" / "parsed_issue_outputs.jsonl"
    for row in _iter_jsonl(issue_outputs_path):
        cid = _norm_str(row.get("custom_id"))
        parts = cid.split("::", 2)
        if len(parts) != 3 or parts[0] != "issue":
            continue
        city_key, issue_id = parts[1], parts[2]
        if city_key not in city_keys:
            continue

        parsed = row.get("parsed_json")
        if not isinstance(parsed, dict):
            continue

        cat = parsed.get("category_shares") if isinstance(parsed.get("category_shares"), dict) else {}
        meas = parsed.get("measures") if isinstance(parsed.get("measures"), dict) else {}
        num = parsed.get("numeric_constraints") if isinstance(parsed.get("numeric_constraints"), dict) else {}

        rec: dict[str, Any] = {
            "city_key": city_key,
            "issue_id": issue_id,
            "status_code": int(_safe_float(row.get("status_code"))) if pd.notna(_safe_float(row.get("status_code"))) else math.nan,
        }

        for c in _FIXED_CATEGORIES:
            rec[c] = _safe_float(cat.get(c))
        rec["category_sum"] = float(np.nansum([rec[c] for c in _FIXED_CATEGORIES]))

        for k in _COMPLEXITY_KEYS:
            rec[k] = _safe_float(meas.get(k))

        rec["max_min_lot_size_sqft"] = _safe_float(num.get("max_min_lot_size_sqft"))
        rec["max_height_ft"] = _safe_float(num.get("max_height_ft"))
        rec["max_front_setback_ft"] = _safe_float(num.get("max_front_setback_ft"))
        rec["max_parking_spaces_per_unit"] = _safe_float(num.get("max_parking_spaces_per_unit"))
        issue_rows.append(rec)

    issue_df = pd.DataFrame.from_records(issue_rows)
    if issue_df.empty:
        raise SystemExit(f"No issue-level parsed rows found in {issue_outputs_path} for selected cities.")

    # Join metadata and normalize shares.
    meta_cols = [
        "city_key",
        "city_name",
        "state_abbr",
        "region",
        "urbanicity_proxy",
        "issue_id",
        "issue_date",
        "classification_label",
        "text_chars",
        "page_count",
    ]
    m = selected_issues[meta_cols].drop_duplicates(["city_key", "issue_id"]).copy()
    merged = m.merge(issue_df, on=["city_key", "issue_id"], how="left")

    # Keep only rows with parsed issue output.
    merged = merged[merged[_FIXED_CATEGORIES].notna().any(axis=1)].copy()

    # Normalize category shares to sum to 1 where possible.
    sums = merged[_FIXED_CATEGORIES].sum(axis=1, skipna=True)
    valid = sums > 0
    merged.loc[valid, _FIXED_CATEGORIES] = merged.loc[valid, _FIXED_CATEGORIES].div(sums[valid], axis=0)
    merged["category_sum_normalized"] = merged[_FIXED_CATEGORIES].sum(axis=1, skipna=True)

    merged = _assign_issue_terciles(merged)

    issue_level_out = merged.sort_values(["city_key", "issue_date", "issue_id"]).reset_index(drop=True)
    issue_level_out.to_csv(out_dir / "city_issue_level_panel.csv", index=False)

    # Deterministic tercile category shares.
    cat_long = issue_level_out.melt(
        id_vars=[
            "city_key",
            "city_name",
            "state_abbr",
            "region",
            "urbanicity_proxy",
            "issue_id",
            "issue_date",
            "classification_label",
            "tercile",
        ],
        value_vars=_FIXED_CATEGORIES,
        var_name="category",
        value_name="share",
    )
    cat_terc = (
        cat_long.groupby(["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "tercile", "category"], dropna=False)["share"]
        .mean()
        .reset_index()
    )
    cat_terc.to_csv(out_dir / "city_tercile_category_shares.csv", index=False)

    # Complexity by tercile.
    cpx_terc = (
        issue_level_out.groupby(["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "tercile"], dropna=False)[_COMPLEXITY_KEYS]
        .mean()
        .reset_index()
    )
    cpx_terc.to_csv(out_dir / "city_tercile_complexity.csv", index=False)

    # Numeric by tercile.
    num_keys = ["max_min_lot_size_sqft", "max_height_ft", "max_front_setback_ft", "max_parking_spaces_per_unit"]
    num_terc = (
        issue_level_out.groupby(["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "tercile"], dropna=False)[num_keys]
        .mean()
        .reset_index()
    )
    num_terc.to_csv(out_dir / "city_tercile_numeric_means.csv", index=False)

    # Category shift metrics and substantive-shift index.
    cat_pivot = (
        cat_terc.pivot_table(
            index=["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "category"],
            columns="tercile",
            values="share",
            aggfunc="mean",
        )
        .reset_index()
    )
    for c in ("early", "mid", "late"):
        if c not in cat_pivot.columns:
            cat_pivot[c] = math.nan
    cat_pivot["late_minus_early"] = cat_pivot["late"] - cat_pivot["early"]
    cat_pivot["late_minus_mid"] = cat_pivot["late"] - cat_pivot["mid"]
    cat_pivot.to_csv(out_dir / "city_category_shift_metrics.csv", index=False)

    city_shift_rows: list[dict[str, Any]] = []
    for city, g in cat_pivot.groupby("city_key", dropna=False):
        gv = g.set_index("category")
        early = np.array([_safe_float(gv["early"].get(c)) for c in _FIXED_CATEGORIES], dtype=float)
        late = np.array([_safe_float(gv["late"].get(c)) for c in _FIXED_CATEGORIES], dtype=float)
        js = _js_divergence(early, late)
        proc_early = float(sum(_safe_float(gv["early"].get(c)) for c in ["procedural_governance", "enforcement_penalties", "definitions"]))
        proc_late = float(sum(_safe_float(gv["late"].get(c)) for c in ["procedural_governance", "enforcement_penalties", "definitions"]))
        use_early = float(sum(_safe_float(gv["early"].get(c)) for c in ["land_use_restrictions", "bulk_dimensional_standards"]))
        use_late = float(sum(_safe_float(gv["late"].get(c)) for c in ["land_use_restrictions", "bulk_dimensional_standards"]))
        proceduralization_early = proc_early - use_early
        proceduralization_late = proc_late - use_late
        flex_early = _safe_float(gv["early"].get("flexible_zoning"))
        flex_late = _safe_float(gv["late"].get("flexible_zoning"))

        row = g.iloc[0]
        city_shift_rows.append(
            {
                "city_key": city,
                "city_name": row["city_name"],
                "state_abbr": row["state_abbr"],
                "region": row["region"],
                "urbanicity_proxy": row["urbanicity_proxy"],
                "substantive_shift_js": js,
                "proceduralization_index_early": proceduralization_early,
                "proceduralization_index_late": proceduralization_late,
                "proceduralization_delta": proceduralization_late - proceduralization_early,
                "flexibility_uptake_index": flex_late - flex_early,
            }
        )
    city_shift = pd.DataFrame.from_records(city_shift_rows)

    # Event lag metrics.
    first_events = _read_csv(longitudinal_dir / "city_first_zoning_events.csv")
    if first_events.empty:
        raise SystemExit(f"Missing first events file: {longitudinal_dir / 'city_first_zoning_events.csv'}")
    first_events = first_events[first_events["city_key"].astype(str).isin(city_keys)].copy()

    d1 = pd.to_datetime(first_events["first_zoning_issue_date"], errors="coerce")
    d2 = pd.to_datetime(first_events["first_full_issue_date"], errors="coerce")
    lag_days = (d2 - d1).dt.days
    first_events["first_signal_to_full_lag_days"] = lag_days
    first_events["first_signal_to_full_lag_years"] = lag_days / 365.25
    first_events["lag_observed"] = ((first_events["first_full_status"].astype(str) == "found") & d1.notna() & d2.notna()).astype(int)
    first_events.to_csv(out_dir / "city_event_lags.csv", index=False)

    # Phase sequence metrics from longitudinal phases.
    phases = _read_csv(longitudinal_dir / "city_evolution_phases.csv")
    phases = phases[phases["city_key"].astype(str).isin(city_keys)].copy() if not phases.empty else phases
    seq_rows: list[dict[str, Any]] = []
    if not phases.empty:
        phases["phase_order"] = pd.to_numeric(phases["phase_order"], errors="coerce")
        phases = phases.sort_values(["city_key", "phase_order", "start_issue_date", "end_issue_date"])
        for city, g in phases.groupby("city_key", dropna=False):
            labels = [str(x).strip() for x in g["phase_label"].tolist() if str(x).strip()]
            uniq_seq = labels
            seq = " > ".join(uniq_seq)
            n_transitions = max(0, len(uniq_seq) - 1)
            row = g.iloc[0]
            seq_rows.append(
                {
                    "city_key": city,
                    "city_name": row["city_name"],
                    "state_abbr": row["state_abbr"],
                    "region": row["region"],
                    "urbanicity_proxy": row["urbanicity_proxy"],
                    "n_phases": len(uniq_seq),
                    "n_phase_transitions": n_transitions,
                    "phase_sequence": seq,
                }
            )
    seq_df = pd.DataFrame.from_records(seq_rows)
    seq_df.to_csv(out_dir / "city_phase_sequences.csv", index=False)

    # Amendment balance metrics.
    waves = _read_csv(longitudinal_dir / "city_amendment_waves.csv")
    waves = waves[waves["city_key"].astype(str).isin(city_keys)].copy() if not waves.empty else waves
    wave_city = pd.DataFrame()
    if not waves.empty:
        waves["change_type"] = waves["change_type"].astype(str).str.strip().str.lower()
        wt = (
            waves.groupby(["city_key", "change_type"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for c in ["expansion", "tightening", "clarification", "reclassification", "mixed", "none", "uncertain"]:
            if c not in wt.columns:
                wt[c] = 0
        wt["wave_count"] = wt[["expansion", "tightening", "clarification", "reclassification", "mixed", "none", "uncertain"]].sum(axis=1)
        wt["expansion_minus_tightening"] = wt["expansion"] - wt["tightening"]
        wt["clarification_share"] = wt["clarification"] / wt["wave_count"].replace(0, np.nan)
        wt["reclassification_share"] = wt["reclassification"] / wt["wave_count"].replace(0, np.nan)

        meta_city = selected[["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy"]].drop_duplicates()
        wt = meta_city.merge(wt, on="city_key", how="left")
        wt = wt.fillna({
            "expansion": 0,
            "tightening": 0,
            "clarification": 0,
            "reclassification": 0,
            "mixed": 0,
            "none": 0,
            "uncertain": 0,
            "wave_count": 0,
            "expansion_minus_tightening": 0,
        })

        # Normalize by panel span.
        span = (
            selected_issues[selected_issues["city_key"].isin(city_keys)]
            .assign(issue_date_parsed=lambda d: pd.to_datetime(d["issue_date"], errors="coerce"))
            .groupby("city_key", dropna=False)
            .agg(min_date=("issue_date_parsed", "min"), max_date=("issue_date_parsed", "max"))
            .reset_index()
        )
        span["span_years"] = (span["max_date"] - span["min_date"]).dt.days / 365.25
        span["span_decades"] = span["span_years"] / 10.0
        wt = wt.merge(span[["city_key", "span_years", "span_decades"]], on="city_key", how="left")
        wt["waves_per_decade"] = wt["wave_count"] / wt["span_decades"].replace(0, np.nan)
        wave_city = wt

    wave_city.to_csv(out_dir / "city_amendment_balance_metrics.csv", index=False)

    # Merge city-level indices.
    insight = selected[["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy"]].drop_duplicates().copy()
    insight = insight.merge(city_shift, on=["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy"], how="left")
    lag_cols = [
        "city_key",
        "first_zoning_issue_date",
        "first_zoning_signal_type",
        "first_full_issue_date",
        "first_full_status",
        "first_signal_to_full_lag_days",
        "first_signal_to_full_lag_years",
        "lag_observed",
    ]
    insight = insight.merge(first_events[lag_cols], on="city_key", how="left")
    if not wave_city.empty:
        insight = insight.merge(
            wave_city[
                [
                    "city_key",
                    "wave_count",
                    "waves_per_decade",
                    "expansion",
                    "tightening",
                    "clarification",
                    "reclassification",
                    "mixed",
                    "expansion_minus_tightening",
                    "clarification_share",
                    "reclassification_share",
                ]
            ],
            on="city_key",
            how="left",
        )
    insight.to_csv(out_dir / "city_insight_indices.csv", index=False)

    # Region-level summaries.
    reg_shift = (
        city_shift.groupby(["region", "urbanicity_proxy"], dropna=False)[
            ["substantive_shift_js", "proceduralization_delta", "flexibility_uptake_index"]
        ]
        .mean()
        .reset_index()
    )
    reg_shift.to_csv(out_dir / "region_shift_summary.csv", index=False)

    reg_wave = pd.DataFrame()
    if not waves.empty:
        reg_wave = (
            waves.groupby(["region", "urbanicity_proxy", "change_type"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "region", "urbanicity_proxy"], ascending=[False, True, True])
        )
    reg_wave.to_csv(out_dir / "region_wave_type_counts.csv", index=False)

    # Manual validation candidates linked to transcript text.
    issue_text_map: dict[tuple[str, str], dict[str, Any]] = {}
    for city in city_keys:
        p = run_dir / "panels" / city / "issue_texts.jsonl"
        for row in _iter_jsonl(p):
            iid = _norm_str(row.get("issue_id"))
            if not iid:
                continue
            issue_text_map[(city, iid)] = {
                "issue_date": _norm_str(row.get("issue_date")),
                "classification_label": _norm_str(row.get("classification_label")),
                "text": _norm_str(row.get("text")),
            }

    candidates: list[dict[str, Any]] = []

    # A) First-full ordinance cases.
    found = first_events[first_events["first_full_status"].astype(str) == "found"].copy()
    found = found.sort_values("first_full_issue_date").head(8)
    for r in found.itertuples(index=False):
        iid = _norm_str(getattr(r, "first_full_issue_id", ""))
        info = issue_text_map.get((r.city_key, iid), {})
        txt = _norm_str(info.get("text"))
        snippet = _extract_snippet(txt, [r"zoning\s+ordinance", r"ordinance\s+no", r"district", r"zone"]) if txt else ""
        candidates.append(
            {
                "claim_group": "first_full_ordinance",
                "claim": "Issue appears to contain full zoning ordinance language",
                "city_key": r.city_key,
                "issue_id": iid,
                "issue_date": _norm_str(info.get("issue_date") or getattr(r, "first_full_issue_date", "")),
                "classification_label": _norm_str(info.get("classification_label")),
                "source_metric": "first_full_status=found",
                "excerpt": snippet,
                "manual_verdict": "",
                "manual_note": "",
            }
        )

    # B) Largest flexibility uptake and proceduralization shifts.
    top_flex = insight.sort_values("flexibility_uptake_index", ascending=False).head(4)
    for r in top_flex.itertuples(index=False):
        city_issues = issue_level_out[issue_level_out["city_key"] == r.city_key].sort_values(["issue_date", "issue_id"])
        if city_issues.empty:
            continue
        flex_pats = [r"special\s+use", r"conditional\s+use", r"planned", r"overlay", r"variance", r"pud", r"mixed\s+use"]
        late_issues = city_issues[city_issues["tercile"] == "late"].copy()
        if late_issues.empty:
            continue
        late_issues["kw_score"] = late_issues["issue_id"].map(
            lambda iid: _keyword_score(_norm_str(issue_text_map.get((r.city_key, str(iid)), {}).get("text")), flex_pats)
        )
        late_issue = late_issues.sort_values(["kw_score", "issue_date", "issue_id"], ascending=[False, False, False]).head(1)
        if late_issue.empty:
            continue
        iid = str(late_issue.iloc[0]["issue_id"])
        info = issue_text_map.get((r.city_key, iid), {})
        txt = _norm_str(info.get("text"))
        snippet = _extract_snippet(txt, flex_pats) if txt else ""
        candidates.append(
            {
                "claim_group": "flexibility_uptake",
                "claim": "Late-period text suggests more flexible/special-use regulation",
                "city_key": r.city_key,
                "issue_id": iid,
                "issue_date": _norm_str(info.get("issue_date")),
                "classification_label": _norm_str(info.get("classification_label")),
                "source_metric": f"flexibility_uptake_index={_safe_float(r.flexibility_uptake_index):.4f}",
                "excerpt": snippet,
                "manual_verdict": "",
                "manual_note": "",
            }
        )

    top_proc = insight.sort_values("proceduralization_delta", ascending=False).head(4)
    for r in top_proc.itertuples(index=False):
        city_issues = issue_level_out[issue_level_out["city_key"] == r.city_key].sort_values(["issue_date", "issue_id"])
        if city_issues.empty:
            continue
        proc_pats = [r"board", r"commission", r"hearing", r"permit", r"variance", r"appeal", r"enforcement", r"certificate"]
        late_issues = city_issues[city_issues["tercile"] == "late"].copy()
        if late_issues.empty:
            continue
        late_issues["kw_score"] = late_issues["issue_id"].map(
            lambda iid: _keyword_score(_norm_str(issue_text_map.get((r.city_key, str(iid)), {}).get("text")), proc_pats)
        )
        late_issue = late_issues.sort_values(["kw_score", "issue_date", "issue_id"], ascending=[False, False, False]).head(1)
        if late_issue.empty:
            continue
        iid = str(late_issue.iloc[0]["issue_id"])
        info = issue_text_map.get((r.city_key, iid), {})
        txt = _norm_str(info.get("text"))
        snippet = _extract_snippet(txt, proc_pats) if txt else ""
        candidates.append(
            {
                "claim_group": "proceduralization",
                "claim": "Late-period text includes stronger procedural/governance language",
                "city_key": r.city_key,
                "issue_id": iid,
                "issue_date": _norm_str(info.get("issue_date")),
                "classification_label": _norm_str(info.get("classification_label")),
                "source_metric": f"proceduralization_delta={_safe_float(r.proceduralization_delta):.4f}",
                "excerpt": snippet,
                "manual_verdict": "",
                "manual_note": "",
            }
        )

    # C) One example for each amendment wave type.
    if not waves.empty:
        for change_type in ["expansion", "tightening", "clarification", "reclassification", "mixed"]:
            g = waves[waves["change_type"] == change_type].copy()
            if g.empty:
                continue
            g = g.sort_values(["city_key", "wave_order"])
            r = g.iloc[0]
            ids = [x.strip() for x in str(r.get("evidence_issue_ids", "")).split("|") if x.strip()]
            wave_pats = [r"amend", r"rezone", r"district", r"classification", r"zoning", r"special\s+use", r"hearing"]
            best_iid = ""
            best_score = -1
            for iid_cand in ids:
                txt_cand = _norm_str(issue_text_map.get((str(r["city_key"]), iid_cand), {}).get("text"))
                sc = _keyword_score(txt_cand, wave_pats)
                if sc > best_score:
                    best_score = sc
                    best_iid = iid_cand
            iid = best_iid if best_iid else (ids[0] if ids else "")
            info = issue_text_map.get((str(r["city_key"]), iid), {})
            txt = _norm_str(info.get("text"))
            snippet = _extract_snippet(txt, wave_pats) if txt else ""
            candidates.append(
                {
                    "claim_group": "amendment_wave_type",
                    "claim": f"Wave classified as {change_type}",
                    "city_key": str(r["city_key"]),
                    "issue_id": iid,
                    "issue_date": _norm_str(info.get("issue_date") or r.get("start_issue_date", "")),
                    "classification_label": _norm_str(info.get("classification_label")),
                    "source_metric": f"change_type={change_type}",
                    "excerpt": snippet,
                    "manual_verdict": "",
                    "manual_note": "",
                }
            )

    cand_df = pd.DataFrame.from_records(candidates).drop_duplicates(["claim_group", "city_key", "issue_id"]).reset_index(drop=True)
    cand_df.to_csv(out_dir / "manual_validation_candidates.csv", index=False)

    # If reviewed file exists, retain and merge with fresh candidates.
    reviewed_path = out_dir / "manual_validation_reviewed.csv"
    if reviewed_path.is_file():
        reviewed = _read_csv(reviewed_path)
        key_cols = ["claim_group", "city_key", "issue_id"]
        for c in key_cols:
            if c not in reviewed.columns:
                reviewed[c] = ""
        keep_cols = key_cols + ["manual_verdict", "manual_note"]
        reviewed = reviewed[keep_cols].drop_duplicates(key_cols)
        merged_review = cand_df.merge(reviewed, on=key_cols, how="left", suffixes=("", "_old"))
        for c in ["manual_verdict", "manual_note"]:
            if f"{c}_old" in merged_review.columns:
                merged_review[c] = merged_review[c].where(merged_review[c].astype(str).str.strip() != "", merged_review[f"{c}_old"])
                merged_review = merged_review.drop(columns=[f"{c}_old"])
        # Preserve manually added rows that are not present in auto candidates.
        cand_key_set = set(zip(cand_df["claim_group"], cand_df["city_key"], cand_df["issue_id"]))
        extra_mask = ~reviewed.apply(
            lambda r: (str(r["claim_group"]), str(r["city_key"]), str(r["issue_id"])) in cand_key_set,
            axis=1,
        )
        extra_rows = reviewed[extra_mask].copy()
        if not extra_rows.empty:
            for c in cand_df.columns:
                if c not in extra_rows.columns:
                    extra_rows[c] = ""
            extra_rows = extra_rows[cand_df.columns]
            merged_review = pd.concat([merged_review, extra_rows], ignore_index=True).drop_duplicates(key_cols, keep="first")
        merged_review.to_csv(reviewed_path, index=False)
    else:
        cand_df.to_csv(reviewed_path, index=False)

    # Summary stats for report.
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "longitudinal_dir": str(longitudinal_dir),
        "output_dir": str(out_dir),
        "selected_city_count": int(len(city_keys)),
        "issue_rows": int(len(issue_level_out)),
        "tercile_category_rows": int(len(cat_terc)),
        "phase_sequence_rows": int(len(seq_df)),
        "wave_balance_rows": int(len(wave_city)),
        "manual_candidates_rows": int(len(cand_df)),
        "metrics": {
            "lag_observed_share": float(first_events["lag_observed"].mean()) if not first_events.empty else None,
            "mean_substantive_shift_js": float(city_shift["substantive_shift_js"].mean()) if not city_shift.empty else None,
            "mean_proceduralization_delta": float(city_shift["proceduralization_delta"].mean()) if not city_shift.empty else None,
            "mean_flexibility_uptake": float(city_shift["flexibility_uptake_index"].mean()) if not city_shift.empty else None,
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"cities={len(city_keys)} "
        f"issue_rows={len(issue_level_out)} "
        f"manual_candidates={len(cand_df)} "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
