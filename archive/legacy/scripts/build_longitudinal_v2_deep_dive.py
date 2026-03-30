#!/usr/bin/env python3
"""
Build focused deep-dive manual validation artifacts for longitudinal v2.

Goal:
- use a small number of high-change towns,
- anchor each town in early vs late panel issues,
- extract long transcript excerpts that can be manually checked against index claims.
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


_GENERAL_PATTERNS = [
    r"zoning\s+ordinance",
    r"\bord(inance)?\b",
    r"\bdistrict\b",
    r"\bzone\b",
    r"\brezoning\b|\brezone\b",
    r"planning\s+commission",
    r"board\s+of\s+zoning\s+appeals|zoning\s+board\s+of\s+appeals|board\s+of\s+appeals",
    r"\bvariance\b",
    r"\bpermit\b",
    r"site\s+plan",
]

_PROCEDURAL_PATTERNS = [
    r"zoning\s+administrator",
    r"planning\s+commission",
    r"board\s+of\s+zoning\s+appeals|zoning\s+board\s+of\s+appeals|board\s+of\s+appeals",
    r"appeal(s)?",
    r"public\s+hearing",
    r"site\s+plan",
    r"certificate\s+of\s+occupancy",
    r"approval\s+subject\s+to\s+conditions|conditional\s+approval",
    r"enforcement|penalt(y|ies)",
    r"\bpermit\b",
]

_FLEX_PATTERNS = [
    r"planned\s+unit\s+development|pud",
    r"special\s+planned\s+development",
    r"special\s+use|conditional\s+use",
    r"large\s+scale\s+development",
    r"development\s+zone",
    r"mixed\s+use",
    r"overlay",
    r"\bvariance\b",
]

_LAND_USE_PATTERNS = [
    r"minimum\s+lot\s+size|min(imum)?\s+lot",
    r"setback|yard",
    r"height",
    r"permitted\s+uses?",
    r"district",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build deep-dive manual validation packets for v2 longitudinal analysis.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/pi_v1_run60",
        help="Base run directory.",
    )
    ap.add_argument(
        "--v2-dir",
        default="",
        help="v2 metrics directory (default: <run-dir>/longitudinal_v2).",
    )
    ap.add_argument(
        "--town-count",
        type=int,
        default=5,
        help="Number of towns for deep dive when --city-keys is not provided.",
    )
    ap.add_argument(
        "--city-keys",
        default="",
        help="Optional comma-separated city_key list to use for deep dive selection.",
    )
    ap.add_argument(
        "--excerpt-chars",
        type=int,
        default=2600,
        help="Target character length for each long excerpt.",
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


def _squash_ws(text: str) -> str:
    return re.sub(r"\s+", " ", _norm_str(text)).strip()


def _keyword_count(text: str, patterns: list[str]) -> int:
    if not text:
        return 0
    t = text.lower()
    score = 0
    for pat in patterns:
        score += len(re.findall(pat, t, flags=re.I))
    return int(score)


def _choose_focus(row: pd.Series) -> tuple[str, str]:
    proc = _safe_float(row.get("proceduralization_delta"))
    flex = _safe_float(row.get("flexibility_uptake_index"))
    if pd.notna(flex) and flex >= 0.08 and (pd.isna(proc) or abs(flex) >= 0.35 * abs(proc)):
        return "flexibility_uptake", "Late panel text shifts toward flexible/special-use instruments."
    if pd.notna(proc) and proc >= 0.15:
        return "proceduralization_up", "Late panel text shifts toward procedural/governance machinery."
    if pd.notna(proc) and proc <= -0.15:
        return "proceduralization_down", "Late panel text shifts away from procedural machinery toward use/bulk content."
    return "mixed_substantive_shift", "Panel shows broad compositional shift across zoning categories."


def _patterns_for_focus(focus: str) -> tuple[list[str], list[str]]:
    if focus.startswith("proceduralization"):
        return _PROCEDURAL_PATTERNS, _GENERAL_PATTERNS
    if focus == "flexibility_uptake":
        return _FLEX_PATTERNS, _GENERAL_PATTERNS
    return _LAND_USE_PATTERNS, _GENERAL_PATTERNS


def _select_anchor_issue(
    city_issues: pd.DataFrame,
    issue_texts: dict[str, str],
    tercile: str,
    primary_patterns: list[str],
    support_patterns: list[str],
    pick_latest: bool,
) -> tuple[str, str, str, int, int]:
    g = city_issues[city_issues["tercile"] == tercile].copy()
    if g.empty:
        return "", "", "", 0, 0
    g["kw_primary"] = g["issue_id"].map(lambda iid: _keyword_count(issue_texts.get(_norm_str(iid), ""), primary_patterns))
    g["kw_support"] = g["issue_id"].map(lambda iid: _keyword_count(issue_texts.get(_norm_str(iid), ""), support_patterns))
    g["issue_date"] = pd.to_datetime(g["issue_date"], errors="coerce")
    if pick_latest:
        g = g.sort_values(["kw_primary", "kw_support", "issue_date", "issue_id"], ascending=[False, False, False, False])
    else:
        g = g.sort_values(["kw_primary", "kw_support", "issue_date", "issue_id"], ascending=[False, False, True, True])
    r = g.head(1)
    if r.empty:
        return "", "", "", 0, 0
    rr = r.iloc[0]
    iid = _norm_str(rr.get("issue_id"))
    date = _norm_str(rr.get("issue_date"))
    label = _norm_str(rr.get("classification_label"))
    pscore = int(_safe_float(rr.get("kw_primary")) if pd.notna(_safe_float(rr.get("kw_primary"))) else 0)
    sscore = int(_safe_float(rr.get("kw_support")) if pd.notna(_safe_float(rr.get("kw_support"))) else 0)
    return iid, date, label, pscore, sscore


def _extract_long_excerpt(text: str, patterns: list[str], excerpt_chars: int) -> tuple[str, str, int]:
    t = _squash_ws(text)
    if not t:
        return "", "", -1
    for pat in patterns:
        m = re.search(pat, t, flags=re.I)
        if not m:
            continue
        pos = m.start()
        start = max(0, pos - int(excerpt_chars * 0.35))
        end = min(len(t), start + excerpt_chars)
        if end - start < excerpt_chars and start > 0:
            start = max(0, end - excerpt_chars)
        return t[start:end], pat, pos
    return t[:excerpt_chars], "start", 0


def _build_markdown_packet(df: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Deep-Dive Transcript Validation Packet")
    lines.append("")
    lines.append("Focus: fewer towns with obvious longitudinal shifts, each backed by long early/late transcript excerpts.")
    lines.append("")
    for _, r in df.iterrows():
        lines.append(f"## {r.get('city_display', r.get('city_key', ''))}")
        lines.append("")
        lines.append(f"- `focus_claim`: {r.get('focus_claim', '')}")
        lines.append(f"- `substantive_shift_js`: {r.get('substantive_shift_js', '')}")
        lines.append(f"- `proceduralization_delta`: {r.get('proceduralization_delta', '')}")
        lines.append(f"- `flexibility_uptake_index`: {r.get('flexibility_uptake_index', '')}")
        lines.append(f"- `manual_verdict`: {r.get('manual_verdict', '')}")
        lines.append(f"- `manual_note`: {r.get('manual_note', '')}")
        lines.append("")
        lines.append(
            f"### Early anchor (`{r.get('early_issue_id','')}`, {r.get('early_issue_date','')}, kw={r.get('early_kw_score','')})"
        )
        lines.append("")
        lines.append(r.get("early_excerpt", ""))
        lines.append("")
        lines.append(
            f"### Late anchor (`{r.get('late_issue_id','')}`, {r.get('late_issue_date','')}, kw={r.get('late_kw_score','')})"
        )
        lines.append("")
        lines.append(r.get("late_excerpt", ""))
        lines.append("")
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    v2_dir = Path(args.v2_dir).expanduser().resolve() if str(args.v2_dir).strip() else (run_dir / "longitudinal_v2")
    if not v2_dir.is_dir():
        raise SystemExit(f"Missing v2 directory: {v2_dir}")

    insight = _read_csv(v2_dir / "city_insight_indices.csv")
    issue_panel = _read_csv(v2_dir / "city_issue_level_panel.csv")
    if insight.empty:
        raise SystemExit(f"Missing city insights: {v2_dir / 'city_insight_indices.csv'}")
    if issue_panel.empty:
        raise SystemExit(f"Missing issue panel: {v2_dir / 'city_issue_level_panel.csv'}")

    d = insight.copy()
    d["selection_score_raw"] = (
        pd.to_numeric(d.get("substantive_shift_js"), errors="coerce").fillna(0.0)
        + 0.6 * pd.to_numeric(d.get("proceduralization_delta"), errors="coerce").abs().fillna(0.0)
        + 0.4 * pd.to_numeric(d.get("flexibility_uptake_index"), errors="coerce").abs().fillna(0.0)
    )

    focus_rows: list[dict[str, Any]] = []
    city_text_cache: dict[str, dict[str, dict[str, Any]]] = {}

    forced_keys = [x.strip() for x in str(args.city_keys).split(",") if x.strip()]
    if forced_keys:
        d = d[d["city_key"].astype(str).isin(forced_keys)].copy()
        if d.empty:
            raise SystemExit("No matching city_key rows found for --city-keys.")
    else:
        d = d.sort_values("selection_score_raw", ascending=False).copy()

    for row in d.itertuples(index=False):
        city_key = _norm_str(getattr(row, "city_key", ""))
        if not city_key:
            continue
        panel_city = issue_panel[issue_panel["city_key"].astype(str) == city_key].copy()
        if panel_city.empty:
            continue

        if city_key not in city_text_cache:
            texts_path = run_dir / "panels" / city_key / "issue_texts.jsonl"
            by_issue: dict[str, dict[str, Any]] = {}
            for rec in _iter_jsonl(texts_path):
                iid = _norm_str(rec.get("issue_id"))
                if not iid:
                    continue
                by_issue[iid] = {
                    "issue_date": _norm_str(rec.get("issue_date")),
                    "classification_label": _norm_str(rec.get("classification_label")),
                    "text": _squash_ws(rec.get("text")),
                }
            city_text_cache[city_key] = by_issue

        by_issue = city_text_cache.get(city_key, {})
        issue_texts = {iid: _norm_str(v.get("text")) for iid, v in by_issue.items()}
        focus_code, focus_claim = _choose_focus(pd.Series(row._asdict()))
        focus_primary_patterns, focus_support_patterns = _patterns_for_focus(focus_code)
        all_focus_patterns = focus_primary_patterns + focus_support_patterns
        early_iid, early_date, early_label, early_kw_primary, early_kw_support = _select_anchor_issue(
            panel_city,
            issue_texts,
            tercile="early",
            primary_patterns=focus_primary_patterns,
            support_patterns=focus_support_patterns,
            pick_latest=False,
        )
        late_iid, late_date, late_label, late_kw_primary, late_kw_support = _select_anchor_issue(
            panel_city,
            issue_texts,
            tercile="late",
            primary_patterns=focus_primary_patterns,
            support_patterns=focus_support_patterns,
            pick_latest=True,
        )
        if not early_iid or not late_iid:
            continue

        early_text = issue_texts.get(early_iid, "")
        late_text = issue_texts.get(late_iid, "")
        early_excerpt, early_anchor_pat, early_anchor_pos = _extract_long_excerpt(early_text, all_focus_patterns, int(args.excerpt_chars))
        late_excerpt, late_anchor_pat, late_anchor_pos = _extract_long_excerpt(late_text, all_focus_patterns, int(args.excerpt_chars))

        early_kw = int(early_kw_primary + early_kw_support)
        late_kw = int(late_kw_primary + late_kw_support)
        kw_min = min(early_kw, late_kw)
        selection_score = float(_safe_float(getattr(row, "selection_score_raw")) if pd.notna(_safe_float(getattr(row, "selection_score_raw"))) else 0.0)
        selection_score *= (1.0 + math.log1p(max(0, kw_min)) / 5.0)

        city_name = _norm_str(getattr(row, "city_name", ""))
        st = _norm_str(getattr(row, "state_abbr", ""))
        city_display = f"{city_name}, {st}" if city_name and st else city_key

        focus_rows.append(
            {
                "city_key": city_key,
                "city_name": city_name,
                "state_abbr": st,
                "city_display": city_display,
                "region": _norm_str(getattr(row, "region", "")),
                "urbanicity_proxy": _norm_str(getattr(row, "urbanicity_proxy", "")),
                "focus_code": focus_code,
                "focus_claim": focus_claim,
                "substantive_shift_js": _safe_float(getattr(row, "substantive_shift_js")),
                "proceduralization_delta": _safe_float(getattr(row, "proceduralization_delta")),
                "flexibility_uptake_index": _safe_float(getattr(row, "flexibility_uptake_index")),
                "selection_score_raw": _safe_float(getattr(row, "selection_score_raw")),
                "selection_score": selection_score,
                "early_issue_id": early_iid,
                "early_issue_date": early_date,
                "early_classification_label": early_label,
                "early_kw_score": int(early_kw),
                "early_kw_focus_score": int(early_kw_primary),
                "early_kw_support_score": int(early_kw_support),
                "early_anchor_pattern": early_anchor_pat,
                "early_anchor_pos": int(early_anchor_pos),
                "late_issue_id": late_iid,
                "late_issue_date": late_date,
                "late_classification_label": late_label,
                "late_kw_score": int(late_kw),
                "late_kw_focus_score": int(late_kw_primary),
                "late_kw_support_score": int(late_kw_support),
                "late_anchor_pattern": late_anchor_pat,
                "late_anchor_pos": int(late_anchor_pos),
                "early_excerpt": early_excerpt,
                "late_excerpt": late_excerpt,
                "manual_verdict": "",
                "manual_note": "",
            }
        )

    deep = pd.DataFrame.from_records(focus_rows)
    if deep.empty:
        raise SystemExit("No deep-dive towns could be assembled from inputs.")

    if forced_keys:
        forced_order = {k: i for i, k in enumerate(forced_keys)}
        deep["forced_ord"] = deep["city_key"].map(lambda x: forced_order.get(str(x), 10_000))
        deep = deep.sort_values(["forced_ord", "selection_score"], ascending=[True, False]).drop(columns=["forced_ord"])
    else:
        deep = deep.sort_values(["selection_score", "selection_score_raw", "city_key"], ascending=[False, False, True]).head(int(args.town_count))

    # Stable outputs.
    selection_cols = [
        "city_key",
        "city_name",
        "state_abbr",
        "city_display",
        "region",
        "urbanicity_proxy",
        "focus_code",
        "focus_claim",
        "substantive_shift_js",
        "proceduralization_delta",
        "flexibility_uptake_index",
        "selection_score_raw",
        "selection_score",
        "early_issue_id",
        "early_issue_date",
        "late_issue_id",
        "late_issue_date",
        "early_kw_score",
        "late_kw_score",
    ]
    deep_sel = deep[selection_cols].copy()
    deep_sel.to_csv(v2_dir / "deep_dive_selection.csv", index=False)

    reviewed_path = v2_dir / "deep_dive_manual_review.csv"
    if reviewed_path.is_file():
        old = _read_csv(reviewed_path)
        key = ["city_key"]
        keep_cols = key + ["manual_verdict", "manual_note"]
        for c in keep_cols:
            if c not in old.columns:
                old[c] = ""
        old = old[keep_cols].drop_duplicates(key)
        merged = deep.merge(old, on="city_key", how="left", suffixes=("", "_old"))
        for c in ["manual_verdict", "manual_note"]:
            if f"{c}_old" in merged.columns:
                merged[c] = merged[c].where(merged[c].astype(str).str.strip() != "", merged[f"{c}_old"])
                merged = merged.drop(columns=[f"{c}_old"])
        deep = merged
    deep.to_csv(reviewed_path, index=False)

    excerpt_rows: list[dict[str, Any]] = []
    for _, r in deep.iterrows():
        for role in ["early", "late"]:
            excerpt_rows.append(
                {
                    "city_key": r["city_key"],
                    "city_display": r["city_display"],
                    "focus_claim": r["focus_claim"],
                    "role": role,
                    "issue_id": r[f"{role}_issue_id"],
                    "issue_date": r[f"{role}_issue_date"],
                    "classification_label": r[f"{role}_classification_label"],
                    "kw_score": r[f"{role}_kw_score"],
                    "anchor_pattern": r[f"{role}_anchor_pattern"],
                    "excerpt": r[f"{role}_excerpt"],
                }
            )
    excerpt_df = pd.DataFrame.from_records(excerpt_rows)
    excerpt_df.to_csv(v2_dir / "deep_dive_excerpts_long.csv", index=False)

    _build_markdown_packet(deep, v2_dir / "deep_dive_manual_packet.md")

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "v2_dir": str(v2_dir),
        "town_count_requested": int(args.town_count),
        "city_keys_forced": forced_keys,
        "excerpt_chars": int(args.excerpt_chars),
        "selected_city_count": int(len(deep)),
    }
    (v2_dir / "deep_dive_provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"selected_cities={len(deep)} "
        f"selection_file={v2_dir / 'deep_dive_selection.csv'} "
        f"review_file={reviewed_path}"
    )


if __name__ == "__main__":
    main()
