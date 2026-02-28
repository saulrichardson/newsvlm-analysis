#!/usr/bin/env python3
"""
Build a LaTeX report for issue-level zoning question extraction outputs.

The report is designed to be human-readable:
- no machine question IDs in visible labels,
- informative chart titles/legends,
- detailed caption notes explaining construction.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INTERNAL_LABEL_FULL = "full_ordinance"
INTERNAL_LABEL_AMEND = "amendment"
V5_LABEL_FULL = "zoning_ordinance_comprehensive"
V5_LABEL_AMEND = "zoning_amendment_or_rezoning"
V5_FRIENDLY_FULL = "V5 zoning ordinance (comprehensive)"
V5_FRIENDLY_AMEND = "V5 zoning amendment or rezoning"
ALL_V5_LABELS = [
    "zoning_ordinance_comprehensive",
    "zoning_ordinance_noncomprehensive",
    "zoning_amendment_or_rezoning",
    "zoning_legal_notice",
    "building_code_or_other_law",
    "zoning_narrative_nonverbatim",
    "non_zoning",
    "uncertain",
]
V5_LABEL_ORDER = [
    "zoning_ordinance_comprehensive",
    "zoning_ordinance_noncomprehensive",
    "zoning_amendment_or_rezoning",
    "zoning_legal_notice",
    "building_code_or_other_law",
    "zoning_narrative_nonverbatim",
    "non_zoning",
    "uncertain",
]
INTERNAL_TO_V5_LABEL = {
    INTERNAL_LABEL_FULL: V5_LABEL_FULL,
    INTERNAL_LABEL_AMEND: V5_LABEL_AMEND,
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build LaTeX report for issue-level numeric/procedural extraction outputs.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/issue_numeric_extraction_full_20260226_020305",
        help="Run directory containing issue_numeric_observations.csv and issue_request_manifest.csv.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output report directory. Default: <run-dir>/latex_issue_question_report",
    )
    ap.add_argument(
        "--newspapers-meta",
        default="/Users/saulrichardson/Dropbox/Inclusionary Zoning/saul/newspapers/page_artifact_packets_v5_full_20260225/metadata_core/newspapers.csv",
        help="Path to newspapers metadata CSV with newspaper_slug and state_abbr.",
    )
    ap.add_argument(
        "--prompt-template",
        default="",
        help="Optional explicit prompt template path. If omitted, appendix pulls prompt text from first request payload in run_dir/requests.",
    )
    ap.add_argument(
        "--v5-issue-labels",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/tmp/page_classify_v5_promptv5_issueprep_outputs_20260227/issue_predictions_parsed.csv",
        help="Issue-level V5 classification CSV with columns issue_id,page_class.",
    )
    ap.add_argument("--compile-pdf", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


QUESTION_LABELS = {
    "q_num_sf_min_lot_area": "Minimum lot area for single-family districts",
    "q_num_sf_min_lot_width": "Minimum lot width/frontage for single-family districts",
    "q_num_sf_front_setback_min": "Minimum front setback for single-family districts",
    "q_num_sf_side_setback_min": "Minimum side setback for single-family districts",
    "q_num_sf_rear_setback_min": "Minimum rear setback for single-family districts",
    "q_num_sf_max_height_ft": "Maximum building height (feet) in single-family districts",
    "q_num_sf_max_height_stories": "Maximum building height (stories) in single-family districts",
    "q_num_sf_max_lot_coverage": "Maximum lot coverage in single-family districts",
    "q_num_sf_min_floor_area": "Minimum floor area in single-family districts",
    "q_num_sf_parking_spaces_per_dwelling": "Minimum off-street parking spaces per dwelling unit",
    "q_num_sf_lot_area_bounds": "Within-issue lot-area range across single-family districts",
    "q_num_sf_front_setback_bounds": "Within-issue front-setback range across single-family districts",
    "q_num_sf_side_setback_bounds": "Within-issue side-setback range across single-family districts",
    "q_proc_permit_conformance_planning_review": "Permit conformance review by planning body",
    "q_proc_site_plan_review_explicit": "Explicit site-plan review requirement",
    "q_proc_subdivision_or_plat_approval": "Subdivision/plat approval requirement",
    "q_proc_use_variance_availability": "Use variance availability",
    "q_proc_use_variance_decider": "Primary decision body for use variance",
    "q_proc_use_variance_standard": "Use variance legal standard",
    "q_proc_area_variance_availability": "Area/bulk variance availability",
    "q_proc_special_permit_or_conditional_use": "Special permit / conditional use requirement",
    "q_proc_public_hearing_required": "Public hearing requirement",
    "q_proc_notice_publication_required": "Published legal-notice requirement",
    "q_proc_certificate_of_occupancy": "Certificate of occupancy/compliance requirement",
    "q_proc_nonconforming_use": "Presence of nonconforming-use framework",
    "q_proc_enforcement_appeals_presence": "Presence of enforcement/appeals framework",
    "q_proc_enforcement_penalty": "Explicit penalty caps (fine/jail days)",
}


YES_NO_MAP = {
    "required": "Required",
    "not_required": "Not required",
    "available": "Available",
    "not_available": "Not available",
    "present": "Present",
    "absent": "Absent",
    "uncertain": "Unclear",
}


DECIDER_MAP = {
    "zba": "Zoning Board of Appeals",
    "board_of_appeals": "Board of Appeals",
    "planning_commission": "Planning Commission",
    "city_council": "City Council",
    "multiple": "Multiple bodies",
    "court_only": "Court only",
    "uncertain": "Unclear",
}


STANDARD_MAP = {
    "unnecessary_hardship": "Unnecessary hardship",
    "practical_difficulty": "Practical difficulty",
    "special_conditions": "Special conditions",
    "not_stated": "Standard not stated",
    "uncertain": "Unclear",
}


def _slug(s: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(s).lower()).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out


def _fmt_int(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def _fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{100.0 * float(x):.{digits}f}\\%"


def _trim_iqr_frame(d: pd.DataFrame, value_col: str = "value_number") -> pd.DataFrame:
    """Trim extreme OCR outliers for trend display only (1st to 99th percentile)."""
    if d.empty:
        return d
    lo = d[value_col].quantile(0.01)
    hi = d[value_col].quantile(0.99)
    return d[(d[value_col] >= lo) & (d[value_col] <= hi)].copy()


def _state_to_region(state_abbr: str) -> str:
    s = str(state_abbr or "").strip().upper()
    northeast = {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"}
    midwest = {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"}
    south = {"DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"}
    west = {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"}
    if s in northeast:
        return "Northeast"
    if s in midwest:
        return "Midwest"
    if s in south:
        return "South"
    if s in west:
        return "West"
    return "Unknown"


def _label_display(internal_label: str) -> str:
    s = str(internal_label).strip()
    if s == INTERNAL_LABEL_FULL:
        return V5_FRIENDLY_FULL
    if s == INTERNAL_LABEL_AMEND:
        return V5_FRIENDLY_AMEND
    return s.replace("_", " ").strip()


def _primary_label_to_v5(s: str) -> str:
    x = str(s).strip()
    return INTERNAL_TO_V5_LABEL.get(x, x)


def _issue_level_binary_share_by_decade(
    obs: pd.DataFrame, *, question_id: str, yes_values: set[str], no_values: set[str]
) -> pd.DataFrame:
    """Compute issue-level binary shares by decade from categorical rows."""
    d = obs[obs["question_id"] == question_id].copy()
    if d.empty:
        return pd.DataFrame(columns=["decade", "share_yes", "resolved_issues"])
    rows: list[dict[str, Any]] = []
    for (issue_id, decade), g in d.groupby(["issue_id", "decade"]):
        vals = {str(v).strip().lower() for v in g["value_category"].tolist() if str(v).strip()}
        yes = any(v in yes_values for v in vals)
        no = any(v in no_values for v in vals)
        if yes and not no:
            b = 1.0
        elif no and not yes:
            b = 0.0
        else:
            b = math.nan
        rows.append({"issue_id": issue_id, "decade": int(decade), "binary": b})
    tmp = pd.DataFrame(rows)
    out = (
        tmp.groupby("decade", as_index=False)
        .agg(share_yes=("binary", "mean"), resolved_issues=("binary", lambda s: int(s.notna().sum())))
        .sort_values("decade")
    )
    return out


def _dominant_category_share_by_decade(
    obs: pd.DataFrame, *, question_id: str, label_map: dict[str, str], top_n: int = 5
) -> pd.DataFrame:
    """Use issue-level dominant category, then compute decade shares."""
    d = obs[obs["question_id"] == question_id].copy()
    if d.empty:
        return pd.DataFrame(columns=["decade", "category_label", "share", "issues"])
    d["value_category"] = d["value_category"].astype(str).str.strip().str.lower()
    d = d[d["value_category"] != ""].copy()
    if d.empty:
        return pd.DataFrame(columns=["decade", "category_label", "share", "issues"])
    per_issue = (
        d.groupby(["issue_id", "decade", "value_category"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["issue_id", "decade", "rows", "value_category"], ascending=[True, True, False, True])
    )
    dom = per_issue.drop_duplicates(["issue_id", "decade"]).copy()
    top_vals = dom["value_category"].value_counts().head(top_n).index.tolist()
    dom["value_group"] = dom["value_category"].where(dom["value_category"].isin(top_vals), "other")
    dom["category_label"] = dom["value_group"].map(label_map).fillna(dom["value_group"].str.replace("_", " ").str.title())
    out = dom.groupby(["decade", "category_label"], as_index=False).size().rename(columns={"size": "issues"})
    totals = out.groupby("decade", as_index=False)["issues"].sum().rename(columns={"issues": "decade_total"})
    out = out.merge(totals, on="decade", how="left")
    out["share"] = out["issues"] / out["decade_total"]
    out = out.sort_values(["decade", "share"], ascending=[True, False]).reset_index(drop=True)
    return out


def _dominant_category_share_by_region(
    obs: pd.DataFrame, *, question_id: str, label_map: dict[str, str], top_n: int = 5
) -> pd.DataFrame:
    """Use issue-level dominant category, then compute region shares."""
    d = obs[obs["question_id"] == question_id].copy()
    if d.empty:
        return pd.DataFrame(columns=["region", "category_label", "share", "issues"])
    d["value_category"] = d["value_category"].astype(str).str.strip().str.lower()
    d = d[d["value_category"] != ""].copy()
    d = d[d["region"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["region", "category_label", "share", "issues"])
    per_issue = (
        d.groupby(["issue_id", "region", "value_category"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["issue_id", "region", "rows", "value_category"], ascending=[True, True, False, True])
    )
    dom = per_issue.drop_duplicates(["issue_id", "region"]).copy()
    top_vals = dom["value_category"].value_counts().head(top_n).index.tolist()
    dom["value_group"] = dom["value_category"].where(dom["value_category"].isin(top_vals), "other")
    dom["category_label"] = dom["value_group"].map(label_map).fillna(dom["value_group"].str.replace("_", " ").str.title())
    out = dom.groupby(["region", "category_label"], as_index=False).size().rename(columns={"size": "issues"})
    totals = out.groupby("region", as_index=False)["issues"].sum().rename(columns={"issues": "region_total"})
    out = out.merge(totals, on="region", how="left")
    out["share"] = out["issues"] / out["region_total"]
    out = out.sort_values(["region", "share"], ascending=[True, False]).reset_index(drop=True)
    return out


def _numeric_trend_by_decade(obs: pd.DataFrame, *, question_id: str, bound_role: str | None = None) -> pd.DataFrame:
    d = obs[(obs["question_id"] == question_id) & obs["value_number"].notna()].copy()
    if bound_role is not None:
        d = d[d["bound_role"] == bound_role].copy()
    if d.empty:
        return pd.DataFrame(columns=["decade", "median_value", "p25", "p75", "rows"])
    d = _trim_iqr_frame(d, value_col="value_number")
    if d.empty:
        return pd.DataFrame(columns=["decade", "median_value", "p25", "p75", "rows"])
    out = (
        d.groupby("decade", as_index=False)
        .agg(
            median_value=("value_number", "median"),
            p25=("value_number", lambda s: s.quantile(0.25)),
            p75=("value_number", lambda s: s.quantile(0.75)),
            rows=("value_number", "size"),
        )
        .sort_values("decade")
    )
    return out


def _share_axis_limits(series_list: list[pd.Series], *, min_span: float = 0.18, pad: float = 0.03) -> tuple[float, float]:
    vals: list[float] = []
    for s in series_list:
        if s is None:
            continue
        arr = pd.to_numeric(s, errors="coerce").dropna()
        if not arr.empty:
            vals.extend(arr.tolist())
    if not vals:
        return (0.0, 1.0)
    lo = max(0.0, float(min(vals)) - pad)
    hi = min(1.0, float(max(vals)) + pad)
    if hi - lo < min_span:
        mid = (lo + hi) / 2.0
        lo = max(0.0, mid - min_span / 2.0)
        hi = min(1.0, mid + min_span / 2.0)
        if hi - lo < min_span:
            if lo <= 0.0:
                hi = min(1.0, lo + min_span)
            elif hi >= 1.0:
                lo = max(0.0, hi - min_span)
    return (lo, hi)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tex_escape(s: str) -> str:
    rep = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(rep.get(ch, ch))
    return "".join(out)


def _norm_geoid7(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    if "." in s:
        s = s.split(".", 1)[0]
    if not s.isdigit():
        s = "".join(ch for ch in s if ch.isdigit())
    return s.zfill(7) if s else ""


def _fetch_acs_place_state(state_fips: str, year: int = 2022) -> pd.DataFrame:
    sf = str(state_fips).zfill(2)
    params = {
        "get": ",".join(
            [
                "B01003_001E",  # population
                "B19013_001E",  # median hh income
                "B17001_001E",  # poverty denom
                "B17001_002E",  # poverty numerator
                "B25003_001E",  # occupied units
                "B25003_002E",  # owner occupied
                "B25024_001E",  # total units by structure
                "B25024_002E",  # 1-unit detached
            ]
        ),
        "for": "place:*",
        "in": f"state:{sf}",
    }
    url = f"https://api.census.gov/data/{year}/acs/acs5?{urlencode(params)}"
    with urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    if not data or len(data) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(data[1:], columns=data[0])
    num_cols = [
        "B01003_001E",
        "B19013_001E",
        "B17001_001E",
        "B17001_002E",
        "B25003_001E",
        "B25003_002E",
        "B25024_001E",
        "B25024_002E",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # ACS missing sentinels are negative.
    for c in num_cols:
        df.loc[df[c] < 0, c] = np.nan
    df["place_geoid7"] = df["state"].astype(str).str.zfill(2) + df["place"].astype(str).str.zfill(5)
    df["acs_population"] = df["B01003_001E"]
    df["acs_median_hh_income"] = df["B19013_001E"]
    df["poverty_rate_pct"] = np.where(
        df["B17001_001E"] > 0,
        100.0 * (df["B17001_002E"] / df["B17001_001E"]),
        np.nan,
    )
    df["homeownership_rate_pct"] = np.where(
        df["B25003_001E"] > 0,
        100.0 * (df["B25003_002E"] / df["B25003_001E"]),
        np.nan,
    )
    df["share_1unit_detached_pct"] = np.where(
        df["B25024_001E"] > 0,
        100.0 * (df["B25024_002E"] / df["B25024_001E"]),
        np.nan,
    )
    keep = [
        "place_geoid7",
        "acs_population",
        "acs_median_hh_income",
        "poverty_rate_pct",
        "homeownership_rate_pct",
        "share_1unit_detached_pct",
    ]
    return df[keep].copy()


def _fetch_acs_place_panel(state_fips_list: list[str], *, cache_csv: Path) -> pd.DataFrame:
    if cache_csv.is_file():
        try:
            d = pd.read_csv(cache_csv, low_memory=False)
            if "place_geoid7" in d.columns:
                d["place_geoid7"] = d["place_geoid7"].astype(str).str.zfill(7)
                return d
        except Exception:
            pass
    frames: list[pd.DataFrame] = []
    for sf in sorted({str(s).zfill(2) for s in state_fips_list if str(s).strip()}):
        try:
            frames.append(_fetch_acs_place_state(sf))
        except Exception:
            continue
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_csv, index=False)
    return out


def _load_place_area_sqmi(*, cache_csv: Path) -> pd.DataFrame:
    if cache_csv.is_file():
        d = pd.read_csv(cache_csv, low_memory=False)
        if "place_geoid7" in d.columns and "aland_sqmi" in d.columns:
            d["place_geoid7"] = d["place_geoid7"].astype(str).str.zfill(7)
            return d[["place_geoid7", "aland_sqmi"]].copy()
    url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_place_national.zip"
    with urlopen(url, timeout=120) as r:
        raw = r.read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    txt_name = [n for n in zf.namelist() if n.lower().endswith(".txt")][0]
    df = pd.read_csv(zf.open(txt_name), sep="\t", low_memory=False)
    df["place_geoid7"] = df["GEOID"].map(_norm_geoid7)
    df["aland_sqmi"] = pd.to_numeric(df["ALAND_SQMI"], errors="coerce")
    out = df[["place_geoid7", "aland_sqmi"]].dropna().drop_duplicates("place_geoid7")
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_csv, index=False)
    return out


def _load_prompt_text_for_appendix(run_dir: Path, explicit_prompt_path: str = "") -> tuple[str, str]:
    if explicit_prompt_path:
        p = Path(explicit_prompt_path).expanduser().resolve()
        if p.is_file():
            return p.read_text(encoding="utf-8"), f"prompt template"
    req_path = run_dir / "requests" / "openai_requests_shard000.jsonl"
    if req_path.is_file():
        with req_path.open("r", encoding="utf-8") as f:
            line = f.readline()
        if line.strip():
            rec = json.loads(line)
            txt = (
                rec.get("body", {})
                .get("input", [{}])[0]
                .get("content", [{}])[0]
                .get("text", "")
            )
            if isinstance(txt, str) and txt.strip():
                head = txt.split("\nIssue metadata:", 1)[0]
                return head, "request payload"
    return "", "unavailable"


def _extract_question_catalog_from_prompt(prompt_text: str) -> list[dict[str, str]]:
    lines = prompt_text.splitlines()
    qpat = re.compile(r"^\s*-\s*(q_[a-z0-9_]+)\s*:\s*$")
    section_pat = re.compile(r"^[A-Za-z][A-Za-z0-9 /()_\-]*:\s*$")
    stop_headers = {
        "bound role guidance:",
        "allowed enums:",
        "allowed value_category enums by question_id:",
        "json output schema:",
        "validation constraints:",
        "issue metadata:",
        "issue pages and ocr transcripts:",
    }
    rows: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        m = qpat.match(lines[i] or "")
        if not m:
            i += 1
            continue
        qid = m.group(1).strip()
        j = i + 1
        chunk: list[str] = []
        while j < len(lines):
            raw = lines[j] or ""
            if qpat.match(raw):
                break
            s = raw.strip()
            sl = s.lower()
            if sl in stop_headers:
                break
            if section_pat.match(s) and not s.startswith("q_"):
                break
            # Stop if we hit unrelated list bullets after the question body.
            if s.startswith("- ") and not qpat.match(s) and chunk:
                break
            if s:
                chunk.append(s)
            j += 1
        # Keep only the human-readable prompt sentence(s), not metric/unit/schema lines.
        desc_lines: list[str] = []
        for s in chunk:
            sl = s.lower()
            if sl.startswith("metric=") or sl.startswith("metric in "):
                break
            if sl.startswith("unit=") or sl.startswith("bound_role in "):
                break
            if sl.startswith("legal_status"):
                break
            desc_lines.append(s)
        qtxt = re.sub(r"\s+", " ", " ".join(desc_lines)).strip()
        if not qtxt and chunk:
            qtxt = re.sub(r"\s+", " ", chunk[0]).strip()
        qtxt = re.sub(r"\s+", " ", qtxt).strip()
        rows.append({"question_id": qid, "question_text": qtxt})
        i = j
    return rows


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir.strip() else (run_dir / "latex_issue_question_report")
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    obs_path = run_dir / "issue_numeric_observations.csv"
    manifest_path = run_dir / "issue_request_manifest.csv"
    v5_labels_path = Path(args.v5_issue_labels).expanduser().resolve()
    newspapers_meta_path = Path(args.newspapers_meta).expanduser().resolve()
    if not obs_path.is_file() or not manifest_path.is_file():
        raise SystemExit(f"Missing required files in run dir: {run_dir}")
    if not v5_labels_path.is_file():
        raise SystemExit(f"Missing --v5-issue-labels file: {v5_labels_path}")

    obs = pd.read_csv(obs_path, low_memory=False)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    obs["issue_date"] = pd.to_datetime(obs["issue_date"], errors="coerce")
    obs = obs[obs["issue_date"].notna()].copy()
    obs["year"] = obs["issue_date"].dt.year.astype(int)
    obs["decade"] = (obs["year"] // 10) * 10
    obs["value_category"] = obs["value_category"].fillna("").astype(str).str.strip().str.lower()
    obs["has_numeric"] = obs["value_number"].notna()
    obs["has_category"] = obs["value_category"].str.len().gt(0)

    manifest["issue_date"] = pd.to_datetime(manifest["issue_date"], errors="coerce")
    manifest = manifest[manifest["issue_date"].notna()].copy()
    manifest["year"] = manifest["issue_date"].dt.year.astype(int)
    manifest["decade"] = (manifest["year"] // 10) * 10

    # True V5-native relabeling:
    # 1) restrict sample to issues with at least one extracted observation row
    # 2) attach issue-level V5 label directly from classifier output
    # 3) drop issues lacking a V5 label in this sample
    issue_ids_with_answers = set(obs["issue_id"].astype(str))
    manifest["issue_id"] = manifest["issue_id"].astype(str)
    manifest = manifest[manifest["issue_id"].isin(issue_ids_with_answers)].copy()
    v5 = pd.read_csv(v5_labels_path, low_memory=False)
    required_v5_cols = {"issue_id", "page_class"}
    if not required_v5_cols.issubset(set(v5.columns)):
        raise SystemExit(f"--v5-issue-labels must include columns {sorted(required_v5_cols)}; got {v5.columns.tolist()}")
    v5 = v5[["issue_id", "page_class"]].copy()
    v5["issue_id"] = v5["issue_id"].astype(str)
    v5["v5_issue_label"] = v5["page_class"].fillna("").astype(str).str.strip()
    v5 = v5.drop(columns=["page_class"]).drop_duplicates("issue_id")
    manifest = manifest.merge(v5, on="issue_id", how="left")
    manifest["v5_issue_label"] = manifest["v5_issue_label"].fillna("").astype(str).str.strip()
    manifest = manifest[(manifest["v5_issue_label"] != "") & (manifest["v5_issue_label"].str.lower() != "nan")].copy()
    obs = obs[obs["issue_id"].astype(str).isin(set(manifest["issue_id"].astype(str)))].copy()

    # Attach region from newspaper metadata via state abbreviation.
    if newspapers_meta_path.is_file():
        pubs = pd.read_csv(newspapers_meta_path, low_memory=False)
        pubs["newspaper_slug"] = pubs["newspaper_slug"].astype(str).str.strip()
        pubs["state_abbr"] = pubs["state_abbr"].fillna("").astype(str).str.strip()
        pubs["region"] = pubs["state_abbr"].map(_state_to_region)
        region_map = pubs[["newspaper_slug", "state_abbr", "region"]].drop_duplicates("newspaper_slug")
        manifest = manifest.merge(region_map, on="newspaper_slug", how="left")
        obs = obs.merge(region_map[["newspaper_slug", "region"]], on="newspaper_slug", how="left")
    else:
        manifest["region"] = "Unknown"
        obs["region"] = "Unknown"
    manifest["region"] = manifest["region"].fillna("Unknown")
    obs["region"] = obs["region"].fillna("Unknown")
    region_order = [r for r in ["Northeast", "Midwest", "South", "West"] if r in set(manifest["region"])]
    if "Unknown" in set(manifest["region"]):
        region_order.append("Unknown")
    plot_regions = [r for r in ["Northeast", "Midwest", "South", "West"] if r in set(region_order)]
    if not plot_regions and region_order:
        plot_regions = region_order[:]

    prompt_text, prompt_source = _load_prompt_text_for_appendix(run_dir, args.prompt_template)
    prompt_question_catalog = _extract_question_catalog_from_prompt(prompt_text) if prompt_text else []
    if prompt_question_catalog:
        pd.DataFrame(prompt_question_catalog).to_csv(
            table_dir / "appendix_question_catalog_from_prompt.csv", index=False
        )
    if prompt_text:
        _write_text(output_dir / "appendix_prompt_verbatim.txt", prompt_text.rstrip() + "\n")

    # ---- Census-linked panel profile (city-level) ----
    census_panel_display_rows: list[tuple[str, str, bool]] = []  # (label, value, is_panel_header)
    census_panel_ready = False
    try:
        meta_root = newspapers_meta_path.parent
        enriched_path = meta_root / "newspapers_enriched.csv"
        city_place_path = meta_root / "census" / "city_place_crosswalk.csv"
        city_keys_path = meta_root / "census" / "city_census_keys.csv"
        if enriched_path.is_file() and city_place_path.is_file() and city_keys_path.is_file():
            run_slugs = manifest["newspaper_slug"].dropna().astype(str).str.strip().unique().tolist()
            ne = pd.read_csv(enriched_path, low_memory=False)
            ne["newspaper_slug"] = ne["newspaper_slug"].astype(str).str.strip()
            run_np = ne[ne["newspaper_slug"].isin(run_slugs)].copy()
            run_np = run_np.drop_duplicates("newspaper_slug")
            city_keys = run_np["city_key"].dropna().astype(str).unique().tolist()

            city_place = pd.read_csv(city_place_path, low_memory=False)
            city_keys_df = pd.read_csv(city_keys_path, low_memory=False)
            city = pd.DataFrame({"city_key": city_keys})
            city = city.merge(
                city_place[
                    [
                        "city_key",
                        "city_name",
                        "state_abbr",
                        "region",
                        "state_fips",
                        "place_pop10",
                        "place_geoid",
                    ]
                ],
                on="city_key",
                how="left",
            )
            city = city.merge(
                city_keys_df[
                    [
                        "city_key",
                        "place20_geoid",
                        "cbsa23_count",
                    ]
                ],
                on="city_key",
                how="left",
            )
            city["state_abbr"] = city["state_abbr"].fillna("").astype(str).str.upper()
            city["region"] = city["state_abbr"].map(_state_to_region)
            city["place_geoid7"] = city["place20_geoid"].map(_norm_geoid7)
            miss_mask = city["place_geoid7"].eq("")
            city.loc[miss_mask, "place_geoid7"] = city.loc[miss_mask, "place_geoid"].map(_norm_geoid7)
            city["state_fips2"] = (
                pd.to_numeric(city["state_fips"], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
                .str.zfill(2)
            )
            city["in_msa"] = pd.to_numeric(city["cbsa23_count"], errors="coerce").fillna(0).gt(0)

            acs = _fetch_acs_place_panel(
                city["state_fips2"].dropna().astype(str).tolist(),
                cache_csv=table_dir / "cache_acs_place_2022.csv",
            )
            area = _load_place_area_sqmi(cache_csv=table_dir / "cache_place_area_sqmi.csv")
            city = city.merge(acs, on="place_geoid7", how="left")
            city = city.merge(area, on="place_geoid7", how="left")
            city["acs_pop_density_per_sqmi"] = np.where(
                pd.to_numeric(city["aland_sqmi"], errors="coerce") > 0,
                pd.to_numeric(city["acs_population"], errors="coerce") / pd.to_numeric(city["aland_sqmi"], errors="coerce"),
                np.nan,
            )

            city.to_csv(table_dir / "census_city_profile_rows.csv", index=False)

            n_cities = int(city["city_key"].nunique())
            n_states = int(city["state_abbr"].replace("", np.nan).dropna().nunique())
            reg_den = float(max(n_cities, 1))
            reg_share = city["region"].value_counts(dropna=False).to_dict()
            ne_share = 100.0 * (reg_share.get("Northeast", 0) / reg_den)
            mw_share = 100.0 * (reg_share.get("Midwest", 0) / reg_den)
            so_share = 100.0 * (reg_share.get("South", 0) / reg_den)
            we_share = 100.0 * (reg_share.get("West", 0) / reg_den)
            unk_share = 100.0 * (reg_share.get("Unknown", 0) / reg_den)
            msa_share = 100.0 * float(city["in_msa"].mean()) if n_cities else math.nan

            pop10_med = pd.to_numeric(city["place_pop10"], errors="coerce").median()
            pop_acs_med = pd.to_numeric(city["acs_population"], errors="coerce").median()
            dens_acs_med = pd.to_numeric(city["acs_pop_density_per_sqmi"], errors="coerce").median()
            homeown_med = pd.to_numeric(city["homeownership_rate_pct"], errors="coerce").median()
            det1_med = pd.to_numeric(city["share_1unit_detached_pct"], errors="coerce").median()
            income_med = pd.to_numeric(city["acs_median_hh_income"], errors="coerce").median()
            pov_med = pd.to_numeric(city["poverty_rate_pct"], errors="coerce").median()

            census_panel_display_rows = [
                ("Panel A. Coverage", "", True),
                ("Municipalities (N)", f"{n_cities:,}", False),
                ("States represented", f"{n_states:,}", False),
                ("Panel B. Geography and metro status", "", True),
                ("Northeast share (%)", f"{ne_share:.1f}", False),
                ("Midwest share (%)", f"{mw_share:.1f}", False),
                ("South share (%)", f"{so_share:.1f}", False),
                ("West share (%)", f"{we_share:.1f}", False),
                ("Unknown region share (%)", f"{unk_share:.1f}", False),
                ("In MSA share (%)", f"{msa_share:.1f}" if not math.isnan(msa_share) else "NA", False),
                ("Panel C. Size and density (2010 and ACS)", "", True),
                ("Population (2010), median", f"{int(round(pop10_med)):,}" if pd.notna(pop10_med) else "NA", False),
                ("Population (ACS), median", f"{int(round(pop_acs_med)):,}" if pd.notna(pop_acs_med) else "NA", False),
                ("Population density (ACS), median", f"{dens_acs_med:,.1f}" if pd.notna(dens_acs_med) else "NA", False),
                ("Panel D. Housing and socioeconomic (ACS)", "", True),
                ("Homeownership rate (%)", f"{homeown_med:.1f}" if pd.notna(homeown_med) else "NA", False),
                ("Share 1-unit detached (%)", f"{det1_med:.1f}" if pd.notna(det1_med) else "NA", False),
                ("Median household income ($)", f"{int(round(income_med)):,}" if pd.notna(income_med) else "NA", False),
                ("Poverty rate (%)", f"{pov_med:.1f}" if pd.notna(pov_med) else "NA", False),
            ]
            pd.DataFrame(
                [
                    {"row_label": label, "value": value, "is_panel_header": is_hdr}
                    for (label, value, is_hdr) in census_panel_display_rows
                ]
            ).to_csv(table_dir / "census_panel_profile_summary.csv", index=False)
            census_panel_ready = True
    except Exception:
        census_panel_ready = False

    plt.style.use("ggplot")

    # ---- Figure 1: Issue counts by decade ----
    counts = (
        manifest.groupby(["decade", "v5_issue_label"], as_index=False)
        .size()
        .rename(columns={"size": "issue_count"})
        .sort_values(["decade", "v5_issue_label"])
    )
    counts.to_csv(table_dir / "issue_counts_by_decade.csv", index=False)
    piv = counts.pivot(index="decade", columns="v5_issue_label", values="issue_count").fillna(0)
    present_v5_labels = [lab for lab in V5_LABEL_ORDER if lab in set(piv.columns)]
    for extra in sorted([c for c in piv.columns if c not in set(present_v5_labels)]):
        present_v5_labels.append(extra)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for lab in present_v5_labels:
        ax.plot(piv.index, piv[lab], marker="o", label=lab)
    ax.set_title("Issue counts over time")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Issue count")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, fig_dir / "fig01_issue_counts_by_decade.png")

    # ---- Figure 1b: Issue counts by decade and region ----
    counts_region = (
        manifest.groupby(["region", "decade", "v5_issue_label"], as_index=False)
        .size()
        .rename(columns={"size": "issue_count"})
        .sort_values(["region", "decade", "v5_issue_label"])
    )
    counts_region.to_csv(table_dir / "issue_counts_by_decade_region.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, reg in enumerate(plot_regions[:4]):
        ax = axes[i]
        d = counts_region[counts_region["region"] == reg].copy()
        if d.empty:
            ax.text(0.5, 0.5, "No observations", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(reg)
            continue
        piv_reg = d.pivot(index="decade", columns="v5_issue_label", values="issue_count").fillna(0).sort_index()
        for lab in [x for x in present_v5_labels if x in set(piv_reg.columns)]:
            ax.plot(piv_reg.index, piv_reg[lab], marker="o", label=lab)
        ax.set_title(reg)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Issue count")
        ax.legend(fontsize=6, ncol=1)
    for j in range(len(plot_regions[:4]), len(axes)):
        axes[j].axis("off")
    _save(fig, fig_dir / "fig01b_issue_counts_by_decade_region.png")

    # ---- Figure 2: Question coverage ----
    cov_rows: list[dict[str, Any]] = []
    issue_total = int(manifest["issue_id"].nunique())
    for qid, label in QUESTION_LABELS.items():
        d = obs[obs["question_id"] == qid]
        cov_rows.append(
            {
                "question_id": qid,
                "question_label": label,
                "issue_coverage_share": (d["issue_id"].nunique() / issue_total) if issue_total else 0.0,
                "rows": int(len(d)),
                "mean_confidence": float(d["confidence_0_to_1"].mean()) if len(d) else math.nan,
            }
        )
    coverage = pd.DataFrame(cov_rows).sort_values("issue_coverage_share", ascending=True)
    coverage.to_csv(table_dir / "question_coverage_table.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.8, 9.5))
    ax.barh(coverage["question_label"], coverage["issue_coverage_share"], color="#4472C4")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of issues with at least one extracted observation")
    ax.set_title("Coverage across all extraction questions")
    _save(fig, fig_dir / "fig02_question_coverage.png")

    # ---- Figure 2b: Question coverage by region (heatmap) ----
    reg_issue_totals = manifest.groupby("region", as_index=False)["issue_id"].nunique().rename(columns={"issue_id": "issues_total"})
    cov_region_rows: list[dict[str, Any]] = []
    for qid, label in QUESTION_LABELS.items():
        d = obs[obs["question_id"] == qid]
        issue_counts = d.groupby("region", as_index=False)["issue_id"].nunique().rename(columns={"issue_id": "issues_with_obs"})
        tmp = reg_issue_totals.merge(issue_counts, on="region", how="left")
        tmp["issues_with_obs"] = tmp["issues_with_obs"].fillna(0)
        tmp["coverage_share"] = np.where(tmp["issues_total"] > 0, tmp["issues_with_obs"] / tmp["issues_total"], np.nan)
        for r in tmp.itertuples(index=False):
            cov_region_rows.append(
                {
                    "question_id": qid,
                    "question_label": label,
                    "region": r.region,
                    "issues_total": int(r.issues_total),
                    "issues_with_obs": int(r.issues_with_obs),
                    "coverage_share": float(r.coverage_share) if pd.notna(r.coverage_share) else np.nan,
                }
            )
    cov_region = pd.DataFrame(cov_region_rows)
    cov_region.to_csv(table_dir / "question_coverage_by_region.csv", index=False)
    heat = (
        cov_region.pivot(index="question_label", columns="region", values="coverage_share")
        .reindex(columns=plot_regions)
        .fillna(0.0)
    )
    heat = heat.loc[coverage.sort_values("issue_coverage_share", ascending=False)["question_label"].tolist()]
    fig, ax = plt.subplots(figsize=(8.8, 10.5))
    im = ax.imshow(heat.values, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="Blues")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, fontsize=9)
    ax.set_title("Question extraction yield by region")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Share of issues with at least one extracted row")
    _save(fig, fig_dir / "fig02b_question_coverage_by_region.png")

    # ---- Figure 3: Core dimensional minima ----
    core_numeric_specs = [
        ("q_num_sf_min_lot_area", "Minimum lot area (sq ft)"),
        ("q_num_sf_min_lot_width", "Minimum lot width/frontage (ft)"),
        ("q_num_sf_front_setback_min", "Minimum front setback (ft)"),
        ("q_num_sf_side_setback_min", "Minimum side setback (ft)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (qid, title) in zip(axes.flatten(), core_numeric_specs):
        d = _numeric_trend_by_decade(obs, question_id=qid)
        d.to_csv(table_dir / f"{_slug(qid)}_numeric_by_decade.csv", index=False)
        if d.empty:
            ax.text(0.5, 0.5, "No observations", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        ax.plot(d["decade"], d["median_value"], marker="o", color="#1f77b4")
        ax.fill_between(d["decade"], d["p25"], d["p75"], alpha=0.25, color="#1f77b4")
        ax.set_title(title)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Median value")
    _save(fig, fig_dir / "fig03_core_dimensional_minima.png")

    # ---- Figure 3b: Core dimensional minima by region ----
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))
    for ax, (qid, title) in zip(axes.flatten(), core_numeric_specs):
        rows: list[pd.DataFrame] = []
        for reg in plot_regions:
            d_reg = obs[(obs["region"] == reg)].copy()
            d = _numeric_trend_by_decade(d_reg, question_id=qid)
            if d.empty:
                continue
            d["region"] = reg
            rows.append(d)
            ax.plot(d["decade"], d["median_value"], marker="o", label=reg)
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(table_dir / f"{_slug(qid)}_numeric_by_decade_region.csv", index=False)
        ax.set_title(title)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Median value")
        ax.legend(fontsize=7)
    _save(fig, fig_dir / "fig03b_core_dimensional_minima_by_region.png")

    # ---- Figure 4: Secondary dimensional standards ----
    sec_numeric_specs = [
        ("q_num_sf_rear_setback_min", "Minimum rear setback (ft)"),
        ("q_num_sf_max_height_ft", "Maximum building height (ft)"),
        ("q_num_sf_max_lot_coverage", "Maximum lot coverage (%)"),
        ("q_num_sf_min_floor_area", "Minimum floor area (sq ft)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (qid, title) in zip(axes.flatten(), sec_numeric_specs):
        d = _numeric_trend_by_decade(obs, question_id=qid)
        d.to_csv(table_dir / f"{_slug(qid)}_numeric_by_decade.csv", index=False)
        if d.empty:
            ax.text(0.5, 0.5, "No observations", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        ax.plot(d["decade"], d["median_value"], marker="o", color="#2ca02c")
        ax.fill_between(d["decade"], d["p25"], d["p75"], alpha=0.25, color="#2ca02c")
        ax.set_title(title)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Median value")
    _save(fig, fig_dir / "fig04_secondary_dimensional_standards.png")

    # ---- Figure 4b: Secondary dimensional standards by region ----
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))
    for ax, (qid, title) in zip(axes.flatten(), sec_numeric_specs):
        rows: list[pd.DataFrame] = []
        for reg in plot_regions:
            d_reg = obs[(obs["region"] == reg)].copy()
            d = _numeric_trend_by_decade(d_reg, question_id=qid)
            if d.empty:
                continue
            d["region"] = reg
            rows.append(d)
            ax.plot(d["decade"], d["median_value"], marker="o", label=reg)
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(table_dir / f"{_slug(qid)}_secondary_numeric_by_decade_region.csv", index=False)
        ax.set_title(title)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Median value")
        ax.legend(fontsize=7)
    _save(fig, fig_dir / "fig04b_secondary_dimensional_standards_by_region.png")

    # ---- Figure 5: Bounds and internal spread (lot area) ----
    lot_small = _numeric_trend_by_decade(obs, question_id="q_num_sf_lot_area_bounds", bound_role="smallest_minimum")
    lot_large = _numeric_trend_by_decade(obs, question_id="q_num_sf_lot_area_bounds", bound_role="largest_minimum")
    lot_small.to_csv(table_dir / "lot_area_bound_smallest_by_decade.csv", index=False)
    lot_large.to_csv(table_dir / "lot_area_bound_largest_by_decade.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    if not lot_small.empty:
        ax.plot(lot_small["decade"], lot_small["median_value"], marker="o", label="Smallest district minimum")
    if not lot_large.empty:
        ax.plot(lot_large["decade"], lot_large["median_value"], marker="o", label="Largest district minimum")
    ax.set_title("Single-family lot-area bounds within issues")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value (sq ft)")
    ax.legend()
    _save(fig, fig_dir / "fig05_lot_area_bounds.png")

    # ---- Figure 5b: Regional split for lot-area minima ----
    lot_region_rows: list[pd.DataFrame] = []
    for reg in plot_regions:
        d = obs[(obs["question_id"] == "q_num_sf_min_lot_area") & (obs["region"] == reg)].copy()
        d = d[d["value_number"].notna()]
        if d.empty:
            continue
        d = _trim_iqr_frame(d, value_col="value_number")
        if d.empty:
            continue
        g = (
            d.groupby("decade", as_index=False)
            .agg(median_value=("value_number", "median"), rows=("value_number", "size"))
            .sort_values("decade")
        )
        g["region"] = reg
        lot_region_rows.append(g)
    lot_region = pd.concat(lot_region_rows, ignore_index=True) if lot_region_rows else pd.DataFrame(columns=["decade", "median_value", "rows", "region"])
    lot_region.to_csv(table_dir / "lot_area_min_by_decade_region.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for reg in plot_regions:
        d = lot_region[lot_region["region"] == reg]
        if d.empty:
            continue
        ax.plot(d["decade"], d["median_value"], marker="o", label=reg)
    ax.set_title("Minimum lot area by region")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value (sq ft)")
    ax.legend()
    _save(fig, fig_dir / "fig05b_lot_area_by_region.png")

    # ---- Figure 6: Bounds and internal spread (front + side setbacks) ----
    fs_small = _numeric_trend_by_decade(obs, question_id="q_num_sf_front_setback_bounds", bound_role="smallest_minimum")
    fs_large = _numeric_trend_by_decade(obs, question_id="q_num_sf_front_setback_bounds", bound_role="largest_minimum")
    ss_small = _numeric_trend_by_decade(obs, question_id="q_num_sf_side_setback_bounds", bound_role="smallest_minimum")
    ss_large = _numeric_trend_by_decade(obs, question_id="q_num_sf_side_setback_bounds", bound_role="largest_minimum")
    fs_small.to_csv(table_dir / "front_setback_bound_smallest_by_decade.csv", index=False)
    fs_large.to_csv(table_dir / "front_setback_bound_largest_by_decade.csv", index=False)
    ss_small.to_csv(table_dir / "side_setback_bound_smallest_by_decade.csv", index=False)
    ss_large.to_csv(table_dir / "side_setback_bound_largest_by_decade.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)
    ax = axes[0]
    if not fs_small.empty:
        ax.plot(fs_small["decade"], fs_small["median_value"], marker="o", label="Smallest minimum")
    if not fs_large.empty:
        ax.plot(fs_large["decade"], fs_large["median_value"], marker="o", label="Largest minimum")
    ax.set_title("Front setback bounds")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value (ft)")
    ax.legend()
    ax = axes[1]
    if not ss_small.empty:
        ax.plot(ss_small["decade"], ss_small["median_value"], marker="o", label="Smallest minimum")
    if not ss_large.empty:
        ax.plot(ss_large["decade"], ss_large["median_value"], marker="o", label="Largest minimum")
    ax.set_title("Side setback bounds")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value (ft)")
    ax.legend()
    _save(fig, fig_dir / "fig06_setback_bounds.png")

    # ---- Figure 6b: Setback spread by region ----
    fs_gap_rows: list[pd.DataFrame] = []
    ss_gap_rows: list[pd.DataFrame] = []
    for reg in plot_regions:
        d_reg = obs[obs["region"] == reg].copy()
        fs_s = _numeric_trend_by_decade(d_reg, question_id="q_num_sf_front_setback_bounds", bound_role="smallest_minimum")
        fs_l = _numeric_trend_by_decade(d_reg, question_id="q_num_sf_front_setback_bounds", bound_role="largest_minimum")
        if not fs_s.empty and not fs_l.empty:
            f = fs_s[["decade", "median_value"]].rename(columns={"median_value": "smallest"})
            f = f.merge(fs_l[["decade", "median_value"]].rename(columns={"median_value": "largest"}), on="decade", how="inner")
            if not f.empty:
                f["spread_value"] = f["largest"] - f["smallest"]
                f["region"] = reg
                fs_gap_rows.append(f[["decade", "spread_value", "region"]])
        ss_s = _numeric_trend_by_decade(d_reg, question_id="q_num_sf_side_setback_bounds", bound_role="smallest_minimum")
        ss_l = _numeric_trend_by_decade(d_reg, question_id="q_num_sf_side_setback_bounds", bound_role="largest_minimum")
        if not ss_s.empty and not ss_l.empty:
            s = ss_s[["decade", "median_value"]].rename(columns={"median_value": "smallest"})
            s = s.merge(ss_l[["decade", "median_value"]].rename(columns={"median_value": "largest"}), on="decade", how="inner")
            if not s.empty:
                s["spread_value"] = s["largest"] - s["smallest"]
                s["region"] = reg
                ss_gap_rows.append(s[["decade", "spread_value", "region"]])
    fs_gap = pd.concat(fs_gap_rows, ignore_index=True) if fs_gap_rows else pd.DataFrame(columns=["decade", "spread_value", "region"])
    ss_gap = pd.concat(ss_gap_rows, ignore_index=True) if ss_gap_rows else pd.DataFrame(columns=["decade", "spread_value", "region"])
    fs_gap.to_csv(table_dir / "front_setback_spread_by_decade_region.csv", index=False)
    ss_gap.to_csv(table_dir / "side_setback_spread_by_decade_region.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=False)
    ax = axes[0]
    for reg in plot_regions:
        d = fs_gap[fs_gap["region"] == reg]
        if d.empty:
            continue
        ax.plot(d["decade"], d["spread_value"], marker="o", label=reg)
    ax.set_title("Front setback spread (largest - smallest)")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median spread (ft)")
    ax.legend(fontsize=7)
    ax = axes[1]
    for reg in plot_regions:
        d = ss_gap[ss_gap["region"] == reg]
        if d.empty:
            continue
        ax.plot(d["decade"], d["spread_value"], marker="o", label=reg)
    ax.set_title("Side setback spread (largest - smallest)")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median spread (ft)")
    ax.legend(fontsize=7)
    _save(fig, fig_dir / "fig06b_setback_spread_by_region.png")

    # ---- Figure 7: Procedural requirements shares ----
    proc_specs = [
        ("q_proc_site_plan_review_explicit", "Site plan review required", {"required"}, {"not_required"}),
        ("q_proc_special_permit_or_conditional_use", "Special permit/conditional use required", {"required"}, {"not_required"}),
        ("q_proc_subdivision_or_plat_approval", "Subdivision/plat approval required", {"required"}, {"not_required"}),
        ("q_proc_public_hearing_required", "Public hearing required", {"required"}, {"not_required"}),
        ("q_proc_notice_publication_required", "Published notice required", {"required"}, {"not_required"}),
        ("q_proc_certificate_of_occupancy", "Certificate of occupancy required", {"required"}, {"not_required"}),
    ]
    proc_table_rows: list[pd.DataFrame] = []
    proc_plot_series: list[pd.Series] = []
    min_resolved_for_plot = 5
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (qid, label, yes_set, no_set) in enumerate(proc_specs):
        d_all = _issue_level_binary_share_by_decade(obs, question_id=qid, yes_values=yes_set, no_values=no_set)
        d_all["series_label"] = label
        d_all["question_id"] = qid
        proc_table_rows.append(d_all)
        d = d_all[d_all["resolved_issues"] >= min_resolved_for_plot].copy()
        if not d.empty:
            ax.plot(d["decade"], d["share_yes"], marker="o", label=label, color=palette[i % len(palette)])
            proc_plot_series.append(d["share_yes"])
    proc_table = pd.concat(proc_table_rows, ignore_index=True) if proc_table_rows else pd.DataFrame()
    proc_table.to_csv(table_dir / "procedural_requirements_share_by_decade.csv", index=False)
    y0, y1 = _share_axis_limits(proc_plot_series)
    ax.set_ylim(y0, y1)
    ax.set_title("Procedural requirement shares by decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Share of resolved issues")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, fig_dir / "fig07_procedural_requirements.png")

    # ---- Figure 7b: Regional split for site plan and use variance ----
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, reg in enumerate(plot_regions[:4]):
        ax = axes[i]
        d_reg = obs[obs["region"] == reg].copy()
        site_reg = _issue_level_binary_share_by_decade(
            d_reg,
            question_id="q_proc_site_plan_review_explicit",
            yes_values={"required"},
            no_values={"not_required"},
        ).rename(columns={"share_yes": "site_plan_required_share"})
        var_reg = _issue_level_binary_share_by_decade(
            d_reg,
            question_id="q_proc_use_variance_availability",
            yes_values={"available"},
            no_values={"not_available"},
        ).rename(columns={"share_yes": "use_variance_available_share"})
        if not site_reg.empty:
            ax.plot(site_reg["decade"], site_reg["site_plan_required_share"], marker="o", label="Site plan required")
        if not var_reg.empty:
            ax.plot(var_reg["decade"], var_reg["use_variance_available_share"], marker="o", label="Use variance available")
        ax.set_title(reg)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Share of issues")
        ax.legend(fontsize=7)
    # Hide extra axes if fewer than 4 regions
    for j in range(len(plot_regions[:4]), len(axes)):
        axes[j].axis("off")
    _save(fig, fig_dir / "fig07b_proc_vs_variance_by_region.png")

    # ---- Figure 12b: Regional composition ----
    region_mix = (
        manifest.groupby(["region", "v5_issue_label"], as_index=False)
        .size()
        .rename(columns={"size": "issues"})
        .sort_values(["region", "v5_issue_label"])
    )
    region_mix.to_csv(table_dir / "issue_counts_by_region_label.csv", index=False)
    piv = region_mix.pivot(index="region", columns="v5_issue_label", values="issues").fillna(0).reindex(region_order)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bottoms = np.zeros(len(piv.index))
    x = np.arange(len(piv.index))
    for lab in [x for x in present_v5_labels if x in set(piv.columns)]:
        vals = piv[lab].values
        ax.bar(x, vals, bottom=bottoms, label=lab)
        bottoms = bottoms + vals
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("Issue count")
    ax.set_title("Issue composition by region (stacked)")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, fig_dir / "fig12b_issue_composition_by_region.png")

    # ---- Figure 8: Flexibility and procedural substitution ----
    min_resolved_for_plot = 5
    site = _issue_level_binary_share_by_decade(
        obs, question_id="q_proc_site_plan_review_explicit", yes_values={"required"}, no_values={"not_required"}
    ).rename(columns={"share_yes": "site_plan_required_share"})
    site = site[site["resolved_issues"] >= min_resolved_for_plot].copy()
    use_var = _issue_level_binary_share_by_decade(
        obs, question_id="q_proc_use_variance_availability", yes_values={"available"}, no_values={"not_available"}
    ).rename(columns={"share_yes": "use_variance_available_share"})
    use_var = use_var[use_var["resolved_issues"] >= min_resolved_for_plot].copy()
    area_var = _issue_level_binary_share_by_decade(
        obs, question_id="q_proc_area_variance_availability", yes_values={"available"}, no_values={"not_available"}
    ).rename(columns={"share_yes": "area_variance_available_share"})
    area_var = area_var[area_var["resolved_issues"] >= min_resolved_for_plot].copy()
    flex = site.merge(use_var[["decade", "use_variance_available_share"]], on="decade", how="outer")
    flex = flex.merge(area_var[["decade", "area_variance_available_share"]], on="decade", how="outer").sort_values("decade")
    flex.to_csv(table_dir / "flexibility_procedure_shares_by_decade.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    if "site_plan_required_share" in flex:
        ax.plot(flex["decade"], flex["site_plan_required_share"], marker="o", label="Site plan review required")
    if "use_variance_available_share" in flex:
        ax.plot(flex["decade"], flex["use_variance_available_share"], marker="o", label="Use variance available")
    if "area_variance_available_share" in flex:
        ax.plot(flex["decade"], flex["area_variance_available_share"], marker="o", label="Area variance available")
    y0, y1 = _share_axis_limits(
        [
            flex["site_plan_required_share"] if "site_plan_required_share" in flex else pd.Series(dtype=float),
            flex["use_variance_available_share"] if "use_variance_available_share" in flex else pd.Series(dtype=float),
            flex["area_variance_available_share"] if "area_variance_available_share" in flex else pd.Series(dtype=float),
        ]
    )
    ax.set_ylim(y0, y1)
    ax.set_title("Procedure versus formal relief tools")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Share of resolved issues")
    ax.legend()
    _save(fig, fig_dir / "fig08_procedure_vs_flexibility.png")

    # ---- Figure 8b: Procedure versus flexibility by region ----
    flex_region_rows: list[pd.DataFrame] = []
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, reg in enumerate(plot_regions[:4]):
        ax = axes[i]
        d_reg = obs[obs["region"] == reg].copy()
        site_reg = _issue_level_binary_share_by_decade(
            d_reg,
            question_id="q_proc_site_plan_review_explicit",
            yes_values={"required"},
            no_values={"not_required"},
        ).rename(columns={"share_yes": "share", "resolved_issues": "resolved"})
        site_reg["metric"] = "Site plan required"
        use_reg = _issue_level_binary_share_by_decade(
            d_reg,
            question_id="q_proc_use_variance_availability",
            yes_values={"available"},
            no_values={"not_available"},
        ).rename(columns={"share_yes": "share", "resolved_issues": "resolved"})
        use_reg["metric"] = "Use variance available"
        area_reg = _issue_level_binary_share_by_decade(
            d_reg,
            question_id="q_proc_area_variance_availability",
            yes_values={"available"},
            no_values={"not_available"},
        ).rename(columns={"share_yes": "share", "resolved_issues": "resolved"})
        area_reg["metric"] = "Area variance available"
        tmp = pd.concat([site_reg, use_reg, area_reg], ignore_index=True)
        if not tmp.empty:
            tmp["region"] = reg
            flex_region_rows.append(tmp[["region", "metric", "decade", "share", "resolved"]])
        site_plot = site_reg[site_reg["resolved"] >= min_resolved_for_plot].copy()
        use_plot = use_reg[use_reg["resolved"] >= min_resolved_for_plot].copy()
        area_plot = area_reg[area_reg["resolved"] >= min_resolved_for_plot].copy()
        if not site_plot.empty:
            ax.plot(site_plot["decade"], site_plot["share"], marker="o", label="Site plan required")
        if not use_plot.empty:
            ax.plot(use_plot["decade"], use_plot["share"], marker="o", label="Use variance available")
        if not area_plot.empty:
            ax.plot(area_plot["decade"], area_plot["share"], marker="o", label="Area variance available")
        ax.set_title(reg)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Share of resolved issues")
        ax.legend(fontsize=7)
    for j in range(len(plot_regions[:4]), len(axes)):
        axes[j].axis("off")
    flex_region = (
        pd.concat(flex_region_rows, ignore_index=True)
        if flex_region_rows
        else pd.DataFrame(columns=["region", "metric", "decade", "share", "resolved"])
    )
    flex_region.to_csv(table_dir / "procedure_vs_flexibility_by_region_decade.csv", index=False)
    _save(fig, fig_dir / "fig08b_procedure_vs_flexibility_by_region.png")

    # ---- Figure 9: Use variance decision body mix ----
    decider_mix = _dominant_category_share_by_decade(
        obs, question_id="q_proc_use_variance_decider", label_map=DECIDER_MAP, top_n=5
    )
    decider_mix.to_csv(table_dir / "use_variance_decider_mix_by_decade.csv", index=False)
    piv = decider_mix.pivot(index="decade", columns="category_label", values="share").fillna(0).sort_index() if not decider_mix.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    if not piv.empty:
        for col in piv.columns:
            ax.plot(piv.index, piv[col], marker="o", label=col)
    ax.set_ylim(0, 1)
    ax.set_title("Who decides use variances")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Share of issues")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, fig_dir / "fig09_use_variance_decider_mix.png")

    # ---- Figure 9b: Use variance decision body mix by region ----
    decider_region_rows: list[pd.DataFrame] = []
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, reg in enumerate(plot_regions[:4]):
        ax = axes[i]
        d_reg = obs[obs["region"] == reg].copy()
        mix_reg = _dominant_category_share_by_decade(
            d_reg, question_id="q_proc_use_variance_decider", label_map=DECIDER_MAP, top_n=3
        )
        if not mix_reg.empty:
            mix_reg["region"] = reg
            decider_region_rows.append(mix_reg[["region", "decade", "category_label", "share", "issues"]])
            piv_reg = mix_reg.pivot(index="decade", columns="category_label", values="share").fillna(0).sort_index()
            for col in piv_reg.columns:
                ax.plot(piv_reg.index, piv_reg[col], marker="o", label=col)
        ax.set_title(reg)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Share of issues")
        ax.legend(fontsize=7)
    for j in range(len(plot_regions[:4]), len(axes)):
        axes[j].axis("off")
    decider_region = (
        pd.concat(decider_region_rows, ignore_index=True)
        if decider_region_rows
        else pd.DataFrame(columns=["region", "decade", "category_label", "share", "issues"])
    )
    decider_region.to_csv(table_dir / "use_variance_decider_mix_by_region_decade.csv", index=False)
    _save(fig, fig_dir / "fig09b_use_variance_decider_mix_by_region.png")

    # ---- Figure 10: Use variance standard mix ----
    std_mix = _dominant_category_share_by_decade(
        obs, question_id="q_proc_use_variance_standard", label_map=STANDARD_MAP, top_n=5
    )
    std_mix.to_csv(table_dir / "use_variance_standard_mix_by_decade.csv", index=False)
    piv = std_mix.pivot(index="decade", columns="category_label", values="share").fillna(0).sort_index() if not std_mix.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    if not piv.empty:
        for col in piv.columns:
            ax.plot(piv.index, piv[col], marker="o", label=col)
    ax.set_ylim(0, 1)
    ax.set_title("Legal standard used for use variances")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Share of issues")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, fig_dir / "fig10_use_variance_standard_mix.png")

    # ---- Figure 10b: Use variance legal standard mix by region ----
    std_region_rows: list[pd.DataFrame] = []
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, reg in enumerate(plot_regions[:4]):
        ax = axes[i]
        d_reg = obs[obs["region"] == reg].copy()
        mix_reg = _dominant_category_share_by_decade(
            d_reg, question_id="q_proc_use_variance_standard", label_map=STANDARD_MAP, top_n=3
        )
        if not mix_reg.empty:
            mix_reg["region"] = reg
            std_region_rows.append(mix_reg[["region", "decade", "category_label", "share", "issues"]])
            piv_reg = mix_reg.pivot(index="decade", columns="category_label", values="share").fillna(0).sort_index()
            for col in piv_reg.columns:
                ax.plot(piv_reg.index, piv_reg[col], marker="o", label=col)
        ax.set_title(reg)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Share of issues")
        ax.legend(fontsize=7)
    for j in range(len(plot_regions[:4]), len(axes)):
        axes[j].axis("off")
    std_region = (
        pd.concat(std_region_rows, ignore_index=True)
        if std_region_rows
        else pd.DataFrame(columns=["region", "decade", "category_label", "share", "issues"])
    )
    std_region.to_csv(table_dir / "use_variance_standard_mix_by_region_decade.csv", index=False)
    _save(fig, fig_dir / "fig10b_use_variance_standard_mix_by_region.png")

    # ---- Figure 11: Enforcement penalties ----
    pen = obs[(obs["question_id"] == "q_proc_enforcement_penalty") & obs["value_number"].notna()].copy()
    pen["metric"] = pen["metric"].astype(str).str.strip().str.lower()
    fine = _trim_iqr_frame(pen[pen["metric"] == "max_fine"], value_col="value_number")
    jail = _trim_iqr_frame(pen[pen["metric"] == "max_jail_days"], value_col="value_number")
    fine_dec = (
        fine.groupby("decade", as_index=False)
        .agg(median_value=("value_number", "median"), rows=("value_number", "size"))
        .sort_values("decade")
        if not fine.empty
        else pd.DataFrame(columns=["decade", "median_value", "rows"])
    )
    jail_dec = (
        jail.groupby("decade", as_index=False)
        .agg(median_value=("value_number", "median"), rows=("value_number", "size"))
        .sort_values("decade")
        if not jail.empty
        else pd.DataFrame(columns=["decade", "median_value", "rows"])
    )
    fine_dec.to_csv(table_dir / "max_fine_by_decade.csv", index=False)
    jail_dec.to_csv(table_dir / "max_jail_days_by_decade.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    ax = axes[0]
    if not fine_dec.empty:
        ax.plot(fine_dec["decade"], fine_dec["median_value"], marker="o", color="#d62728")
    ax.set_title("Maximum fines")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value (USD)")
    ax = axes[1]
    if not jail_dec.empty:
        ax.plot(jail_dec["decade"], jail_dec["median_value"], marker="o", color="#9467bd")
    ax.set_title("Maximum jail days")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value (days)")
    _save(fig, fig_dir / "fig11_enforcement_penalties.png")

    # ---- Figure 11b: Enforcement penalties by region ----
    fine_region_rows: list[pd.DataFrame] = []
    jail_region_rows: list[pd.DataFrame] = []
    for reg in plot_regions:
        pen_reg = pen[pen["region"] == reg].copy()
        if pen_reg.empty:
            continue
        fine_reg = _trim_iqr_frame(pen_reg[pen_reg["metric"] == "max_fine"], value_col="value_number")
        jail_reg = _trim_iqr_frame(pen_reg[pen_reg["metric"] == "max_jail_days"], value_col="value_number")
        if not fine_reg.empty:
            f = (
                fine_reg.groupby("decade", as_index=False)
                .agg(median_value=("value_number", "median"), rows=("value_number", "size"))
                .sort_values("decade")
            )
            f["region"] = reg
            fine_region_rows.append(f[["region", "decade", "median_value", "rows"]])
        if not jail_reg.empty:
            j = (
                jail_reg.groupby("decade", as_index=False)
                .agg(median_value=("value_number", "median"), rows=("value_number", "size"))
                .sort_values("decade")
            )
            j["region"] = reg
            jail_region_rows.append(j[["region", "decade", "median_value", "rows"]])
    fine_region = (
        pd.concat(fine_region_rows, ignore_index=True)
        if fine_region_rows
        else pd.DataFrame(columns=["region", "decade", "median_value", "rows"])
    )
    jail_region = (
        pd.concat(jail_region_rows, ignore_index=True)
        if jail_region_rows
        else pd.DataFrame(columns=["region", "decade", "median_value", "rows"])
    )
    fine_region.to_csv(table_dir / "max_fine_by_region_decade.csv", index=False)
    jail_region.to_csv(table_dir / "max_jail_days_by_region_decade.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), sharex=True, sharey=False)
    ax = axes[0]
    for reg in plot_regions:
        d = fine_region[fine_region["region"] == reg]
        if d.empty:
            continue
        ax.plot(d["decade"], d["median_value"], marker="o", label=reg)
    ax.set_title("Maximum fines by region")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value (USD)")
    ax.legend(fontsize=7)
    ax = axes[1]
    for reg in plot_regions:
        d = jail_region[jail_region["region"] == reg]
        if d.empty:
            continue
        ax.plot(d["decade"], d["median_value"], marker="o", label=reg)
    ax.set_title("Maximum jail days by region")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value (days)")
    ax.legend(fontsize=7)
    _save(fig, fig_dir / "fig11b_enforcement_penalties_by_region.png")

    # ---- Figure 12: Parking + height panel ----
    parking = _numeric_trend_by_decade(obs, question_id="q_num_sf_parking_spaces_per_dwelling")
    height_ft = _numeric_trend_by_decade(obs, question_id="q_num_sf_max_height_ft")
    height_st = _numeric_trend_by_decade(obs, question_id="q_num_sf_max_height_stories")
    parking.to_csv(table_dir / "parking_spaces_per_dwelling_by_decade.csv", index=False)
    height_ft.to_csv(table_dir / "max_height_ft_by_decade.csv", index=False)
    height_st.to_csv(table_dir / "max_height_stories_by_decade.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    ax = axes[0]
    if not parking.empty:
        ax.plot(parking["decade"], parking["median_value"], marker="o")
        ax.fill_between(parking["decade"], parking["p25"], parking["p75"], alpha=0.25)
    ax.set_title("Parking spaces per dwelling")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value")
    ax = axes[1]
    if not height_ft.empty:
        ax.plot(height_ft["decade"], height_ft["median_value"], marker="o", color="#2ca02c")
        ax.fill_between(height_ft["decade"], height_ft["p25"], height_ft["p75"], alpha=0.25, color="#2ca02c")
    ax.set_title("Maximum height (ft)")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value")
    ax = axes[2]
    if not height_st.empty:
        ax.plot(height_st["decade"], height_st["median_value"], marker="o", color="#8c564b")
        ax.fill_between(height_st["decade"], height_st["p25"], height_st["p75"], alpha=0.25, color="#8c564b")
    ax.set_title("Maximum height (stories)")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median value")
    _save(fig, fig_dir / "fig12_parking_and_height.png")

    # ---- Figure 12c: Parking + height by region ----
    region_metric_rows: list[pd.DataFrame] = []
    metric_specs = [
        ("q_num_sf_parking_spaces_per_dwelling", "Parking spaces per dwelling"),
        ("q_num_sf_max_height_ft", "Maximum height (ft)"),
        ("q_num_sf_max_height_stories", "Maximum height (stories)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9), sharex=True, sharey=False)
    for ax, (qid, title) in zip(axes, metric_specs):
        for reg in plot_regions:
            d_reg = obs[obs["region"] == reg].copy()
            d = _numeric_trend_by_decade(d_reg, question_id=qid)
            if d.empty:
                continue
            d["region"] = reg
            d["metric_label"] = title
            region_metric_rows.append(d[["region", "metric_label", "decade", "median_value", "p25", "p75", "rows"]])
            ax.plot(d["decade"], d["median_value"], marker="o", label=reg)
        ax.set_title(title)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Median value")
        ax.legend(fontsize=7)
    region_metric = (
        pd.concat(region_metric_rows, ignore_index=True)
        if region_metric_rows
        else pd.DataFrame(columns=["region", "metric_label", "decade", "median_value", "p25", "p75", "rows"])
    )
    region_metric.to_csv(table_dir / "parking_height_by_region_decade.csv", index=False)
    _save(fig, fig_dir / "fig12c_parking_and_height_by_region.png")

    # ---- Table: data summary ----
    label_counts_v5 = (
        manifest["v5_issue_label"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    present_v5_summary = [lab for lab in V5_LABEL_ORDER if lab in label_counts_v5]
    for extra in sorted([lab for lab in label_counts_v5.keys() if lab not in set(present_v5_summary)]):
        present_v5_summary.append(extra)
    summary = {
        "issues_total": int(manifest["issue_id"].nunique()),
        "newspapers_total": int(manifest["newspaper_slug"].nunique()),
        "v5_label_counts": label_counts_v5,
        "v5_labels_present": present_v5_summary,
        "observation_rows": int(len(obs)),
        "questions_covered": int(obs["question_id"].nunique()),
        "mean_confidence": float(obs["confidence_0_to_1"].mean()),
    }
    _write_text(output_dir / "summary.json", json.dumps(summary, indent=2))

    # ---- Table: question coverage (friendly labels) ----
    cov = coverage.copy()
    cov["question_label"] = cov["question_label"].map(_tex_escape)
    cov["issue_coverage_pct"] = cov["issue_coverage_share"].map(lambda x: round(100.0 * x, 1))
    cov["mean_confidence"] = cov["mean_confidence"].round(3)
    cov_show = cov.sort_values("issue_coverage_share", ascending=False)[
        ["question_label", "rows", "issue_coverage_pct", "mean_confidence"]
    ].copy()
    cov_show.to_csv(table_dir / "question_coverage_friendly.csv", index=False)

    # ---- Build LaTeX ----
    tex = []
    tex.append(r"\documentclass[11pt]{article}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{longtable}")
    tex.append(r"\usepackage{array}")
    tex.append(r"\usepackage{verbatim}")
    tex.append(r"\usepackage{float}")
    tex.append(r"\usepackage{placeins}")
    tex.append(r"\floatplacement{figure}{H}")
    tex.append(r"\floatplacement{table}{H}")
    tex.append(r"\usepackage{caption}")
    tex.append(r"\captionsetup{font=small,labelfont=bf}")
    tex.append(r"\usepackage[hidelinks]{hyperref}")
    tex.append(r"\title{Issue-Level Zoning Question Extraction (V5 Label Set)}")
    tex.append(r"\author{}")
    tex.append(r"\date{\today}")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")

    tex.append(r"\section*{Data Summary}")
    tex.append(
        r"This report summarizes issue-level answers using issue labels taken directly from the V5 classifier output. "
        r"The analysis sample includes only issues that returned at least one extracted observation row and have a non-missing V5 issue label. "
        r"Each point in the charts is built from extracted issue-level rows, grouped by issue decade."
    )
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\begin{tabular}{lr}")
    tex.append(r"\toprule")
    tex.append(r"Measure & Value\\")
    tex.append(r"\midrule")
    tex.append(f"Total issues analyzed & {_fmt_int(summary['issues_total'])}\\\\")
    tex.append(f"Distinct newspapers & {_fmt_int(summary['newspapers_total'])}\\\\")
    for lab in summary["v5_labels_present"]:
        tex.append(f"{_tex_escape(lab)} issues & {_fmt_int(summary['v5_label_counts'].get(lab, 0))}\\\\")
    tex.append(f"Extracted observation rows & {_fmt_int(summary['observation_rows'])}\\\\")
    tex.append(f"Questions with nonzero coverage & {_fmt_int(summary['questions_covered'])}\\\\")
    tex.append(f"Mean extraction confidence & {summary['mean_confidence']:.3f}\\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(
        r"\caption{Run-level summary. \textit{Notes:} Requests are issued once per newspaper issue (date-level input), "
        r"but each issue can return multiple extracted records because several questions can be answered in the same issue "
        r"(and some questions yield more than one row, such as district-level values or bound pairs). "
        r"The label rows shown here are exactly the V5 labels present in the analysis sample for this run. "
        r"Total issues analyzed equals distinct issue identifiers in the issue manifest. "
        r"Extracted observation rows equals the number of normalized extraction rows returned across all issues. "
        r"Questions with nonzero coverage equals the number of question IDs with at least one extracted row in the run. "
        r"Figures below use decade bins derived from issue date.}"
    )
    tex.append(r"\end{table}")

    if census_panel_ready and census_panel_display_rows:
        tex.append(r"\begin{table}[H]")
        tex.append(r"\centering")
        tex.append(r"\begin{tabular}{p{0.72\textwidth}r}")
        tex.append(r"\toprule")
        tex.append(r"Metric & Value\\")
        tex.append(r"\midrule")
        for label, value, is_hdr in census_panel_display_rows:
            if is_hdr:
                tex.append(rf"\multicolumn{{2}}{{l}}{{\textbf{{{_tex_escape(label)}}}}}\\")
            else:
                tex.append(rf"{_tex_escape(label)} & {_tex_escape(value)}\\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        tex.append(
            r"\caption{Census-derived descriptive profile for municipalities represented in this run. "
            r"\textit{Table notes:} Unit of analysis is municipality (city key), not issue. "
            r"Municipality universe is built by mapping run newspapers to city keys, then to place identifiers. "
            r"Region shares are municipality shares. In-MSA share equals "
            r"$100\times\#(\mathrm{municipalities\ with\ CBSA\ linkage})/\#(\mathrm{municipalities})$. "
            r"ACS values are 2022 ACS 5-year place-level values joined by place geoid; "
            r"population density uses ACS population divided by Census Gazetteer place land area (square miles).}"
        )
        tex.append(r"\end{table}")

    def fig_block(filename: str, caption: str) -> None:
        tex.append(r"\begin{figure}[H]")
        tex.append(r"\centering")
        tex.append(rf"\includegraphics[width=0.95\textwidth]{{figures/{filename}}}")
        tex.append(rf"\caption{{{caption}}}")
        tex.append(r"\end{figure}")

    tex.append(r"\FloatBarrier")
    tex.append(r"\section*{Coverage and Volume}")
    fig_block(
        "fig01_issue_counts_by_decade.png",
        "Issue counts over time by V5 issue label. "
        r"\textit{Notes:} Counts are issue rows grouped by decade from issue date, using the V5 issue label attached to each issue in the analysis sample. "
        r"This view is intentionally pre-dedup and shows observed publication volume.",
    )
    fig_block(
        "fig01b_issue_counts_by_decade_region.png",
        "Issue counts over time by region. "
        r"\textit{Notes:} Same issue-row counting logic as the prior figure, split into regional panels. "
        r"Each panel shows decade counts by V5 issue label before event deduplication.",
    )
    fig_block(
        "fig02_question_coverage.png",
        "Extraction yield across all questions. "
        r"\textit{Notes:} For each question, coverage share is "
        r"$\#(\mathrm{issues\ with\ at\ least\ one\ row\ for\ question})\ /\ \#(\mathrm{all\ analyzed\ issues})$. "
        r"Each issue contributes at most one unit to a question's numerator and denominator even when that issue yields multiple extracted rows for that question. "
        r"A value near 0 means the question was asked but almost no issues produced an extractable structured row; it does not mean the question was omitted.",
    )
    fig_block(
        "fig12b_issue_composition_by_region.png",
        "Issue-label composition by region. "
        r"\textit{Notes:} Regions are derived from newspaper state abbreviation using U.S. Census region groups (Northeast, Midwest, South, West). "
        r"Bars are issue-level stacked counts within each region by V5 issue label and remain pre-dedup publication rows.",
    )

    tex.append(r"\FloatBarrier")
    tex.append(r"\section*{Regional Splits}")
    tex.append(
        r"Regional views split the same issue-level extraction outputs by U.S. Census regions "
        r"(Northeast, Midwest, South, West), mapped from each newspaper's state."
    )
    fig_block(
        "fig02b_question_coverage_by_region.png",
        "Question extraction-yield heatmap by region. "
        r"\textit{Notes:} Within each region, each cell is "
        r"$\#(\mathrm{issues\ in\ region\ with\ at\ least\ one\ row\ for\ question})\ /\ \#(\mathrm{all\ analyzed\ issues\ in\ region})$. "
        r"Issue-level counting is used; multiple rows from the same issue do not increase the numerator. "
        r"All questions are asked for all issues. Therefore a 0.00 cell means asked-but-no-extractable-row (0 of N issues), not ``question not asked.'' "
        r"Low values can reflect true non-applicability, absence in text, or unresolved extraction ambiguity.",
    )

    tex.append(r"\FloatBarrier")
    tex.append(r"\section*{Dimensional Standards Over Time}")
    fig_block(
        "fig03_core_dimensional_minima.png",
        "Core single-family dimensional minima. "
        r"\textit{Notes:} Each panel plots decade medians; shaded ribbons show interquartile range (25th--75th percentile). "
        r"Values are issue-level numeric rows after 1st--99th percentile trimming within question to reduce OCR outlier influence in visual trends.",
    )
    fig_block(
        "fig03b_core_dimensional_minima_by_region.png",
        "Core single-family dimensional minima by region. "
        r"\textit{Notes:} For each metric panel, lines are decade medians computed separately within each region from issue-level numeric rows.",
    )
    fig_block(
        "fig04_secondary_dimensional_standards.png",
        "Secondary dimensional standards. "
        r"\textit{Notes:} Same construction as the prior figure, using issue-level numeric rows and decade medians with interquartile ribbons.",
    )
    fig_block(
        "fig04b_secondary_dimensional_standards_by_region.png",
        "Secondary dimensional standards by region. "
        r"\textit{Notes:} Same metric definitions as the national chart, split into region-specific decade-median lines.",
    )
    fig_block(
        "fig05_lot_area_bounds.png",
        "Within-issue lot-area spread across districts. "
        r"\textit{Notes:} For issues that report both smallest and largest district minima, the chart tracks decade medians of those two bounds.",
    )
    fig_block(
        "fig05b_lot_area_by_region.png",
        "Minimum lot-area trends split by region. "
        r"\textit{Notes:} Lines are decade medians of minimum lot area for single-family districts, computed separately within each region.",
    )
    fig_block(
        "fig06_setback_bounds.png",
        "Within-issue front and side setback spread across districts. "
        r"\textit{Notes:} Each line is a decade median bound value (smallest versus largest district minimum) among issues providing bound information.",
    )
    fig_block(
        "fig06b_setback_spread_by_region.png",
        "Within-issue front and side setback spread by region. "
        r"\textit{Notes:} Spread equals largest minus smallest district minimum within issue; lines are decade medians computed separately by region.",
    )
    fig_block(
        "fig12_parking_and_height.png",
        "Parking and building-height standards. "
        r"\textit{Notes:} Panels report decade medians with interquartile ribbons for parking spaces, maximum height in feet, and maximum height in stories.",
    )
    fig_block(
        "fig12c_parking_and_height_by_region.png",
        "Parking and building-height standards by region. "
        r"\textit{Notes:} For each metric panel, lines are region-specific decade medians from issue-level numeric extraction rows.",
    )

    tex.append(r"\FloatBarrier")
    tex.append(r"\section*{Procedural Governance and Flexibility}")
    fig_block(
        "fig07_procedural_requirements.png",
        "Procedural requirement shares by decade. "
        r"\textit{Notes:} For each question, an issue is coded as yes/no from categorical responses (conflicts kept unresolved). "
        r"Lines show the yes-share among resolved issues by decade. Decades with fewer than 5 resolved issues are omitted from the plotted lines. "
        r"The y-axis is data-scaled to the observed range to improve readability.",
    )
    fig_block(
        "fig07b_proc_vs_variance_by_region.png",
        "Site-plan review and use-variance availability by region. "
        r"\textit{Notes:} Small multiples show yes-shares by decade for each region, using the same issue-level binary coding as the national procedural chart.",
    )
    fig_block(
        "fig08_procedure_vs_flexibility.png",
        "Procedure versus formal relief tools. "
        r"\textit{Notes:} The figure jointly tracks site-plan requirements and variance availability to show how procedural review and variance channels move over time in issue-level shares. "
        r"Series include only decades with at least 5 resolved issues per question; the y-axis is data-scaled for interpretability.",
    )
    fig_block(
        "fig08b_procedure_vs_flexibility_by_region.png",
        "Procedure versus formal relief tools by region. "
        r"\textit{Notes:} Regional panels apply the same issue-level yes-share construction and 5-resolved-issue threshold used in the national chart.",
    )
    fig_block(
        "fig09_use_variance_decider_mix.png",
        "Who decides use variances. "
        r"\textit{Notes:} For each issue, dominant decision-body category is selected, then decade shares are computed across issues.",
    )
    fig_block(
        "fig09b_use_variance_decider_mix_by_region.png",
        "Who decides use variances by region. "
        r"\textit{Notes:} Within each region, each issue is assigned one dominant decision-body category; lines track decade shares of those dominant categories.",
    )
    fig_block(
        "fig10_use_variance_standard_mix.png",
        "Legal standard used for use variances. "
        r"\textit{Notes:} Same dominant-category methodology as prior figure, using extracted standard categories (for example unnecessary hardship, practical difficulty, or not stated).",
    )
    fig_block(
        "fig10b_use_variance_standard_mix_by_region.png",
        "Legal standard used for use variances by region. "
        r"\textit{Notes:} Same dominant-category approach as the national standard-mix chart, calculated within each region by decade.",
    )
    fig_block(
        "fig11_enforcement_penalties.png",
        "Enforcement penalty magnitudes. "
        r"\textit{Notes:} Left panel shows maximum fines; right panel shows maximum jail-day caps. "
        r"Series are decade medians from issue-level numeric rows after 1st--99th percentile trimming.",
    )
    fig_block(
        "fig11b_enforcement_penalties_by_region.png",
        "Enforcement penalty magnitudes by region. "
        r"\textit{Notes:} Regional lines are decade medians for extracted maximum fines and jail-day caps, using the same outlier-trimming rule as the national figure.",
    )
    tex.append(r"\FloatBarrier")
    tex.append(r"\section*{Question Coverage Table}")
    tex.append(r"\small")
    tex.append(r"\begin{longtable}{p{0.53\textwidth}rrr}")
    tex.append(
        r"\caption{Question-level extraction coverage and confidence. "
        r"\textit{Table notes:} Rows are normalized extraction rows. "
        r"Issue coverage (\%) is "
        r"$100\times\#(\mathrm{issues\ with\ at\ least\ one\ row\ for\ question})/\#(\mathrm{all\ analyzed\ issues})$. "
        r"Mean confidence is the arithmetic mean of model confidence values across rows for that question. "
        r"All questions are sent to all issues; low or zero issue coverage means few or no issues returned an extractable row for that question (not that the question was skipped).}\\"
    )
    tex.append(r"\toprule")
    tex.append(r"Question & Rows & Issue coverage (\%) & Mean confidence\\")
    tex.append(r"\midrule")
    tex.append(r"\endfirsthead")
    tex.append(r"\toprule")
    tex.append(r"Question & Rows & Issue coverage (\%) & Mean confidence\\")
    tex.append(r"\midrule")
    tex.append(r"\endhead")
    for r in cov_show.itertuples(index=False):
        tex.append(
            f"{r.question_label} & {_fmt_int(r.rows)} & {r.issue_coverage_pct:.1f} & {r.mean_confidence:.3f}\\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{longtable}")
    tex.append(r"\normalsize")

    tex.append(r"\FloatBarrier")
    tex.append(r"\appendix")
    tex.append(r"\section*{Appendix: Exact Questions Asked}")
    if prompt_question_catalog:
        tex.append(
            r"The table below is parsed from the extraction prompt used in this run and lists the exact question identifiers and question text."
        )
        tex.append(r"\footnotesize")
        tex.append(r"\setlength{\LTleft}{0pt}")
        tex.append(r"\setlength{\LTright}{0pt}")
        tex.append(
            r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.26\textwidth}>{\raggedright\arraybackslash}p{0.69\textwidth}}"
        )
        tex.append(
            r"\caption{Exact question catalog parsed from the run prompt. "
            r"\textit{Table notes:} Each row is one question bullet (q\_*) extracted from the prompt text that was sent with requests in this run. "
            r"Question text is shown as cleaned prose for readability and excludes schema-only boilerplate.}\\"
        )
        tex.append(r"\toprule")
        tex.append(r"Question ID & Question text\\")
        tex.append(r"\midrule")
        tex.append(r"\endfirsthead")
        tex.append(r"\toprule")
        tex.append(r"Question ID & Question text\\")
        tex.append(r"\midrule")
        tex.append(r"\endhead")
        for r in prompt_question_catalog:
            tex.append(f"{_tex_escape(r['question_id'])} & {_tex_escape(r['question_text'])}\\\\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{longtable}")
        tex.append(rf"\normalsize\noindent\textit{{Prompt source:}} {_tex_escape(prompt_source)}.")
    else:
        tex.append(r"No prompt-derived question catalog was recoverable for this run.")

    tex.append(r"\section*{Appendix: Full Prompt (Verbatim)}")
    if prompt_text:
        tex.append(
            r"The full prompt text used for this run is embedded below exactly as sent (before issue metadata/text injection)."
        )
        tex.append(r"\begingroup\footnotesize")
        tex.append(r"\verbatiminput{appendix_prompt_verbatim.txt}")
        tex.append(r"\endgroup")
        tex.append(rf"\noindent\textit{{Prompt source:}} {_tex_escape(prompt_source)}.")
    else:
        tex.append(r"No prompt text was recoverable for this run.")

    tex.append(r"\end{document}")
    tex_text = "\n".join(tex) + "\n"
    _write_text(output_dir / "report.tex", tex_text)

    if args.compile_pdf:
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"]
        for _ in range(2):
            subprocess.run(cmd, cwd=output_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    print(json.dumps({"output_dir": str(output_dir), "pdf": str(output_dir / "report.pdf"), "figures": str(fig_dir)}, indent=2))


if __name__ == "__main__":
    main()
