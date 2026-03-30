#!/usr/bin/env python3
"""
Time-series analysis over questionnaire PCA outputs.

Assumes you already ran:
  - scripts/compute_questionnaire_pca.py with group keys including a year field

This script:
  - loads the PCA workbook (scores + group_meta + params)
  - builds a clean long table (entity, year, PC1.., Overall_Index, coverage)
  - computes per-entity time-series metrics:
      * n_years, min/max year
      * total pages/boxes (from group_meta)
      * slope_per_year (simple linear regression)
      * swing (max-min)
      * max_abs_yoy (max absolute year-to-year change)
      * diff_vol (std of year-to-year diffs)
  - writes an XLSX report and (optionally) plots a few top series.

This is intentionally lightweight: it is a triage tool to identify “interesting”
places/years worth deeper inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from os.path import expanduser
from pathlib import Path
from typing import Any


def _eprint(msg: str) -> None:
    print(msg, flush=True)


def _json_dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_params(*, xlsx: Path) -> dict[str, Any]:
    import pandas as pd  # type: ignore

    try:
        df = pd.read_excel(xlsx, sheet_name="params")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to read params sheet from {xlsx}: {exc}") from exc

    if "k" not in df.columns or "v" not in df.columns:
        raise SystemExit(f"params sheet in {xlsx} does not have expected columns (k, v). Have: {list(df.columns)}")

    out: dict[str, Any] = {}
    for _, r in df.iterrows():
        k = r.get("k")
        v = r.get("v")
        if not isinstance(k, str) or not k.strip():
            continue
        if isinstance(v, str) and v.strip().startswith(("{", "[", '"')):
            try:
                out[k] = json.loads(v)
                continue
            except Exception:
                pass
        out[k] = v
    return out


def _ffill_group_keys(df, keys: list[str]):
    # Excel outputs may have blank repeated index values; forward-fill makes them usable.
    for k in keys:
        if k in df.columns:
            df[k] = df[k].ffill()
    return df


def _metric_sort_key(col: str) -> tuple[int, str]:
    m = re.match(r"^Principal_Component_(\d+)$", str(col))
    if m:
        return (int(m.group(1)), col)
    if col == "Overall_Index":
        return (10_000, col)
    return (99_999, col)


def _metric_label(metric: str) -> str:
    m = re.match(r"^Principal_Component_(\d+)$", str(metric))
    if m:
        return f"PC{m.group(1)}"
    if metric == "Overall_Index":
        return "Overall"
    return re.sub(r"[^A-Za-z0-9]+", "_", str(metric)).strip("_")[:20]


def _safe_filename_stem(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(s)).strip("_")
    out = re.sub(r"_+", "_", out)
    return out or "unknown"


ORDINANCE_AMENDMENT_LABELS = {"full_ordinance", "amendment_substantial", "amendment_targeted"}


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


def _coerce_json(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", stripped)
        return json.loads(cleaned)


def _summarize_zoning_box_labels(
    *,
    classification_request_dir: Path,
    classification_results_dir: Path,
    min_confidence: float,
) -> dict[str, Any]:
    """Compute page- and box-level label counts from the box-level classifier run.

    This mirrors scripts/tabulate_zoning_box_labels_by_page.py so we can embed a
    stable summary into the LaTeX report without shelling out.
    """
    req_dir = classification_request_dir
    res_dir = classification_results_dir

    mapping_paths = sorted(req_dir.glob("mapping_shard*.jsonl"))
    result_paths = sorted(res_dir.glob("openai_results_shard*.jsonl"))
    error_paths = sorted(res_dir.glob("openai_errors_shard*.jsonl"))
    if not mapping_paths:
        raise SystemExit(f"No mapping_shard*.jsonl found in {req_dir}")
    if not result_paths:
        raise SystemExit(f"No openai_results_shard*.jsonl found in {res_dir}")
    if not error_paths:
        raise SystemExit(f"No openai_errors_shard*.jsonl found in {res_dir}")

    pages_total: set[str] = set()
    pages_with_any_ok_box: set[str] = set()
    pages_with_any_failure: set[str] = set()

    mapping_by_id: dict[str, str] = {}
    boxes_total = 0

    for mp in mapping_paths:
        for raw in mp.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("id")
            page_id = obj.get("page_id")
            if not isinstance(cid, str) or not cid:
                continue
            if not isinstance(page_id, str) or not page_id:
                continue
            mapping_by_id[cid] = page_id
            pages_total.add(page_id)
            boxes_total += 1

    if not mapping_by_id:
        raise SystemExit(f"Loaded 0 mapping ids from {req_dir}")

    box_label_counts: Counter[str] = Counter()
    page_label_sets: dict[str, set[str]] = defaultdict(set)
    page_ord_label_sets: dict[str, set[str]] = defaultdict(set)

    boxes_ok = 0
    boxes_failed = 0
    fail_reasons: Counter[str] = Counter()

    def note_failure(custom_id: str, reason: str) -> None:
        nonlocal boxes_failed
        boxes_failed += 1
        fail_reasons[reason] += 1
        page_id = mapping_by_id.get(custom_id)
        if page_id is not None:
            pages_with_any_failure.add(page_id)

    def note_ok(custom_id: str, label: str) -> None:
        nonlocal boxes_ok
        boxes_ok += 1
        box_label_counts[label] += 1
        page_id = mapping_by_id.get(custom_id)
        if page_id is None:
            return
        pages_with_any_ok_box.add(page_id)
        page_label_sets[page_id].add(label)
        if label in ORDINANCE_AMENDMENT_LABELS:
            page_ord_label_sets[page_id].add(label)

    min_conf = float(min_confidence)

    # Process successful result lines (may still be incomplete/parse-fail)
    for rp in result_paths:
        with rp.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                custom_id = obj.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id:
                    continue

                resp = obj.get("response")
                if not isinstance(resp, dict):
                    note_failure(custom_id, "no_response")
                    continue

                status_code = resp.get("status_code")
                if status_code != 200:
                    note_failure(custom_id, f"http_{status_code}")
                    continue

                body = resp.get("body")
                if not isinstance(body, dict) or not body:
                    note_failure(custom_id, "empty_body")
                    continue

                if body.get("status") != "completed":
                    note_failure(custom_id, f"body_status_{body.get('status')}")
                    continue

                output_text = _extract_openai_output_text(body)
                if not output_text.strip():
                    note_failure(custom_id, "empty_output_text")
                    continue

                try:
                    parsed = _coerce_json(output_text)
                except Exception:  # noqa: BLE001
                    note_failure(custom_id, "json_parse_fail")
                    continue

                label = parsed.get("label")
                conf = parsed.get("confidence")

                try:
                    conf_f = float(conf) if isinstance(conf, (int, float)) else 0.0
                except Exception:
                    conf_f = 0.0

                if conf_f < min_conf:
                    continue

                if not isinstance(label, str) or not label:
                    note_failure(custom_id, "missing_label")
                    continue

                note_ok(custom_id, label)

    # Process OpenAI error-file lines (failed requests)
    for ep in error_paths:
        with ep.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                custom_id = obj.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id:
                    continue
                note_failure(custom_id, "openai_error_file")

    page_label_counts: Counter[str] = Counter()
    for pid, labels in page_label_sets.items():
        for lab in labels:
            page_label_counts[lab] += 1

    pages_any_ord = len(page_ord_label_sets)
    pages_none_ord = len(pages_total) - pages_any_ord

    combo_counts: Counter[str] = Counter()
    for labels in page_ord_label_sets.values():
        combo_counts[",".join(sorted(labels))] += 1

    label_summary: dict[str, dict[str, int]] = {}
    for label in sorted(ORDINANCE_AMENDMENT_LABELS):
        label_summary[label] = {
            "boxes": int(box_label_counts.get(label, 0)),
            "pages": int(page_label_counts.get(label, 0)),
        }

    return {
        "boxes_total": int(boxes_total),
        "boxes_ok_counted": int(boxes_ok),
        "boxes_failed": int(boxes_failed),
        "unique_pages_total": int(len(pages_total)),
        "unique_pages_with_any_ok_box": int(len(pages_with_any_ok_box)),
        "unique_pages_with_any_failure": int(len(pages_with_any_failure)),
        "unique_pages_with_any_ordinance_or_amendment": int(pages_any_ord),
        "unique_pages_with_no_ordinance_or_amendment": int(pages_none_ord),
        "ordinance_amendment_combo_counts": dict(combo_counts),
        "fail_reason_counts": dict(fail_reasons),
        "label_summary": label_summary,
    }


@dataclass(frozen=True)
class _SeriesStats:
    entity: str
    n_years: int
    min_year: int
    max_year: int
    pages_total: int | None
    boxes_total: int | None
    slope_per_year: float
    swing: float
    max_abs_yoy: float
    diff_vol: float


def _series_stats_for_metric(
    df,
    *,
    entity_key: str,
    year_key: str,
    metric: str,
    pages_col: str | None,
    boxes_col: str | None,
    min_years: int,
):
    import numpy as np  # type: ignore

    rows: list[_SeriesStats] = []
    for entity, g in df.dropna(subset=[entity_key, year_key, metric]).groupby(entity_key):
        g = g.sort_values(year_key)
        years = g[year_key].dropna().unique().tolist()
        n_years = int(len(years))
        if n_years < int(min_years):
            continue

        x = g[year_key].to_numpy(dtype=float)
        y = g[metric].to_numpy(dtype=float)

        x0 = x - float(np.mean(x))
        denom = float(np.sum(x0 * x0))
        slope = float(np.sum(x0 * y) / denom) if denom != 0 else 0.0

        diffs = np.diff(y)
        diff_vol = float(np.std(diffs)) if len(diffs) > 0 else 0.0
        swing = float(np.max(y) - np.min(y))
        max_abs_yoy = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0

        pages_total = None
        boxes_total = None
        if pages_col and pages_col in g.columns:
            try:
                pages_total = int(float(g[pages_col].fillna(0).sum()))
            except Exception:
                pages_total = None
        if boxes_col and boxes_col in g.columns:
            try:
                boxes_total = int(float(g[boxes_col].fillna(0).sum()))
            except Exception:
                boxes_total = None

        rows.append(
            _SeriesStats(
                entity=str(entity),
                n_years=n_years,
                min_year=int(min(years)),
                max_year=int(max(years)),
                pages_total=pages_total,
                boxes_total=boxes_total,
                slope_per_year=slope,
                swing=swing,
                max_abs_yoy=max_abs_yoy,
                diff_vol=diff_vol,
            )
        )
    return rows


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze PCA time series and rank “interesting” entities.")
    ap.add_argument(
        "--pca-xlsx",
        required=True,
        help="Path to PCA workbook produced by scripts/compute_questionnaire_pca.py",
    )
    ap.add_argument("--scores-sheet", default="scores")
    ap.add_argument("--meta-sheet", default="group_meta")
    ap.add_argument("--out-xlsx", required=True, help="Path to write output workbook.")
    ap.add_argument(
        "--out-tex",
        default="",
        help="Optional: write a LaTeX report embedding all generated plots (blank disables).",
    )
    ap.add_argument("--entity-key", default="city_state", help="Entity/group column name (e.g. city_state)")
    ap.add_argument("--year-key", default="page_year", help="Year column name (e.g. page_year)")
    ap.add_argument("--min-years", type=int, default=5, help="Minimum distinct years per entity to include in rankings.")
    ap.add_argument(
        "--min-pages-total",
        type=int,
        default=30,
        help="Coverage filter: require at least this many total pages per entity across included years (0 disables).",
    )
    ap.add_argument(
        "--min-boxes-total",
        type=int,
        default=0,
        help="Coverage filter: require at least this many total boxes per entity across included years (0 disables).",
    )
    ap.add_argument("--top-n", type=int, default=5, help="How many top rows to include per ranking view (and plots).")
    ap.add_argument(
        "--plots-dir",
        default="",
        help="Optional directory to write PNG plots for top series (blank disables plotting).",
    )
    ap.add_argument(
        "--zoning-classification-request-dir",
        default="",
        help=(
            "Optional: directory containing mapping_shard*.jsonl from the box-level zoning classifier export. "
            "If set (and results dir is also set), the LaTeX intro includes a summary table for "
            "{full_ordinance, amendment_substantial, amendment_targeted}."
        ),
    )
    ap.add_argument(
        "--zoning-classification-results-dir",
        default="",
        help=(
            "Optional: directory containing openai_results_shard*.jsonl and openai_errors_shard*.jsonl "
            "from the box-level zoning classifier download."
        ),
    )
    ap.add_argument(
        "--zoning-min-confidence",
        type=float,
        default=0.0,
        help="Optional: only count zoning classifier boxes with confidence >= this value (default: 0.0).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Missing dependency: pandas (pip install pandas)") from exc

    pca_xlsx = Path(expanduser(args.pca_xlsx)).resolve()
    if not pca_xlsx.is_file():
        raise SystemExit(f"--pca-xlsx not found: {pca_xlsx}")

    out_xlsx = Path(expanduser(args.out_xlsx)).resolve()
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    params = _read_params(xlsx=pca_xlsx)
    group_keys = params.get("group_keys")
    if not isinstance(group_keys, list) or not all(isinstance(x, str) for x in group_keys):
        group_keys = [str(args.entity_key), str(args.year_key)]

    scores = pd.read_excel(pca_xlsx, sheet_name=str(args.scores_sheet))
    meta = pd.read_excel(pca_xlsx, sheet_name=str(args.meta_sheet))

    scores = _ffill_group_keys(scores, group_keys)
    meta = _ffill_group_keys(meta, group_keys)

    entity_key = str(args.entity_key)
    year_key = str(args.year_key)
    if entity_key not in scores.columns or year_key not in scores.columns:
        raise SystemExit(
            f"scores sheet missing entity/year columns: entity_key={entity_key!r} year_key={year_key!r}. "
            f"Have: {list(scores.columns)}"
        )

    scores[year_key] = pd.to_numeric(scores[year_key], errors="coerce")
    meta[year_key] = pd.to_numeric(meta[year_key], errors="coerce")

    merged = scores.merge(meta, how="left", on=group_keys, suffixes=("", "_meta"))

    # Find likely coverage columns.
    pages_col = "n_pages" if "n_pages" in merged.columns else None
    boxes_col = "n_boxes" if "n_boxes" in merged.columns else None

    # Metrics we rank by (if present).
    pc_cols = [c for c in merged.columns if isinstance(c, str) and c.startswith("Principal_Component_")]
    metrics = sorted(pc_cols, key=_metric_sort_key)
    if "Overall_Index" in merged.columns:
        metrics.append("Overall_Index")
    if not metrics:
        raise SystemExit("No principal component columns found (expected Principal_Component_1, etc).")

    summary_rows: list[dict[str, Any]] = []
    for metric in metrics:
        stats = _series_stats_for_metric(
            merged,
            entity_key=entity_key,
            year_key=year_key,
            metric=metric,
            pages_col=pages_col,
            boxes_col=boxes_col,
            min_years=int(args.min_years),
        )
        for st in stats:
            summary_rows.append(
                {
                    entity_key: st.entity,
                    "metric": metric,
                    "n_years": st.n_years,
                    "min_year": st.min_year,
                    "max_year": st.max_year,
                    "pages_total": st.pages_total,
                    "boxes_total": st.boxes_total,
                    "slope_per_year": st.slope_per_year,
                    "swing": st.swing,
                    "max_abs_yoy": st.max_abs_yoy,
                    "diff_vol": st.diff_vol,
                }
            )

    summary_all = pd.DataFrame(summary_rows)
    if summary_all.empty:
        raise SystemExit(f"No entities with >= {args.min_years} distinct years found.")

    # Apply basic coverage filters to reduce pathological series.
    summary = summary_all.copy()
    if int(args.min_pages_total) > 0:
        summary = summary[summary["pages_total"].fillna(-1).astype(float) >= float(args.min_pages_total)].copy()
    if int(args.min_boxes_total) > 0:
        summary = summary[summary["boxes_total"].fillna(-1).astype(float) >= float(args.min_boxes_total)].copy()

    if summary.empty:
        raise SystemExit(
            "No entities survived coverage filtering.\n"
            f"  min_years={int(args.min_years)}\n"
            f"  min_pages_total={int(args.min_pages_total)}\n"
            f"  min_boxes_total={int(args.min_boxes_total)}"
        )

    # Views: top by swing, top by max_abs_yoy, top by abs(slope)
    views: dict[str, "pd.DataFrame"] = {}
    for metric in metrics:
        sub = summary[summary["metric"] == metric].copy()
        sub["abs_slope"] = sub["slope_per_year"].abs()
        lbl = _metric_label(metric)
        views[f"top_swing_{lbl}"] = sub.sort_values(["swing", "n_years"], ascending=[False, False]).head(int(args.top_n))
        views[f"top_yoy_{lbl}"] = sub.sort_values(["max_abs_yoy", "n_years"], ascending=[False, False]).head(int(args.top_n))
        views[f"top_slope_abs_{lbl}"] = sub.sort_values(["abs_slope", "n_years"], ascending=[False, False]).head(int(args.top_n))

    # Optional plotting: plot the top-swing entities for each metric.
    plots_dir_s = str(args.plots_dir).strip()
    if plots_dir_s:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"--plots-dir requested but matplotlib is unavailable: {exc}") from exc

        plots_dir = Path(expanduser(plots_dir_s)).resolve()
        plots_dir.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            lbl = _metric_label(metric)
            top_entities = views[f"top_swing_{lbl}"][entity_key].dropna().astype(str).tolist()
            for entity in top_entities[: min(int(args.top_n), len(top_entities))]:
                g = merged[(merged[entity_key] == entity) & merged[metric].notna() & merged[year_key].notna()].copy()
                if g.empty:
                    continue
                g = g.sort_values(year_key)
                x = g[year_key].astype(float).to_numpy()
                y = g[metric].astype(float).to_numpy()
                sizes = None
                if pages_col and pages_col in g.columns:
                    try:
                        sizes = (g[pages_col].fillna(0).astype(float).to_numpy() * 8.0) + 10.0
                    except Exception:
                        sizes = None

                plt.figure(figsize=(8, 3))
                if sizes is None:
                    plt.plot(x, y, marker="o", linewidth=1)
                else:
                    plt.scatter(x, y, s=sizes, alpha=0.8)
                    plt.plot(x, y, linewidth=1, alpha=0.5)
                plt.title(f"{entity} — {metric}")
                plt.xlabel(year_key)
                plt.ylabel(metric)
                plt.tight_layout()
                safe_entity = _safe_filename_stem(entity)
                safe_metric = _safe_filename_stem(metric)
                out = plots_dir / f"{safe_entity}__{safe_metric}.png"
                plt.savefig(out, dpi=160)
                plt.close()

    # Optional LaTeX report (embeds the plots_dir images).
    out_tex_s = str(args.out_tex).strip()
    if out_tex_s:
        if not plots_dir_s:
            raise SystemExit("--out-tex requires --plots-dir (so we have images to embed).")

        out_tex = Path(expanduser(out_tex_s)).resolve()
        out_tex.parent.mkdir(parents=True, exist_ok=True)

        def _tex_escape(s: str) -> str:
            # Minimal escaping for common special chars.
            return (
                s.replace("\\", "\\textbackslash{}")
                .replace("&", "\\&")
                .replace("%", "\\%")
                .replace("$", "\\$")
                .replace("#", "\\#")
                .replace("_", "\\_")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("~", "\\textasciitilde{}")
                .replace("^", "\\textasciicircum{}")
            )

        plots_dir = Path(expanduser(plots_dir_s)).resolve()

        # Render TeX
        zoning_req_s = str(args.zoning_classification_request_dir or "").strip()
        zoning_res_s = str(args.zoning_classification_results_dir or "").strip()
        zoning_summary: dict[str, Any] | None = None
        if zoning_req_s or zoning_res_s:
            if not (zoning_req_s and zoning_res_s):
                raise SystemExit(
                    "--zoning-classification-request-dir and --zoning-classification-results-dir must be provided together"
                )
            zoning_summary = _summarize_zoning_box_labels(
                classification_request_dir=Path(expanduser(zoning_req_s)).resolve(),
                classification_results_dir=Path(expanduser(zoning_res_s)).resolve(),
                min_confidence=float(args.zoning_min_confidence),
            )

        lines: list[str] = []
        lines.append(r"\documentclass[11pt]{article}")
        lines.append(r"\usepackage[margin=1in]{geometry}")
        lines.append(r"\usepackage{graphicx}")
        lines.append(r"\usepackage{booktabs}")
        lines.append(r"\usepackage{float}")
        lines.append(r"\begin{document}")
        if zoning_summary is not None:
            total_pages = int(zoning_summary["unique_pages_total"])
            boxes_total = int(zoning_summary["boxes_total"])
            boxes_ok = int(zoning_summary["boxes_ok_counted"])
            boxes_failed = int(zoning_summary["boxes_failed"])
            pages_any = int(zoning_summary["unique_pages_with_any_ordinance_or_amendment"])
            pages_none = int(zoning_summary.get("unique_pages_with_no_ordinance_or_amendment", 0))
            combo_counts = zoning_summary.get("ordinance_amendment_combo_counts") or {}
            if not isinstance(combo_counts, dict):
                combo_counts = {}

            ls: dict[str, dict[str, int]] = zoning_summary["label_summary"]
            full_boxes = int(ls["full_ordinance"]["boxes"])
            full_pages = int(ls["full_ordinance"]["pages"])
            sub_boxes = int(ls["amendment_substantial"]["boxes"])
            sub_pages = int(ls["amendment_substantial"]["pages"])
            tar_boxes = int(ls["amendment_targeted"]["boxes"])
            tar_pages = int(ls["amendment_targeted"]["pages"])

            any_boxes = full_boxes + sub_boxes + tar_boxes

            def _pct(n: int, d: int) -> str:
                return f"{(n / d * 100.0):.1f}\\%" if d else "0.0\\%"

            lines.append(
                r"\noindent "
                + _tex_escape(
                    f"I processed {total_pages:,} unique newspaper page images (PNGs). "
                    f"Across those pages, I issued {boxes_total:,} box-level zoning-classification requests. "
                    f"{boxes_ok:,} completed and parsed successfully; {boxes_failed:,} failed (incomplete/parse/API errors)."
                )
            )
            lines.append(
                r"\noindent "
                + _tex_escape(
                    "The table below summarizes the three zoning-relevant labels used downstream "
                    "(full ordinance + two amendment types). Counts are shown both by box and by page."
                )
            )
            lines.append(r"")
            lines.append(r"\begin{table}[H]")
            lines.append(r"\centering")
            lines.append(r"\small")
            lines.append(r"\begin{tabular}{lrrr}")
            lines.append(r"\toprule")
            lines.append(r"Label & Boxes & Pages (PNGs) & \% of PNGs \\")
            lines.append(r"\midrule")
            lines.append(rf"Full ordinance (full\_ordinance) & {full_boxes:,} & {full_pages:,} & {_pct(full_pages, total_pages)} \\")
            lines.append(
                rf"Amendment — substantial (amendment\_substantial) & {sub_boxes:,} & {sub_pages:,} & {_pct(sub_pages, total_pages)} \\"
            )
            lines.append(
                rf"Amendment — targeted (amendment\_targeted) & {tar_boxes:,} & {tar_pages:,} & {_pct(tar_pages, total_pages)} \\"
            )
            lines.append(r"\midrule")
            lines.append(
                rf"Any of the above (page has $\geq$1 such box) & {any_boxes:,} & {pages_any:,} & {_pct(pages_any, total_pages)} \\"
            )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append(r"")

            # Mutually exclusive page-level breakdown (by combination of labels present on the page).
            lines.append(
                r"\noindent "
                + _tex_escape(
                    "For a mutually exclusive view at the page level, each page is assigned to exactly one category "
                    "based on which of the three labels appear at least once on that page:"
                )
            )
            lines.append(r"")

            label_human = {
                "full_ordinance": "Full ordinance",
                "amendment_substantial": "Amendment — substantial",
                "amendment_targeted": "Amendment — targeted",
            }

            def _combo_display(labels: set[str]) -> str:
                if not labels:
                    return "None of the three labels"
                ordered = sorted(labels, key=lambda s: (s != "full_ordinance", s))
                parts = [label_human.get(x, x) for x in ordered]
                if len(parts) == 1:
                    return parts[0] + " only"
                if len(parts) == 2:
                    return parts[0] + " + " + parts[1]
                return " + ".join(parts)

            def _combo_key_to_set(k: str) -> set[str]:
                s = str(k).strip()
                if not s:
                    return set()
                return {p.strip() for p in s.split(",") if p.strip()}

            desired_combos: list[set[str]] = [
                set(),
                {"full_ordinance"},
                {"amendment_substantial"},
                {"amendment_targeted"},
                {"full_ordinance", "amendment_substantial"},
                {"full_ordinance", "amendment_targeted"},
                {"amendment_substantial", "amendment_targeted"},
                {"full_ordinance", "amendment_substantial", "amendment_targeted"},
            ]

            # Build counts in a stable order; include "none" explicitly.
            combo_count_by_set: dict[frozenset[str], int] = {}
            for k, v in combo_counts.items():
                try:
                    n = int(v)
                except Exception:
                    continue
                combo_count_by_set[frozenset(_combo_key_to_set(str(k)))] = n

            combo_count_by_set[frozenset()] = int(pages_none)

            lines.append(r"\begin{table}[H]")
            lines.append(r"\centering")
            lines.append(r"\small")
            lines.append(r"\begin{tabular}{lrr}")
            lines.append(r"\toprule")
            lines.append(r"Mutually exclusive category & Pages (PNGs) & \% of PNGs \\")
            lines.append(r"\midrule")
            for labels in desired_combos:
                n_pages = int(combo_count_by_set.get(frozenset(labels), 0))
                lines.append(rf"{_tex_escape(_combo_display(labels))} & {n_pages:,} & {_pct(n_pages, total_pages)} \\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append(r"")

            lines.append(
                r"\noindent\textbf{Nuances.} "
                + _tex_escape(
                    "Page-level counts are not mutually exclusive: a single page can contribute to multiple rows if different boxes "
                    "on that page receive different labels. Each box, however, has exactly one label. "
                    "A page is counted as having a label if any box on that page has that label, even if other boxes are unrelated. "
                    "Interpret page-level counts as a flag (at least one qualifying box), not as a dominant-page label. "
                    "Other classifier labels (e.g., unrelated/public_hearing) are omitted from this summary table."
                )
            )
        else:
            lines.append(
                r"\noindent "
                + _tex_escape(
                    "This report ranks “interesting” PCA time series by swing (max–min) for each metric. "
                    "Pass --zoning-classification-request-dir/--zoning-classification-results-dir to include a zoning label summary."
                )
            )

        for metric in metrics:
            lbl = _metric_label(metric)
            top = views[f"top_swing_{lbl}"]
            lines.append(r"\clearpage")
            lines.append(rf"\section*{{{_tex_escape(metric)} — Top {int(args.top_n)} swings}}")

            # Summary table
            lines.append(r"\begin{table}[H]")
            lines.append(r"\centering")
            lines.append(r"\small")
            lines.append(r"\begin{tabular}{llrrrr}")
            lines.append(r"\toprule")
            lines.append(r"Entity & Years & n\_years & pages & swing & max\_abs\_yoy \\")
            lines.append(r"\midrule")
            for _, r in top.iterrows():
                entity = str(r.get(entity_key))
                years = f"{int(r.get('min_year'))}-{int(r.get('max_year'))}"
                pages_val = r.get("pages_total")
                pages = "" if pages_val is None or pd.isna(pages_val) else int(pages_val)
                lines.append(
                    rf"{_tex_escape(entity)} & {years} & {int(r.get('n_years'))} & {pages} & {float(r.get('swing')):.3f} & {float(r.get('max_abs_yoy')):.3f} \\"
                )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")

            # Figures
            for _, r in top.iterrows():
                entity = str(r.get(entity_key))
                safe_entity = _safe_filename_stem(entity)
                safe_metric = _safe_filename_stem(metric)
                fig_path = plots_dir / f"{safe_entity}__{safe_metric}.png"
                try:
                    rel = fig_path.relative_to(out_tex.parent)
                    rel_s = str(rel)
                except Exception:
                    rel_s = str(fig_path)

                pages_val = r.get("pages_total")
                pages_total = None if pages_val is None or pd.isna(pages_val) else int(pages_val)
                pages_total_s = str(pages_total) if pages_total is not None else "NA"

                cap = (
                    f"{entity} ({int(r.get('min_year'))}-{int(r.get('max_year'))}, "
                    f"n_years={int(r.get('n_years'))}, pages_total={pages_total_s}, "
                    f"swing={float(r.get('swing')):.3f})"
                )

                lines.append(r"\begin{figure}[H]")
                lines.append(r"\centering")
                lines.append(rf"\includegraphics[width=\linewidth]{{{_tex_escape(rel_s)}}}")
                lines.append(rf"\caption{{{_tex_escape(cap)}}}")
                lines.append(r"\end{figure}")

        lines.append(r"\end{document}")
        out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Write workbook
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        merged.sort_values(group_keys).to_excel(xw, sheet_name="timeseries_long", index=False)
        summary_all.sort_values(["metric", "swing"], ascending=[True, False]).to_excel(
            xw, sheet_name="entity_metric_summary_all", index=False
        )
        summary.sort_values(["metric", "swing"], ascending=[True, False]).to_excel(
            xw, sheet_name="entity_metric_summary", index=False
        )
        for sheet, dfv in views.items():
            dfv.to_excel(xw, sheet_name=sheet[:31], index=False)

        pd.DataFrame([{"k": k, "v": _json_dumps(v) if isinstance(v, (dict, list)) else v} for k, v in params.items()]).to_excel(
            xw, sheet_name="pca_params", index=False
        )

    _eprint(f"out_xlsx\t{out_xlsx}")
    _eprint(f"metrics\t{_json_dumps(metrics)}")
    _eprint(f"entities_with_min_years\t{int(summary[summary['metric'] == metrics[0]][entity_key].nunique())}")
    _eprint(f"min_years\t{int(args.min_years)}")
    _eprint(f"min_pages_total\t{int(args.min_pages_total)}")
    _eprint(f"min_boxes_total\t{int(args.min_boxes_total)}")
    _eprint(f"top_n\t{int(args.top_n)}")


if __name__ == "__main__":
    main()
