#!/usr/bin/env python3
"""
Render a *full* prompt-level deep dive for the slug-year grouped questionnaire workflow.

Compared to scripts/render_questionnaire_pca_prompt_deepdive.py (v1), this script is "full" in two ways:
  1) It embeds the exact prompt text sent to the model for *every* (slug,year) in the request dir.
  2) It attributes the largest year-to-year swing per slug for *each* PCA metric:
        - Principal_Component_1..N
        - Overall_Index

Design goals (artifact-first)
----------------------------
- Do not reconstruct prompts: extract exact model input text from openai_requests_shard*.jsonl.
- Ground any interpretation in those prompt texts (via BLOCK + page_id provenance).
- Make the "why did this PC move?" chain explicit:
    answers/evidence -> located OCR snippet -> standardized feature deltas -> PC contributions.

Notes on attribution
--------------------
PCA is computed on standardized (z-scored) features.
Let z_i be the standardized value of feature i, and loading_{k,i} the PCA component weight for PCk.

Then for two years y1,y2:
  Δz_i = (x_i(y2) - x_i(y1)) / std_i
  ΔPCk = Σ_i loading_{k,i} * Δz_i
  Overall_Index is defined as Σ_i z_i, so:
    ΔOverall = Σ_i Δz_i

We rank "drivers" by |loading_{k,i} * Δz_i| (for PCk) or |Δz_i| (Overall_Index).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from os.path import expanduser
from pathlib import Path
from typing import Any, Iterable


def _eprint(msg: str) -> None:
    print(msg, flush=True)


def _json_loads_line(line: str, *, path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return obj


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip("\n")
        if not line.strip():
            continue
        out.append(_json_loads_line(line, path=path))
    return out


def _tex_escape(s: str) -> str:
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


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _normalize_quotes_and_dashes(s: str) -> str:
    return (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("…", "...")
        .replace("—", "-")
        .replace("–", "-")
    )


def _dehyphenate_linebreaks(s: str) -> str:
    # Join OCR split words like "classifi- cation" -> "classification".
    return re.sub(r"([A-Za-z])\-\s+([a-z])", r"\1\2", s)


def _normalize_for_search(s: str) -> str:
    s = _normalize_quotes_and_dashes(s)
    s = s.strip().strip('"')
    s = s.replace("...", " ")
    s = _dehyphenate_linebreaks(s)
    s = _norm_ws(s)
    # Remove spaces after hyphens: "Two- Family" -> "Two-Family"
    s = re.sub(r"\-\s+", "-", s)
    return s


@dataclass(frozen=True)
class _PromptBundle:
    custom_id: str
    prompt_text: str
    # Mapping from BLOCK index (1-based) -> page_id (provenance).
    block_page_ids: dict[int, str]


@dataclass(frozen=True)
class _EvidenceHit:
    block_idx: int
    page_id: str | None
    snippet: str


def _extract_openai_prompt_text(req_obj: dict[str, Any]) -> str:
    body = req_obj.get("body")
    if not isinstance(body, dict):
        raise SystemExit("OpenAI request JSONL line missing body object")
    inp = body.get("input")
    if not isinstance(inp, list) or not inp:
        raise SystemExit("OpenAI request JSONL line missing body.input[]")
    msg0 = inp[0]
    if not isinstance(msg0, dict):
        raise SystemExit("OpenAI request JSONL body.input[0] is not an object")
    content = msg0.get("content")
    if not isinstance(content, list) or not content:
        raise SystemExit("OpenAI request JSONL body.input[0].content[] missing")
    c0 = content[0]
    if not isinstance(c0, dict):
        raise SystemExit("OpenAI request JSONL body.input[0].content[0] is not an object")
    text = c0.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SystemExit("OpenAI request JSONL missing input_text.text")
    return text


def _parse_blocks_from_prompt(prompt_text: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    pat = re.compile(r"^BLOCK (\d+):\s*$", re.MULTILINE)
    matches = list(pat.finditer(prompt_text))
    for i, m in enumerate(matches):
        bidx = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt_text)
        block_text = prompt_text[start:end].strip("\n")
        blocks.append((bidx, block_text))
    return blocks


def _find_evidence_in_prompt(*, prompt: _PromptBundle, evidence: str) -> _EvidenceHit | None:
    ev = _normalize_for_search(evidence)
    if not ev:
        return None
    ev_l = ev.lower()
    ev_l_nocomma = ev_l.replace(",", "")

    keys = [ev_l, ev_l_nocomma]
    if len(ev_l) > 200:
        keys.append(ev_l[:200])
        keys.append(ev_l_nocomma[:200])
    if len(ev_l) > 120:
        keys.append(ev_l[:120])
        keys.append(ev_l_nocomma[:120])
    if len(ev_l) > 80:
        keys.append(ev_l[:80])
        keys.append(ev_l_nocomma[:80])

    for bidx, btxt in _parse_blocks_from_prompt(prompt.prompt_text):
        bt = _normalize_for_search(btxt)
        bt_l = bt.lower()
        bt_l_nocomma = bt_l.replace(",", "")
        for k in keys:
            if not k:
                continue
            pos = bt_l.find(k)
            used_nocomma = False
            if pos < 0:
                pos = bt_l_nocomma.find(k)
                used_nocomma = pos >= 0
            if pos < 0:
                continue
            bt_snip_src = bt.replace(",", "") if used_nocomma else bt
            start = max(0, pos - 320)
            end = min(len(bt_snip_src), pos + min(len(k), 160) + 520)
            page_id = prompt.block_page_ids.get(bidx)
            return _EvidenceHit(block_idx=bidx, page_id=page_id, snippet=bt_snip_src[start:end])

    # Fallback: token match (long tokens must all appear in a block)
    toks = [w for w in re.split(r"[^a-zA-Z0-9]+", ev_l) if len(w) >= 7][:8]
    if toks:
        for bidx, btxt in _parse_blocks_from_prompt(prompt.prompt_text):
            bt = _normalize_for_search(btxt).lower()
            if all(tok in bt for tok in toks):
                page_id = prompt.block_page_ids.get(bidx)
                return _EvidenceHit(block_idx=bidx, page_id=page_id, snippet=_norm_ws(btxt)[:900])

    return None


@dataclass(frozen=True)
class QuestionMeta:
    qid: str
    short: str
    full: str
    qtype: str
    positive_means_stricter: bool | None


def _load_question_meta(*, questions_xlsx: Path, processed_sheet: str) -> dict[str, QuestionMeta]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Missing dependency: pandas (pip install pandas)") from exc

    if not questions_xlsx.is_file():
        raise SystemExit(f"Questions workbook not found: {questions_xlsx}")

    df = pd.read_excel(questions_xlsx, sheet_name=processed_sheet)
    if "Include" not in df.columns or "ID" not in df.columns:
        raise SystemExit(f"Questions sheet missing required columns (Include, ID). Have: {list(df.columns)}")

    df = df[df["Include"].astype(str).str.strip().str.lower() == "yes"].copy()
    df["ID"] = df["ID"].astype(str)

    out: dict[str, QuestionMeta] = {}
    for _, r in df.iterrows():
        qid = str(r.get("ID") or "").strip()
        if not qid:
            continue
        short = str(r.get("Short Question") or "").strip()
        full = str(r.get("Full Question") or "").strip()
        qtype = str(r.get("Question Type") or "").strip() or "Unknown"
        pms = r.get("Positive Means Stricter")
        pms_b: bool | None = None
        if isinstance(pms, bool):
            pms_b = pms
        elif isinstance(pms, (int, float)) and not (isinstance(pms, float) and math.isnan(pms)):
            pms_b = bool(int(pms))
        elif isinstance(pms, str) and pms.strip():
            pms_b = pms.strip().lower() in {"true", "t", "yes", "y", "1"}
        out[qid] = QuestionMeta(qid=qid, short=short, full=full, qtype=qtype, positive_means_stricter=pms_b)
    return out


def _fmt_answer(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        if isinstance(v, float) and abs(v - int(v)) < 1e-9:
            return str(int(v))
        return f"{float(v):g}"
    return str(v)


def _metric_label(metric: str) -> str:
    m = re.match(r"^Principal_Component_(\d+)$", str(metric))
    if m:
        return f"PC{m.group(1)}"
    if metric == "Overall_Index":
        return "Overall"
    return re.sub(r"[^A-Za-z0-9]+", "_", str(metric)).strip("_")[:20] or "metric"


def _parse_metric_list(metrics_arg: str, scores_cols: Iterable[str]) -> list[str]:
    cols = list(scores_cols)
    default = [c for c in cols if c.startswith("Principal_Component_")] + (["Overall_Index"] if "Overall_Index" in cols else [])
    if not metrics_arg.strip() or metrics_arg.strip().lower() == "all":
        return default
    wanted = [m.strip() for m in metrics_arg.split(",") if m.strip()]
    missing = [m for m in wanted if m not in cols]
    if missing:
        raise SystemExit(f"--metrics requested missing columns in scores sheet: {missing}; available={cols}")
    return wanted


def main() -> None:
    ap = argparse.ArgumentParser(description="Render FULL deep dive: exact questionnaire prompts + PC swing attribution.")
    ap.add_argument("--pca-xlsx", required=True, help="PCA workbook produced by scripts/compute_questionnaire_pca.py")
    ap.add_argument(
        "--questionnaire-request-dir",
        required=True,
        help="Questionnaire request dir (mapping_shard*.jsonl + openai_requests_shard*.jsonl)",
    )
    ap.add_argument("--questionnaire-results-dir", required=True, help="Questionnaire results dir (normalized.jsonl)")
    ap.add_argument("--slugs", default="", help="Optional comma-separated slug allowlist (default: all in mapping)")
    ap.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated scores metrics to analyze (default: all Principal_Component_* plus Overall_Index).",
    )
    ap.add_argument(
        "--top-detail-features",
        type=int,
        default=10,
        help="Number of top contributing features to expand with OCR evidence quotes per swing (default: 10).",
    )
    ap.add_argument("--out-tex", required=True, help="Output .tex path")
    ap.add_argument(
        "--out-prompt-dir",
        required=True,
        help="Directory to write exact prompt text files (referenced by LaTeX via VerbatimInput)",
    )
    args = ap.parse_args()

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Missing dependency: pandas (pip install pandas)") from exc

    pca_xlsx = Path(expanduser(str(args.pca_xlsx))).resolve()
    req_dir = Path(expanduser(str(args.questionnaire_request_dir))).resolve()
    res_dir = Path(expanduser(str(args.questionnaire_results_dir))).resolve()
    out_tex = Path(expanduser(str(args.out_tex))).resolve()
    out_prompt_dir = Path(expanduser(str(args.out_prompt_dir))).resolve()

    if not pca_xlsx.is_file():
        raise SystemExit(f"--pca-xlsx not found: {pca_xlsx}")
    if not req_dir.is_dir():
        raise SystemExit(f"--questionnaire-request-dir not found: {req_dir}")
    if not res_dir.is_dir():
        raise SystemExit(f"--questionnaire-results-dir not found: {res_dir}")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_prompt_dir.mkdir(parents=True, exist_ok=True)

    # Load PCA workbook.
    scores = pd.read_excel(pca_xlsx, sheet_name="scores")
    imputed = pd.read_excel(pca_xlsx, sheet_name="imputed")
    pivot = pd.read_excel(pca_xlsx, sheet_name="pivot_numeric")
    loadings = pd.read_excel(pca_xlsx, sheet_name="loadings", index_col=0)

    if "slug" not in scores.columns or "page_year" not in scores.columns:
        raise SystemExit("PCA scores sheet must include columns: slug, page_year")

    metrics = _parse_metric_list(str(args.metrics), scores.columns)

    # Feature columns are those present in imputed and in loadings columns, excluding the variance column.
    feature_cols = [c for c in loadings.columns if c != "Variance Explained" and c in imputed.columns]
    if "Standard_Deviation" not in loadings.index:
        raise SystemExit("PCA loadings sheet missing Standard_Deviation row (expected from compute_questionnaire_pca.py).")
    std = loadings.loc["Standard_Deviation", feature_cols].astype(float)

    # Load mapping + prompt text (exact) from request dir.
    mapping_rows: dict[str, dict[str, Any]] = {}
    prompt_by_cid: dict[str, str] = {}

    mapping_paths = sorted(req_dir.glob("mapping_shard*.jsonl"))
    if not mapping_paths:
        raise SystemExit(f"No mapping_shard*.jsonl found in {req_dir}")
    for mp in mapping_paths:
        for obj in _read_jsonl(mp):
            cid = obj.get("id") or obj.get("custom_id")
            if isinstance(cid, str) and cid.strip():
                mapping_rows[cid] = obj

    req_paths = sorted(req_dir.glob("openai_requests_shard*.jsonl"))
    if not req_paths:
        raise SystemExit(f"No openai_requests_shard*.jsonl found in {req_dir}")
    for rp in req_paths:
        for obj in _read_jsonl(rp):
            cid = obj.get("custom_id")
            if not isinstance(cid, str) or not cid.strip():
                continue
            prompt_by_cid[cid] = _extract_openai_prompt_text(obj)

    if not mapping_rows:
        raise SystemExit(f"Loaded 0 mapping rows from {req_dir}")
    if not prompt_by_cid:
        raise SystemExit(f"Loaded 0 OpenAI request prompts from {req_dir}")

    # Normalize answers (evidence + answer values).
    norm_path = res_dir / "normalized.jsonl"
    if not norm_path.is_file():
        raise SystemExit(f"Expected normalized.jsonl in results dir: {norm_path}")

    normalized_by_cid: dict[str, dict[str, Any]] = {}
    for obj in _read_jsonl(norm_path):
        cid = obj.get("custom_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        ans = ((obj.get("normalized") or {}).get("answers_by_id") or {})
        if isinstance(ans, dict):
            normalized_by_cid[cid] = ans

    # Determine consistent Questions.xlsx path for label/type lookup (required for robust output).
    questions_xlsx: Path | None = None
    processed_sheet: str | None = None
    for m in mapping_rows.values():
        qx = m.get("questions_xlsx")
        ps = m.get("questions_processed_sheet")
        if isinstance(qx, str) and qx.strip():
            p = Path(expanduser(qx)).resolve()
            if questions_xlsx is None:
                questions_xlsx = p
            elif questions_xlsx != p:
                raise SystemExit(f"Inconsistent questions_xlsx across mapping rows: {questions_xlsx} vs {p}")
        if isinstance(ps, str) and ps.strip():
            if processed_sheet is None:
                processed_sheet = ps
            elif processed_sheet != ps:
                raise SystemExit(f"Inconsistent questions_processed_sheet across mapping rows: {processed_sheet} vs {ps}")
    if questions_xlsx is None or processed_sheet is None:
        raise SystemExit("Mapping rows missing questions_xlsx / questions_processed_sheet; cannot render labels/types.")
    qmeta = _load_question_meta(questions_xlsx=questions_xlsx, processed_sheet=processed_sheet)

    # Slug allowlist
    slug_allow: set[str] = set()
    if str(args.slugs).strip():
        slug_allow = {s.strip() for s in str(args.slugs).split(",") if s.strip()}

    # Cohort slug-years come from the request mapping (this is what we can embed).
    cohort_rows: list[tuple[str, int, str]] = []
    for cid, m in mapping_rows.items():
        slug = m.get("slug")
        year = m.get("page_year")
        if not isinstance(slug, str) or not slug.strip():
            continue
        try:
            year_i = int(year)
        except Exception:
            continue
        if slug_allow and slug not in slug_allow:
            continue
        cohort_rows.append((slug, year_i, cid))
    if not cohort_rows:
        raise SystemExit("No cohort rows after --slugs filter; nothing to render.")

    slugs = sorted({s for s, _, _ in cohort_rows})
    min_year = min(y for _, y, _ in cohort_rows)
    max_year = max(y for _, y, _ in cohort_rows)

    # Helper: map BLOCK index to page_id for a given custom_id.
    def block_page_ids_for(cid: str) -> dict[int, str]:
        m = mapping_rows.get(cid) or {}
        pages = m.get("pages")
        if not isinstance(pages, list):
            return {}
        out: dict[int, str] = {}
        for i, p in enumerate(pages, start=1):
            if not isinstance(p, dict):
                continue
            pid = p.get("page_id")
            if isinstance(pid, str) and pid.strip():
                out[i] = pid
        return out

    # Prompt bundles for evidence localization.
    prompt_bundles: dict[str, _PromptBundle] = {}
    for cid in mapping_rows.keys():
        ptext = prompt_by_cid.get(cid)
        if ptext is None:
            raise SystemExit(f"Missing OpenAI request prompt for custom_id={cid} (not in openai_requests_shard*.jsonl)")
        prompt_bundles[cid] = _PromptBundle(custom_id=cid, prompt_text=ptext, block_page_ids=block_page_ids_for(cid))

    # Write exact prompt text files for ALL cohort slug-years (and only those).
    prompt_files: dict[str, Path] = {}
    for slug, year, cid in sorted(cohort_rows, key=lambda t: (t[0], t[1])):
        prompt_text = prompt_by_cid.get(cid)
        if prompt_text is None:
            raise SystemExit(f"Missing OpenAI request prompt for custom_id={cid}")
        out_path = out_prompt_dir / f"{cid}.txt"
        out_path.write_text(prompt_text.rstrip("\n") + "\n", encoding="utf-8")
        prompt_files[cid] = out_path

    # Index imputed/pivot by (slug,year)
    def _keyed(df):
        out = {}
        for _, r in df.iterrows():
            slug = str(r.get("slug"))
            year = r.get("page_year")
            try:
                year_i = int(year)
            except Exception:
                continue
            out[(slug, year_i)] = r
        return out

    imputed_by_key = _keyed(imputed)
    pivot_by_key = _keyed(pivot)

    # Determine which cohort slug-years are in PCA.
    cohort_in_pca: set[tuple[str, int]] = set(imputed_by_key.keys())

    # Compute per-slug swings for each metric.
    @dataclass(frozen=True)
    class Swing:
        slug: str
        metric: str
        y1: int
        y2: int
        delta: float

    swings: list[Swing] = []
    for slug in slugs:
        g = scores[scores["slug"].astype(str) == slug].dropna(subset=["page_year"]).copy()
        if g.empty:
            continue
        g["page_year"] = g["page_year"].astype(int)
        g = g.sort_values("page_year")
        years = g["page_year"].tolist()
        if len(years) < 2:
            continue
        for metric in metrics:
            if metric not in g.columns:
                continue
            vals_s = g[metric].tolist()
            # drop NaNs aligned with years
            years2: list[int] = []
            vals2: list[float] = []
            for y, v in zip(years, vals_s, strict=False):
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isnan(fv):
                    continue
                years2.append(int(y))
                vals2.append(fv)
            if len(vals2) < 2:
                continue
            diffs = [vals2[i + 1] - vals2[i] for i in range(len(vals2) - 1)]
            idx = max(range(len(diffs)), key=lambda i: abs(diffs[i]))
            swings.append(Swing(slug=slug, metric=metric, y1=int(years2[idx]), y2=int(years2[idx + 1]), delta=float(diffs[idx])))

    if not swings:
        raise SystemExit("No (slug,metric) had >=2 years with non-null values; nothing to write.")

    # LaTeX render.
    lines: list[str] = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{float}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{iftex}")
    lines.append(r"\ifPDFTeX")
    lines.append(r"  \usepackage[utf8]{inputenc}")
    lines.append(r"  \usepackage[T1]{fontenc}")
    lines.append(r"\else")
    lines.append(r"  \usepackage{fontspec}")
    # Ensure verbatim blocks render broad Unicode from OCR; pick the first available.
    lines.append(
        r"  \IfFontExistsTF{Menlo}{\setmonofont{Menlo}}{"
        r"\IfFontExistsTF{DejaVu Sans Mono}{\setmonofont{DejaVu Sans Mono}}{"
        r"\IfFontExistsTF{Arial Unicode MS}{\setmonofont{Arial Unicode MS}}{}"
        r"}}"
    )
    lines.append(r"\fi")
    lines.append(r"\usepackage{fvextra}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\begin{document}")

    lines.append(r"\section*{Prompt-level deep dive (FULL): PCA swings grounded in model-visible OCR text}")
    lines.append(
        r"\noindent "
        + _tex_escape(
            "This document is an artifact-first deep dive for the slug-year grouped ordinance questionnaire workflow. "
            "It embeds the exact prompt text sent to the model for every (slug,year) request in the cohort, and "
            "for each slug it explains the largest year-to-year swing for each PCA metric by ranking contributing "
            "questionnaire features and locating the model-provided evidence inside the model-visible OCR blocks."
        )
    )
    lines.append("")

    lines.append(r"\subsection*{Pipeline change (context)}")
    lines.append(
        r"\noindent "
        + _tex_escape(
            "Previously (baseline), questionnaire answers were produced per bounding box and aggregated downstream into (slug,year). "
            "In the new workflow, the model is asked once per (slug,year) and is given concatenated OCR page blocks ordered oldest to newest, "
            "with an explicit “latest wins” rule for resolving conflicts."
        )
    )
    lines.append("")

    lines.append(r"\subsection*{Cohort}")
    lines.append(r"\noindent " + _tex_escape(f"Slugs: {len(slugs)}; years in request mapping: {min_year}-{max_year}."))
    lines.append(r"\noindent " + _tex_escape(f"Total (slug,year) prompts embedded: {len(cohort_rows)}."))
    lines.append(r"\noindent " + _tex_escape(f"PCA workbook rows (usable feature rows): {len(imputed_by_key)}."))
    lines.append(
        r"\noindent "
        + _tex_escape(
            "Some (slug,year) prompts may be absent from PCA because the group had no usable feature values (all answers null after filtering). "
            "Prompts are still embedded so you can inspect what the model saw."
        )
    )
    lines.append("")

    lines.append(r"\subsection*{Attribution math (what a “driver” means)}")
    lines.append(
        r"\noindent "
        + _tex_escape(
            "Let x_i be the oriented (higher=stricter) feature value for question i (after imputation), and std_i the PCA scaler standard deviation. "
            "We compute Δz_i = (x_i(y2) - x_i(y1)) / std_i. For PCk, the per-feature contribution is Δz_i × loading_{k,i}, which sums to the total ΔPCk. "
            "For Overall_Index (defined as Σ z_i), the per-feature contribution is Δz_i."
        )
    )
    lines.append("")

    lines.append(r"\subsection*{Questions workbook}")
    lines.append(r"\noindent " + _tex_escape(f"{questions_xlsx} (sheet={processed_sheet})"))
    lines.append("")

    # Organize swings: per slug, order metrics stably (PC1.. then Overall).
    def _metric_sort_key(m: str) -> tuple[int, str]:
        mm = re.match(r"^Principal_Component_(\d+)$", m)
        if mm:
            return (int(mm.group(1)), m)
        if m == "Overall_Index":
            return (10_000, m)
        return (99_999, m)

    swings_by_slug: dict[str, list[Swing]] = {}
    for sw in swings:
        swings_by_slug.setdefault(sw.slug, []).append(sw)
    for s in swings_by_slug:
        swings_by_slug[s] = sorted(swings_by_slug[s], key=lambda sw: _metric_sort_key(sw.metric))

    # Helper: observed flag (best-effort) from pivot sheet.
    def observed_flag(qid: str, *, slug: str, year: int) -> str:
        r = pivot_by_key.get((slug, year))
        if r is None or qid not in r.index:
            return "?"
        v = r.get(qid)
        qm = qmeta.get(qid)
        if qm is not None and qm.qtype.lower() == "binary":
            try:
                fv = float(v)
            except Exception:
                return "?"
            if math.isnan(fv):
                return "missing"
            if abs(fv - 0.5) < 1e-9:
                return "unknown"
            return "observed"
        # numeric-ish: NaN => missing
        try:
            fv = float(v)
        except Exception:
            return "?"
        if math.isnan(fv):
            return "missing"
        return "observed"

    # Per slug sections
    for slug in slugs:
        lines.append(r"\clearpage")
        lines.append(rf"\section*{{{_tex_escape(slug)}}}")

        years_for_slug = sorted({y for s, y, _ in cohort_rows if s == slug})
        years_s = ", ".join(str(y) for y in years_for_slug)
        in_pca_years = sorted({y for (s, y) in cohort_in_pca if s == slug})
        in_pca_s = ", ".join(str(y) for y in in_pca_years) if in_pca_years else "(none)"

        lines.append(r"\noindent " + _tex_escape(f"Years in request mapping: {years_s}"))
        lines.append(r"\noindent " + _tex_escape(f"Years present in PCA workbook: {in_pca_s}"))
        lines.append("")

        for sw in swings_by_slug.get(slug, []):
            metric = sw.metric
            lbl = _metric_label(metric)
            lines.append(rf"\subsection*{{{_tex_escape(metric)} — largest YoY swing}}")
            lines.append(r"\noindent " + _tex_escape(f"Swing: {sw.y1} → {sw.y2} (Δ{lbl} = {sw.delta:+.3f})."))

            # Rows must exist in PCA to attribute contributions.
            row1 = imputed_by_key.get((slug, sw.y1))
            row2 = imputed_by_key.get((slug, sw.y2))
            if row1 is None or row2 is None:
                lines.append("")
                lines.append(
                    r"\noindent "
                    + _tex_escape(
                        "Cannot compute feature contributions for this swing because at least one endpoint year is not present in the PCA workbook "
                        "(likely all features were missing for that (slug,year)). See the exact prompt texts in the appendix."
                    )
                )
                lines.append("")
                continue

            # Compute Δz and contributions.
            dz_s = ((row2[feature_cols] - row1[feature_cols]) / std).astype(float)
            if metric == "Overall_Index":
                contrib_s = dz_s.copy()
                loading_s = None
            else:
                m = re.match(r"^Principal_Component_(\d+)$", metric)
                if not m:
                    lines.append("")
                    lines.append(r"\noindent " + _tex_escape("Unknown metric shape; skipping attribution."))
                    lines.append("")
                    continue
                pc_row = f"PC{m.group(1)}"
                if pc_row not in loadings.index:
                    lines.append("")
                    lines.append(r"\noindent " + _tex_escape(f"Missing loadings row {pc_row} in PCA workbook; skipping attribution."))
                    lines.append("")
                    continue
                loading_s = loadings.loc[pc_row, feature_cols].astype(float)
                contrib_s = (dz_s * loading_s).astype(float)

            contrib_sorted = contrib_s.sort_values(key=lambda s: s.abs(), ascending=False)

            # Table: all features, ranked by |contribution|.
            lines.append("")
            lines.append(r"\begin{table}[H]")
            lines.append(r"\centering")
            lines.append(r"\tiny")
            # Columns: rank, qid, label, type, ans1->ans2, obs flags, Dz, loading, contribution, evidence loc
            if loading_s is None:
                lines.append(r"\begin{tabular}{rlll l l rr r}")
                lines.append(r"\toprule")
                lines.append(r"\# & QID & Label & Type & Answers & Obs(y1/y2) & Dz & Contrib & Evidence \\")
            else:
                lines.append(r"\begin{tabular}{rlll l l rrr r}")
                lines.append(r"\toprule")
                lines.append(r"\# & QID & Label & Type & Answers & Obs(y1/y2) & Dz & Loading & Contrib & Evidence \\")
            lines.append(r"\midrule")

            # Collect for detailed sections.
            ranked_qids = [str(q) for q in contrib_sorted.index.tolist()]
            top_detail = ranked_qids[: max(0, int(args.top_detail_features))]

            for rank, qid in enumerate(ranked_qids, start=1):
                qm = qmeta.get(qid)
                label = qm.short if (qm and qm.short) else (qid)
                qtype = qm.qtype if qm else "Unknown"

                a1_all = normalized_by_cid.get(f"{slug}__{sw.y1}", {})
                a2_all = normalized_by_cid.get(f"{slug}__{sw.y2}", {})
                a1 = a1_all.get(qid) if isinstance(a1_all.get(qid), dict) else {}
                a2 = a2_all.get(qid) if isinstance(a2_all.get(qid), dict) else {}
                ans1 = a1.get("answer") if isinstance(a1, dict) else None
                ans2 = a2.get("answer") if isinstance(a2, dict) else None

                obs1 = observed_flag(qid, slug=slug, year=sw.y1)
                obs2 = observed_flag(qid, slug=slug, year=sw.y2)
                obs = f"{obs1}/{obs2}"

                dz = float(dz_s.get(qid) or 0.0)
                contrib = float(contrib_s.get(qid) or 0.0)

                # Evidence location summary (prefer later year evidence).
                ev_year = None
                ev_text = None
                for y in (sw.y2, sw.y1):
                    a = a2 if y == sw.y2 else a1
                    ev = a.get("evidence") if isinstance(a, dict) else None
                    if isinstance(ev, str) and ev.strip():
                        ev_year = y
                        ev_text = ev
                        break
                ev_loc = ""
                if ev_year is not None and ev_text is not None:
                    pb = prompt_bundles.get(f"{slug}__{ev_year}")
                    hit = _find_evidence_in_prompt(prompt=pb, evidence=ev_text) if pb is not None else None
                    if hit is not None:
                        ev_loc = f"{ev_year} BLOCK {hit.block_idx}" + (f" {hit.page_id}" if hit.page_id else "")
                    else:
                        ev_loc = f"{ev_year} (not found)"

                ans_s = f"{_fmt_answer(ans1)}→{_fmt_answer(ans2)}"

                if loading_s is None:
                    lines.append(
                        rf"{rank} & {_tex_escape(qid)} & {_tex_escape(label)[:60]} & {_tex_escape(qtype)[:10]} & "
                        rf"{_tex_escape(ans_s)[:40]} & {_tex_escape(obs)[:16]} & {dz:+.3f} & {contrib:+.3f} & {_tex_escape(ev_loc)[:35]} \\"
                    )
                else:
                    loading = float(loading_s.get(qid) or 0.0)
                    lines.append(
                        rf"{rank} & {_tex_escape(qid)} & {_tex_escape(label)[:60]} & {_tex_escape(qtype)[:10]} & "
                        rf"{_tex_escape(ans_s)[:40]} & {_tex_escape(obs)[:16]} & {dz:+.3f} & {loading:+.3f} & {contrib:+.3f} & {_tex_escape(ev_loc)[:35]} \\"
                    )

            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append("")

            # Detailed per-feature notes with direct OCR quotes.
            lines.append(r"\noindent " + _tex_escape(f"Top {len(top_detail)} contributing questions (expanded with evidence + OCR context):"))
            lines.append("")

            for qid in top_detail:
                qm = qmeta.get(qid)
                short = qm.short if qm else ""
                full = qm.full if qm else ""
                qtype = qm.qtype if qm else "Unknown"
                dz = float(dz_s.get(qid) or 0.0)
                contrib = float(contrib_s.get(qid) or 0.0)
                loading = None if loading_s is None else float(loading_s.get(qid) or 0.0)

                a1_all = normalized_by_cid.get(f"{slug}__{sw.y1}", {})
                a2_all = normalized_by_cid.get(f"{slug}__{sw.y2}", {})
                a1 = a1_all.get(qid) if isinstance(a1_all.get(qid), dict) else {}
                a2 = a2_all.get(qid) if isinstance(a2_all.get(qid), dict) else {}
                ans1 = a1.get("answer") if isinstance(a1, dict) else None
                ans2 = a2.get("answer") if isinstance(a2, dict) else None
                conf1 = a1.get("confidence") if isinstance(a1, dict) else None
                conf2 = a2.get("confidence") if isinstance(a2, dict) else None
                ev1 = a1.get("evidence") if isinstance(a1, dict) else None
                ev2 = a2.get("evidence") if isinstance(a2, dict) else None

                obs1 = observed_flag(qid, slug=slug, year=sw.y1)
                obs2 = observed_flag(qid, slug=slug, year=sw.y2)

                lines.append(rf"\subsubsection*{{{_tex_escape(qid)}{(' — ' + _tex_escape(short)) if short else ''}}}")
                if full and full != short:
                    lines.append(r"\noindent " + _tex_escape(full))
                    lines.append("")

                head = f"type={qtype}; obs({sw.y1})={obs1}, obs({sw.y2})={obs2}; Dz={dz:+.3f}"
                if loading is not None:
                    head += f"; loading={loading:+.3f}"
                head += f"; contrib={contrib:+.3f}"
                lines.append(r"\noindent " + _tex_escape(head))
                lines.append(r"\noindent " + _tex_escape(f"answers: {sw.y1}={ans1!r} (conf={conf1!r}), {sw.y2}={ans2!r} (conf={conf2!r})."))
                lines.append("")

                def render_ev(year: int, ev: Any) -> None:
                    if not isinstance(ev, str) or not ev.strip():
                        lines.append(r"\noindent " + _tex_escape(f"Evidence quote ({year}): null"))
                        lines.append("")
                        return
                    ev_clean = _norm_ws(ev)
                    lines.append(r"\noindent " + _tex_escape(f"Evidence quote ({year}):") + " " + _tex_escape(f"\"{ev_clean}\""))
                    pb = prompt_bundles.get(f"{slug}__{year}")
                    hit = _find_evidence_in_prompt(prompt=pb, evidence=ev) if pb is not None else None
                    if hit is not None:
                        where = f"BLOCK {hit.block_idx}"
                        if hit.page_id:
                            where += f" (page_id={hit.page_id})"
                        lines.append(r"\noindent " + _tex_escape(f"Located in model-visible prompt: {where}."))
                        lines.append(r"\begin{quote}\small\ttfamily " + _tex_escape(hit.snippet) + r"\end{quote}")
                    else:
                        lines.append(
                            r"\noindent "
                            + _tex_escape(
                                "Could not locate the evidence quote verbatim inside the stored prompt text "
                                "(likely due to OCR hyphenation/linebreak differences)."
                            )
                        )
                    lines.append("")

                render_ev(sw.y1, ev1)
                render_ev(sw.y2, ev2)

            lines.append("")

        # Appendix: exact prompt texts for all years of this slug.
        lines.append(r"\subsection*{Appendix: exact prompt texts (as sent to the model)}")
        for year in years_for_slug:
            cid = f"{slug}__{year}"
            pf = prompt_files.get(cid)
            if pf is None:
                continue
            try:
                rel = pf.relative_to(out_tex.parent)
                rel_s = str(rel)
            except Exception:
                rel_s = str(pf)
            lines.append(r"\noindent " + _tex_escape(f"{cid}:"))
            lines.append(rf"\VerbatimInput[breaklines=true,fontsize=\scriptsize,frame=single]{{\detokenize{{{rel_s}}}}}")
            lines.append("")

    lines.append(r"\end{document}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _eprint(f"out_tex\t{out_tex}")
    _eprint(f"out_prompt_dir\t{out_prompt_dir}")
    _eprint(f"slugs\t{len(slugs)}")
    _eprint(f"cohort_slug_years\t{len(cohort_rows)}")
    _eprint(f"metrics\t{','.join(metrics)}")


if __name__ == "__main__":
    main()
