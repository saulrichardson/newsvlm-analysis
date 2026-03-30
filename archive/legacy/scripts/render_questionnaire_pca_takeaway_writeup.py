#!/usr/bin/env python3
"""
Render a narrative "takeaway" writeup after the prompt-level deep dive.

This is designed to be readable (unlike the FULL 700+ page deep dive), while still being
artifact-first and fully auditable:
  - It embeds the *exact* prompts sent to the LLM for a few representative examples.
  - It extracts the exact OCR BLOCK text that contains the model's evidence quote.
  - It shows the PCA attribution math outputs (Dz, loading, contribution) that explain the PC swing.

The intended message of the writeup is:
  - Many large PCA swings are driven by answerability (null ↔ non-null) rather than clear policy change.
  - Drivers concentrate in a small set of questions (e.g., Z11, 28Min, 4, 14, 5).

Inputs
------
- PCA workbook produced by scripts/compute_questionnaire_pca.py:
    sheets: scores, imputed, pivot_numeric, loadings
- Questionnaire request dir:
    mapping_shard*.jsonl + openai_requests_shard*.jsonl (exact prompts)
- Questionnaire results dir:
    normalized.jsonl (answers/evidence)

Output
------
- A LaTeX file plus an inputs directory containing:
    - exact prompt text files for embedded examples
    - extracted OCR block text files for key drivers
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
    return re.sub(r"([A-Za-z])\-\s+([a-z])", r"\1\2", s)


def _normalize_for_search(s: str) -> str:
    s = _normalize_quotes_and_dashes(s)
    s = s.strip().strip('"')
    s = s.replace("...", " ")
    s = _dehyphenate_linebreaks(s)
    s = _norm_ws(s)
    s = re.sub(r"\-\s+", "-", s)
    return s


@dataclass(frozen=True)
class _PromptBundle:
    custom_id: str
    prompt_text: str
    block_page_ids: dict[int, str]


@dataclass(frozen=True)
class _EvidenceHit:
    block_idx: int
    page_id: str | None
    snippet: str
    full_block_text: str


@dataclass(frozen=True)
class QuestionMeta:
    qid: str
    short: str
    full: str
    qtype: str


@dataclass(frozen=True)
class Swing:
    slug: str
    metric: str
    y1: int
    y2: int
    delta: float
    top8_missingness_flips: int


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
            return _EvidenceHit(
                block_idx=bidx,
                page_id=page_id,
                snippet=bt_snip_src[start:end],
                full_block_text=btxt,
            )

    # Fallback: token match (long tokens must all appear in a block)
    toks = [w for w in re.split(r"[^a-zA-Z0-9]+", ev_l) if len(w) >= 7][:8]
    if toks:
        for bidx, btxt in _parse_blocks_from_prompt(prompt.prompt_text):
            bt = _normalize_for_search(btxt).lower()
            if all(tok in bt for tok in toks):
                page_id = prompt.block_page_ids.get(bidx)
                return _EvidenceHit(
                    block_idx=bidx,
                    page_id=page_id,
                    snippet=_norm_ws(btxt)[:900],
                    full_block_text=btxt,
                )
    return None


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


def _parse_metric_list(scores_cols: Iterable[str]) -> list[str]:
    cols = list(scores_cols)
    pcs = [c for c in cols if c.startswith("Principal_Component_")]
    pcs = sorted(pcs, key=lambda c: int(c.split("_")[-1]))
    return pcs + (["Overall_Index"] if "Overall_Index" in cols else [])


def _load_question_meta(*, questions_xlsx: Path, processed_sheet: str) -> dict[str, QuestionMeta]:
    import pandas as pd  # type: ignore

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
        out[qid] = QuestionMeta(
            qid=qid,
            short=str(r.get("Short Question") or "").strip(),
            full=str(r.get("Full Question") or "").strip(),
            qtype=str(r.get("Question Type") or "").strip() or "Unknown",
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Render narrative takeaway writeup with fully embedded examples.")
    ap.add_argument("--pca-xlsx", required=True, help="PCA workbook produced by scripts/compute_questionnaire_pca.py")
    ap.add_argument("--questionnaire-request-dir", required=True, help="Questionnaire request dir (mapping + openai_requests)")
    ap.add_argument("--questionnaire-results-dir", required=True, help="Questionnaire results dir (normalized.jsonl)")
    ap.add_argument("--out-tex", required=True, help="Output .tex path")
    ap.add_argument("--out-dir", required=True, help="Directory to write embedded prompt+block text files")
    ap.add_argument("--top-driver-rows", type=int, default=25, help="Rows to show in per-example ranked driver table (default: 25 = all)")
    ap.add_argument("--top-driver-blocks", type=int, default=3, help="How many top drivers to expand with full OCR BLOCK text (default: 3)")
    args = ap.parse_args()

    import pandas as pd  # type: ignore

    pca_xlsx = Path(expanduser(str(args.pca_xlsx))).resolve()
    req_dir = Path(expanduser(str(args.questionnaire_request_dir))).resolve()
    res_dir = Path(expanduser(str(args.questionnaire_results_dir))).resolve()
    out_tex = Path(expanduser(str(args.out_tex))).resolve()
    out_dir = Path(expanduser(str(args.out_dir))).resolve()

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PCA artifacts.
    scores = pd.read_excel(pca_xlsx, sheet_name="scores")
    imputed = pd.read_excel(pca_xlsx, sheet_name="imputed")
    pivot = pd.read_excel(pca_xlsx, sheet_name="pivot_numeric")
    loadings = pd.read_excel(pca_xlsx, sheet_name="loadings", index_col=0)

    if "slug" not in scores.columns or "page_year" not in scores.columns:
        raise SystemExit("PCA scores sheet must include columns: slug, page_year")

    metrics = _parse_metric_list(scores.columns)

    feature_cols = [c for c in loadings.columns if c != "Variance Explained" and c in imputed.columns]
    if "Standard_Deviation" not in loadings.index:
        raise SystemExit("PCA loadings sheet missing Standard_Deviation row.")
    std = loadings.loc["Standard_Deviation", feature_cols].astype(float)

    # Load mapping + exact prompts.
    mapping_rows: dict[str, dict[str, Any]] = {}
    prompt_by_cid: dict[str, str] = {}

    mapping_paths = sorted(req_dir.glob("mapping_shard*.jsonl"))
    req_paths = sorted(req_dir.glob("openai_requests_shard*.jsonl"))
    if not mapping_paths:
        raise SystemExit(f"No mapping_shard*.jsonl found in {req_dir}")
    if not req_paths:
        raise SystemExit(f"No openai_requests_shard*.jsonl found in {req_dir}")

    for mp in mapping_paths:
        for obj in _read_jsonl(mp):
            cid = obj.get("id") or obj.get("custom_id")
            if isinstance(cid, str) and cid.strip():
                mapping_rows[cid] = obj

    for rp in req_paths:
        for obj in _read_jsonl(rp):
            cid = obj.get("custom_id")
            if isinstance(cid, str) and cid.strip():
                prompt_by_cid[cid] = _extract_openai_prompt_text(obj)

    # Answers/evidence.
    norm_path = res_dir / "normalized.jsonl"
    if not norm_path.is_file():
        raise SystemExit(f"Expected normalized.jsonl: {norm_path}")
    normalized_by_cid: dict[str, dict[str, Any]] = {}
    for obj in _read_jsonl(norm_path):
        cid = obj.get("custom_id")
        ans = ((obj.get("normalized") or {}).get("answers_by_id") or {})
        if isinstance(cid, str) and cid.strip() and isinstance(ans, dict):
            normalized_by_cid[cid] = ans

    # Questions workbook (for labels).
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
        raise SystemExit("Mapping missing questions_xlsx / questions_processed_sheet; cannot label questions.")
    qmeta = _load_question_meta(questions_xlsx=questions_xlsx, processed_sheet=processed_sheet)

    # Index imputed/pivot by (slug,year).
    def _keyed(df):
        out = {}
        for _, r in df.iterrows():
            try:
                year_i = int(r.get("page_year"))
            except Exception:
                continue
            out[(str(r.get("slug")), year_i)] = r
        return out

    imputed_by_key = _keyed(imputed)
    pivot_by_key = _keyed(pivot)

    # Prompt bundles.
    def block_page_ids_for(cid: str) -> dict[int, str]:
        m = mapping_rows.get(cid) or {}
        pages = m.get("pages")
        if not isinstance(pages, list):
            return {}
        out: dict[int, str] = {}
        for i, p in enumerate(pages, start=1):
            if isinstance(p, dict):
                pid = p.get("page_id")
                if isinstance(pid, str) and pid.strip():
                    out[i] = pid
        return out

    prompt_bundles: dict[str, _PromptBundle] = {}
    for cid in mapping_rows.keys():
        ptext = prompt_by_cid.get(cid)
        if ptext is None:
            raise SystemExit(f"Missing stored prompt text for custom_id={cid}")
        prompt_bundles[cid] = _PromptBundle(custom_id=cid, prompt_text=ptext, block_page_ids=block_page_ids_for(cid))

    # Helper: observed/unknown/missing flag from pivot sheet (best-effort).
    def observed_flag(qid: str, *, slug: str, year: int) -> str:
        r = pivot_by_key.get((slug, year))
        if r is None:
            return "?"
        if qid not in r.index:
            return "?"
        v = r.get(qid)
        # binary: unknown is 0.5 in pivot_numeric for this pipeline; missing is NaN.
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
        try:
            fv = float(v)
        except Exception:
            return "?"
        if math.isnan(fv):
            return "missing"
        return "observed"

    # Compute swings per slug/metric (largest YoY by abs delta), plus missingness-flip score.
    swings: list[Swing] = []
    for slug in sorted(scores["slug"].dropna().astype(str).unique().tolist()):
        g = scores[scores["slug"].astype(str) == slug].copy()
        g = g.dropna(subset=["page_year"]).copy()
        if g.empty:
            continue
        g["page_year"] = g["page_year"].astype(int)
        g = g.sort_values("page_year")
        years_all = g["page_year"].tolist()
        if len(years_all) < 2:
            continue
        for metric in metrics:
            if metric not in g.columns:
                continue
            # filter non-nan values aligned with years
            years: list[int] = []
            vals: list[float] = []
            for y, v in zip(years_all, g[metric].tolist(), strict=False):
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isnan(fv):
                    continue
                years.append(int(y))
                vals.append(fv)
            if len(vals) < 2:
                continue
            diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            idx = max(range(len(diffs)), key=lambda i: abs(diffs[i]))
            y1, y2 = int(years[idx]), int(years[idx + 1])
            delta = float(diffs[idx])

            # missingness flips among top-8 drivers
            r1 = imputed_by_key.get((slug, y1))
            r2 = imputed_by_key.get((slug, y2))
            flips = 0
            if r1 is not None and r2 is not None:
                dz = ((r2[feature_cols] - r1[feature_cols]) / std).astype(float)
                if metric == "Overall_Index":
                    contrib = dz
                else:
                    m = re.match(r"^Principal_Component_(\d+)$", metric)
                    if m:
                        pc_row = f"PC{m.group(1)}"
                        loading = loadings.loc[pc_row, feature_cols].astype(float)
                        contrib = dz * loading
                    else:
                        contrib = dz
                top8 = contrib.abs().sort_values(ascending=False).head(8).index.tolist()
                cid1 = f"{slug}__{y1}"
                cid2 = f"{slug}__{y2}"
                a1_all = normalized_by_cid.get(cid1, {})
                a2_all = normalized_by_cid.get(cid2, {})
                for qid in top8:
                    a1 = a1_all.get(qid) if isinstance(a1_all.get(qid), dict) else {}
                    a2 = a2_all.get(qid) if isinstance(a2_all.get(qid), dict) else {}
                    ans1 = a1.get("answer") if isinstance(a1, dict) else None
                    ans2 = a2.get("answer") if isinstance(a2, dict) else None
                    if (ans1 is None) != (ans2 is None):
                        flips += 1

            swings.append(
                Swing(
                    slug=slug,
                    metric=metric,
                    y1=y1,
                    y2=y2,
                    delta=delta,
                    top8_missingness_flips=int(flips),
                )
            )

    if not swings:
        raise SystemExit("No swings computed; check PCA workbook.")

    # Summary stats used in the writeup.
    total_swings = len(swings)
    missingness_heavy = sum(1 for s in swings if s.top8_missingness_flips >= 4)

    flip_hist: dict[int, int] = {}
    for s in swings:
        flip_hist[int(s.top8_missingness_flips)] = flip_hist.get(int(s.top8_missingness_flips), 0) + 1

    # Driver concentration: count qids appearing in top-5 drivers across swings.
    from collections import Counter

    qid_top5_counts: Counter[str] = Counter()
    for s in swings:
        r1 = imputed_by_key.get((s.slug, s.y1))
        r2 = imputed_by_key.get((s.slug, s.y2))
        if r1 is None or r2 is None:
            continue
        dz = ((r2[feature_cols] - r1[feature_cols]) / std).astype(float)
        if s.metric == "Overall_Index":
            contrib = dz
        else:
            m = re.match(r"^Principal_Component_(\d+)$", s.metric)
            if not m:
                continue
            pc_row = f"PC{m.group(1)}"
            loading = loadings.loc[pc_row, feature_cols].astype(float)
            contrib = dz * loading
        top5 = contrib.abs().sort_values(ascending=False).head(5).index.tolist()
        for qid in top5:
            qid_top5_counts[str(qid)] += 1

    # Choose representative examples (auto-selected from the computed swings).
    def pick_max_abs(metric: str) -> Swing:
        cand = [s for s in swings if s.metric == metric]
        if not cand:
            raise SystemExit(f"No swings available for metric={metric}")
        return max(cand, key=lambda s: abs(s.delta))

    example_overall = pick_max_abs("Overall_Index")
    example_pc1 = pick_max_abs("Principal_Component_1")
    # Typical missingness-driven example: pick Ames PC1 if present, else highest missingness flips.
    ames_pc1 = next((s for s in swings if s.slug == "ames-daily-tribune" and s.metric == "Principal_Component_1"), None)
    if ames_pc1 is None:
        ames_pc1 = max(swings, key=lambda s: (s.top8_missingness_flips, abs(s.delta)))
    # Lower-missingness example: choose a swing with flips<=3 and largest abs delta among those.
    low_flip = [s for s in swings if s.top8_missingness_flips <= 3]
    if not low_flip:
        low_flip = swings
    example_low_missing = max(low_flip, key=lambda s: abs(s.delta))

    examples: list[Swing] = [example_overall, example_pc1, ames_pc1, example_low_missing]
    # de-dupe (keep order)
    seen = set()
    uniq_examples: list[Swing] = []
    for ex in examples:
        k = (ex.slug, ex.metric, ex.y1, ex.y2)
        if k in seen:
            continue
        seen.add(k)
        uniq_examples.append(ex)
    examples = uniq_examples

    # Write prompt files for examples (exact text).
    prompt_files: dict[str, Path] = {}
    for ex in examples:
        for year in (ex.y1, ex.y2):
            cid = f"{ex.slug}__{year}"
            ptext = prompt_by_cid.get(cid)
            if ptext is None:
                raise SystemExit(f"Missing prompt text for custom_id={cid}")
            out_path = out_dir / f"{cid}.txt"
            out_path.write_text(ptext.rstrip("\n") + "\n", encoding="utf-8")
            prompt_files[cid] = out_path

    # LaTeX
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
    # Broad unicode in verbatim blocks (OCR): pick the first available monofont.
    lines.append(
        r"  \IfFontExistsTF{Menlo}{\setmonofont{Menlo}}{"
        r"\IfFontExistsTF{DejaVu Sans Mono}{\setmonofont{DejaVu Sans Mono}}{"
        r"\IfFontExistsTF{Arial Unicode MS}{\setmonofont{Arial Unicode MS}}{}"
        r"}}"
    )
    lines.append(r"\fi")
    lines.append(r"\usepackage{fvextra}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\begin{document}")

    lines.append(r"\section*{Takeaway after prompt-level deep dive: why PCA swings are often answerability-driven}")
    lines.append(
        r"\noindent "
        + _tex_escape(
            "This writeup summarizes what the prompt-level deep dive shows for the slug-year grouped questionnaire workflow: "
            "large PCA movements are frequently driven by whether key questions become answerable (null <-> non-null) in a given year, "
            "rather than clear municipality-wide policy changes. "
            "All examples below embed the exact prompt text that was sent to the model, and the exact OCR BLOCKs that contain the model's evidence quotes."
        )
    )
    lines.append("")

    lines.append(r"\subsection*{Cohort and measurement}")
    lines.append(r"\noindent " + _tex_escape(f"PCA workbook: {pca_xlsx}"))
    lines.append(r"\noindent " + _tex_escape(f"Questions workbook: {questions_xlsx} (sheet={processed_sheet})"))
    lines.append(r"\noindent " + _tex_escape(f"Swings analyzed: {total_swings} (one per slug × metric; metrics={', '.join(metrics)})."))
    lines.append("")

    lines.append(r"\subsection*{Key quantitative observation: missingness dominates many large swings}")
    lines.append(
        r"\noindent "
        + _tex_escape(
            "For each (slug, metric) we locate the largest year-to-year swing and then rank features by absolute contribution. "
            "We then count, among the top 8 drivers, how many flip between having a non-null answer and being null (answerability flips). "
            f"In this cohort, {missingness_heavy} / {total_swings} swings have >=4 answerability flips among their top 8 drivers."
        )
    )
    lines.append("")

    # Missingness flip histogram table
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{rr}")
    lines.append(r"\toprule")
    lines.append(r"Top-8 answerability flips & Count of swings \\")
    lines.append(r"\midrule")
    for k in sorted(flip_hist):
        lines.append(rf"{k} & {int(flip_hist[k])} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"\subsection*{Driver concentration: a small set of questions repeatedly moves PCs}")
    lines.append(
        r"\noindent "
        + _tex_escape(
            "Across all (slug, metric) largest-swing events, the same questions appear repeatedly among the top contributors. "
            "This is expected when: (a) those questions are frequently answerable from ordinance-like text, and (b) unknowns "
            "collapse to a common imputed value, creating large standardized deltas when a real answer appears/disappears."
        )
    )
    lines.append("")

    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llr}")
    lines.append(r"\toprule")
    lines.append(r"Question ID & Label & Times in top-5 \\")
    lines.append(r"\midrule")
    for qid, n in qid_top5_counts.most_common(15):
        lbl = qmeta.get(qid).short if qid in qmeta and qmeta[qid].short else ""
        lines.append(rf"{_tex_escape(qid)} & {_tex_escape(lbl)} & {int(n)} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"\subsection*{Why this happens (mechanism)}")
    lines.append(
        r"\noindent "
        + _tex_escape(
            "PCA is computed on standardized features. In this pipeline, binary unknowns often behave like a common midpoint "
            "(e.g., 0.5), and missing numeric values are imputed from neighbors. When a year contains ordinance-like text that "
            "lets the model answer a question with evidence, the feature value moves away from that midpoint, producing a large "
            "Δz. When multiple highly-weighted features do this together, a principal component can move substantially even if "
            "the underlying policy did not change by the same magnitude."
        )
    )
    lines.append("")

    # Embedded examples
    lines.append(r"\section*{Fully embedded examples (exact prompts + OCR BLOCKs + contributions)}")

    def write_example(ex: Swing) -> None:
        slug = ex.slug
        metric = ex.metric
        lbl = _metric_label(metric)
        cid1 = f"{slug}__{ex.y1}"
        cid2 = f"{slug}__{ex.y2}"

        lines.append(r"\clearpage")
        lines.append(rf"\subsection*{{Example: {_tex_escape(slug)} — {_tex_escape(metric)} swing}}")
        lines.append(
            r"\noindent "
            + _tex_escape(
                f"Swing years: {ex.y1} → {ex.y2}. Δ{lbl} = {ex.delta:+.3f}. "
                f"Top-8 answerability flips = {ex.top8_missingness_flips}."
            )
        )
        lines.append("")

        r1 = imputed_by_key.get((slug, ex.y1))
        r2 = imputed_by_key.get((slug, ex.y2))
        if r1 is None or r2 is None:
            lines.append(r"\noindent " + _tex_escape("Cannot attribute features: one endpoint year missing from PCA workbook."))
            lines.append("")
            return

        dz_s = ((r2[feature_cols] - r1[feature_cols]) / std).astype(float)
        if metric == "Overall_Index":
            contrib_s = dz_s.copy()
            loading_s = None
        else:
            m = re.match(r"^Principal_Component_(\d+)$", metric)
            pc_row = f"PC{m.group(1)}" if m else ""
            loading_s = loadings.loc[pc_row, feature_cols].astype(float) if pc_row in loadings.index else None
            contrib_s = (dz_s * loading_s).astype(float) if loading_s is not None else dz_s.copy()

        contrib_sorted = contrib_s.sort_values(key=lambda s: s.abs(), ascending=False)
        ranked_qids = [str(q) for q in contrib_sorted.index.tolist()]
        top_rows = ranked_qids[: min(len(ranked_qids), int(args.top_driver_rows))]
        top_expand = ranked_qids[: max(0, int(args.top_driver_blocks))]

        # Ranked driver table
        lines.append(r"\noindent Ranked contributions (drivers):")
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\tiny")
        if loading_s is None:
            lines.append(r"\begin{tabular}{rlll l l rr}")
            lines.append(r"\toprule")
            lines.append(r"\# & QID & Label & Type & Answers & Obs(y1/y2) & Dz & Contrib \\")
        else:
            lines.append(r"\begin{tabular}{rlll l l rrr}")
            lines.append(r"\toprule")
            lines.append(r"\# & QID & Label & Type & Answers & Obs(y1/y2) & Dz & Loading & Contrib \\")
        lines.append(r"\midrule")

        a1_all = normalized_by_cid.get(cid1, {})
        a2_all = normalized_by_cid.get(cid2, {})

        for rank, qid in enumerate(top_rows, start=1):
            qm = qmeta.get(qid)
            label = qm.short if (qm and qm.short) else qid
            qtype = qm.qtype if qm else "Unknown"

            a1 = a1_all.get(qid) if isinstance(a1_all.get(qid), dict) else {}
            a2 = a2_all.get(qid) if isinstance(a2_all.get(qid), dict) else {}
            ans1 = a1.get("answer") if isinstance(a1, dict) else None
            ans2 = a2.get("answer") if isinstance(a2, dict) else None

            obs = f"{observed_flag(qid, slug=slug, year=ex.y1)}/{observed_flag(qid, slug=slug, year=ex.y2)}"
            dz = float(dz_s.get(qid) or 0.0)
            contrib = float(contrib_s.get(qid) or 0.0)
            ans_s = f"{_fmt_answer(ans1)}→{_fmt_answer(ans2)}"

            if loading_s is None:
                lines.append(
                    rf"{rank} & {_tex_escape(qid)} & {_tex_escape(label)[:60]} & {_tex_escape(qtype)[:10]} & "
                    rf"{_tex_escape(ans_s)[:40]} & {_tex_escape(obs)[:16]} & {dz:+.3f} & {contrib:+.3f} \\"
                )
            else:
                loading = float(loading_s.get(qid) or 0.0)
                lines.append(
                    rf"{rank} & {_tex_escape(qid)} & {_tex_escape(label)[:60]} & {_tex_escape(qtype)[:10]} & "
                    rf"{_tex_escape(ans_s)[:40]} & {_tex_escape(obs)[:16]} & {dz:+.3f} & {loading:+.3f} & {contrib:+.3f} \\"
                )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        # Expanded drivers with embedded BLOCK text
        lines.append(r"\noindent Key drivers (expanded with evidence + exact OCR BLOCK text):")
        lines.append("")

        for qid in top_expand:
            qm = qmeta.get(qid)
            short = qm.short if qm else ""
            full = qm.full if qm else ""
            qtype = qm.qtype if qm else "Unknown"

            a1 = a1_all.get(qid) if isinstance(a1_all.get(qid), dict) else {}
            a2 = a2_all.get(qid) if isinstance(a2_all.get(qid), dict) else {}
            ans1 = a1.get("answer") if isinstance(a1, dict) else None
            ans2 = a2.get("answer") if isinstance(a2, dict) else None
            ev1 = a1.get("evidence") if isinstance(a1, dict) else None
            ev2 = a2.get("evidence") if isinstance(a2, dict) else None

            dz = float(dz_s.get(qid) or 0.0)
            contrib = float(contrib_s.get(qid) or 0.0)
            loading = None if loading_s is None else float(loading_s.get(qid) or 0.0)

            lines.append(rf"\subsubsection*{{{_tex_escape(qid)}{(' — ' + _tex_escape(short)) if short else ''}}}")
            if full and full != short:
                lines.append(r"\noindent " + _tex_escape(full))
                lines.append("")
            head = f"type={qtype}; Dz={dz:+.3f}"
            if loading is not None:
                head += f"; loading={loading:+.3f}"
            head += f"; contrib={contrib:+.3f}"
            lines.append(r"\noindent " + _tex_escape(head))
            lines.append(r"\noindent " + _tex_escape(f"answers: {ex.y1}={ans1!r}, {ex.y2}={ans2!r}."))
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
                if hit is None:
                    lines.append(r"\noindent " + _tex_escape("Could not locate evidence inside stored prompt text."))
                    lines.append("")
                    return

                where = f"BLOCK {hit.block_idx}"
                if hit.page_id:
                    where += f" (page_id={hit.page_id})"
                lines.append(r"\noindent " + _tex_escape(f"Located in exact prompt: {where}."))
                lines.append(r"\begin{quote}\small\ttfamily " + _tex_escape(hit.snippet) + r"\end{quote}")
                lines.append("")

                # Write and embed the full block text as an artifact.
                block_fname = f"{slug}__{year}__BLOCK_{hit.block_idx}.txt"
                block_path = out_dir / block_fname
                block_path.write_text(hit.full_block_text.rstrip("\n") + "\n", encoding="utf-8")
                try:
                    rel = block_path.relative_to(out_tex.parent)
                    rel_s = str(rel)
                except Exception:
                    rel_s = str(block_path)
                lines.append(r"\noindent " + _tex_escape(f"Exact OCR BLOCK text ({year}, BLOCK {hit.block_idx}):"))
                lines.append(rf"\VerbatimInput[breaklines=true,fontsize=\scriptsize,frame=single]{{\detokenize{{{rel_s}}}}}")
                lines.append("")

            render_ev(ex.y1, ev1)
            render_ev(ex.y2, ev2)

        # Embed full exact prompts for both years.
        lines.append(r"\subsubsection*{Exact prompts (as sent to the model)}")
        for cid in (cid1, cid2):
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

    # Write all examples
    for ex in examples:
        write_example(ex)

    lines.append(r"\end{document}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _eprint(f"out_tex\t{out_tex}")
    _eprint(f"out_dir\t{out_dir}")
    _eprint(f"examples\t{len(examples)}")
    for ex in examples:
        _eprint(f"example\t{ex.slug}\t{ex.metric}\t{ex.y1}->{ex.y2}\t{ex.delta:+.3f}\tflips={ex.top8_missingness_flips}")


if __name__ == "__main__":
    main()
