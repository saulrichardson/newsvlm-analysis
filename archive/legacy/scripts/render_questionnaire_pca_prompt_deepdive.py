#!/usr/bin/env python3
"""
Render a prompt-level deep dive for the slug-year grouped questionnaire workflow.

Goal
----
Given:
  - A questionnaire PCA workbook (scores + imputed + loadings)
  - A slug-year grouped questionnaire request dir (mapping + openai_requests JSONLs)
  - A questionnaire results dir with normalized outputs (normalized.jsonl)

This script:
  1) picks a fixed set of slugs (default: all slugs in the PCA workbook)
  2) for each slug, finds the largest year-to-year swing in a metric (default: Overall_Index)
  3) attributes that swing to the most-changing oriented standardized features
  4) embeds the EXACT prompt text that was sent to the LLM for each involved slug-year
  5) annotates which OCR passages (BLOCKs) support the changed answers.

This is intentionally artifact-first:
  - Prompt text is extracted from openai_requests_shard*.jsonl (not reconstructed).
  - Question text comes from Questions.xlsx (if available in the request mapping).
  - Evidence and answers come from normalized.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from os.path import expanduser
from pathlib import Path
from typing import Any


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
    # Prompt contains blocks like:
    #   OCR_PAGE_BLOCKS_ORDERED_OLDEST_TO_NEWEST:
    #   BLOCK 1:
    #   ...
    #   BLOCK 2:
    #   ...
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
    if len(ev_l) > 160:
        keys.append(ev_l[:160])
        keys.append(ev_l_nocomma[:160])
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
            # Context from a whitespace-normalized (and de-hyphenated) version of the exact
            # model-visible block text. This keeps the quote readable while ensuring the
            # returned snippet actually contains the matched evidence anchor.
            bt_snip_src = bt.replace(",", "") if used_nocomma else bt
            start = max(0, pos - 220)
            end = min(len(bt_snip_src), pos + min(len(k), 120) + 320)
            page_id = prompt.block_page_ids.get(bidx)
            return _EvidenceHit(block_idx=bidx, page_id=page_id, snippet=bt_snip_src[start:end])

    # Fallback: token match (long tokens must all appear in a block)
    toks = [w for w in re.split(r"[^a-zA-Z0-9]+", ev_l) if len(w) >= 7][:6]
    if toks:
        for bidx, btxt in _parse_blocks_from_prompt(prompt.prompt_text):
            bt = _normalize_for_search(btxt).lower()
            if all(tok in bt for tok in toks):
                page_id = prompt.block_page_ids.get(bidx)
                return _EvidenceHit(block_idx=bidx, page_id=page_id, snippet=_norm_ws(btxt)[:600])

    return None


def _load_question_texts(*, questions_xlsx: Path, processed_sheet: str) -> tuple[dict[str, str], dict[str, str]]:
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

    short: dict[str, str] = {}
    full: dict[str, str] = {}
    for _, r in df.iterrows():
        qid = str(r.get("ID") or "").strip()
        if not qid:
            continue
        s = str(r.get("Short Question") or "").strip()
        f = str(r.get("Full Question") or "").strip()
        if s:
            short[qid] = s
        if f:
            full[qid] = f
    return short, full


def main() -> None:
    ap = argparse.ArgumentParser(description="Render deep dive: exact questionnaire LLM prompts + PCA swing attribution.")
    ap.add_argument("--pca-xlsx", required=True, help="PCA workbook produced by scripts/compute_questionnaire_pca.py")
    ap.add_argument("--questionnaire-request-dir", required=True, help="Questionnaire request dir (mapping_shard*.jsonl + openai_requests_shard*.jsonl)")
    ap.add_argument("--questionnaire-results-dir", required=True, help="Questionnaire results dir (normalized.jsonl)")
    ap.add_argument("--metric", default="Overall_Index", help="Metric column in PCA scores sheet (default: Overall_Index)")
    ap.add_argument("--slugs", default="", help="Optional comma-separated slug allowlist (default: all in PCA workbook)")
    ap.add_argument("--top-features", type=int, default=6, help="Number of driving features to show per slug (default: 6)")
    ap.add_argument("--out-tex", required=True, help="Output .tex path")
    ap.add_argument(
        "--out-prompt-dir",
        required=True,
        help="Directory to write exact prompt text files (referenced by LaTeX via lstinputlisting)",
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

    # Load PCA workbook (scores + imputed + loadings).
    scores = pd.read_excel(pca_xlsx, sheet_name="scores")
    imputed = pd.read_excel(pca_xlsx, sheet_name="imputed")
    loadings = pd.read_excel(pca_xlsx, sheet_name="loadings", index_col=0)

    metric = str(args.metric).strip()
    if metric not in scores.columns:
        raise SystemExit(f"--metric {metric!r} not found in PCA scores columns: {list(scores.columns)}")

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

    # Determine consistent questions workbook path for label lookup (optional but helpful).
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
    q_short: dict[str, str] = {}
    q_full: dict[str, str] = {}
    if questions_xlsx is not None and processed_sheet is not None:
        q_short, q_full = _load_question_texts(questions_xlsx=questions_xlsx, processed_sheet=processed_sheet)

    # Choose slugs to render.
    slug_allow: set[str] = set()
    if str(args.slugs).strip():
        slug_allow = {s.strip() for s in str(args.slugs).split(",") if s.strip()}

    if "slug" not in scores.columns or "page_year" not in scores.columns:
        raise SystemExit("PCA scores sheet must include columns: slug, page_year")

    slugs_all = sorted({str(s) for s in scores["slug"].dropna().astype(str).tolist()})
    slugs = [s for s in slugs_all if (not slug_allow or s in slug_allow)]
    if not slugs:
        raise SystemExit("No slugs selected (check --slugs filter vs PCA workbook).")

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

    # Compute largest year-to-year swing per slug.
    @dataclass(frozen=True)
    class Swing:
        slug: str
        y1: int
        y2: int
        delta: float

    swings: list[Swing] = []
    for slug in slugs:
        g = scores[scores["slug"].astype(str) == slug].dropna(subset=[metric, "page_year"]).copy()
        if g.empty:
            continue
        g["page_year"] = g["page_year"].astype(int)
        g = g.sort_values("page_year")
        years = g["page_year"].tolist()
        vals = g[metric].astype(float).tolist()
        if len(vals) < 2:
            continue
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        idx = max(range(len(diffs)), key=lambda i: abs(diffs[i]))
        swings.append(Swing(slug=slug, y1=int(years[idx]), y2=int(years[idx + 1]), delta=float(diffs[idx])))

    if not swings:
        raise SystemExit("No slugs had >=2 years with non-null metric; nothing to write.")

    # Write exact prompt text files needed for appendix.
    prompt_files: dict[str, Path] = {}
    for sw in swings:
        for year in (sw.y1, sw.y2):
            cid = f"{sw.slug}__{year}"
            if cid in prompt_files:
                continue
            prompt_text = prompt_by_cid.get(cid)
            if prompt_text is None:
                raise SystemExit(f"Missing OpenAI request prompt for custom_id={cid} (not in openai_requests_shard*.jsonl)")
            out_path = out_prompt_dir / f"{cid}.txt"
            out_path.write_text(prompt_text.rstrip("\n") + "\n", encoding="utf-8")
            prompt_files[cid] = out_path

    # Build prompt bundles for evidence localization.
    prompt_bundles: dict[str, _PromptBundle] = {}
    for cid, ptext in prompt_by_cid.items():
        if cid not in mapping_rows:
            continue
        prompt_bundles[cid] = _PromptBundle(custom_id=cid, prompt_text=ptext, block_page_ids=block_page_ids_for(cid))

    # Render LaTeX.
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
    lines.append(r"\fi")
    lines.append(r"\usepackage{fvextra}")
    lines.append(r"\begin{document}")
    lines.append(r"\section*{Prompt-level deep dive: PCA swings grounded in model-visible OCR text}")
    lines.append(
        r"\noindent "
        + _tex_escape(
            "This document is a qualitative deep dive for the slug-year grouped questionnaire workflow. "
            "For each plotted slug, it identifies the largest year-to-year swing in the requested metric "
            f"({metric}) and then attributes that swing to the questionnaire features that changed most. "
            "For each driving feature, it quotes the exact OCR passage (from the exact prompt text sent to the model) "
            "that supports the model's answer/evidence."
        )
    )
    lines.append("")

    if questions_xlsx is not None:
        lines.append(r"\noindent " + _tex_escape(f"Questions workbook: {questions_xlsx} (sheet={processed_sheet})"))
        lines.append("")

    lines.append(r"\noindent Cohort slugs included in this writeup:")
    lines.append(r"\begin{itemize}")
    for s in sorted({sw.slug for sw in swings}):
        lines.append(rf"\item {_tex_escape(s)}")
    lines.append(r"\end{itemize}")
    lines.append("")

    for sw in swings:
        cid1 = f"{sw.slug}__{sw.y1}"
        cid2 = f"{sw.slug}__{sw.y2}"

        # Pull imputed feature rows for the two years.
        r1 = imputed[(imputed["slug"].astype(str) == sw.slug) & (imputed["page_year"].astype(int) == sw.y1)]
        r2 = imputed[(imputed["slug"].astype(str) == sw.slug) & (imputed["page_year"].astype(int) == sw.y2)]
        if r1.empty or r2.empty:
            raise SystemExit(f"Missing imputed feature row for {cid1} or {cid2} in PCA workbook.")

        row1 = r1.iloc[0]
        row2 = r2.iloc[0]

        contrib = ((row2[feature_cols] - row1[feature_cols]) / std).astype(float)
        contrib = contrib.sort_values(key=lambda s: s.abs(), ascending=False)

        # Prefer features that have at least one non-null model answer in either year.
        a1_all = normalized_by_cid.get(cid1, {})
        a2_all = normalized_by_cid.get(cid2, {})

        driving: list[str] = []
        for qid in contrib.index.tolist():
            if len(driving) >= int(args.top_features):
                break
            a1 = a1_all.get(qid) if isinstance(a1_all.get(qid), dict) else {}
            a2 = a2_all.get(qid) if isinstance(a2_all.get(qid), dict) else {}
            ans1 = a1.get("answer") if isinstance(a1, dict) else None
            ans2 = a2.get("answer") if isinstance(a2, dict) else None
            if ans1 is None and ans2 is None:
                continue
            driving.append(str(qid))
        if not driving:
            # Fall back to top contributors even if answers are null (still reflects imputation effects).
            driving = [str(q) for q in contrib.index.tolist()[: int(args.top_features)]]

        lines.append(r"\clearpage")
        lines.append(rf"\section*{{{_tex_escape(sw.slug)}}}")
        lines.append(
            r"\noindent "
            + _tex_escape(
                f"Largest year-to-year swing in {metric}: {sw.y1} -> {sw.y2} (Delta = {sw.delta:+.3f})."
            )
        )
        lines.append("")

        # Summary table of driving features.
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{llrr}")
        lines.append(r"\toprule")
        lines.append(r"Question ID & Label & Dz & Answer change \\")
        lines.append(r"\midrule")
        for qid in driving:
            short = q_short.get(qid) or ""
            a1 = a1_all.get(qid) if isinstance(a1_all.get(qid), dict) else {}
            a2 = a2_all.get(qid) if isinstance(a2_all.get(qid), dict) else {}
            ans1 = a1.get("answer") if isinstance(a1, dict) else None
            ans2 = a2.get("answer") if isinstance(a2, dict) else None
            dz = float(contrib.get(qid) or 0.0)
            ans1_s = _tex_escape(repr(ans1))
            ans2_s = _tex_escape(repr(ans2))
            lines.append(
                rf"{_tex_escape(qid)} & {_tex_escape(short) if short else ''} & {dz:+.3f} & {ans1_s} -> {ans2_s} \\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        # Detailed per-feature notes with direct OCR quotes.
        for qid in driving:
            short = q_short.get(qid) or ""
            full = q_full.get(qid) or ""
            dz = float(contrib.get(qid) or 0.0)

            a1 = a1_all.get(qid) if isinstance(a1_all.get(qid), dict) else {}
            a2 = a2_all.get(qid) if isinstance(a2_all.get(qid), dict) else {}
            ans1 = a1.get("answer") if isinstance(a1, dict) else None
            ans2 = a2.get("answer") if isinstance(a2, dict) else None
            ev1 = a1.get("evidence") if isinstance(a1, dict) else None
            ev2 = a2.get("evidence") if isinstance(a2, dict) else None

            # Prefer the evidence from the later year if present.
            ev_year = sw.y2 if isinstance(ev2, str) and ev2.strip() else (sw.y1 if isinstance(ev1, str) and ev1.strip() else None)
            ev = ev2 if isinstance(ev2, str) and ev2.strip() else (ev1 if isinstance(ev1, str) and ev1.strip() else None)

            lines.append(rf"\subsection*{{{_tex_escape(qid)}{(' — ' + _tex_escape(short)) if short else ''}}}")
            if full and full != short:
                lines.append(r"\noindent " + _tex_escape(full))
                lines.append("")
            lines.append(r"\noindent " + _tex_escape(f"Dz = {dz:+.3f}; model answers: {sw.y1}={ans1!r}, {sw.y2}={ans2!r}."))
            lines.append("")

            if isinstance(ev, str) and ev.strip() and ev_year is not None:
                ev_clean = _norm_ws(ev)
                lines.append(r"\noindent " + _tex_escape(f"Model evidence quote ({ev_year}):") + r" " + _tex_escape(f"\"{ev_clean}\""))
                lines.append("")

                pb = prompt_bundles.get(f"{sw.slug}__{ev_year}")
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
            else:
                lines.append(r"\noindent Evidence was null in both years for this question.")
                lines.append("")

        # Appendix links for exact prompt texts.
        lines.append(r"\subsection*{Exact prompt texts (as sent to the model)}")
        rel1 = prompt_files[cid1].relative_to(out_tex.parent)
        rel2 = prompt_files[cid2].relative_to(out_tex.parent)
        lines.append(r"\noindent " + _tex_escape(f"{cid1}:"))
        lines.append(rf"\VerbatimInput[breaklines=true,fontsize=\footnotesize,frame=single]{{\detokenize{{{str(rel1)}}}}}")
        lines.append(r"\noindent " + _tex_escape(f"{cid2}:"))
        lines.append(rf"\VerbatimInput[breaklines=true,fontsize=\footnotesize,frame=single]{{\detokenize{{{str(rel2)}}}}}")
        lines.append("")

    lines.append(r"\end{document}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _eprint(f"out_tex\t{out_tex}")
    _eprint(f"out_prompt_dir\t{out_prompt_dir}")
    _eprint(f"metric\t{metric}")
    _eprint(f"slugs_written\t{len(swings)}")


if __name__ == "__main__":
    main()
