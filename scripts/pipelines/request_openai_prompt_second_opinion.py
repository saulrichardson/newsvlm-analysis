#!/usr/bin/env python3
"""
Request a grounded second opinion from OpenAI on the current ordinance
reconstruction prompt using embedded prompt text, testing evidence, example
outputs, and example PNGs in a single Responses API payload.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fullnewspaper_v3_common import extract_openai_output_text, parse_env_file


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type or 'image/png'};base64,{payload}"


def _load_openai_key(env_file: Path) -> str:
    env = parse_env_file(env_file)
    key = (env.get("OPENAI_KEY") or "").strip()
    if not key:
        raise SystemExit(f"Missing OPENAI_KEY in {env_file}")
    return key


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Send a grounded prompt-review packet to OpenAI Responses API.")
    ap.add_argument("--env-file", default=str(repo_root / ".env"))
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--api-url", default="https://api.openai.com/v1/responses")
    ap.add_argument("--output-dir", default=str(repo_root / "artifacts/scratch" / "prompt_second_opinion_20260306" / "run_001"))
    ap.add_argument("--prompt-path", default=str(repo_root / "prompts" / "transcription_v12_issue_multimage_postgate.txt"))
    ap.add_argument("--supporting-prompt-path", default=str(repo_root / "prompts" / "transcription_v12_issue_ocr_only_postgate.txt"))
    ap.add_argument("--issue-report-path", default=str(repo_root / "reports" / "issue_image_multimodal_pilot_20260306.md"))
    ap.add_argument("--postgate-report-path", default=str(repo_root / "reports" / "postgate_operative_transcription_models_20260306.md"))
    ap.add_argument("--oxnard-issue-transcript-path", default=str(repo_root / "artifacts/scratch" / "stage2_prompt_eval_full_inputs_20260305" / "issue_txt" / "oxnard-press-courier__1968-05-24.txt"))
    ap.add_argument("--oxnard-ocr-only-path", default=str(repo_root / "artifacts/scratch" / "issue_image_ocr_pilot_20260306" / "run_004_gemini31pro_oxnard" / "rendered_outputs" / "ocr_only" / "oxnard-press-courier__1968-05-24.txt"))
    ap.add_argument("--oxnard-ocr-images-path", default=str(repo_root / "artifacts/scratch" / "issue_image_ocr_pilot_20260306" / "run_004_gemini31pro_oxnard" / "rendered_outputs" / "ocr_plus_images" / "oxnard-press-courier__1968-05-24.txt"))
    ap.add_argument("--oxnard-image-a", default=str(repo_root / "artifacts/scratch" / "issue_image_ocr_pilot_20260306" / "run_004_gemini31pro_oxnard" / "images" / "oxnard-press-courier__1968-05-24" / "oxnard-press-courier-may-24-1968-p-43.png"))
    ap.add_argument("--oxnard-image-b", default=str(repo_root / "artifacts/scratch" / "issue_image_ocr_pilot_20260306" / "run_004_gemini31pro_oxnard" / "images" / "oxnard-press-courier__1968-05-24" / "oxnard-press-courier-may-24-1968-p-46.png"))
    ap.add_argument("--park-ocr-path", default=str(repo_root / "artifacts/scratch" / "page_image_ocr_pilot_20260305" / "run_009_gemini31pro_v11_positive" / "page_ocr" / "park-forest-star-dec-18-1969-p-152.txt"))
    ap.add_argument("--park-ocr-only-path", default=str(repo_root / "artifacts/scratch" / "page_image_ocr_pilot_20260305" / "run_009_gemini31pro_v11_positive" / "rendered_outputs" / "ocr_only" / "park-forest-star-dec-18-1969-p-152.txt"))
    ap.add_argument("--park-ocr-image-path", default=str(repo_root / "artifacts/scratch" / "page_image_ocr_pilot_20260305" / "run_009_gemini31pro_v11_positive" / "rendered_outputs" / "ocr_plus_image" / "park-forest-star-dec-18-1969-p-152.txt"))
    ap.add_argument("--park-image-path", default=str(repo_root / "artifacts/scratch" / "page_image_ocr_pilot_20260305" / "run_009_gemini31pro_v11_positive" / "images" / "park-forest-star-dec-18-1969-p-152.png"))
    return ap.parse_args()


def _supporting_summary(issue_report: str, postgate_report: str) -> str:
    return (
        "Grounded testing summary from local artifacts follows.\n\n"
        "1. The current candidate under review is an issue-level prompt that assumes a post-gate setting and instructs the model to use the full issue OCR transcript plus all issue page images in order.\n"
        "2. A completed issue-level multimodal Gemini run on oxnard-press-courier__1968-05-24 showed that OCR+all-issue-images was materially cleaner than OCR-only on the same two-page issue.\n"
        "3. In that completed Oxnard run, the image-backed version recovered ORDINANCE NO. 2052, County of Ventura, California, ARTICLE 24, a complete Sec. 8142-1 Purpose clause, missing subsection headers, and removed all [ILLEGIBLE] and [MISSING TEXT] markers that remained in OCR-only.\n"
        "4. Earlier post-gate page-level testing showed that multimodal repair can correct genuine OCR damage, including the Park Forest numeric correction M-10 -> M-180 when the image is available.\n"
        "5. The main architectural conclusion from the tests so far is: strict gating first, then a more permissive multimodal reconstruction prompt for issues already believed to contain operative zoning ordinance text.\n\n"
        "Supporting report excerpt A:\n"
        f"{issue_report[:12000]}\n\n"
        "Supporting report excerpt B:\n"
        f"{postgate_report[:12000]}"
    )


def _build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_text = _read_text(Path(args.prompt_path))
    supporting_prompt_text = _read_text(Path(args.supporting_prompt_path))
    issue_report = _read_text(Path(args.issue_report_path))
    postgate_report = _read_text(Path(args.postgate_report_path))
    oxnard_issue_transcript = _read_text(Path(args.oxnard_issue_transcript_path))
    oxnard_ocr_only = _read_text(Path(args.oxnard_ocr_only_path))
    oxnard_ocr_images = _read_text(Path(args.oxnard_ocr_images_path))
    park_ocr = _read_text(Path(args.park_ocr_path))
    park_ocr_only = _read_text(Path(args.park_ocr_only_path))
    park_ocr_image = _read_text(Path(args.park_ocr_image_path))

    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "You are giving a second-opinion review of a prompt for post-gate operative zoning ordinance reconstruction.\n\n"
                "Review the candidate prompt using the embedded prompt text, grounded testing summaries, example OCR transcripts, example reconstructed ordinance outputs, and example PNGs.\n\n"
                "Your task:\n"
                "1. Judge whether the current issue-level prompt is the right direction.\n"
                "2. Identify the most important prompt-level improvements, grounded in the supplied evidence.\n"
                "3. Say whether the prompt is too permissive, too restrictive, or roughly right for a post-gate reconstruction stage.\n"
                "4. Produce one improved revised prompt if you think changes are needed.\n\n"
                "Important constraints:\n"
                "- Ground every material claim in the embedded evidence. Do not give generic prompt advice that is not tied to the examples.\n"
                "- Assume over-reconstruction is less concerning than under-reconstruction at this post-gate stage, but do not recommend fabricating large unseen passages.\n"
                "- Pay attention to whether issue-level all-image input is materially better than OCR-only for multi-page ordinance reconstruction.\n"
                "- Return markdown with exactly these sections:\n"
                "## Overall Assessment\n"
                "## Evidence-Grounded Observations\n"
                "## Recommended Prompt Changes\n"
                "## Revised Prompt\n"
                "## Confidence And Limits"
            ),
        },
        {
            "type": "input_text",
            "text": (
                "Candidate issue-level multimodal prompt under review:\n\n"
                f"{prompt_text}"
            ),
        },
        {
            "type": "input_text",
            "text": (
                "Companion OCR-only issue-level prompt used for comparison:\n\n"
                f"{supporting_prompt_text}"
            ),
        },
        {
            "type": "input_text",
            "text": _supporting_summary(issue_report=issue_report, postgate_report=postgate_report),
        },
        {
            "type": "input_text",
            "text": (
                "Example A: full issue OCR transcript for oxnard-press-courier__1968-05-24.\n\n"
                f"{oxnard_issue_transcript}"
            ),
        },
        {
            "type": "input_text",
            "text": (
                "Example A OCR-only reconstructed ordinance output for the same full issue.\n\n"
                f"{oxnard_ocr_only}"
            ),
        },
        {
            "type": "input_text",
            "text": (
                "Example A OCR+all-issue-images reconstructed ordinance output for the same full issue.\n\n"
                f"{oxnard_ocr_images}"
            ),
        },
        {
            "type": "input_text",
            "text": "Example A image 1 of 2. This is the first issue page image, corresponding to oxnard-press-courier-may-24-1968-p-43.",
        },
        {"type": "input_image", "image_url": _image_data_url(Path(args.oxnard_image_a))},
        {
            "type": "input_text",
            "text": "Example A image 2 of 2. This is the second issue page image, corresponding to oxnard-press-courier-may-24-1968-p-46.",
        },
        {"type": "input_image", "image_url": _image_data_url(Path(args.oxnard_image_b))},
        {
            "type": "input_text",
            "text": (
                "Example B: single-page OCR input for Park Forest numeric-repair case.\n\n"
                f"{park_ocr}"
            ),
        },
        {
            "type": "input_text",
            "text": (
                "Example B OCR-only reconstructed ordinance output.\n\n"
                f"{park_ocr_only}"
            ),
        },
        {
            "type": "input_text",
            "text": (
                "Example B OCR+image reconstructed ordinance output.\n\n"
                f"{park_ocr_image}"
            ),
        },
        {
            "type": "input_text",
            "text": "Example B image. This page was used to test whether the model can repair numeric OCR damage from the image.",
        },
        {"type": "input_image", "image_url": _image_data_url(Path(args.park_image_path))},
    ]

    payload: dict[str, Any] = {
        "model": str(args.model),
        "reasoning": {"effort": "medium"},
        "input": [{"role": "user", "content": content}],
        "stream": False,
    }
    preview = {
        "model": payload["model"],
        "reasoning": payload["reasoning"],
        "content_parts": [
            {"type": part["type"], **({"text_len": len(part["text"])} if part["type"] == "input_text" else {"image_url_len": len(part["image_url"])})}
            for part in content
        ],
    }
    return payload, preview


def main() -> None:
    args = _parse_args()
    env_file = Path(args.env_file).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    key = _load_openai_key(env_file)
    payload, preview = _build_payload(args)
    (out_dir / "request_preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")

    req = urllib.request.Request(
        url=str(args.api_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        (out_dir / "response_error.txt").write_text(body, encoding="utf-8")
        raise SystemExit(f"OpenAI API error {exc.code}: {body[:1000]}")

    (out_dir / "response.json").write_text(raw, encoding="utf-8")
    body = json.loads(raw)
    text = extract_openai_output_text(body)
    (out_dir / "response.txt").write_text(text, encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "response_chars": len(text)}, indent=2))


if __name__ == "__main__":
    main()
