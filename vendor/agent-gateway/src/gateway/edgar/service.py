from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd

from edgar_filing_pipeline.processing import normalize_html
from edgar_filing_pipeline.segment import SegmentExtractor, TarSegmentReader
from edgar_filing_pipeline.workflow import read_manifest

from .job_store import JobRecord, JobStore
from .openai_client import OpenAIResponsesClient
from .schemas import JobRequest

DEFAULT_USER_TEMPLATE = "<<BEGIN DOC>>\n{{document}}\n<<END DOC>>"


class PipelineGatewayService:
    def __init__(
        self,
        *,
        tar_root: Path,
        manifest_path: Path,
        openai_client: OpenAIResponsesClient,
        job_store: JobStore,
    ) -> None:
        self.tar_root = Path(tar_root)
        self.openai_client = openai_client
        self.job_store = job_store
        self.manifest_path = Path(manifest_path)
        self.manifest_df = self._load_manifest(self.manifest_path)

    @staticmethod
    def _load_manifest(path: Path) -> pd.DataFrame:
        df = read_manifest(path)
        df = df.copy()
        if "segment_id" not in df.columns:
            from edgar_filing_pipeline.identifiers import SegmentKey

            df["segment_id"] = df.apply(
                lambda row: SegmentKey(
                    str(row["tarfile"]),
                    str(row["file"] if "file" in row else row["file_x"]),
                    int(row["segment_no"] if "segment_no" in row else row["segment_no_x"]),
                ).id,
                axis=1,
            )
        return df

    async def submit_job(self, request: JobRequest) -> JobRecord:
        job_id = str(uuid4())
        job = JobRecord(job_id=job_id, request=request, status="pending", model=request.model)
        self.job_store.create(job)
        asyncio.create_task(self._run_job(job_id))
        return job

    async def _run_job(self, job_id: str) -> None:
        await asyncio.to_thread(self._process_job, job_id)

    def _process_job(self, job_id: str) -> None:
        record = self.job_store.update(job_id, status="running")
        if record is None:
            return
        try:
            manifest_row = self._resolve_manifest_row(record.request)
            segment_payload = self._load_segment(manifest_row)
            document_text, tables_for_prompt = self._build_document_text(
                record.request, manifest_row, segment_payload
            )
            payload = self._build_openai_payload(
                record.request,
                document_text,
            )
            response_data = self.openai_client.create_response(payload)
            response_text = self.openai_client.extract_text(response_data)
            self.job_store.update(
                job_id,
                status="succeeded",
                response_text=response_text,
                response_payload=response_data,
                document_text=document_text,
                document_tables=tables_for_prompt,
                segment_id=manifest_row.get("segment_id"),
            )
        except Exception as exc:  # noqa: BLE001
            self.job_store.update(job_id, status="failed", error=str(exc))

    def _resolve_manifest_row(self, request: JobRequest) -> Dict[str, Any]:
        selector = request.segment
        df = self.manifest_df
        if selector.segment_id:
            matches = df[df["segment_id"] == selector.segment_id]
        else:
            file_col = "file" if "file" in df.columns else "file_x"
            seg_col = "segment_no" if "segment_no" in df.columns else "segment_no_x"
            matches = df[
                (df["tarfile"] == selector.tarfile)
                & (df[file_col] == selector.file)
                & (df[seg_col] == selector.segment_no)
            ]
        if matches.empty:
            raise ValueError("Segment not found in manifest")
        row = matches.iloc[0].to_dict()
        if "segment_id" not in row:
            row["segment_id"] = selector.segment_id
        return row

    def _load_segment(self, manifest_row: Dict[str, Any]) -> Dict[str, Any]:
        tarfile = manifest_row["tarfile"]
        file_name = manifest_row.get("file") or manifest_row.get("file_x")
        segment_no = manifest_row.get("segment_no") or manifest_row.get("segment_no_x")
        if file_name is None or segment_no is None:
            raise ValueError("Manifest row missing file or segment number")
        tar_path = self.tar_root / tarfile
        reader = TarSegmentReader(tar_path)
        try:
            extractor = SegmentExtractor(reader.read_member(file_name))
            index = int(segment_no) - 1
            html = extractor.get_segment_html(index)
        finally:
            reader.close()
        normalized = normalize_html(html)
        tables = [
            markdown
            for markdown in normalized.tables.markdown
            if markdown and markdown.strip()
        ]
        return {
            "html": html,
            "normalized": normalized,
            "tables_markdown": tables,
        }

    def _build_document_text(
        self,
        request: JobRequest,
        manifest_row: Dict[str, Any],
        segment_payload: Dict[str, Any],
    ) -> tuple[str, list[str]]:
        normalized = segment_payload["normalized"]
        tables = segment_payload["tables_markdown"]
        prompt_cfg = request.prompt
        metadata_section: list[str] = []
        if prompt_cfg.include_metadata:
            meta_lines = []
            doc_type = manifest_row.get("doc_type")
            if doc_type:
                meta_lines.append(f"Document type: {doc_type}")
            meta_lines.append(f"Tarfile: {manifest_row.get('tarfile')}")
            meta_lines.append(f"File: {manifest_row.get('file') or manifest_row.get('file_x')}")
            meta_lines.append(f"Segment #: {manifest_row.get('segment_no') or manifest_row.get('segment_no_x')}")
            header_json = manifest_row.get("header_json")
            if header_json:
                try:
                    header = json.loads(header_json)
                    filer = header.get("company-name") or header.get("conformed-name")
                    if filer:
                        meta_lines.append(f"Company: {filer}")
                except json.JSONDecodeError:
                    pass
            metadata_section.append("\n".join(meta_lines))
        document_parts = []
        if metadata_section:
            document_parts.extend(metadata_section)
        document_parts.append(normalized.text)
        tables_used: list[str] = []
        if prompt_cfg.include_tables and tables:
            for idx, markdown in enumerate(tables, start=1):
                marker = f"Table {idx}:"
                tables_used.append(markdown)
                document_parts.append(f"{marker}\n{markdown}")
        document_text = "\n\n".join(part.strip() for part in document_parts if part and part.strip())
        return document_text, tables_used

    def _build_openai_payload(self, request: JobRequest, document_text: str) -> Dict[str, Any]:
        prompt_cfg = request.prompt
        template = prompt_cfg.user_template or DEFAULT_USER_TEMPLATE
        if "{{document}}" in template:
            user_text = template.replace("{{document}}", document_text)
        else:
            user_text = f"{template.strip()}\n\n{document_text}"
        messages = []
        if prompt_cfg.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prompt_cfg.system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            }
        )
        payload: Dict[str, Any] = {
            "model": request.model,
            "input": messages,
        }
        response_opts = request.response
        if response_opts.reasoning_effort:
            payload["reasoning"] = {"effort": response_opts.reasoning_effort}
        if response_opts.max_output_tokens is not None:
            payload["max_output_tokens"] = response_opts.max_output_tokens
        if response_opts.temperature is not None:
            payload["temperature"] = response_opts.temperature
        if response_opts.top_p is not None:
            payload["top_p"] = response_opts.top_p
        if response_opts.extra_parameters:
            payload.update(response_opts.extra_parameters)
        return payload
