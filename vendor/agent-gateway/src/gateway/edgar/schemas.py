from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SegmentSelector(BaseModel):
    segment_id: Optional[str] = Field(
        default=None,
        description="Canonical segment identifier (tarfile::member::segment).",
    )
    tarfile: Optional[str] = Field(default=None, description="Tar archive file name.")
    file: Optional[str] = Field(default=None, description="Filename inside the tar archive.")
    segment_no: Optional[int] = Field(default=None, ge=1, description="1-based segment number.")

    @model_validator(mode="after")
    def validate_selector(self) -> "SegmentSelector":  # noqa: D401
        # Ensure either segment_id or the tarfile/file/segment trio is present.
        if self.segment_id:
            return self
        if self.tarfile and self.file and self.segment_no:
            return self
        raise ValueError(
            "Provide either segment_id or tarfile + file + segment_no to locate a segment."
        )


class PromptConfig(BaseModel):
    user_template: Optional[str] = Field(
        default=None,
        description="Template that wraps the document; include '{{document}}' where the text should go.",
    )
    system_prompt: Optional[str] = Field(default=None, description="System prompt delivered to the model.")
    include_tables: bool = Field(
        default=True,
        description="Append Markdown-rendered tables after the normalized text in the prompt.",
    )
    include_metadata: bool = Field(
        default=False,
        description="Prepend key metadata (doc type, tarfile, etc.) ahead of the document text.",
    )


class ResponseOptions(BaseModel):
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="Reasoning effort hint for reasoning-capable models (OpenAI Responses API).",
    )
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    extra_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw key/value pairs forwarded to the OpenAI Responses API payload.",
    )


class JobRequest(BaseModel):
    segment: SegmentSelector
    model: str = Field(..., description="OpenAI model to invoke (e.g., gpt-5.5-mini).")
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    response: ResponseOptions = Field(default_factory=ResponseOptions)


class JobSubmitResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "succeeded", "failed"]


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    segment_id: Optional[str]
    model: Optional[str]
    response_text: Optional[str]
    response_payload: Optional[Dict[str, Any]]
    error: Optional[str]
    document_text: Optional[str]
    document_tables: Optional[List[str]]
    created_at: datetime
    updated_at: datetime
