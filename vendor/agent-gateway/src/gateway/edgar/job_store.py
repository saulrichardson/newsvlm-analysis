from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

from .schemas import JobRequest

JobStatusLiteral = Literal["pending", "running", "succeeded", "failed"]


@dataclass
class JobRecord:
    job_id: str
    request: JobRequest
    status: JobStatusLiteral = "pending"
    segment_id: Optional[str] = None
    model: Optional[str] = None
    response_text: Optional[str] = None
    response_payload: Optional[Dict[str, object]] = None
    error: Optional[str] = None
    document_text: Optional[str] = None
    document_tables: Optional[list[str]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "segment_id": self.segment_id,
            "model": self.model,
            "response_text": self.response_text,
            "response_payload": self.response_payload,
            "error": self.error,
            "document_text": self.document_text,
            "document_tables": self.document_tables,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobStore:
    def __init__(self) -> None:
        self._records: Dict[str, JobRecord] = {}
        self._lock = threading.RLock()

    def create(self, job: JobRecord) -> JobRecord:
        with self._lock:
            self._records[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._records.get(job_id)

    def update(self, job_id: str, **changes) -> Optional[JobRecord]:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return None
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = datetime.now(timezone.utc)
            return record

    def list_recent(self, limit: int = 50) -> list[JobRecord]:
        with self._lock:
            records = sorted(
                self._records.values(), key=lambda rec: rec.created_at, reverse=True
            )
            return records[:limit]
