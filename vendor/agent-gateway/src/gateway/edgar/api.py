from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .job_store import JobStore
from .openai_client import OpenAIResponsesClient
from .schemas import JobRequest, JobStatusResponse, JobSubmitResponse
from .service import PipelineGatewayService
from .settings import GatewaySettings


def _record_to_status(record) -> JobStatusResponse:
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    data = record.to_dict()
    return JobStatusResponse(**data)


def create_app(settings: GatewaySettings) -> FastAPI:
    job_store = JobStore()
    openai_client = OpenAIResponsesClient(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
        initial_backoff=settings.openai_initial_backoff,
    )
    service = PipelineGatewayService(
        tar_root=settings.tar_root,
        manifest_path=settings.manifest_path,
        openai_client=openai_client,
        job_store=job_store,
    )

    app = FastAPI(title="EDGAR Pipeline Gateway", version="0.1.0")

    @app.post("/jobs", response_model=JobSubmitResponse)
    async def submit_job(request: JobRequest) -> JobSubmitResponse:
        job = await service.submit_job(request)
        return JobSubmitResponse(job_id=job.job_id, status=job.status)

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(job_id: str) -> JobStatusResponse:
        record = job_store.get(job_id)
        return _record_to_status(record)

    @app.get("/jobs", response_model=list[JobStatusResponse])
    async def list_jobs(limit: int = 25) -> list[JobStatusResponse]:
        records = job_store.list_recent(limit=limit)
        return [JobStatusResponse(**record.to_dict()) for record in records]

    @app.get("/healthz")
    async def healthcheck() -> dict:
        return {
            "status": "ok",
            "manifest_rows": len(service.manifest_df),
            "tar_root": str(settings.tar_root),
        }

    return app
