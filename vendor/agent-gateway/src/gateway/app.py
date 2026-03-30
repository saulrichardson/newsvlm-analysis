"""FastAPI application factory for the gateway."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from .api.routes import router
from .logging import bind_trace, configure_logging
from .services.gateway import GatewayService
from .settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    gateway = GatewayService(settings=settings)

    app = FastAPI(title="LLM Gateway", version="0.1.0")
    app.state.gateway = gateway

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - framework hook
        await gateway.shutdown()

    @app.middleware("http")
    async def inject_request_context(  # pragma: no cover
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
        request.state.request_id = request_id
        bind_trace(request_id=request_id)
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    app.include_router(router)
    return app
