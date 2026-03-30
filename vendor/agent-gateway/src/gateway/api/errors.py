"""Utilities for translating gateway errors into HTTP responses."""

from __future__ import annotations

from fastapi import HTTPException, status

from ..providers import ProviderError, ProviderNotConfiguredError


def map_exception(exc: Exception, provider: str | None = None) -> HTTPException:
    if isinstance(exc, ProviderNotConfiguredError):
        return HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail={
                "error": {
                    "message": str(exc),
                    "code": "provider_not_configured",
                    "provider": provider,
                }
            },
        )

    if isinstance(exc, ProviderError):
        code = "provider_error"
        http_status = status.HTTP_502_BAD_GATEWAY
        if exc.status_code in (401, 403):
            code = "upstream_auth_error"
        elif exc.status_code == 429:
            code = "upstream_rate_limited"
            http_status = status.HTTP_429_TOO_MANY_REQUESTS
        elif exc.status_code and exc.status_code >= 500:
            code = "upstream_unavailable"

        return HTTPException(
            status_code=http_status,
            detail={
                "error": {
                    "message": str(exc),
                    "code": code,
                    "provider": provider,
                    "upstream_status": exc.status_code,
                    "provider_request_id": exc.provider_request_id,
                }
            },
        )

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"error": {"message": str(exc), "code": "internal_error", "provider": provider}},
    )
