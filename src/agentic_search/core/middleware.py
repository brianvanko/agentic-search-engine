"""FastAPI middleware for request tracking and logging.

This module provides:
- RequestIDMiddleware: Automatically assigns and tracks request IDs
- Request/response logging with timing information

Example:
    from fastapi import FastAPI
    from agentic_search.core.middleware import RequestIDMiddleware

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
"""

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from agentic_search.core.logging import (
    get_logger,
    set_request_id,
    clear_request_id,
    get_request_id,
)

logger = get_logger(__name__)

# Header name for request ID (standard conventions)
REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns and tracks request IDs.

    Features:
    - Extracts existing X-Request-ID from incoming requests
    - Generates new request ID if not provided
    - Adds request ID to response headers
    - Sets request ID in logging context for correlation
    - Logs request start/end with timing

    Example:
        app.add_middleware(RequestIDMiddleware)

        # All logs within request handling will include request_id
        # Response will have X-Request-ID header
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with request ID tracking."""
        # Extract or generate request ID
        request_id = request.headers.get(REQUEST_ID_HEADER)
        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        # Set request ID in context
        set_request_id(request_id)

        # Store in request state for access in handlers
        request.state.request_id = request_id

        # Log request start
        start_time = time.perf_counter()
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": self._get_client_ip(request),
            },
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log request completion
            logger.info(
                "Request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            # Add request ID to response headers
            response.headers[REQUEST_ID_HEADER] = request_id

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log error
            logger.error(
                "Request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration_ms": round(duration_ms, 2),
                },
                exc_info=True,
            )
            raise

        finally:
            # Clear request ID from context
            clear_request_id()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, handling proxies."""
        # Check X-Forwarded-For header (for proxied requests)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP in the chain (original client)
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client
        if request.client:
            return request.client.host

        return "unknown"


def get_request_id_from_request(request: Request) -> str:
    """Get request ID from a request object.

    Args:
        request: FastAPI/Starlette request object.

    Returns:
        Request ID string.
    """
    return getattr(request.state, "request_id", get_request_id() or "unknown")
