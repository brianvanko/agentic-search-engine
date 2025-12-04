"""Structured logging with JSON format and request ID support.

This module provides:
- JSON-formatted logs for production environments
- Request ID tracking via context variables
- Configurable log levels and formats
- Thread-safe context management

Example:
    # Configure logging on startup
    from agentic_search.core.logging import configure_logging, get_logger, set_request_id

    configure_logging(json_format=True, level="INFO")
    logger = get_logger(__name__)

    # In a request handler
    set_request_id("req-123")
    logger.info("Processing request", extra={"user_id": 42})
    # Output: {"timestamp": "...", "level": "INFO", "request_id": "req-123", ...}
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Context variable for request ID (thread-safe)
_request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context.

    Returns:
        The current request ID or None if not set.
    """
    return _request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set the request ID in context.

    Args:
        request_id: Optional request ID. If None, generates a new UUID.

    Returns:
        The request ID that was set.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    _request_id_var.set(request_id)
    return request_id


def clear_request_id() -> None:
    """Clear the request ID from context."""
    _request_id_var.set(None)


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured output.

    Produces logs in the format:
    {
        "timestamp": "2024-01-15T10:30:00.123456Z",
        "level": "INFO",
        "logger": "agentic_search.api",
        "message": "Request received",
        "request_id": "abc12345",
        "extra": {...}
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log structure
        log_dict: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            log_dict["request_id"] = request_id

        # Add source location for DEBUG level
        if record.levelno <= logging.DEBUG:
            log_dict["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        # Add extra fields (excluding standard LogRecord attributes)
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "taskName", "message",
        }

        extra = {}
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                try:
                    # Ensure value is JSON serializable
                    json.dumps(value)
                    extra[key] = value
                except (TypeError, ValueError):
                    extra[key] = str(value)

        if extra:
            log_dict["extra"] = extra

        return json.dumps(log_dict, default=str)


class StandardFormatter(logging.Formatter):
    """Standard text formatter with request ID support.

    Format: [LEVEL] [request_id] logger - message
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text with request ID."""
        request_id = get_request_id()
        request_id_str = f"[{request_id}] " if request_id else ""

        # Create base format
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base = f"{timestamp} [{record.levelname:8}] {request_id_str}{record.name} - {record.getMessage()}"

        # Add exception if present
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    include_uvicorn: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, use JSON format. Otherwise, use standard text format.
        include_uvicorn: If True, also configure uvicorn loggers.

    Example:
        # Development (readable logs)
        configure_logging(level="DEBUG", json_format=False)

        # Production (JSON logs for log aggregation)
        configure_logging(level="INFO", json_format=True)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Set formatter based on format preference
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(StandardFormatter())

    # Configure root logger for agentic_search
    root_logger = logging.getLogger("agentic_search")
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    root_logger.propagate = False

    # Configure uvicorn loggers if requested
    if include_uvicorn:
        for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
            uvicorn_logger = logging.getLogger(logger_name)
            uvicorn_logger.handlers.clear()
            uvicorn_logger.addHandler(handler)
            uvicorn_logger.setLevel(log_level)
            uvicorn_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started", extra={"item_count": 10})
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for request ID scope.

    Example:
        with LogContext(request_id="req-123"):
            logger.info("Inside request context")
        # request_id is cleared after the block
    """

    def __init__(self, request_id: Optional[str] = None):
        """Initialize log context.

        Args:
            request_id: Optional request ID. If None, generates a new one.
        """
        self._request_id = request_id
        self._token = None

    def __enter__(self) -> str:
        """Enter context and set request ID."""
        request_id = set_request_id(self._request_id)
        return request_id

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and clear request ID."""
        clear_request_id()


# Convenience functions for structured logging
def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **kwargs,
) -> None:
    """Log a structured event with additional context.

    Args:
        logger: Logger instance.
        level: Log level (e.g., logging.INFO).
        event: Event name/description.
        **kwargs: Additional context to include in the log.

    Example:
        log_event(logger, logging.INFO, "cache_hit", query="test", similarity=0.95)
    """
    logger.log(level, event, extra=kwargs)
