"""
Structured JSON logging configuration.

Usage
-----
    from src.core.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Affiliates ingested", extra={"count": 10})
    logger.error("DB connection failed", extra={"error": str(exc)})

Output format
-------------
    {
      "timestamp": "2026-06-12T10:30:00.123456+00:00",
      "level": "INFO",
      "module": "src.ingestion.etl_pipeline",
      "message": "Affiliates ingested",
      "extra": {"count": 10}
    }

Log level is controlled by the LOG_LEVEL environment variable (default: INFO).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


# Standard LogRecord attributes — excluded from the "extra" field so we don't
# pollute every log line with Python internals.
_SKIP: frozenset[str] = frozenset({
    "args", "created", "exc_info", "exc_text", "filename", "funcName",
    "id", "levelname", "levelno", "lineno", "message", "module", "msecs",
    "msg", "name", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName", "taskName",
})


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        extra = {k: v for k, v in record.__dict__.items() if k not in _SKIP}

        entry: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.message,
        }
        if extra:
            entry["extra"] = extra
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


def configure_logging() -> None:
    """
    Configure the root logger with a JSON formatter and the level from
    the LOG_LEVEL environment variable.

    Safe to call multiple times — subsequent calls are no-ops once the
    root logger already has a JSON handler attached.
    """
    root = logging.getLogger()

    # Avoid adding duplicate handlers on hot-reload
    if any(isinstance(h.formatter, _JSONFormatter) for h in root.handlers):
        return

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    handler.setLevel(log_level)

    root.setLevel(log_level)
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger.  Import and call at module level:

        logger = get_logger(__name__)
    """
    return logging.getLogger(name)