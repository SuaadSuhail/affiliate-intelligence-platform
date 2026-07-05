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

File persistence
----------------
Alongside the stdout handler (unchanged — `docker logs` keeps working exactly
as before), the same JSON lines are also written to `logs/app.jsonl` via a
size-based `RotatingFileHandler` (10MB per file, 5 backups kept — 60MB ceiling
total). Size-based rotation was chosen over time-based because the goal is
bounding disk usage during a long-running demo session, which is a size
concern, not a calendar one. This is a demo-appropriate substitute for a real
log aggregator (CloudWatch/ELK/Datadog) — a production deployment should swap
to one of those rather than scale this file-based approach further. See
`read_log_entries()` for the query-side counterpart, used by `GET /admin/logs`.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.jsonl"

MAX_BYTES = 10 * 1024 * 1024  # 10MB per file
BACKUP_COUNT = 5  # app.jsonl.1 .. app.jsonl.5


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

    # Second, additive handler — same JSON lines, persisted to a rotating file
    # so they're queryable after the fact (see read_log_entries / GET /admin/logs).
    # The stdout handler above is untouched by this addition.
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(_JSONFormatter())
    file_handler.setLevel(log_level)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger.  Import and call at module level:

        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def _candidate_log_files(log_file: Path, backup_count: int) -> list[Path]:
    """Current file first (newest data), then rotated backups oldest-numbered
    last (.1 is the most recently rotated, per RotatingFileHandler's naming)."""
    files = [log_file]
    for i in range(1, backup_count + 1):
        files.append(log_file.with_name(f"{log_file.name}.{i}"))
    return files


def read_log_entries(
    level: str | None = None,
    search: str | None = None,
    limit: int = 100,
    log_file: Path | None = None,
    backup_count: int | None = None,
) -> list[dict]:
    """
    Read parsed JSON log entries newest-first, optionally filtered by level
    and/or a case-insensitive substring search over `message`.

    Reads each candidate file (current file, then rotated backups) forward
    once, keeping only the most recent `limit` matching entries per file in a
    bounded deque — never holding an entire file's parsed contents in memory
    at once. Falls back to the next (older) file only if the current one
    doesn't yield enough matches, so a request right after rotation still
    finds older entries in the freshly-rotated backup.
    """
    path = log_file if log_file is not None else LOG_FILE
    n_backups = backup_count if backup_count is not None else BACKUP_COUNT
    level_upper = level.upper() if level else None
    search_lower = search.lower() if search else None

    results: list[dict] = []
    for candidate in _candidate_log_files(path, n_backups):
        if len(results) >= limit:
            break
        if not candidate.exists():
            continue

        file_matches: deque = deque(maxlen=limit)
        with open(candidate, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a partial line from an in-progress write

                if level_upper and entry.get("level") != level_upper:
                    continue
                if search_lower and search_lower not in entry.get("message", "").lower():
                    continue
                file_matches.append(entry)

        # file_matches is oldest-to-newest (deque only drops from the left
        # once past maxlen) — reverse so the newest entries come first.
        results.extend(reversed(file_matches))

    return results[:limit]