"""
Admin Router
============
Protected endpoints for database administration.

POST /admin/migrate          — run all pending Alembic migrations
GET  /admin/migration-status — check current migration revision
GET  /admin/logs             — read recent structured log entries from disk

These endpoints exist so migrations can be triggered manually after
deployment, without running them automatically on startup (which would
cause lock contention when multiple instances start simultaneously).

GET /admin/logs is general operational visibility (HTTP requests, pipeline
steps, errors) — it does NOT replace GET /audit, which stays reserved for
decision-linked entries (rulebook/signals/approval stages). It reads from the
file-based log handler configured in src.core.logging_config; a real
deployment should point at CloudWatch/ELK/Datadog instead of scaling this
file-based approach further.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.api.auth import get_api_key
from src.core.logging_config import get_logger, read_log_entries

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

_ALEMBIC_INI = Path(__file__).parent.parent.parent.parent / "alembic.ini"


@router.post("/migrate", dependencies=[Depends(get_api_key)])
def run_migrations() -> dict:
    """
    Run all pending Alembic migrations.

    Call this once after deployment or after a schema change.
    Never call simultaneously from multiple instances — Alembic
    acquires schema locks that will deadlock under concurrent execution.
    """
    try:
        from alembic.config import Config
        from alembic import command

        alembic_cfg = Config(str(_ALEMBIC_INI))
        command.upgrade(alembic_cfg, "head")
        logger.info("Migrations applied successfully")
        return {"status": "complete", "message": "All migrations applied successfully"}
    except Exception as exc:
        logger.error(f"Migration failed: {exc}")
        return {"status": "failed", "error": str(exc)}


@router.get("/migration-status", dependencies=[Depends(get_api_key)])
def migration_status() -> dict:
    """Return the current Alembic migration revision."""
    try:
        from alembic.runtime.migration import MigrationContext
        from sqlalchemy import create_engine

        engine = create_engine(os.getenv("DATABASE_URL", ""))
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current = context.get_current_revision()
        return {"current_revision": current, "status": "ok"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@router.get("/logs", dependencies=[Depends(get_api_key)])
def get_logs(
    level: Optional[str] = Query(None, description="Filter by level, e.g. ERROR, WARNING, INFO"),
    limit: int = Query(100, ge=1, le=1000, description="Max number of entries to return"),
    search: Optional[str] = Query(None, description="Case-insensitive substring search over the message field"),
) -> dict:
    """
    Return recent structured log entries (newest first) from the file-based
    log handler, optionally filtered by level and/or a message substring.

    General operational visibility only — see GET /audit for decision-linked
    entries (rulebook/signals/approval stages), which this does not replace.
    """
    entries = read_log_entries(level=level, search=search, limit=limit)
    return {"count": len(entries), "entries": entries}