"""
Log Persistence Tests
======================
Tests for the file-based log handler added to src.core.logging_config
(RotatingFileHandler alongside the existing stdout handler) and the
read-side query logic (read_log_entries, used by GET /admin/logs).

All tests use a throwaway tmp_path log file with a freshly attached handler
— they never touch the app's real logs/app.jsonl or mutate the root logger,
so they can't interfere with each other or with the app's own logging.

Run:
    pytest tests/test_logging_config.py -v
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import pytest


def _make_isolated_logger(name: str, log_file, max_bytes=10 * 1024 * 1024, backup_count=5):
    """Attach a fresh RotatingFileHandler + _JSONFormatter to a uniquely
    named, non-propagating logger — isolated from the app's real logging
    setup and from other tests."""
    from src.core.logging_config import _JSONFormatter

    handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setFormatter(_JSONFormatter())

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers = [handler]
    return logger, handler


# ─── File handler writes real, parseable JSON lines ───────────────────────────

def test_log_line_written_via_logger_appears_in_file_handler_output(tmp_path):
    from src.core.logging_config import read_log_entries

    log_file = tmp_path / "test.jsonl"
    logger, handler = _make_isolated_logger("test_logging_config.appears", log_file)
    try:
        logger.info("hello from the file handler test")
    finally:
        handler.close()

    entries = read_log_entries(log_file=log_file, backup_count=5, limit=10)
    assert any(e["message"] == "hello from the file handler test" for e in entries)
    assert all(e["level"] == "INFO" for e in entries)


# ─── read_log_entries: level filter ────────────────────────────────────────────

def test_read_log_entries_filters_by_level(tmp_path):
    from src.core.logging_config import read_log_entries

    log_file = tmp_path / "test.jsonl"
    logger, handler = _make_isolated_logger("test_logging_config.level_filter", log_file)
    try:
        logger.info("an info line")
        logger.warning("a warning line")
        logger.error("an error line")
    finally:
        handler.close()

    warnings = read_log_entries(level="WARNING", log_file=log_file, backup_count=5, limit=10)
    assert len(warnings) == 1
    assert warnings[0]["message"] == "a warning line"

    errors = read_log_entries(level="error", log_file=log_file, backup_count=5, limit=10)
    assert len(errors) == 1
    assert errors[0]["message"] == "an error line"


# ─── read_log_entries: search filter ───────────────────────────────────────────

def test_read_log_entries_filters_by_search_term_case_insensitive(tmp_path):
    from src.core.logging_config import read_log_entries

    log_file = tmp_path / "test.jsonl"
    logger, handler = _make_isolated_logger("test_logging_config.search_filter", log_file)
    try:
        logger.info("Affiliate Marcus Williams flagged for leak")
        logger.info("Affiliate Sarah Chen search trend updated")
        logger.info("unrelated pipeline step complete")
    finally:
        handler.close()

    matches = read_log_entries(search="MARCUS", log_file=log_file, backup_count=5, limit=10)
    assert len(matches) == 1
    assert "Marcus Williams" in matches[0]["message"]


# ─── read_log_entries: limit + newest-first ordering ───────────────────────────

def test_read_log_entries_respects_limit_and_returns_newest_first(tmp_path):
    from src.core.logging_config import read_log_entries

    log_file = tmp_path / "test.jsonl"
    logger, handler = _make_isolated_logger("test_logging_config.limit_order", log_file)
    try:
        for i in range(10):
            logger.info(f"entry number {i}")
    finally:
        handler.close()

    limited = read_log_entries(log_file=log_file, backup_count=5, limit=3)
    assert len(limited) == 3
    # Newest-first: entry 9 was logged last, so it must come first.
    assert [e["message"] for e in limited] == [
        "entry number 9",
        "entry number 8",
        "entry number 7",
    ]


# ─── read_log_entries tolerates a malformed/partial line ──────────────────────

def test_read_log_entries_skips_malformed_lines_without_crashing(tmp_path):
    from src.core.logging_config import read_log_entries

    log_file = tmp_path / "test.jsonl"
    logger, handler = _make_isolated_logger("test_logging_config.malformed", log_file)
    try:
        logger.info("a clean valid line")
    finally:
        handler.close()

    # Simulate a partial write (e.g. process killed mid-line) by appending a
    # truncated, non-JSON fragment after the real line.
    with open(log_file, "a", encoding="utf-8") as f:
        f.write('{"timestamp": "2026-01-01T00:00:00", "level": "INFO", "mes')

    entries = read_log_entries(log_file=log_file, backup_count=5, limit=10)
    assert len(entries) == 1
    assert entries[0]["message"] == "a clean valid line"


# ─── Rotation does not lose in-progress writes ─────────────────────────────────

def test_rotation_does_not_lose_entries_across_rollover(tmp_path):
    """A small maxBytes forces several rotations while writing; every entry
    written must still be readable afterwards across the current file plus
    its rotated backups — a basic correctness check, not a stress test."""
    from src.core.logging_config import read_log_entries

    # backup_count is sized generously above what rotation will actually use
    # (~20 lines at ~130 bytes each, 500 bytes/file => ~7-8 files) so that no
    # data is purged for exceeding the backup window — this test is about
    # rotation not losing writes, not about the retention-window trade-off.
    log_file = tmp_path / "test.jsonl"
    logger, handler = _make_isolated_logger(
        "test_logging_config.rotation", log_file, max_bytes=500, backup_count=20
    )
    try:
        for i in range(20):
            logger.info(f"rotation marker line {i:03d}")
    finally:
        handler.close()

    # Confirm rotation actually happened, or the test isn't exercising anything.
    assert (tmp_path / "test.jsonl.1").exists()

    entries = read_log_entries(
        search="rotation marker line", log_file=log_file, backup_count=20, limit=50
    )
    assert len(entries) == 20
    found_indices = {e["message"].rsplit(" ", 1)[-1] for e in entries}
    assert found_indices == {f"{i:03d}" for i in range(20)}


# ─── GET /admin/logs endpoint wiring ───────────────────────────────────────────

def test_get_logs_endpoint_forwards_params_and_wraps_result(monkeypatch):
    from src.api.routers import admin

    captured = {}

    def fake_read_log_entries(level=None, search=None, limit=100):
        captured["level"] = level
        captured["search"] = search
        captured["limit"] = limit
        return [{"message": "one"}, {"message": "two"}]

    monkeypatch.setattr(admin, "read_log_entries", fake_read_log_entries)

    result = admin.get_logs(level="ERROR", limit=5, search="two")

    assert captured == {"level": "ERROR", "search": "two", "limit": 5}
    assert result == {"count": 2, "entries": [{"message": "one"}, {"message": "two"}]}


def test_get_logs_endpoint_default_params(monkeypatch):
    from src.api.routers import admin

    monkeypatch.setattr(admin, "read_log_entries", lambda level, search, limit: [])

    result = admin.get_logs(level=None, limit=100, search=None)
    assert result == {"count": 0, "entries": []}
