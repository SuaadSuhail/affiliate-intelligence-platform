"""
SEO Signal Tests
=================
Tests for src.seo.api_client (fixture-first data source), src.seo.analyze
(pure trend derivation), and src.seo.checker.check_seo (the orchestrator).

Tests (a)-(d): pure unit / fixture-read — no database required.
Tests (e)-(h): require a live PostgreSQL database (Docker must be running).

Run:
    pytest tests/test_seo.py -v
"""

from __future__ import annotations

import uuid

import pytest


# ─── (a) fetch_seo_data reads the real fixture ────────────────────────────────

def test_fetch_seo_data_reads_fixture():
    """kind='fixture' must read the real JSON fixture and return the expected
    shape — a list of dicts with the core rank-tracking fields."""
    from src.seo.api_client import fetch_seo_data

    rows = fetch_seo_data(kind="fixture")

    assert isinstance(rows, list)
    assert len(rows) >= 4
    for row in rows:
        assert "keyword" in row
        assert "position" in row
        assert "search_volume" in row
        assert "checked_at" in row

    keywords = {row["keyword"] for row in rows}
    assert "marcus williams promo codes" in keywords
    assert "sarah chen exclusive deals" in keywords


def test_fetch_seo_data_live_not_implemented():
    """kind='live' must raise NotImplementedError, not silently fall back to
    the fixture or fail some other way — no real SEO API exists yet."""
    from src.seo.api_client import fetch_seo_data

    with pytest.raises(NotImplementedError):
        fetch_seo_data(kind="live")


# ─── (b) derive_search_trend boundaries ───────────────────────────────────────

def test_derive_search_trend_empty_list_is_stable():
    from src.seo.analyze import derive_search_trend
    assert derive_search_trend([]) == "stable"


def test_derive_search_trend_none_rank_change_is_stable():
    from src.seo.analyze import derive_search_trend
    assert derive_search_trend([{"rank_change": None, "checked_at": "2026-07-01"}]) == "stable"


def test_derive_search_trend_boundaries():
    from src.seo.analyze import (
        DECLINING_THRESHOLD,
        IMPROVING_THRESHOLD,
        derive_search_trend,
    )

    def _signal(rank_change):
        return [{"rank_change": rank_change, "checked_at": "2026-07-01"}]

    assert derive_search_trend(_signal(DECLINING_THRESHOLD + 1)) == "stable"
    assert derive_search_trend(_signal(DECLINING_THRESHOLD)) == "declining"
    assert derive_search_trend(_signal(DECLINING_THRESHOLD - 1)) == "declining"

    assert derive_search_trend(_signal(IMPROVING_THRESHOLD - 1)) == "stable"
    assert derive_search_trend(_signal(IMPROVING_THRESHOLD)) == "improving"
    assert derive_search_trend(_signal(IMPROVING_THRESHOLD + 1)) == "improving"

    assert derive_search_trend(_signal(0)) == "stable"


def test_derive_search_trend_uses_most_recent_signal_only():
    """An old declining signal must not override a newer stable/improving
    one — only the most recently checked_at signal's rank_change counts."""
    from src.seo.analyze import derive_search_trend

    signals = [
        {"rank_change": -10, "checked_at": "2026-06-01T00:00:00+00:00"},  # old, declining
        {"rank_change": 5, "checked_at": "2026-07-01T00:00:00+00:00"},   # newest, improving
    ]
    assert derive_search_trend(signals) == "improving"


def test_derive_search_trend_accepts_orm_like_objects():
    """Must work with attribute-access objects (real SeoSignal rows), not
    just dicts — same dual-mode pattern as recommend()'s leak handling."""
    from types import SimpleNamespace

    from src.seo.analyze import derive_search_trend

    signal = SimpleNamespace(rank_change=-8, checked_at="2026-07-01T00:00:00+00:00")
    assert derive_search_trend([signal]) == "declining"


# ─── (e) check_seo: end-to-end writes expected row + trend ────────────────────

def test_check_seo_end_to_end_writes_expected_signal_and_trend():
    """
    Real DB: tracking a keyword that matches a declining fixture entry must
    persist one SeoSignal row with the correct rank/rank_change and update
    affiliates.search_trend to 'declining'.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, SeoSignal
    from src.seo.checker import check_seo

    db = SessionLocal()
    aff = None
    original_keyword = None
    original_trend = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        original_keyword = aff.tracked_keyword
        original_trend = aff.search_trend

        db.query(SeoSignal).filter(SeoSignal.affiliate_id == aff.id).delete(
            synchronize_session=False
        )
        aff.tracked_keyword = "marcus williams promo codes"  # matches fixture, declining
        aff.search_trend = "stable"
        db.commit()

        result = check_seo(db, scan_type="on_demand")
        own = [s for s in result["signals"] if s["affiliate_id"] == str(aff.id)]
        assert len(own) == 1
        assert own[0]["rank"] == 27
        assert own[0]["rank_change"] == 8 - 27

        row = (
            db.query(SeoSignal)
            .filter(SeoSignal.affiliate_id == aff.id, SeoSignal.keyword == "marcus williams promo codes")
            .first()
        )
        assert row is not None
        assert row.rank == 27
        assert row.rank_change == -19
        assert row.search_volume == 320

        db.refresh(aff)
        assert aff.search_trend == "declining"
    finally:
        if aff is not None:
            db.query(SeoSignal).filter(SeoSignal.affiliate_id == aff.id).delete(
                synchronize_session=False
            )
            from src.storage.models import AuditLog
            db.query(AuditLog).filter(
                AuditLog.record_id == aff.id, AuditLog.rule_or_tool == "check_seo"
            ).delete(synchronize_session=False)
            aff.tracked_keyword = original_keyword
            aff.search_trend = original_trend if original_trend is not None else "stable"
            db.commit()
        db.close()


def test_check_seo_second_call_does_not_duplicate_identical_measurement():
    """
    Root-cause regression: check_seo() must not insert a second SeoSignal row
    when the fetched measurement's (affiliate_id, keyword, checked_at) is
    identical to one already recorded. This is exactly what happened via
    seed_demo_seo_scan() being called on every POST /ingest/full re-run: the
    fixture source's checked_at is a fixed timestamp (not "now"), so every
    repeat call inserted a byte-for-byte duplicate row — confirmed live as
    176 total seo_signals rows collapsing to 4 unique measurements. A
    genuinely new measurement (a different checked_at) must still always be
    recorded — this is an exact-measurement guard, not a time-window dedup.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, AuditLog, SeoSignal
    from src.seo.checker import check_seo

    db = SessionLocal()
    aff = None
    original_keyword = None
    original_trend = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        original_keyword = aff.tracked_keyword
        original_trend = aff.search_trend

        db.query(SeoSignal).filter(SeoSignal.affiliate_id == aff.id).delete(
            synchronize_session=False
        )
        aff.tracked_keyword = "priya sharma discount codes"  # matches fixture, improving
        aff.search_trend = "stable"
        db.commit()

        check_seo(db, scan_type="on_demand")
        after_first = db.query(SeoSignal).filter(SeoSignal.affiliate_id == aff.id).count()
        assert after_first == 1

        check_seo(db, scan_type="on_demand")
        after_second = db.query(SeoSignal).filter(SeoSignal.affiliate_id == aff.id).count()

        assert after_second == 1, (
            f"Re-running check_seo() with identical fixture data duplicated "
            f"the measurement: {after_first} -> {after_second} rows"
        )
    finally:
        if aff is not None:
            db.query(SeoSignal).filter(SeoSignal.affiliate_id == aff.id).delete(
                synchronize_session=False
            )
            db.query(AuditLog).filter(
                AuditLog.record_id == aff.id, AuditLog.rule_or_tool == "check_seo"
            ).delete(synchronize_session=False)
            # check_seo()'s recompute is a bulk Query.update(synchronize_session=
            # False) — refresh before reassigning, or the restore silently no-ops
            # whenever it coincidentally matches the stale in-memory value.
            db.refresh(aff)
            aff.tracked_keyword = original_keyword
            aff.search_trend = original_trend if original_trend is not None else "stable"
            db.commit()
        db.close()


# ─── (f) check_seo: tracked keyword not found in source is recorded, not silently dropped ──

def test_check_seo_writes_audit_entry_when_keyword_not_found():
    """
    Real DB: a tracked_keyword with no matching row in the SEO data source
    must not create a SeoSignal row, but must still produce an audit_log
    entry noting the miss — "checked, no data available" is on record too.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, AuditLog, SeoSignal
    from src.seo.checker import check_seo

    db = SessionLocal()
    aff = None
    original_keyword = None
    original_trend = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        original_keyword = aff.tracked_keyword
        original_trend = aff.search_trend

        aff.tracked_keyword = "this keyword does not exist in the mock fixture at all"
        db.commit()

        result = check_seo(db, scan_type="on_demand")
        own = [s for s in result["signals"] if s["affiliate_id"] == str(aff.id)]
        assert len(own) == 0
        assert str(aff.id) in result["not_found"]

        row_count = db.query(SeoSignal).filter(SeoSignal.affiliate_id == aff.id).count()
        assert row_count == 0

        entry = (
            db.query(AuditLog)
            .filter(AuditLog.record_id == aff.id, AuditLog.rule_or_tool == "check_seo")
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert entry is not None
        assert entry.stage == "signals"
        assert entry.record_type == "affiliate"
        assert entry.output_snapshot["rank"] is None
        assert entry.output_snapshot["note"] == "keyword not found in source"

        # Referential sanity: record_id must resolve to a real affiliates row
        # — a regression check for a future bug writing an audit entry that
        # points nowhere, same pattern already applied to check_leakage.
        resolved = db.query(Affiliate).filter(Affiliate.id == entry.record_id).first()
        assert resolved is not None
        assert resolved.id == aff.id
    finally:
        if aff is not None:
            db.query(AuditLog).filter(
                AuditLog.record_id == aff.id, AuditLog.rule_or_tool == "check_seo"
            ).delete(synchronize_session=False)
            # check_seo()'s recompute is a bulk Query.update(synchronize_session=
            # False) — this session's cached aff object is stale until refreshed,
            # so the restore below must happen after a refresh, or it silently
            # no-ops whenever it coincidentally matches the stale in-memory value.
            db.refresh(aff)
            aff.tracked_keyword = original_keyword
            aff.search_trend = original_trend if original_trend is not None else "stable"
            db.commit()
        db.close()


# ─── (g) check_seo: audit entry content round-trips correctly for a real hit ──

def test_check_seo_audit_entry_content_for_a_found_keyword():
    """
    Real DB: the audit_log entry for an affiliate whose keyword WAS found
    must carry the actual rank/rank_change/search_trend — not just "a row
    exists" — and its input_snapshot must list the keyword checked.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, AuditLog, SeoSignal
    from src.seo.checker import check_seo

    db = SessionLocal()
    aff = None
    original_keyword = None
    original_trend = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        original_keyword = aff.tracked_keyword
        original_trend = aff.search_trend

        db.query(SeoSignal).filter(SeoSignal.affiliate_id == aff.id).delete(
            synchronize_session=False
        )
        aff.tracked_keyword = "sarah chen exclusive deals"  # matches fixture, declining
        aff.search_trend = "stable"
        db.commit()

        check_seo(db, scan_type="on_demand")

        entry = (
            db.query(AuditLog)
            .filter(AuditLog.record_id == aff.id, AuditLog.rule_or_tool == "check_seo")
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert entry is not None
        assert entry.input_snapshot == {"keywords_checked": ["sarah chen exclusive deals"]}
        assert entry.output_snapshot["rank"] == 22
        assert entry.output_snapshot["rank_change"] == 5 - 22
        assert entry.output_snapshot["search_trend"] == "declining"
    finally:
        if aff is not None:
            db.query(SeoSignal).filter(SeoSignal.affiliate_id == aff.id).delete(
                synchronize_session=False
            )
            db.query(AuditLog).filter(
                AuditLog.record_id == aff.id, AuditLog.rule_or_tool == "check_seo"
            ).delete(synchronize_session=False)
            # check_seo()'s recompute is a bulk Query.update(synchronize_session=
            # False) — this session's cached aff object is stale until refreshed,
            # so the restore below must happen after a refresh, or it silently
            # no-ops whenever it coincidentally matches the stale in-memory value
            # (confirmed live: this exact test left an unrelated affiliate stuck
            # at search_trend='declining' before this fix was added).
            db.refresh(aff)
            aff.tracked_keyword = original_keyword
            aff.search_trend = original_trend if original_trend is not None else "stable"
            db.commit()
        db.close()
