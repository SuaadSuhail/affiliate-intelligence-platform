"""
ETL Pipeline Tests
==================
Confirms src.ingestion.etl_pipeline._derive_status (ingest-time status
assignment) agrees with src.rulebook.recommend.categorize() (the rulebook's
tier classification) for the same churn/growth inputs — both now read the
same canonical thresholds instead of keeping private copies.

Run:
    pytest tests/test_etl_pipeline.py -v
"""

from __future__ import annotations

import uuid

import pytest

from src.ingestion.etl_pipeline import _derive_status
from src.rulebook.recommend import (
    CHURN_AT_RISK_THRESHOLD,
    CHURN_CRITICAL_THRESHOLD,
    GROWTH_HIGH_THRESHOLD,
    categorize,
)

_BOUNDARY_CASES = [
    (0.0, 0.0),
    (CHURN_AT_RISK_THRESHOLD - 0.01, 0.0),
    (CHURN_AT_RISK_THRESHOLD, 0.0),
    (CHURN_AT_RISK_THRESHOLD + 0.01, 0.0),
    (CHURN_CRITICAL_THRESHOLD - 0.01, 0.0),
    (CHURN_CRITICAL_THRESHOLD, 0.0),
    (CHURN_CRITICAL_THRESHOLD + 0.01, 0.0),
    (0.0, GROWTH_HIGH_THRESHOLD - 0.01),
    (0.0, GROWTH_HIGH_THRESHOLD),
    (0.0, GROWTH_HIGH_THRESHOLD + 0.01),
    (CHURN_AT_RISK_THRESHOLD + 0.01, GROWTH_HIGH_THRESHOLD + 0.01),  # at_risk wins over high_growth
    (0.95, 0.95),
]


@pytest.mark.parametrize("churn,growth", _BOUNDARY_CASES)
def test_derive_status_agrees_with_rulebook_categorize(churn, growth):
    """A status assigned at ingest time must equal the tier recommend()/categorize()
    would compute for the same churn/growth inputs — same thresholds, same precedence."""
    assert _derive_status(churn, growth) == categorize(churn, growth)


# ─── ML score preservation across re-ingest ───────────────────────────────────

def test_reingest_existing_affiliate_does_not_reset_computed_scores():
    """
    Root-cause regression: ingest_affiliates_csv() must NOT reset an existing
    affiliate's churn_risk_score/growth_potential_score/health_score back to
    the 0.5/0.5/50.0 defaults on re-ingest. Those fields belong to the scoring
    step (update_all_scores) — ingest's job is contact/revenue/promo/keyword
    data. The current CSV format has no score columns, so re-running
    POST /ingest/full after POST /ml/score must leave already-computed scores
    untouched; only a brand-new affiliate should get the defaults.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate
    from src.ingestion.etl_pipeline import DATA_DIR, ingest_affiliates_csv

    db = SessionLocal()
    aff = None
    original_churn = original_growth = original_health = None
    try:
        aff = db.query(Affiliate).filter(Affiliate.name == "Rachel Torres").first()
        if aff is None:
            pytest.skip("Rachel Torres not in DB — run POST /ingest/full first")

        original_churn = aff.churn_risk_score
        original_growth = aff.growth_potential_score
        original_health = aff.health_score

        # Simulate a real post-scoring state that must survive a re-ingest.
        aff.churn_risk_score = 0.37
        aff.growth_potential_score = 0.91
        aff.health_score = 77.3
        db.commit()

        ingest_affiliates_csv(DATA_DIR / "affiliates.csv")

        db.refresh(aff)
        assert aff.churn_risk_score == 0.37, (
            "Re-ingest reset churn_risk_score to a default — ingest must not "
            "overwrite scores it didn't compute"
        )
        assert aff.growth_potential_score == 0.91
        assert aff.health_score == 77.3
    finally:
        if aff is not None:
            aff.churn_risk_score = original_churn
            aff.growth_potential_score = original_growth
            aff.health_score = original_health
            db.commit()
        db.close()


# ─── Communication ingest idempotency ─────────────────────────────────────────

def test_reingest_communications_file_does_not_duplicate():
    """
    Root-cause regression: re-running ingest_communications_file() on the
    same mock file must not insert a second copy of each communication.
    occurred_at is derived from "N days ago" relative to now at parse time
    and is NOT stable across runs, so (affiliate_id, source, raw_text) is
    the identity key instead — a repeat call must be a no-op for content
    already present, not a fresh insert with a newer occurred_at that
    silently buries older tagged history (see get_affiliate_summary's
    driver-lookback fix for why that mattered).
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Communication
    from src.ingestion.etl_pipeline import DATA_DIR, ingest_communications_file

    db = SessionLocal()
    try:
        ingest_communications_file(DATA_DIR / "emails.txt")  # ensure content exists at least once
        after_first = db.query(Communication).count()

        second_run_ids = ingest_communications_file(DATA_DIR / "emails.txt")
        after_second = db.query(Communication).count()

        assert second_run_ids == [], (
            "Re-ingesting identical content must create zero new rows, got "
            f"{len(second_run_ids)}"
        )
        assert after_second == after_first, (
            f"Re-ingesting the same file duplicated rows: {after_first} -> {after_second}"
        )
    finally:
        db.close()


def test_ingest_communications_file_still_inserts_genuinely_new_content(tmp_path):
    """
    The idempotency dedup key (affiliate_id, source, raw_text) must only
    block exact re-ingests of already-seen content — genuinely new content
    for a real affiliate must still be inserted normally, and a second
    ingest of that same new content must then also be a no-op.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, Communication
    from src.ingestion.etl_pipeline import ingest_communications_file

    db = SessionLocal()
    aff = None
    created_id = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        unique_marker = f"UNIQUE-TEST-MARKER-{uuid.uuid4()}"
        mock_file = tmp_path / "new_comm.txt"
        mock_file.write_text(
            f"[AFFILIATE: {aff.name}]\n[DATE: 1 days ago]\n[SOURCE: email]\n\n"
            f"{unique_marker}\n"
        )

        before = db.query(Communication).count()
        created_ids = ingest_communications_file(mock_file)
        after = db.query(Communication).count()

        assert len(created_ids) == 1
        assert after == before + 1
        created_id = created_ids[0]

        row = db.query(Communication).filter(Communication.id == created_id).first()
        assert row is not None
        assert unique_marker in row.raw_text

        second_ids = ingest_communications_file(mock_file)
        assert second_ids == []
        assert db.query(Communication).count() == after
    finally:
        if created_id is not None:
            db.query(Communication).filter(Communication.id == created_id).delete(
                synchronize_session=False
            )
            db.commit()
        db.close()


# ─── Demo leak seed ───────────────────────────────────────────────────────────

def test_seed_demo_leak_scan_flags_exactly_the_two_seeded_affiliates():
    """
    Real DB / real fixtures: after ingest_affiliates_csv (which sets
    active_promo_code for Rachel Torres and Marcus Williams from the real
    data/mock/affiliates.csv) + seed_demo_leak_scan(), those two affiliates
    must have has_active_leak=True with a leaked_codes row referencing a
    real fixture site, and every other affiliate must be False.

    Deliberately NOT cleaned up in a finally block — this is the intended
    persistent demo baseline (data/mock/affiliates.csv permanently seeds
    these two codes), not synthetic test pollution to revert. Every other
    real-DB leakage test in this suite reverts its own temporary
    active_promo_code and has_active_leak changes, so this holds as long as
    this test runs against a DB where those have already cleaned up after
    themselves.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, LeakedCode
    from src.ingestion.etl_pipeline import DATA_DIR, ingest_affiliates_csv, seed_demo_leak_scan

    db = SessionLocal()
    try:
        ingest_affiliates_csv(DATA_DIR / "affiliates.csv")
        db.expire_all()  # this session's Affiliate objects may be stale after ingest committed

        seed_demo_leak_scan()

        affiliates = db.query(Affiliate).all()
        if not affiliates:
            pytest.skip("No affiliates in DB — ingest_affiliates_csv should have seeded them")

        flagged = {a.name for a in affiliates if a.has_active_leak}
        assert flagged == {"Rachel Torres", "Marcus Williams"}, (
            f"Expected exactly Rachel Torres and Marcus Williams flagged, got {flagged}"
        )

        rachel = next(a for a in affiliates if a.name == "Rachel Torres")
        marcus = next(a for a in affiliates if a.name == "Marcus Williams")

        rachel_leaks = db.query(LeakedCode).filter(LeakedCode.affiliate_id == rachel.id).all()
        marcus_leaks = db.query(LeakedCode).filter(LeakedCode.affiliate_id == marcus.id).all()

        assert any(l.code == "TOMB-EXCL20" and l.site == "voucherslug-mock" for l in rachel_leaks), (
            f"Expected TOMB-EXCL20 on voucherslug-mock for Rachel Torres, "
            f"got {[(l.code, l.site) for l in rachel_leaks]}"
        )
        assert any(l.code == "CSRLEAK99" and l.site == "csr-shell-mock" for l in marcus_leaks), (
            f"Expected CSRLEAK99 on csr-shell-mock for Marcus Williams, "
            f"got {[(l.code, l.site) for l in marcus_leaks]}"
        )
    finally:
        db.close()


def test_seed_demo_leak_scan_skips_when_a_live_site_is_configured(caplog):
    """
    If any site in site_config.SITES is not a local file:// fixture (i.e. a
    real site has been enabled), seed_demo_leak_scan() must skip itself and
    log a warning rather than silently running a live scan as a side effect
    of routine ingestion. SiteConfig.kind is deliberately NOT what gates
    this — csr-shell-mock is kind="live" but is still a safe local fixture
    (file:// url); only the url scheme is trustworthy here.
    """
    from src.ingestion.etl_pipeline import seed_demo_leak_scan
    from src.scraping.site_config import SITES, SiteConfig
    from src.storage.database import SessionLocal
    from src.storage.models import LeakedCode

    live_site = SiteConfig(
        name="not-actually-mock",
        kind="live",
        url="https://example.com/vouchers/",
        code_selectors=[".code"],
        merchant_selectors=[".merchant"],
    )

    db = SessionLocal()
    try:
        before_count = db.query(LeakedCode).count()

        SITES.append(live_site)
        try:
            with caplog.at_level("WARNING"):
                result = seed_demo_leak_scan()
        finally:
            SITES.remove(live_site)

        assert result is None, "Must skip (return None) rather than run when a live site is present"
        assert "seed_demo_leak_scan skipped" in caplog.text, (
            "Expected a warning log explaining the skip"
        )

        after_count = db.query(LeakedCode).count()
        assert after_count == before_count, (
            "No scan should have run at all — leaked_codes must be untouched"
        )
    finally:
        db.close()


# ─── Demo SEO seed ────────────────────────────────────────────────────────────

def test_seed_demo_seo_scan_flags_exactly_the_seeded_affiliates():
    """
    Real DB / real fixture: after ingest_affiliates_csv (which sets
    tracked_keyword for Rachel Torres, Priya Sharma, Sarah Chen, and Marcus
    Williams from the real data/mock/affiliates.csv) + seed_demo_seo_scan(),
    those four affiliates must have the exact search_trend the fixture data
    implies (Marcus/Sarah declining, Priya improving, Rachel stable), each
    with a seo_signals row referencing their real tracked keyword, and every
    untracked affiliate must remain at the 'stable' default.

    Deliberately NOT cleaned up in a finally block — same reasoning as
    test_seed_demo_leak_scan_flags_exactly_the_two_seeded_affiliates: this
    is the intended persistent demo baseline, not synthetic test pollution.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, SeoSignal
    from src.ingestion.etl_pipeline import DATA_DIR, ingest_affiliates_csv, seed_demo_seo_scan

    db = SessionLocal()
    try:
        ingest_affiliates_csv(DATA_DIR / "affiliates.csv")
        db.expire_all()

        seed_demo_seo_scan()

        affiliates = db.query(Affiliate).all()
        if not affiliates:
            pytest.skip("No affiliates in DB — ingest_affiliates_csv should have seeded them")

        by_name = {a.name: a for a in affiliates}

        expected_trends = {
            "Marcus Williams": "declining",
            "Sarah Chen": "declining",
            "Priya Sharma": "improving",
            "Rachel Torres": "stable",
        }
        for name, expected in expected_trends.items():
            assert by_name[name].search_trend == expected, (
                f"Expected {name} search_trend={expected}, got {by_name[name].search_trend}"
            )

        untracked_names = set(by_name) - set(expected_trends)
        for name in untracked_names:
            assert by_name[name].search_trend == "stable", (
                f"Untracked affiliate {name} should stay at the 'stable' default, "
                f"got {by_name[name].search_trend}"
            )

        marcus_signals = db.query(SeoSignal).filter(SeoSignal.affiliate_id == by_name["Marcus Williams"].id).all()
        assert any(s.keyword == "marcus williams promo codes" and s.rank == 27 for s in marcus_signals), (
            f"Expected a signal for Marcus Williams's tracked keyword, got {[(s.keyword, s.rank) for s in marcus_signals]}"
        )
    finally:
        db.close()


def test_seed_demo_seo_scan_skips_when_a_live_api_is_configured(caplog, monkeypatch):
    """
    If src.seo.api_client.LIVE_API_CONFIGURED is True (a real SEO API has
    been wired up), seed_demo_seo_scan() must skip itself and log a warning
    rather than silently running a live check as a side effect of routine
    ingestion.
    """
    import src.seo.api_client as api_client
    from src.ingestion.etl_pipeline import seed_demo_seo_scan
    from src.storage.database import SessionLocal
    from src.storage.models import SeoSignal

    monkeypatch.setattr(api_client, "LIVE_API_CONFIGURED", True)

    db = SessionLocal()
    try:
        before_count = db.query(SeoSignal).count()

        with caplog.at_level("WARNING"):
            result = seed_demo_seo_scan()

        assert result is None, "Must skip (return None) rather than run when a live API is configured"
        assert "seed_demo_seo_scan skipped" in caplog.text, (
            "Expected a warning log explaining the skip"
        )

        after_count = db.query(SeoSignal).count()
        assert after_count == before_count, (
            "No scan should have run at all — seo_signals must be untouched"
        )
    finally:
        db.close()
