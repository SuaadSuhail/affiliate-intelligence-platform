"""
Leakage Scraper Tests
=====================
Tests for fetch_html + extractor, matcher, and the check_leakage orchestrator.

Tests (a)–(e): pure unit / integration — no database required.
Tests (f)–(h): require a live PostgreSQL database (Docker must be running).

Run:
    pytest tests/test_leakage_scraper.py -v
"""

from __future__ import annotations

import pytest


# ─── (a) Extractor: TESTLEAK20 and TOMB-EXCL20 found in voucherslug fixture ──

def test_extractor_finds_leaked_code_in_fixture():
    """
    CSS-selector path extracts all codes from voucherslug_mock.html with the
    correct merchant context.  Specifically asserts TESTLEAK20 (basic case) and
    TOMB-EXCL20 (hyphenated regression — would silently truncate under \\b regex).
    """
    from src.scraping.site_config import SITES
    from src.scraping.fetcher import fetch_html
    from src.scraping.extractor import extract_candidate_codes

    site = next(s for s in SITES if s.name == "voucherslug-mock")
    html = fetch_html(site.url, kind=site.kind)
    candidates = extract_candidate_codes(html, site)

    codes = {c.code for c in candidates}
    assert "TESTLEAK20" in codes, f"TESTLEAK20 missing from {codes}"
    assert "TOMB-EXCL20" in codes, f"TOMB-EXCL20 missing from {codes} (hyphen regression)"

    testleak = next(c for c in candidates if c.code == "TESTLEAK20")
    assert testleak.merchant_context == "FashionHub", (
        f"Expected merchant 'FashionHub', got {testleak.merchant_context!r}"
    )

    tomb = next(c for c in candidates if c.code == "TOMB-EXCL20")
    assert tomb.merchant_context == "PartnerBrand", (
        f"Expected merchant 'PartnerBrand', got {tomb.merchant_context!r}"
    )


# ─── (b) Extractor: exactly 3 codes from dealsden, no cross-contamination ─────

def test_extractor_no_false_positive_on_clean_fixture():
    """
    dealsden_mock.html contains exactly 3 deal codes (BOOKS30, EXTRA5, PETB2G1).
    Asserts the count is exactly 3 and no codes from other fixtures bleed in.
    """
    from src.scraping.site_config import SITES
    from src.scraping.fetcher import fetch_html
    from src.scraping.extractor import extract_candidate_codes

    site = next(s for s in SITES if s.name == "dealsden-mock")
    html = fetch_html(site.url, kind=site.kind)
    candidates = extract_candidate_codes(html, site)

    codes = [c.code for c in candidates]
    assert len(candidates) == 3, f"Expected exactly 3 candidates, got {len(candidates)}: {codes}"
    assert "BOOKS30" in codes
    assert "EXTRA5" in codes
    assert "PETB2G1" in codes

    # No cross-contamination from other fixtures
    assert "TESTLEAK20" not in codes
    assert "TOMB-EXCL20" not in codes
    assert "CSRLEAK99" not in codes


# ─── (c) Extractor: Playwright renders JS and CSRLEAK99 is injected ───────────

def test_extractor_handles_csr_rendered_content():
    """
    csr_shell_mock.html is an empty shell — a static read returns only <div id="app">.
    Playwright executes csr_shell_mock.js, which injects a voucher card with
    code CSRLEAK99.  Asserts the code is found, proving Playwright rendering is active.
    """
    from src.scraping.site_config import SITES
    from src.scraping.fetcher import fetch_html
    from src.scraping.extractor import extract_candidate_codes

    site = next(s for s in SITES if s.name == "csr-shell-mock")
    html = fetch_html(site.url, kind=site.kind)
    candidates = extract_candidate_codes(html, site)

    codes = {c.code for c in candidates}
    assert "CSRLEAK99" in codes, (
        f"CSRLEAK99 not found — Playwright may not have executed the JS. "
        f"Got: {codes}"
    )


# ─── (d) Extractor: <title> text does not produce JS-R false positive ─────────

def test_extractor_title_text_not_falsely_matched():
    """
    Regression test for the JS-R false positive:
    <title>CSR Shell — Mock JS-Rendered Page</title> contains 'JS-Rendered',
    which the original soup.get_text() would extract as 'JS-R'.
    Fixed by using soup.body.get_text() in the regex fallback.
    Asserts JS-R is absent and CSRLEAK99 is still correctly found.
    """
    from src.scraping.site_config import SITES
    from src.scraping.fetcher import fetch_html
    from src.scraping.extractor import extract_candidate_codes

    site = next(s for s in SITES if s.name == "csr-shell-mock")
    html = fetch_html(site.url, kind=site.kind)
    candidates = extract_candidate_codes(html, site)

    codes = {c.code for c in candidates}
    assert "JS-R" not in codes, (
        "JS-R false positive from <title> text must not appear (soup.body fix regression)"
    )
    assert "CSRLEAK99" in codes, "CSRLEAK99 must still be found after the title-exclusion fix"


# ─── (e) Matcher: exact case-normalised match, near-misses correctly rejected ─

def test_matcher_exact_match_only():
    """
    Matcher normalises both sides (strip + upper) before comparing.
    - lowercase 'testleak20' normalises to TESTLEAK20 → matches
    - 'TESTLEAK20 ' (trailing space) strips to TESTLEAK20 → matches
    - 'TESTLEAK21' is a genuinely different code → must NOT match
    - 'TOMB-EXCL20' matches exactly
    """
    from src.scraping.extractor import CandidateCode
    from src.scraping.matcher import match_candidates_to_affiliates

    candidates = [
        CandidateCode(code="TESTLEAK20",  merchant_context="TestMerchant", snippet="exact"),
        CandidateCode(code="testleak20",  merchant_context="",             snippet="lowercase"),
        CandidateCode(code="TESTLEAK20 ", merchant_context="",             snippet="trailing space"),
        CandidateCode(code="TESTLEAK21",  merchant_context="",             snippet="off by one"),
        CandidateCode(code="TOMB-EXCL20", merchant_context="Partner",      snippet="hyphenated"),
    ]

    affiliate_codes = {
        "aff-uuid-001": "TESTLEAK20",
        "aff-uuid-002": "TOMB-EXCL20",
    }

    matches = match_candidates_to_affiliates(
        candidates, affiliate_codes, "test-site", "https://test.example.com"
    )

    matched_affiliate_codes = [m.affiliate_code for m in matches]

    # All three TESTLEAK20 surface forms normalise and match
    testleak_matches = [m for m in matches if m.affiliate_code == "TESTLEAK20"]
    assert len(testleak_matches) == 3, (
        f"Expected 3 TESTLEAK20 matches (exact/lowercase/trailing-space), got {len(testleak_matches)}"
    )

    # TESTLEAK21 is a different code and must not match
    assert not any(m.affiliate_code == "TESTLEAK21" for m in matches), (
        "TESTLEAK21 must not match TESTLEAK20 — off-by-one should be rejected"
    )

    # TOMB-EXCL20 matches exactly once
    tomb_matches = [m for m in matches if m.affiliate_code == "TOMB-EXCL20"]
    assert len(tomb_matches) == 1, f"Expected 1 TOMB-EXCL20 match, got {len(tomb_matches)}"

    # Affiliate ID wiring is correct
    assert all(m.affiliate_id == "aff-uuid-001" for m in testleak_matches)
    assert tomb_matches[0].affiliate_id == "aff-uuid-002"
    assert tomb_matches[0].site_name == "test-site"
    assert tomb_matches[0].source_url == "https://test.example.com"


# ─── (f) check_leakage: writes exactly one LeakedCode row ────────────────────

def test_check_leakage_end_to_end_writes_expected_rows():
    """
    Sets one affiliate's active_promo_code to TESTLEAK20, runs check_leakage,
    and asserts exactly one LeakedCode row is persisted with correct fields.
    Cleans up the row and reverts the affiliate code in a finally block.
    Requires a live PostgreSQL database.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, LeakedCode
    from src.scraping.leakage_scraper import check_leakage

    db = SessionLocal()
    aff = None
    original_code = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        original_code = aff.active_promo_code
        aff.active_promo_code = "TESTLEAK20"
        db.flush()

        result = check_leakage(db, scan_type="on_demand")

        # Filter to this test's own affiliate — the DB may have other
        # affiliates with a real active_promo_code too (the demo seed:
        # Rachel Torres / Marcus Williams, see data/mock/affiliates.csv),
        # so global new_leaks is no longer guaranteed to be just this one.
        own_leaks = [l for l in result["new_leaks"] if l["affiliate_id"] == str(aff.id)]
        assert len(own_leaks) == 1, (
            f"Expected 1 new_leak for this affiliate, got {len(own_leaks)}: {own_leaks}"
        )
        leak = own_leaks[0]
        assert leak["affiliate_id"] == str(aff.id)
        assert leak["code"] == "TESTLEAK20"
        assert leak["site"] == "voucherslug-mock"
        assert leak["source_url"], "source_url must be non-empty"
        assert leak["found_at"], "found_at must be non-empty"

        # DB row assertions
        row = db.query(LeakedCode).filter(
            LeakedCode.affiliate_id == aff.id,
            LeakedCode.code == "TESTLEAK20",
        ).first()
        assert row is not None, "LeakedCode row must exist in DB after check_leakage"
        assert row.scan_type == "on_demand"
        assert row.site == "voucherslug-mock"
        assert row.source_url
        assert row.found_at is not None

    finally:
        if aff is not None:
            db.query(LeakedCode).filter(
                LeakedCode.affiliate_id == aff.id,
                LeakedCode.code == "TESTLEAK20",
            ).delete(synchronize_session=False)
            # check_leakage()'s has_active_leak recompute is a bulk
            # Query.update(synchronize_session=False), which bypasses the
            # ORM — this session's cached aff object is stale until
            # refreshed. Refresh FIRST, before making any reassignment below:
            # refresh() overwrites in-memory attributes with the current DB
            # row, discarding any not-yet-flushed pending changes — refreshing
            # in between two assignments (rather than before both) would
            # silently drop the first one.
            db.refresh(aff)
            aff.active_promo_code = original_code
            # Recompute has_active_leak now that this test's row is gone —
            # check_leakage() set it True during the test; leaving it stuck
            # True with zero underlying leaked_codes rows would be exactly
            # the kind of "points nowhere" inconsistency Phase 5A's
            # referential-integrity tests exist to catch elsewhere.
            aff.has_active_leak = (
                db.query(LeakedCode).filter(LeakedCode.affiliate_id == aff.id).count() > 0
            )
            db.commit()
        db.close()


# ─── (g) check_leakage: 20-hour dedup window prevents duplicate row ───────────

def test_check_leakage_dedup_window_prevents_duplicate():
    """
    Calls check_leakage twice in immediate succession with the same promo code.
    First call must create 1 new_leak; second call must return 0 new_leaks.
    Exactly 1 LeakedCode row must exist in the DB after both runs.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, LeakedCode
    from src.scraping.leakage_scraper import check_leakage

    db = SessionLocal()
    aff = None
    original_code = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        original_code = aff.active_promo_code
        aff.active_promo_code = "TESTLEAK20"
        db.flush()

        result1 = check_leakage(db, scan_type="on_demand")
        result2 = check_leakage(db, scan_type="on_demand")

        # Filter to this test's own affiliate — the DB may have other
        # affiliates with a real active_promo_code too (the demo seed:
        # Rachel Torres / Marcus Williams), so global new_leaks is no longer
        # guaranteed to be just this one.
        own1 = [l for l in result1["new_leaks"] if l["affiliate_id"] == str(aff.id)]
        own2 = [l for l in result2["new_leaks"] if l["affiliate_id"] == str(aff.id)]

        assert len(own1) == 1, f"Run 1: expected 1 new_leak for this affiliate, got {len(own1)}"
        assert len(own2) == 0, (
            f"Run 2: expected 0 new_leaks for this affiliate (dedup window), got {len(own2)}"
        )

        count = db.query(LeakedCode).filter(
            LeakedCode.affiliate_id == aff.id,
            LeakedCode.code == "TESTLEAK20",
        ).count()
        assert count == 1, f"Expected exactly 1 row after dedup, found {count}"

    finally:
        if aff is not None:
            db.query(LeakedCode).filter(
                LeakedCode.affiliate_id == aff.id,
                LeakedCode.code == "TESTLEAK20",
            ).delete(synchronize_session=False)
            # check_leakage()'s recompute uses Query.update(synchronize_session=False)
            # — a bulk UPDATE that bypasses the ORM, so this session's cached aff
            # object is now stale. Refresh FIRST, before either reassignment below:
            # refresh() overwrites in-memory attributes with the current DB row,
            # discarding any not-yet-flushed pending changes — refreshing in
            # between the two assignments (rather than before both) would
            # silently drop the first one.
            db.refresh(aff)
            aff.active_promo_code = original_code
            aff.has_active_leak = (
                db.query(LeakedCode).filter(LeakedCode.affiliate_id == aff.id).count() > 0
            )
            db.commit()
        db.close()


# ─── (h) check_leakage: one failing site is isolated, others still process ────

def test_check_leakage_isolates_site_failures():
    """
    Injects a fourth SiteConfig pointing at a nonexistent file into SITES for
    the duration of this test only (restored in finally).
    Asserts that:
      - sites_failed has exactly 1 entry for the broken site with a non-empty reason
      - sites_checked reflects only the 3 working sites
      - new_leaks still contains results from working fixtures (no abort on failure)
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, LeakedCode
    from src.scraping.leakage_scraper import check_leakage
    from src.scraping.site_config import SITES, SiteConfig, FIXTURES_DIR

    broken = SiteConfig(
        name="broken-site",
        kind="fixture",
        url=(FIXTURES_DIR / "does_not_exist.html").as_uri(),
        code_selectors=[".voucher-code"],
        merchant_selectors=[".merchant-name"],
    )

    db = SessionLocal()
    aff = None
    original_code = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        original_code = aff.active_promo_code
        aff.active_promo_code = "TESTLEAK20"
        db.flush()

        SITES.append(broken)
        try:
            result = check_leakage(db, scan_type="on_demand")
        finally:
            SITES.remove(broken)

        failed_names = [e["site"] for e in result["sites_failed"]]
        assert len(result["sites_failed"]) == 1, (
            f"Expected 1 failed site, got {result['sites_failed']}"
        )
        assert "broken-site" in failed_names
        assert result["sites_failed"][0]["reason"], "Failure reason must be non-empty"

        assert result["sites_checked"] == 3, (
            f"Expected 3 sites_checked (only working sites), got {result['sites_checked']}"
        )
        assert len(result["new_leaks"]) >= 1, (
            "Working fixtures must still produce new_leaks despite the one failed site"
        )

    finally:
        if aff is not None:
            db.query(LeakedCode).filter(
                LeakedCode.affiliate_id == aff.id,
                LeakedCode.code == "TESTLEAK20",
            ).delete(synchronize_session=False)
            # check_leakage()'s recompute uses Query.update(synchronize_session=False)
            # — a bulk UPDATE that bypasses the ORM, so this session's cached aff
            # object is now stale. Refresh FIRST, before either reassignment below:
            # refresh() overwrites in-memory attributes with the current DB row,
            # discarding any not-yet-flushed pending changes — refreshing in
            # between the two assignments (rather than before both) would
            # silently drop the first one.
            db.refresh(aff)
            aff.active_promo_code = original_code
            aff.has_active_leak = (
                db.query(LeakedCode).filter(LeakedCode.affiliate_id == aff.id).count() > 0
            )
            db.commit()


# ─── (i) check_leakage: writes an audit_log entry even with zero leaks ────────

def test_check_leakage_writes_audit_entry_even_with_zero_leaks():
    """
    check_leakage must write one audit_log entry per affiliate in scope
    (stage='signals', rule_or_tool='check_leakage') — even when zero leaks
    are found, so "checked and found nothing" is on record too.
    Requires a live PostgreSQL database.
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, AuditLog
    from src.scraping.leakage_scraper import check_leakage

    db = SessionLocal()
    aff = None
    original_code = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        original_code = aff.active_promo_code
        # Guaranteed not to appear in any fixture — exercises the
        # "checked and found nothing" path specifically.
        aff.active_promo_code = "NO-SUCH-LEAK-CODE-XYZ"
        db.flush()

        result = check_leakage(db, scan_type="on_demand")
        # Filter to this test's own affiliate — the DB may have other
        # affiliates with a real active_promo_code too (the demo seed:
        # Rachel Torres / Marcus Williams), so global new_leaks is no longer
        # guaranteed to be zero.
        own_leaks = [l for l in result["new_leaks"] if l["affiliate_id"] == str(aff.id)]
        assert len(own_leaks) == 0

        entry = (
            db.query(AuditLog)
            .filter(
                AuditLog.record_id == aff.id,
                AuditLog.rule_or_tool == "check_leakage",
            )
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert entry is not None
        assert entry.stage == "signals"
        assert entry.record_type == "affiliate"
        assert entry.input_snapshot["scan_type"] == "on_demand"
        assert isinstance(entry.input_snapshot["sites_checked"], list)
        assert len(entry.input_snapshot["sites_checked"]) == 3  # 3 fixture sites
        assert entry.output_snapshot == {"leaks_found": 0, "codes": []}

        # Referential sanity: record_id must resolve to a real affiliates row
        # — a regression check for a future bug writing an audit entry that
        # points nowhere. Not a schema constraint (record_type stays
        # polymorphic by design), just a check on this specific wiring point.
        resolved = db.query(Affiliate).filter(Affiliate.id == entry.record_id).first()
        assert resolved is not None, (
            f"audit_log record_id {entry.record_id} does not exist in affiliates"
        )
        assert resolved.id == aff.id
    finally:
        if aff is not None:
            db.query(AuditLog).filter(
                AuditLog.record_id == aff.id,
                AuditLog.rule_or_tool == "check_leakage",
            ).delete(synchronize_session=False)
            aff.active_promo_code = original_code
            db.commit()
        db.close()


# ─── (j) has_active_leak: flips true on a leak, stays true on a clean rescan ──

def test_has_active_leak_flips_true_and_stays_true_on_clean_rescan():
    """
    Real DB: affiliates.has_active_leak starts False for a clean affiliate,
    flips True after a scan finds a leak, and stays True on a subsequent scan
    that finds nothing new for it — this is the "has ever had a leak on
    record" definition (see src.scraping.leakage_scraper module docstring for
    the judgment call: no resolution/expiry workflow exists yet, so the flag
    does not go back to False on its own).
    """
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, AuditLog, LeakedCode
    from src.scraping.leakage_scraper import check_leakage

    db = SessionLocal()
    aff = None
    original_code = None
    original_flag = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        original_code = aff.active_promo_code
        original_flag = aff.has_active_leak

        # Force a genuinely clean starting state for this affiliate.
        db.query(LeakedCode).filter(LeakedCode.affiliate_id == aff.id).delete(
            synchronize_session=False
        )
        aff.has_active_leak = False
        aff.active_promo_code = "TESTLEAK20"  # matches voucherslug-mock fixture
        db.commit()

        # Filter to this test's own affiliate — the DB may have other
        # affiliates with a real active_promo_code too (the demo seed:
        # Rachel Torres / Marcus Williams), so global new_leaks is no longer
        # guaranteed to be just this one.
        result1 = check_leakage(db, scan_type="on_demand")
        own1 = [l for l in result1["new_leaks"] if l["affiliate_id"] == str(aff.id)]
        assert len(own1) == 1

        db.refresh(aff)
        assert aff.has_active_leak is True

        # Second scan: the dedup window means no NEW leaked_codes row, but the
        # existing one must keep the flag True — recompute must look at the
        # full table, not just this run's new_leaks.
        result2 = check_leakage(db, scan_type="on_demand")
        own2 = [l for l in result2["new_leaks"] if l["affiliate_id"] == str(aff.id)]
        assert len(own2) == 0

        db.refresh(aff)
        assert aff.has_active_leak is True
    finally:
        if aff is not None:
            db.query(AuditLog).filter(
                AuditLog.record_id == aff.id,
                AuditLog.rule_or_tool == "check_leakage",
            ).delete(synchronize_session=False)
            db.query(LeakedCode).filter(
                LeakedCode.affiliate_id == aff.id,
                LeakedCode.code == "TESTLEAK20",
            ).delete(synchronize_session=False)
            aff.active_promo_code = original_code
            aff.has_active_leak = original_flag if original_flag is not None else False
            db.commit()
        db.close()


# ─── (k) has_active_leak: exposed correctly in AffiliateOut ───────────────────

def test_affiliate_out_includes_has_active_leak_for_both_states():
    """AffiliateOut.from_orm must surface has_active_leak with the correct
    value for both a leaked and a non-leaked affiliate — not silently
    dropped from the affiliate-list-facing schema."""
    import uuid as _uuid
    from src.api.main import AffiliateOut
    from src.storage.models import Affiliate

    def _aff(has_leak: bool) -> Affiliate:
        a = Affiliate()
        a.id = _uuid.uuid4()
        a.name = "Test"
        a.status = "active"
        a.churn_risk_score = 0.3
        a.growth_potential_score = 0.5
        a.health_score = 60.0
        a.revenue_30d = 1000.0
        a.days_since_contact = 5
        a.last_contact_at = None
        a.has_active_leak = has_leak
        return a

    leaked_out = AffiliateOut.from_orm(_aff(True))
    clean_out = AffiliateOut.from_orm(_aff(False))

    assert leaked_out.has_active_leak is True
    assert clean_out.has_active_leak is False