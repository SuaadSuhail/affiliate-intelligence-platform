"""
Audit Log Tests
================
Real DB: confirms GET /audit's record_id filter returns entries written by
all three wiring points (score_updater, leakage_scraper, approvals) together
for one record, and excludes entries for other records.

Also confirms referential sanity: the record_id each wiring point writes
must resolve to a real row in the table implied by record_type. This is not
a schema constraint — record_type stays polymorphic and unconstrained by
design — just a regression check for a future bug that writes an audit
entry pointing at a nonexistent record (e.g. the wrong field passed as
record_id). The leakage_scraper check lives in test_leakage_scraper.py
instead, as one extra assertion on its existing real-DB audit test, rather
than duplicating that test's setup here.

Run:
    pytest tests/test_audit.py -v
"""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import patch

import pytest
from sqlalchemy import func


def test_list_audit_log_filters_by_record_id_across_all_sources():
    """Write one entry shaped like each of the three wiring points against
    the same record_id, plus one for a different record_id, then confirm
    list_audit_log's record_id filter returns exactly the three matching
    entries — newest first — and none of the unrelated one.
    Requires a live PostgreSQL database.
    """
    from src.api.routers.audit import list_audit_log
    from src.audit.log import write_audit_entry
    from src.storage.database import SessionLocal
    from src.storage.models import AuditLog

    db = SessionLocal()
    target_id = uuid.uuid4()
    other_id = uuid.uuid4()
    written_ids: list = []
    try:
        e1 = write_audit_entry(
            db,
            stage="rulebook",
            record_type="affiliate",
            record_id=target_id,
            rule_or_tool="recommend",
            input_snapshot={"days_since_contact": 5},
            output_snapshot={"tier": "active", "evidence": ["churn_risk_score=0.10 ..."]},
        )
        e2 = write_audit_entry(
            db,
            stage="signals",
            record_type="affiliate",
            record_id=target_id,
            rule_or_tool="check_leakage",
            input_snapshot={"scan_type": "on_demand", "sites_checked": ["voucherslug-mock"]},
            output_snapshot={"leaks_found": 0, "codes": []},
        )
        e3 = write_audit_entry(
            db,
            stage="approval",
            record_type="approval_request",
            record_id=target_id,
            rule_or_tool="approvals.approve",
            input_snapshot={"to": "x", "subject": "y", "body": "z"},
            output_snapshot={
                "status": "approved",
                "decided_by": "api",
                "decided_at": "2026-01-01T00:00:00+00:00",
            },
        )
        # Unrelated record — must not appear when filtering by target_id.
        e4 = write_audit_entry(
            db,
            stage="rulebook",
            record_type="affiliate",
            record_id=other_id,
            rule_or_tool="recommend",
            input_snapshot={},
            output_snapshot={"tier": "active", "evidence": []},
        )
        db.commit()
        written_ids = [e1.id, e2.id, e3.id, e4.id]

        result = list_audit_log(
            record_type=None, record_id=str(target_id), stage=None, limit=50, offset=0, db=db
        )

        assert len(result) == 3
        assert all(r.record_id == str(target_id) for r in result)
        assert {r.rule_or_tool for r in result} == {
            "recommend",
            "check_leakage",
            "approvals.approve",
        }
        # newest first
        timestamps = [r.timestamp for r in result]
        assert timestamps == sorted(timestamps, reverse=True)

        # Round-trip check on one entry — not just "a row exists".
        leak_entry = next(r for r in result if r.rule_or_tool == "check_leakage")
        assert leak_entry.input_snapshot == {
            "scan_type": "on_demand",
            "sites_checked": ["voucherslug-mock"],
        }
        assert leak_entry.output_snapshot == {"leaks_found": 0, "codes": []}
    finally:
        if written_ids:
            db.query(AuditLog).filter(AuditLog.id.in_(written_ids)).delete(synchronize_session=False)
            db.commit()
        db.close()


def test_list_audit_log_filters_by_stage():
    """The stage filter must narrow results independently of record_id."""
    from src.api.routers.audit import list_audit_log
    from src.audit.log import write_audit_entry
    from src.storage.database import SessionLocal
    from src.storage.models import AuditLog

    db = SessionLocal()
    record_id = uuid.uuid4()
    written_ids: list = []
    try:
        e1 = write_audit_entry(
            db, stage="rulebook", record_type="affiliate", record_id=record_id,
            rule_or_tool="recommend", input_snapshot={}, output_snapshot={"tier": "active", "evidence": []},
        )
        e2 = write_audit_entry(
            db, stage="signals", record_type="affiliate", record_id=record_id,
            rule_or_tool="check_leakage", input_snapshot={}, output_snapshot={"leaks_found": 0, "codes": []},
        )
        db.commit()
        written_ids = [e1.id, e2.id]

        result = list_audit_log(
            record_type=None, record_id=str(record_id), stage="signals", limit=50, offset=0, db=db
        )

        assert len(result) == 1
        assert result[0].rule_or_tool == "check_leakage"
    finally:
        if written_ids:
            db.query(AuditLog).filter(AuditLog.id.in_(written_ids)).delete(synchronize_session=False)
            db.commit()
        db.close()


# ─── Referential sanity: record_id resolves to a real row ─────────────────────
# (leakage_scraper's version of this check lives in test_leakage_scraper.py,
# as one extra assertion on its existing real-DB audit test.)

def test_score_updater_audit_entry_record_id_resolves_to_real_affiliate():
    """
    Real DB: after update_all_scores runs for a real, previously-unscored-today
    affiliate, its audit_log entry's record_id must actually exist in the
    affiliates table — a regression check for a future bug that writes an
    audit entry pointing at a nonexistent record (e.g. the wrong field passed
    as record_id). Snapshots/restores the affiliate's score fields; deletes
    only the specific score_history/audit_log rows this test creates.
    """
    from src.ml.score_updater import update_all_scores
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, AuditLog, ScoreHistory

    db = SessionLocal()
    aff = None
    original = None
    created_score_history_id = None
    created_audit_id = None
    try:
        today = date.today()
        scored_today_ids = {
            row[0]
            for row in db.query(ScoreHistory.affiliate_id)
            .filter(func.date(ScoreHistory.scored_at) == today)
            .all()
        }
        all_affiliates = db.query(Affiliate).order_by(Affiliate.name).all()
        if not all_affiliates:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        aff = next((a for a in all_affiliates if a.id not in scored_today_ids), None)
        if aff is None:
            pytest.skip(
                "Every affiliate already has a score_history row for today — "
                "cannot isolate a fresh scoring run without disturbing existing data"
            )

        original = {
            "churn_risk_score": aff.churn_risk_score,
            "growth_potential_score": aff.growth_potential_score,
            "health_score": aff.health_score,
        }

        update_all_scores(db)
        db.commit()

        new_score_row = (
            db.query(ScoreHistory)
            .filter(ScoreHistory.affiliate_id == aff.id, func.date(ScoreHistory.scored_at) == today)
            .order_by(ScoreHistory.scored_at.desc())
            .first()
        )
        assert new_score_row is not None
        created_score_history_id = new_score_row.id

        entry = (
            db.query(AuditLog)
            .filter(
                AuditLog.record_type == "affiliate",
                AuditLog.record_id == aff.id,
                AuditLog.rule_or_tool == "recommend",
            )
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert entry is not None
        created_audit_id = entry.id

        resolved = db.query(Affiliate).filter(Affiliate.id == entry.record_id).first()
        assert resolved is not None, (
            f"audit_log record_id {entry.record_id} does not exist in affiliates"
        )
        assert resolved.id == aff.id
    finally:
        if created_audit_id is not None:
            db.query(AuditLog).filter(AuditLog.id == created_audit_id).delete(synchronize_session=False)
        if created_score_history_id is not None:
            db.query(ScoreHistory).filter(
                ScoreHistory.id == created_score_history_id
            ).delete(synchronize_session=False)
        if aff is not None and original is not None:
            aff.churn_risk_score = original["churn_risk_score"]
            aff.growth_potential_score = original["growth_potential_score"]
            aff.health_score = original["health_score"]
        db.commit()
        db.close()


def test_approve_request_audit_entry_record_id_resolves_to_real_approval_request():
    """
    Real DB: after approve_request runs for a real approval_requests row,
    the audit_log entry's record_id must actually exist in approval_requests
    — catches a future bug writing the wrong id (e.g. the affiliate's id
    instead of the request's own id) into the audit entry.
    """
    from src.api.routers.approvals import approve_request
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, ApprovalRequest, AuditLog

    db = SessionLocal()
    req = None
    try:
        aff = db.query(Affiliate).order_by(Affiliate.name).first()
        if aff is None:
            pytest.skip("No affiliates in DB — run POST /ingest/full first")

        req = ApprovalRequest(
            kind="email",
            affiliate_id=aff.id,
            payload={"to": "test", "subject": "resolve check", "body": "x"},
            status="waiting_for_review",
        )
        db.add(req)
        db.commit()
        db.refresh(req)

        with patch("src.notifications.sender.send_email"):
            approve_request(str(req.id), db=db)

        entry = (
            db.query(AuditLog)
            .filter(
                AuditLog.record_type == "approval_request",
                AuditLog.record_id == req.id,
                AuditLog.rule_or_tool == "approvals.approve",
            )
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert entry is not None

        resolved = db.query(ApprovalRequest).filter(ApprovalRequest.id == entry.record_id).first()
        assert resolved is not None, (
            f"audit_log record_id {entry.record_id} does not exist in approval_requests"
        )
        assert resolved.id == req.id
    finally:
        if req is not None:
            db.query(AuditLog).filter(AuditLog.record_id == req.id).delete(synchronize_session=False)
            db.query(ApprovalRequest).filter(ApprovalRequest.id == req.id).delete(synchronize_session=False)
            db.commit()
        db.close()
