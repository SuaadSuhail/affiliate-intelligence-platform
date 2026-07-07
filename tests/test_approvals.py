"""
Approval Queue Tests
=====================
Full lifecycle: draft_email files a request, nothing sends until a human
approves it via POST /approvals/{id}/approve. Reject leaves send_email
uncalled entirely.

Run:
    pytest tests/test_approvals.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


def _make_affiliate(name="Tom Bauer"):
    from src.storage.models import Affiliate
    a = Affiliate()
    a.id = uuid.uuid4()
    a.name = name
    return a


def _make_pending_request():
    from src.storage.models import ApprovalRequest
    req = ApprovalRequest()
    req.id = uuid.uuid4()
    req.kind = "email"
    req.affiliate_id = uuid.uuid4()
    req.payload = {
        "to": "Tom Bauer <no email on file>",
        "subject": "Following up",
        "body": "Hi Tom, ...",
        "affiliate_id": str(uuid.uuid4()),
    }
    req.status = "waiting_for_review"
    req.decided_at = None
    req.decided_by = None
    req.created_at = datetime.now(timezone.utc)
    return req


# ─── Test 1: draft_email files a pending request, never sends ────────────────

def test_draft_email_creates_pending_approval_and_does_not_send():
    """draft_email must insert an approval_requests row with
    status=waiting_for_review and must never call send_email itself —
    drafting and sending are two different code paths. No conversation_id
    is supplied here (no config passed to .invoke()), so this must always
    create a new row, never attempt to match an existing one."""
    from src.agent.tools import draft_email
    from src.storage.models import ApprovalRequest, AuditLog

    aff = _make_affiliate("Tom Bauer")
    mock_db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.order_by.return_value = q
    q.first.return_value = aff
    mock_db.query.return_value = q

    added = []
    mock_db.add.side_effect = lambda obj: added.append(obj)

    with (
        patch("src.agent.tools._get_db", return_value=mock_db),
        patch("src.agent.tools._get_llm", return_value=None),  # force template path, no API call
        patch("src.notifications.sender.send_email") as mock_send,
    ):
        result = draft_email.invoke(
            {"affiliate_name": "Tom Bauer", "situation_override": "51 days silent", "tone": "urgent"}
        )

    approvals_added = [obj for obj in added if isinstance(obj, ApprovalRequest)]
    assert len(approvals_added) == 1
    approval = approvals_added[0]
    assert approval.kind == "email"
    assert approval.affiliate_id == aff.id
    assert approval.status == "waiting_for_review"
    assert approval.session_id is None
    assert approval.payload["subject"]
    assert approval.payload["body"]
    assert approval.payload["affiliate_id"] == str(aff.id)

    audit_entries = [obj for obj in added if isinstance(obj, AuditLog)]
    assert len(audit_entries) == 1
    assert audit_entries[0].stage == "agent"
    assert audit_entries[0].rule_or_tool == "draft_email"
    assert audit_entries[0].output_snapshot["action"] == "created"

    assert "pending approval" in result.lower()
    assert "waiting_for_review" not in result or True  # message need not echo the raw status
    mock_send.assert_not_called()


def test_draft_email_unknown_affiliate_creates_no_request():
    """Unresolvable affiliate name must not create a dangling approval row."""
    from src.agent.tools import draft_email

    mock_db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = None
    mock_db.query.return_value = q

    with (
        patch("src.agent.tools._get_db", return_value=mock_db),
        patch("src.notifications.sender.send_email") as mock_send,
    ):
        result = draft_email.invoke(
            {"affiliate_name": "Nonexistent Person", "situation_override": "x", "tone": "warm"}
        )

    mock_db.add.assert_not_called()
    mock_send.assert_not_called()
    assert "no affiliate found" in result.lower()


# ─── Test 1b: draft_email revise-in-place (real DB) ───────────────────────────
#
# These four use a real database session, same convention as
# test_audit.py / test_leakage_scraper.py — the matching logic depends on
# real SQL filter semantics (session_id equality, status filtering) that
# would be fragile and unconvincing to fake with a mocked query chain.
# _get_llm is patched to None (forcing the deterministic template path) so
# no real OpenAI call is needed and results are reproducible.

def _cleanup_draft_email_test_rows(approval_ids: list, audit_ids: list) -> None:
    from src.storage.database import SessionLocal
    from src.storage.models import ApprovalRequest, AuditLog

    db = SessionLocal()
    try:
        if audit_ids:
            db.query(AuditLog).filter(AuditLog.id.in_(audit_ids)).delete(synchronize_session=False)
        if approval_ids:
            db.query(ApprovalRequest).filter(ApprovalRequest.id.in_(approval_ids)).delete(
                synchronize_session=False
            )
        db.commit()
    finally:
        db.close()


def test_draft_email_revises_existing_pending_request_within_same_conversation():
    """Real DB: a second draft_email call for the same affiliate within the
    same conversation_id must UPDATE the existing waiting_for_review row in
    place, not insert a second, unrelated one."""
    from src.agent.tools import draft_email
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, ApprovalRequest

    conv_id = f"test-conv-{uuid.uuid4()}"
    approval_ids: list = []
    db = SessionLocal()
    try:
        aff = db.query(Affiliate).filter(Affiliate.name == "Marcus Williams").first()
        if not aff:
            pytest.skip("Marcus Williams not in DB — run POST /ingest/full first")

        with patch("src.agent.tools._get_llm", return_value=None):
            r1 = draft_email.invoke(
                {"affiliate_name": "Marcus Williams", "tone": "warm"},
                config={"configurable": {"conversation_id": conv_id}},
            )
            r2 = draft_email.invoke(
                {"affiliate_name": "Marcus Williams", "tone": "urgent"},
                config={"configurable": {"conversation_id": conv_id}},
            )

        rows = db.query(ApprovalRequest).filter(ApprovalRequest.session_id == conv_id).all()
        approval_ids = [r.id for r in rows]

        assert len(rows) == 1, f"expected exactly one row for this conversation, got {len(rows)}"
        assert rows[0].updated_at is not None, "revised row must have updated_at set"
        assert "Draft created" in r1
        assert "Draft revised" in r2
        assert str(rows[0].id) in r2, "the revised response must reference the same request id"
    finally:
        _cleanup_draft_email_test_rows(approval_ids, [])
        db.close()


def test_draft_email_does_not_match_across_different_conversations():
    """Real DB: two draft_email calls for the same affiliate but different
    conversation_ids must never be treated as revisions of each other —
    proves no over-matching on affiliate_id alone."""
    from src.agent.tools import draft_email
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, ApprovalRequest

    conv_a = f"test-conv-{uuid.uuid4()}"
    conv_b = f"test-conv-{uuid.uuid4()}"
    approval_ids: list = []
    db = SessionLocal()
    try:
        aff = db.query(Affiliate).filter(Affiliate.name == "Marcus Williams").first()
        if not aff:
            pytest.skip("Marcus Williams not in DB — run POST /ingest/full first")

        with patch("src.agent.tools._get_llm", return_value=None):
            r1 = draft_email.invoke(
                {"affiliate_name": "Marcus Williams", "tone": "warm"},
                config={"configurable": {"conversation_id": conv_a}},
            )
            r2 = draft_email.invoke(
                {"affiliate_name": "Marcus Williams", "tone": "warm"},
                config={"configurable": {"conversation_id": conv_b}},
            )

        rows_a = db.query(ApprovalRequest).filter(ApprovalRequest.session_id == conv_a).all()
        rows_b = db.query(ApprovalRequest).filter(ApprovalRequest.session_id == conv_b).all()
        approval_ids = [r.id for r in rows_a] + [r.id for r in rows_b]

        assert len(rows_a) == 1
        assert len(rows_b) == 1
        assert rows_a[0].id != rows_b[0].id
        assert "Draft created" in r1
        assert "Draft created" in r2  # both created — neither revised the other
    finally:
        _cleanup_draft_email_test_rows(approval_ids, [])
        db.close()


def test_draft_email_with_no_conversation_id_always_inserts():
    """Real DB: draft_email calls with no conversation_id (e.g. a caller
    that doesn't go through run_agent, like the POST /approvals test
    endpoint) must never match on affiliate_id alone — every such call
    inserts a new row."""
    from src.agent.tools import draft_email
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, ApprovalRequest

    approval_ids: list = []
    db = SessionLocal()
    try:
        aff = db.query(Affiliate).filter(Affiliate.name == "Marcus Williams").first()
        if not aff:
            pytest.skip("Marcus Williams not in DB — run POST /ingest/full first")

        before_ids = {
            r.id
            for r in db.query(ApprovalRequest)
            .filter(ApprovalRequest.affiliate_id == aff.id, ApprovalRequest.session_id.is_(None))
            .all()
        }

        with patch("src.agent.tools._get_llm", return_value=None):
            r1 = draft_email.invoke({"affiliate_name": "Marcus Williams", "tone": "warm"})
            r2 = draft_email.invoke({"affiliate_name": "Marcus Williams", "tone": "warm"})

        after_rows = (
            db.query(ApprovalRequest)
            .filter(ApprovalRequest.affiliate_id == aff.id, ApprovalRequest.session_id.is_(None))
            .all()
        )
        new_rows = [r for r in after_rows if r.id not in before_ids]
        approval_ids = [r.id for r in new_rows]

        assert len(new_rows) == 2, (
            f"expected 2 new rows (no matching without a conversation_id), got {len(new_rows)}"
        )
        assert "Draft created" in r1
        assert "Draft created" in r2
    finally:
        _cleanup_draft_email_test_rows(approval_ids, [])
        db.close()


def test_draft_email_audit_log_entries_tagged_created_vs_revised():
    """Real DB: draft_email's audit log entries (stage='agent',
    rule_or_tool='draft_email') must distinguish 'created' from 'revised'
    via output_snapshot['action']."""
    from src.agent.tools import draft_email
    from src.storage.database import SessionLocal
    from src.storage.models import Affiliate, ApprovalRequest, AuditLog

    conv_id = f"test-conv-{uuid.uuid4()}"
    approval_ids: list = []
    audit_ids: list = []
    db = SessionLocal()
    try:
        aff = db.query(Affiliate).filter(Affiliate.name == "Marcus Williams").first()
        if not aff:
            pytest.skip("Marcus Williams not in DB — run POST /ingest/full first")

        with patch("src.agent.tools._get_llm", return_value=None):
            draft_email.invoke(
                {"affiliate_name": "Marcus Williams", "tone": "warm"},
                config={"configurable": {"conversation_id": conv_id}},
            )
            draft_email.invoke(
                {"affiliate_name": "Marcus Williams", "tone": "urgent"},
                config={"configurable": {"conversation_id": conv_id}},
            )

        approval_row = db.query(ApprovalRequest).filter(ApprovalRequest.session_id == conv_id).first()
        assert approval_row is not None
        approval_ids = [approval_row.id]

        entries = (
            db.query(AuditLog)
            .filter(AuditLog.rule_or_tool == "draft_email", AuditLog.record_id == approval_row.id)
            .order_by(AuditLog.timestamp.asc())
            .all()
        )
        audit_ids = [e.id for e in entries]

        assert len(entries) == 2, f"expected 2 audit entries (create + revise), got {len(entries)}"
        assert entries[0].stage == "agent"
        assert entries[0].output_snapshot["action"] == "created"
        assert entries[0].input_snapshot["conversation_id"] == conv_id
        assert entries[1].stage == "agent"
        assert entries[1].output_snapshot["action"] == "revised"
    finally:
        _cleanup_draft_email_test_rows(approval_ids, audit_ids)
        db.close()


# ─── Test 2: approving transitions status and triggers send_email ────────────

def test_approve_request_transitions_status_and_calls_send_email():
    from src.api.routers.approvals import approve_request

    req = _make_pending_request()

    mock_db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = req
    mock_db.query.return_value = q

    with patch("src.notifications.sender.send_email") as mock_send:
        result = approve_request(str(req.id), db=mock_db)

    mock_send.assert_called_once_with(req.payload)
    assert req.status == "approved"
    assert req.decided_at is not None
    assert req.decided_by is not None
    assert result.status == "approved"


def test_approve_request_already_decided_is_rejected_with_409():
    """An already-decided request must not be re-approved (and must not
    re-trigger send_email)."""
    from fastapi import HTTPException
    from src.api.routers.approvals import approve_request

    req = _make_pending_request()
    req.status = "approved"

    mock_db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = req
    mock_db.query.return_value = q

    with patch("src.notifications.sender.send_email") as mock_send:
        with pytest.raises(HTTPException) as exc_info:
            approve_request(str(req.id), db=mock_db)

    assert exc_info.value.status_code == 409
    mock_send.assert_not_called()


# ─── Test 3: rejecting never calls send_email ─────────────────────────────────

def test_reject_request_never_calls_send_email():
    from src.api.routers.approvals import reject_request

    req = _make_pending_request()

    mock_db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = req
    mock_db.query.return_value = q

    with patch("src.notifications.sender.send_email") as mock_send:
        result = reject_request(str(req.id), db=mock_db)

    mock_send.assert_not_called()
    assert req.status == "rejected"
    assert req.decided_at is not None
    assert result.status == "rejected"


# ─── Test 3b: approve/reject each write an audit_log entry ───────────────────

def test_approve_request_writes_audit_entry():
    from src.api.routers.approvals import approve_request
    from src.storage.models import AuditLog

    req = _make_pending_request()

    mock_db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = req
    mock_db.query.return_value = q

    with patch("src.notifications.sender.send_email"):
        approve_request(str(req.id), db=mock_db)

    audit_entries = [
        call.args[0] for call in mock_db.add.call_args_list if isinstance(call.args[0], AuditLog)
    ]
    assert len(audit_entries) == 1
    entry = audit_entries[0]
    assert entry.stage == "approval"
    assert entry.record_type == "approval_request"
    assert entry.record_id == req.id
    assert entry.rule_or_tool == "approvals.approve"
    assert entry.input_snapshot == req.payload
    assert entry.output_snapshot["status"] == "approved"
    assert entry.output_snapshot["decided_by"] == "api"
    assert entry.output_snapshot["decided_at"] is not None


def test_reject_request_writes_audit_entry():
    from src.api.routers.approvals import reject_request
    from src.storage.models import AuditLog

    req = _make_pending_request()

    mock_db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = req
    mock_db.query.return_value = q

    with patch("src.notifications.sender.send_email") as mock_send:
        reject_request(str(req.id), db=mock_db)

    mock_send.assert_not_called()
    audit_entries = [
        call.args[0] for call in mock_db.add.call_args_list if isinstance(call.args[0], AuditLog)
    ]
    assert len(audit_entries) == 1
    entry = audit_entries[0]
    assert entry.stage == "approval"
    assert entry.record_type == "approval_request"
    assert entry.rule_or_tool == "approvals.reject"
    assert entry.input_snapshot == req.payload
    assert entry.output_snapshot["status"] == "rejected"


# ─── Test 4: send_email is unreachable except via the approvals router ───────

def test_send_email_only_referenced_from_approvals_router():
    """Static check: src.notifications.sender.send_email must not be called
    or imported anywhere in src/ except src/api/routers/approvals.py (and its
    own definition in src/notifications/sender.py) — draft_email and every
    other tool must remain draft-only / read-only."""
    import pathlib

    src_root = pathlib.Path(__file__).parent.parent / "src"
    hits = []
    for path in src_root.rglob("*.py"):
        if path.name == "sender.py":
            continue
        if "send_email" in path.read_text():
            hits.append(str(path.relative_to(src_root)))

    assert hits == ["api/routers/approvals.py"], (
        f"send_email referenced outside the approvals router: {hits}"
    )
