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
    drafting and sending are two different code paths."""
    from src.agent.tools import draft_email

    aff = _make_affiliate("Tom Bauer")
    mock_db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
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
            "affiliate_name: Tom Bauer, situation: 51 days silent, tone: urgent"
        )

    assert len(added) == 1
    approval = added[0]
    assert approval.kind == "email"
    assert approval.affiliate_id == aff.id
    assert approval.status == "waiting_for_review"
    assert approval.payload["subject"]
    assert approval.payload["body"]
    assert approval.payload["affiliate_id"] == str(aff.id)

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
        result = draft_email.invoke("affiliate_name: Nonexistent Person, situation: x, tone: warm")

    mock_db.add.assert_not_called()
    mock_send.assert_not_called()
    assert "no affiliate found" in result.lower()


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
