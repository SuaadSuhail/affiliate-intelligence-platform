"""
Approvals Router
================
Human-in-the-loop approval queue. Every action that would leave the system
(send email, notify partner, ...) is created here with
status='waiting_for_review' and stays inert until a human approves or
rejects it — nothing external fires without that.

POST /approvals               — create a request (used internally by tools; exposed for testing)
GET  /approvals                — list, filterable by ?status=
POST /approvals/{id}/approve  — approve; triggers the real action (the only path that does)
POST /approvals/{id}/reject   — reject; nothing fires
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.auth import get_api_key
from src.audit.log import write_audit_entry
from src.core.logging_config import get_logger
from src.storage.database import get_db
from src.storage.models import ApprovalRequest

logger = get_logger(__name__)
router = APIRouter()

_VALID_STATUSES = {"waiting_for_review", "approved", "rejected"}

# There is no per-user identity system in this app — src.api.auth.get_api_key
# only confirms a valid shared API key was presented, it doesn't identify who
# holds it. Storing the raw key value in decided_by would echo a secret into
# stored data, so a fixed placeholder is used instead until real auth exists.
_DECIDED_BY_PLACEHOLDER = "api"


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class ApprovalRequestIn(BaseModel):
    kind: str
    affiliate_id: str
    payload: dict


class ApprovalRequestOut(BaseModel):
    id: str
    kind: str
    affiliate_id: str
    payload: dict
    status: str
    created_at: str
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None

    @classmethod
    def from_orm(cls, r: ApprovalRequest) -> "ApprovalRequestOut":
        return cls(
            id=str(r.id),
            kind=r.kind,
            affiliate_id=str(r.affiliate_id),
            payload=r.payload,
            status=r.status,
            created_at=r.created_at.isoformat() if r.created_at else "",
            decided_at=r.decided_at.isoformat() if r.decided_at else None,
            decided_by=r.decided_by,
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_request_or_404(request_id: str, db: Session) -> ApprovalRequest:
    try:
        rid = _uuid.UUID(request_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request id: must be a UUID")
    req = db.query(ApprovalRequest).filter(ApprovalRequest.id == rid).first()
    if not req:
        raise HTTPException(status_code=404, detail=f"Approval request {request_id} not found")
    return req


# ─── POST /approvals ──────────────────────────────────────────────────────────

@router.post("", response_model=ApprovalRequestOut, dependencies=[Depends(get_api_key)])
def create_approval_request(
    body: ApprovalRequestIn,
    db: Session = Depends(get_db),
) -> ApprovalRequestOut:
    """
    Create a pending approval request. Not meant for routine external use —
    tools such as draft_email (src/agent/tools.py) call this internally when
    they compose a draft. Exposed as its own endpoint so the approve/reject
    lifecycle can be tested and demoed directly.
    """
    try:
        aff_id = _uuid.UUID(body.affiliate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid affiliate_id: must be a UUID")

    req = ApprovalRequest(
        kind=body.kind,
        affiliate_id=aff_id,
        payload=body.payload,
        status="waiting_for_review",
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    logger.info(
        "Approval request created",
        extra={"id": str(req.id), "kind": req.kind, "affiliate_id": str(aff_id)},
    )
    return ApprovalRequestOut.from_orm(req)


# ─── GET /approvals ───────────────────────────────────────────────────────────

@router.get("", response_model=list[ApprovalRequestOut])
def list_approval_requests(
    status: Optional[str] = None,
    db: Session = Depends(get_db),
) -> list[ApprovalRequestOut]:
    """List approval requests, newest first, optionally filtered by status."""
    if status is not None and status not in _VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status filter '{status}'. Must be one of {sorted(_VALID_STATUSES)}",
        )
    q = db.query(ApprovalRequest).order_by(ApprovalRequest.created_at.desc())
    if status:
        q = q.filter(ApprovalRequest.status == status)
    return [ApprovalRequestOut.from_orm(r) for r in q.all()]


# ─── POST /approvals/{id}/approve ─────────────────────────────────────────────

@router.post(
    "/{request_id}/approve", response_model=ApprovalRequestOut, dependencies=[Depends(get_api_key)]
)
def approve_request(request_id: str, db: Session = Depends(get_db)) -> ApprovalRequestOut:
    """
    Approve a pending request and trigger the real action. This is the only
    path in the codebase that calls src.notifications.sender — nothing else
    is permitted to fire an action that leaves the system.
    """
    req = _get_request_or_404(request_id, db)
    if req.status != "waiting_for_review":
        raise HTTPException(
            status_code=409,
            detail=f"Request {request_id} is already '{req.status}', not waiting_for_review",
        )

    if req.kind == "email":
        from src.notifications.sender import send_email
        send_email(req.payload)
    else:
        logger.warning(
            "Approved request has an unrecognised kind — no sender wired up for it",
            extra={"id": str(req.id), "kind": req.kind},
        )

    req.status = "approved"
    req.decided_at = datetime.now(timezone.utc)
    req.decided_by = _DECIDED_BY_PLACEHOLDER

    write_audit_entry(
        db,
        stage="approval",
        record_type="approval_request",
        record_id=req.id,
        rule_or_tool="approvals.approve",
        input_snapshot=req.payload,
        output_snapshot={
            "status": req.status,
            "decided_by": req.decided_by,
            "decided_at": req.decided_at.isoformat(),
        },
    )

    db.commit()
    db.refresh(req)
    logger.info("Approval request approved", extra={"id": str(req.id), "kind": req.kind})
    return ApprovalRequestOut.from_orm(req)


# ─── POST /approvals/{id}/reject ──────────────────────────────────────────────

@router.post(
    "/{request_id}/reject", response_model=ApprovalRequestOut, dependencies=[Depends(get_api_key)]
)
def reject_request(request_id: str, db: Session = Depends(get_db)) -> ApprovalRequestOut:
    """Reject a pending request. Nothing fires."""
    req = _get_request_or_404(request_id, db)
    if req.status != "waiting_for_review":
        raise HTTPException(
            status_code=409,
            detail=f"Request {request_id} is already '{req.status}', not waiting_for_review",
        )

    req.status = "rejected"
    req.decided_at = datetime.now(timezone.utc)
    req.decided_by = _DECIDED_BY_PLACEHOLDER

    write_audit_entry(
        db,
        stage="approval",
        record_type="approval_request",
        record_id=req.id,
        rule_or_tool="approvals.reject",
        input_snapshot=req.payload,
        output_snapshot={
            "status": req.status,
            "decided_by": req.decided_by,
            "decided_at": req.decided_at.isoformat(),
        },
    )

    db.commit()
    db.refresh(req)
    logger.info("Approval request rejected", extra={"id": str(req.id)})
    return ApprovalRequestOut.from_orm(req)
