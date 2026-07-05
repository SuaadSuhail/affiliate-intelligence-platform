"""
Notification senders — the only code in this repo allowed to take an action
that leaves the system.

send_email() must be called ONLY from POST /approvals/{id}/approve
(src/api/routers/approvals.py). It is not imported anywhere else — not from
src/agent/tools.py, not from any scheduled job. A drafted email sits in
approval_requests as status='waiting_for_review' until a human approves it;
only that approval path may call this module.
"""

from __future__ import annotations

from src.core.logging_config import get_logger

logger = get_logger(__name__)


def send_email(payload: dict) -> None:
    """
    Placeholder sender — no real email provider is integrated yet.
    Logs what would be sent via the existing structured logger.
    """
    logger.info(
        "Would send email (no real provider configured)",
        extra={
            "to": payload.get("to"),
            "subject": payload.get("subject"),
            "affiliate_id": payload.get("affiliate_id"),
        },
    )
