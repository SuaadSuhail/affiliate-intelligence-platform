"""
Audit log — append-only record linking a stored decision/outcome back to the
specific inputs and rule/tool that produced it.

write_audit_entry() is deliberately dumb: it builds one AuditLog row and
adds() it to the given session — no logic, no decisions about what or when
to log. It does not commit; the caller already owns the transaction, same
as every other insert in this codebase (e.g. score_history rows in
src.ml.score_updater are add()-ed and committed by the caller too).

Call sites (exactly four — see each module for why):
  - src.ml.score_updater      (stage="rulebook", after recommend())
  - src.scraping.leakage_scraper (stage="signals", after a scan completes)
  - src.seo.checker           (stage="signals", after a rank check completes)
  - src.api.routers.approvals (stage="approval", on approve/reject)
"""

from __future__ import annotations

import uuid as _uuid

from sqlalchemy.orm import Session

from src.storage.models import AuditLog


def write_audit_entry(
    db: Session,
    stage: str,
    record_type: str,
    record_id,
    rule_or_tool: str,
    input_snapshot: dict,
    output_snapshot: dict,
) -> AuditLog:
    """Insert one audit_log row. Caller commits. record_id may be a UUID or a UUID string."""
    if isinstance(record_id, str):
        record_id = _uuid.UUID(record_id)

    entry = AuditLog(
        stage=stage,
        record_type=record_type,
        record_id=record_id,
        rule_or_tool=rule_or_tool,
        input_snapshot=input_snapshot,
        output_snapshot=output_snapshot,
    )
    db.add(entry)
    return entry
