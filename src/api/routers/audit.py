"""
Audit Router
============
Read-only access to the append-only audit_log table — traces any stored
recommendation, signal check, or approval decision back to the record and
rule/tool that produced it. See src/audit/log.py for the single write path.

GET /audit?record_type=&record_id=&stage=&limit=&offset=
"""

from __future__ import annotations

import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.auth import get_api_key
from src.storage.database import get_db
from src.storage.models import AuditLog

router = APIRouter()


class AuditLogOut(BaseModel):
    id: str
    timestamp: str
    stage: str
    record_type: str
    record_id: str
    rule_or_tool: str
    input_snapshot: dict
    output_snapshot: dict

    @classmethod
    def from_orm(cls, a: AuditLog) -> "AuditLogOut":
        return cls(
            id=str(a.id),
            timestamp=a.timestamp.isoformat() if a.timestamp else "",
            stage=a.stage,
            record_type=a.record_type,
            record_id=str(a.record_id),
            rule_or_tool=a.rule_or_tool,
            input_snapshot=a.input_snapshot,
            output_snapshot=a.output_snapshot,
        )


@router.get("", response_model=list[AuditLogOut], dependencies=[Depends(get_api_key)])
def list_audit_log(
    record_type: Optional[str] = Query(None, description="Filter by record type, e.g. 'affiliate'"),
    record_id: Optional[str] = Query(None, description="Filter by record UUID"),
    stage: Optional[str] = Query(None, description="Filter by stage: signals|rulebook|agent|approval"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[AuditLogOut]:
    """List audit_log entries, newest first, filterable by record_type/record_id/stage."""
    q = db.query(AuditLog)

    if record_type:
        q = q.filter(AuditLog.record_type == record_type)

    if record_id:
        try:
            rid = _uuid.UUID(record_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid record_id: must be a UUID")
        q = q.filter(AuditLog.record_id == rid)

    if stage:
        q = q.filter(AuditLog.stage == stage)

    q = q.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit)
    return [AuditLogOut.from_orm(a) for a in q.all()]
