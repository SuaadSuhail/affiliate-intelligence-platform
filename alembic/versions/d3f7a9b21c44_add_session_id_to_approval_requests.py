"""add session_id to approval_requests

Revision ID: d3f7a9b21c44
Revises: 3ec97360dce2
Create Date: 2026-07-07 19:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd3f7a9b21c44'
down_revision: Union[str, Sequence[str], None] = '3ec97360dce2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: existing rows (created before this column existed) and any
    # non-chat-originated approval request (e.g. POST /approvals used
    # directly for testing) have no conversation to tie to — NULL means
    # "not part of a tracked chat session", not "unknown session".
    op.add_column(
        'approval_requests',
        sa.Column('session_id', sa.String(length=64), nullable=True),
    )
    # Nullable, left unset on creation — only written when draft_email
    # revises an existing row in place, so NULL vs non-NULL doubles as a
    # cheap "was this ever revised" signal without a separate boolean.
    op.add_column(
        'approval_requests',
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('approval_requests', 'updated_at')
    op.drop_column('approval_requests', 'session_id')
