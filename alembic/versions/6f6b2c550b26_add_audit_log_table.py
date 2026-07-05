"""add audit_log table

Revision ID: 6f6b2c550b26
Revises: c44d4aa7e33a
Create Date: 2026-07-03 20:58:36.067917

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '6f6b2c550b26'
down_revision: Union[str, Sequence[str], None] = 'c44d4aa7e33a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'audit_log',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('stage', sa.String(length=20), nullable=False),
        sa.Column('record_type', sa.String(length=32), nullable=False),
        sa.Column('record_id', sa.UUID(), nullable=False),
        sa.Column('rule_or_tool', sa.String(length=64), nullable=False),
        sa.Column('input_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('output_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_audit_log_record_type_record_id', 'audit_log', ['record_type', 'record_id'], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_audit_log_record_type_record_id', table_name='audit_log')
    op.drop_table('audit_log')
