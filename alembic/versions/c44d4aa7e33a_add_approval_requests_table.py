"""add approval_requests table

Revision ID: c44d4aa7e33a
Revises: c4476aa6f3e0
Create Date: 2026-07-03 20:19:19.691372

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c44d4aa7e33a'
down_revision: Union[str, Sequence[str], None] = 'c4476aa6f3e0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'approval_requests',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('kind', sa.String(length=32), nullable=False),
        sa.Column('affiliate_id', sa.UUID(), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('decided_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('decided_by', sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(['affiliate_id'], ['affiliates.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_approval_requests_affiliate_id', 'approval_requests', ['affiliate_id'], unique=False
    )
    op.create_index(
        'ix_approval_requests_status', 'approval_requests', ['status'], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_approval_requests_status', table_name='approval_requests')
    op.drop_index('ix_approval_requests_affiliate_id', table_name='approval_requests')
    op.drop_table('approval_requests')
