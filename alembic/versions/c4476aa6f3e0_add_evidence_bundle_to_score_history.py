"""add evidence_bundle to score_history

Revision ID: c4476aa6f3e0
Revises: 339d15d8734c
Create Date: 2026-07-03 20:03:51.774024

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c4476aa6f3e0'
down_revision: Union[str, Sequence[str], None] = '339d15d8734c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'score_history',
        sa.Column('evidence_bundle', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('score_history', 'evidence_bundle')
