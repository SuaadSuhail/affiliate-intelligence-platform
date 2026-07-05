"""add has_active_leak to affiliates

Revision ID: c19c2f2fc727
Revises: 6f6b2c550b26
Create Date: 2026-07-04 16:11:23.460201

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c19c2f2fc727'
down_revision: Union[str, Sequence[str], None] = '6f6b2c550b26'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'affiliates',
        sa.Column('has_active_leak', sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('affiliates', 'has_active_leak')
