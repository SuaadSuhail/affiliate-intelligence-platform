"""add seo_signals table and affiliate seo fields

Revision ID: 3ec97360dce2
Revises: c19c2f2fc727
Create Date: 2026-07-04 17:17:27.419157

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3ec97360dce2'
down_revision: Union[str, Sequence[str], None] = 'c19c2f2fc727'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'seo_signals',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('affiliate_id', sa.UUID(), nullable=False),
        sa.Column('keyword', sa.String(length=255), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('rank_change', sa.Integer(), nullable=True),
        sa.Column('search_volume', sa.Integer(), nullable=True),
        sa.Column('checked_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['affiliate_id'], ['affiliates.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_seo_signals_affiliate_id', 'seo_signals', ['affiliate_id'], unique=False)
    op.create_index('ix_seo_signals_keyword', 'seo_signals', ['keyword'], unique=False)

    op.add_column('affiliates', sa.Column('tracked_keyword', sa.String(length=255), nullable=True))
    op.add_column(
        'affiliates',
        sa.Column('search_trend', sa.String(length=20), nullable=False, server_default='stable'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('affiliates', 'search_trend')
    op.drop_column('affiliates', 'tracked_keyword')
    op.drop_index('ix_seo_signals_keyword', table_name='seo_signals')
    op.drop_index('ix_seo_signals_affiliate_id', table_name='seo_signals')
    op.drop_table('seo_signals')
