"""initial schema - affiliates communications score_history

Revision ID: 13ea16583831
Revises:
Create Date: 2026-06-13 21:31:58.829333

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision: str = '13ea16583831'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ENUM types
    affiliate_status = postgresql.ENUM(
        'active', 'at_risk', 'churned', 'high_growth',
        name='affiliate_status',
        create_type=False,
    )
    affiliate_status.create(op.get_bind(), checkfirst=True)

    communication_source = postgresql.ENUM(
        'email', 'call', 'api_event',
        name='communication_source',
        create_type=False,
    )
    communication_source.create(op.get_bind(), checkfirst=True)

    # affiliates
    op.create_table(
        'affiliates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('status', sa.Enum('active', 'at_risk', 'churned', 'high_growth',
                                    name='affiliate_status'),
                  nullable=False, server_default='active'),
        sa.Column('churn_risk_score', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('growth_potential_score', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('health_score', sa.Float(), nullable=False, server_default='50.0'),
        sa.Column('revenue_30d', sa.Numeric(10, 2), nullable=False, server_default='0.00'),
        sa.Column('ctr_trend_pct', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('last_contact_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('days_since_contact', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()')),
    )
    op.create_index('ix_affiliates_status', 'affiliates', ['status'])
    op.create_index('ix_affiliates_churn_risk', 'affiliates', ['churn_risk_score'])
    op.create_index('ix_affiliates_growth', 'affiliates', ['growth_potential_score'])

    # communications
    op.create_table(
        'communications',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('affiliate_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('affiliates.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source', sa.Enum('email', 'call', 'api_event',
                                    name='communication_source'), nullable=False),
        sa.Column('raw_text', sa.Text(), nullable=False),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=False,
                  server_default='{}'),
        sa.Column('sentiment_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('embedding_id', sa.String(255), nullable=True),
        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_comms_affiliate_id', 'communications', ['affiliate_id'])
    op.create_index('ix_comms_occurred_at', 'communications', ['occurred_at'])
    op.create_index('ix_comms_source', 'communications', ['source'])

    # score_history
    op.create_table(
        'score_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('affiliate_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('affiliates.id', ondelete='CASCADE'), nullable=False),
        sa.Column('churn_risk_score', sa.Float(), nullable=False),
        sa.Column('growth_potential_score', sa.Float(), nullable=False),
        sa.Column('health_score', sa.Float(), nullable=False),
        sa.Column('scored_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()')),
    )
    op.create_index('ix_score_history_affiliate_id', 'score_history', ['affiliate_id'])
    op.create_index('ix_score_history_scored_at', 'score_history', ['scored_at'])


def downgrade() -> None:
    op.drop_table('score_history')
    op.drop_table('communications')
    op.drop_table('affiliates')
    op.execute("DROP TYPE IF EXISTS communication_source")
    op.execute("DROP TYPE IF EXISTS affiliate_status")