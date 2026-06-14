"""add pgvector extension and embeddings table

Revision ID: ccc1c19d5237
Revises: 13ea16583831
Create Date: 2026-06-14 17:28:08.076215

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ccc1c19d5237'
down_revision: Union[str, Sequence[str], None] = '13ea16583831'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            affiliate_id UUID REFERENCES affiliates(id) ON DELETE CASCADE,
            affiliate_name TEXT,
            source TEXT,
            chunk_text TEXT,
            tags TEXT[],
            occurred_at TIMESTAMPTZ,
            embedding vector(384)
        )
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10)
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS embeddings_affiliate_id_idx
        ON embeddings (affiliate_id)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS embeddings_affiliate_id_idx")
    op.execute("DROP INDEX IF EXISTS embeddings_vector_idx")
    op.execute("DROP TABLE IF EXISTS embeddings")
    op.execute("DROP EXTENSION IF EXISTS vector")