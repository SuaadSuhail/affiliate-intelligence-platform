"""
Embedding Generator
===================
Converts communication text to 384-dim vectors using
sentence-transformers/all-MiniLM-L6-v2 and stores them in ChromaDB
with full metadata.  Only records where embedding_id IS NULL are processed,
making the pipeline safe to re-run at any time.

Usage
-----
    from src.ingestion.embedding_generator import embed_all_communications
    from src.storage.database import db_session
    from src.storage.vector_store import vector_store

    with db_session() as db:
        result = embed_all_communications(db, vector_store)
        print(result)
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise RuntimeError(
        "sentence-transformers model not found. "
        "Run: pip install sentence-transformers"
    ) from exc

from src.storage.models import Affiliate, Communication
from src.storage.vector_store import VectorStore, vector_store

# ─── Model: loaded once at module level ──────────────────────────────────────

model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")

# ─── Chunking ─────────────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Parameters
    ----------
    text       : raw text to split
    chunk_size : words per chunk
    overlap    : words shared between consecutive chunks

    Returns
    -------
    List of chunk strings.  If the text is shorter than chunk_size it is
    returned as a single-element list with no splitting.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


# ─── Core embedding functions ─────────────────────────────────────────────────


def embed_communication(
    comm: Communication,
    db: Session,
    vs: VectorStore,
) -> dict:
    """
    Embed a single communication record and store all chunks in ChromaDB.

    Steps
    -----
    1. Look up the affiliate name from PostgreSQL
    2. Chunk the raw_text
    3. Encode each chunk and upsert to ChromaDB via vs.add_document()
    4. Write the first chunk's doc_id back to comm.embedding_id

    Parameters
    ----------
    comm : Communication ORM instance (must be in the current db session)
    db   : active SQLAlchemy session
    vs   : VectorStore instance

    Returns
    -------
    {comm_id, chunks_created, embedding_id}
    """
    # 1. Affiliate name
    aff: Optional[Affiliate] = (
        db.query(Affiliate).filter(Affiliate.id == comm.affiliate_id).first()
    )
    affiliate_name = aff.name if aff else "Unknown"

    # 2. Chunk
    chunks = chunk_text(comm.raw_text or "")

    # 3. Embed + store
    first_doc_id: Optional[str] = None
    for i, chunk in enumerate(chunks):
        doc_id = f"{comm.id}_chunk_{i}"
        embedding = model.encode(chunk).tolist()
        vs.add_document(
            doc_id=doc_id,
            text=chunk,
            embedding=embedding,
            affiliate_id=str(comm.affiliate_id),
            affiliate_name=affiliate_name,
            source=comm.source,
            tags=comm.tags or [],
            occurred_at=str(comm.occurred_at),
        )
        if i == 0:
            first_doc_id = doc_id

    # 4. Persist embedding_id
    comm.embedding_id = first_doc_id

    return {
        "comm_id": str(comm.id),
        "chunks_created": len(chunks),
        "embedding_id": first_doc_id,
    }


def embed_all_communications(
    db: Session,
    vs: VectorStore,
) -> dict:
    """
    Embed every communication that has no embedding_id yet.

    Only processes records where embedding_id IS NULL — safe to re-run.
    Caller is responsible for committing the session.

    Parameters
    ----------
    db : active SQLAlchemy session
    vs : VectorStore instance

    Returns
    -------
    {total_processed, total_chunks_created, already_embedded}
    """
    already_embedded = (
        db.query(Communication)
        .filter(Communication.embedding_id.isnot(None))
        .count()
    )
    unembedded = (
        db.query(Communication)
        .filter(Communication.embedding_id.is_(None))
        .all()
    )

    total_chunks = 0
    for comm in unembedded:
        result = embed_communication(comm, db, vs)
        total_chunks += result["chunks_created"]

    return {
        "total_processed": len(unembedded),
        "total_chunks_created": total_chunks,
        "already_embedded": already_embedded,
    }