"""
ChromaDB client — vector storage for affiliate communications.

Collection
----------
affiliate_comms  one document per Communication row, cosine similarity

The client is initialised lazily on first use so that importing this
module never fails even when ChromaDB is not running.

Public API
----------
add_document(...)         store / overwrite one document
search_similar(...)       semantic search with optional metadata filters
get_by_affiliate(...)     retrieve all documents for one affiliate
delete_by_affiliate(...)  remove all documents for one affiliate
"""

import os
from typing import Optional

import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8001"))

COLLECTION_NAME = "affiliate_comms"

# ── Lazy module-level state (monkeypatch-friendly for tests) ──────────────────
_client: Optional[chromadb.HttpClient] = None
_collection = None


def _get_collection():
    """
    Return the module-level ChromaDB collection, creating the client and
    the collection on the first call.

    Raises ConnectionError with a clear message if ChromaDB is unreachable.
    """
    global _client, _collection
    if _collection is not None:
        return _collection
    try:
        _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        _client.heartbeat()  # fail fast if server is not reachable
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return _collection
    except Exception as exc:
        raise ConnectionError(
            f"\n[vector_store] ✗ Could not connect to ChromaDB.\n"
            f"  Host : {CHROMA_HOST}:{CHROMA_PORT}\n"
            f"  Why  : {exc}\n\n"
            f"  Fix  : start ChromaDB with\n"
            f"         docker-compose up -d chromadb\n"
            f"         and then retry.\n"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────

def add_document(
    doc_id: str,
    text: str,
    affiliate_id: str,
    affiliate_name: str,
    source: str,
    tags: list[str],
    occurred_at: str,
) -> None:
    """
    Store (or overwrite) a communication document in ChromaDB.

    ChromaDB embeds the text automatically using the default embedding
    function (ONNXMiniLM / all-MiniLM-L6-v2).

    Parameters
    ----------
    doc_id        Stable unique identifier — use f"comm_{uuid}" to match
                  the Communication.embedding_id column.
    text          Full raw text of the communication.
    affiliate_id  UUID string of the parent Affiliate row.
    affiliate_name  Human-readable affiliate name (for display in results).
    source        One of: email | call | api_event.
    tags          List of NLP tag strings, e.g. ["churn_signal", "urgency"].
    occurred_at   ISO-8601 datetime string.
    """
    col = _get_collection()
    metadata = {
        "affiliate_id": affiliate_id,
        "affiliate_name": affiliate_name,
        "source": source,
        "tags": "|".join(tags),  # pipe-joined; ChromaDB metadata must be primitive
        "occurred_at": occurred_at,
    }
    col.upsert(
        ids=[doc_id],
        documents=[text],
        metadatas=[metadata],
    )


def search_similar(
    query: str,
    n_results: int = 5,
    filter_tags: Optional[list[str]] = None,
    filter_affiliate_id: Optional[str] = None,
) -> list[dict]:
    """
    Semantic search over all stored communications.

    Parameters
    ----------
    query               Natural-language search query.
    n_results           Maximum number of results to return.
    filter_tags         If provided, only return results that contain
                        *all* of the listed tags.  Filtering is applied
                        client-side after the vector search.
    filter_affiliate_id If provided, restrict results to one affiliate.

    Returns
    -------
    List of dicts, each with keys:
        id, document, affiliate_id, affiliate_name,
        source, tags (list), occurred_at, distance
    """
    col = _get_collection()

    # Build ChromaDB where-clause for server-side affiliate filtering.
    # Tag filtering is handled client-side (ChromaDB has no substring op).
    where: Optional[dict] = None
    if filter_affiliate_id:
        where = {"affiliate_id": {"$eq": filter_affiliate_id}}

    query_kwargs: dict = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    raw = col.query(**query_kwargs)

    results: list[dict] = []
    for i, doc_id in enumerate(raw["ids"][0]):
        meta = raw["metadatas"][0][i]
        doc_tags = [t for t in meta.get("tags", "").split("|") if t]

        # Client-side tag filter — skip if any required tag is missing
        if filter_tags and not all(t in doc_tags for t in filter_tags):
            continue

        results.append(
            {
                "id": doc_id,
                "document": raw["documents"][0][i],
                "affiliate_id": meta.get("affiliate_id"),
                "affiliate_name": meta.get("affiliate_name"),
                "source": meta.get("source"),
                "tags": doc_tags,
                "occurred_at": meta.get("occurred_at"),
                "distance": raw["distances"][0][i],
            }
        )
    return results


def get_by_affiliate(affiliate_id: str, limit: int = 20) -> list[dict]:
    """
    Return up to `limit` documents stored for one affiliate.

    Parameters
    ----------
    affiliate_id  UUID string of the affiliate.
    limit         Maximum number of documents to return (default 20).

    Returns
    -------
    List of dicts with keys: id, document, source, tags, occurred_at.
    """
    col = _get_collection()
    raw = col.get(
        where={"affiliate_id": {"$eq": affiliate_id}},
        limit=limit,
        include=["documents", "metadatas"],
    )
    results = []
    for i, doc_id in enumerate(raw["ids"]):
        meta = raw["metadatas"][i]
        doc_tags = [t for t in meta.get("tags", "").split("|") if t]
        results.append(
            {
                "id": doc_id,
                "document": raw["documents"][i],
                "source": meta.get("source"),
                "tags": doc_tags,
                "occurred_at": meta.get("occurred_at"),
            }
        )
    return results


def delete_by_affiliate(affiliate_id: str) -> None:
    """
    Remove all documents belonging to one affiliate.

    Called when an affiliate is deleted from PostgreSQL (the cascade
    removes relational data; this cleans the vector store).

    Parameters
    ----------
    affiliate_id  UUID string of the affiliate whose documents to remove.
    """
    col = _get_collection()
    col.delete(where={"affiliate_id": {"$eq": affiliate_id}})