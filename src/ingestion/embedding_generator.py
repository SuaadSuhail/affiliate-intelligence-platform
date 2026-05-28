"""
Embedding Generator
===================
Converts text to 384-dim vectors using sentence-transformers/all-MiniLM-L6-v2
and provides index helpers for ChromaDB via the vector_store module.

ChromaDB's default embedding function (ONNXMiniLM / all-MiniLM-L6-v2) handles
upserts automatically, so index_communication() delegates directly to
vector_store.add_document() rather than pre-computing vectors.  The embed() /
embed_batch() methods remain available for tasks that need raw vectors
(feature engineering, external ranking, etc.).

Usage
-----
    from src.ingestion.embedding_generator import get_generator

    gen = get_generator()
    doc_id = gen.index_communication(
        comm_id="abc-123", text="...", affiliate_id="...",
        affiliate_name="Priya Sharma", source="email",
        tags=["growth_intent"], occurred_at="2026-05-19T09:14:00",
    )
    results = gen.search_communications("payment issue", n_results=5)
"""

import os
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import src.storage.vector_store as vs

load_dotenv()

MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


class EmbeddingGenerator:
    """
    Wraps a SentenceTransformer model and provides index helpers for ChromaDB.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        print(f"[embedding_generator] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    # ── Raw embedding helpers ──────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Return a 384-dim embedding for a single piece of text."""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode a list of texts. More efficient than looping embed()."""
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return embeddings.tolist()

    # ── Communication indexing ─────────────────────────────────────────────────

    def index_communication(
        self,
        comm_id: str,
        text: str,
        affiliate_id: str,
        affiliate_name: str,
        source: str,           # email | call | api_event
        tags: list[str],
        occurred_at: str,
    ) -> str:
        """
        Upsert a communication document into the affiliate_comms ChromaDB
        collection.  ChromaDB embeds the text automatically via its built-in
        ONNXMiniLM function — no manual embedding step needed here.

        Parameters
        ----------
        comm_id        : UUID of the Communication row
        text           : full raw text of the communication
        affiliate_id   : UUID string of the parent Affiliate row
        affiliate_name : human-readable affiliate name (displayed in search results)
        source         : one of email | call | api_event
        tags           : list of NLP tag strings
        occurred_at    : ISO-8601 datetime string

        Returns the ChromaDB document ID (``f"comm_{comm_id}"``).
        """
        doc_id = f"comm_{comm_id}"
        vs.add_document(
            doc_id=doc_id,
            text=text,
            affiliate_id=affiliate_id,
            affiliate_name=affiliate_name,
            source=source,
            tags=tags,
            occurred_at=occurred_at,
        )
        return doc_id

    # ── Semantic search ────────────────────────────────────────────────────────

    def search_communications(
        self,
        query: str,
        n_results: int = 5,
        affiliate_id: Optional[str] = None,
        filter_tags: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Semantic search over indexed communications.

        Parameters
        ----------
        query        : natural-language search query
        n_results    : number of results to return
        affiliate_id : optional UUID string — restrict to one affiliate
        filter_tags  : optional list of tags; all must be present (client-side)

        Returns a list of dicts with keys:
            id, document, affiliate_id, affiliate_name, source, tags,
            occurred_at, distance
        """
        return vs.search_similar(
            query=query,
            n_results=n_results,
            filter_tags=filter_tags,
            filter_affiliate_id=affiliate_id,
        )


# ── Module-level singleton ─────────────────────────────────────────────────────

_generator: Optional[EmbeddingGenerator] = None


def get_generator() -> EmbeddingGenerator:
    """Return (or lazily create) the module-level EmbeddingGenerator."""
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator()
    return _generator