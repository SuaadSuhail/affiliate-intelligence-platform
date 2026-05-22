"""
Embedding Generator
===================
Converts text to 384-dim vectors using sentence-transformers/all-MiniLM-L6-v2
and upserts them into ChromaDB via the VectorStore wrapper.

Usage
-----
    from src.ingestion.embedding_generator import EmbeddingGenerator
    gen = EmbeddingGenerator()
    embedding = gen.embed("some text")
    gen.index_communication(comm, affiliate_id="aff-001")
"""

import os
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.storage.vector_store import vector_store

load_dotenv()

MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


class EmbeddingGenerator:
    """
    Wraps a SentenceTransformer model and provides index helpers for
    both ChromaDB collections.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        print(f"[embedding_generator] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

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
        content: str,
        affiliate_id: str,
        channel: str,
        direction: str,
        sentiment_label: str,
        tags: list[str],
        occurred_at: str,
    ) -> str:
        """
        Embed a communication and upsert it into ChromaDB.

        Parameters
        ----------
        comm_id        : UUID of the Communication row (will be prefixed with "comm_")
        content        : full text of the communication
        affiliate_id   : UUID of the parent Affiliate
        channel        : email | call | chat | ticket
        direction      : inbound | outbound
        sentiment_label: positive | neutral | negative
        tags           : list of NLP tag strings
        occurred_at    : ISO datetime string

        Returns the ChromaDB document ID.
        """
        doc_id = f"comm_{comm_id}"
        embedding = self.embed(content)
        metadata = {
            "affiliate_id": str(affiliate_id),
            "channel": channel,
            "direction": direction,
            "sentiment_label": sentiment_label,
            "tags": "|".join(tags),  # ChromaDB metadata must be primitive
            "occurred_at": occurred_at,
        }
        vector_store.upsert_communication(
            comm_id=doc_id,
            embedding=embedding,
            document=content,
            metadata=metadata,
        )
        return doc_id

    # ── Affiliate profile indexing ─────────────────────────────────────────────

    def index_affiliate_profile(
        self,
        affiliate_id: str,
        name: str,
        company: Optional[str],
        niche: Optional[str],
        traffic_source: Optional[str],
        tier: str,
        monthly_revenue: float,
        churn_risk_score: float,
        growth_potential_score: float,
        health_score: float,
    ) -> str:
        """
        Build a summary string for an affiliate, embed it, and upsert into
        the affiliate_profiles ChromaDB collection.

        Returns the ChromaDB document ID.
        """
        document = (
            f"{name} | {company or 'unknown'} | {niche or 'unknown'} | "
            f"{traffic_source or 'unknown'} | tier={tier} | "
            f"revenue={monthly_revenue:.2f}"
        )
        embedding = self.embed(document)
        metadata = {
            "affiliate_id": str(affiliate_id),
            "tier": tier,
            "niche": niche or "",
            "churn_risk_score": round(churn_risk_score, 4),
            "growth_potential_score": round(growth_potential_score, 4),
            "health_score": round(health_score, 2),
        }
        vector_store.upsert_affiliate_profile(
            affiliate_id=str(affiliate_id),
            embedding=embedding,
            document=document,
            metadata=metadata,
        )
        return f"aff_{affiliate_id}"

    # ── Semantic search helpers ────────────────────────────────────────────────

    def search_communications(
        self,
        query: str,
        n_results: int = 5,
        affiliate_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search over indexed communications.

        Parameters
        ----------
        query        : natural-language search query
        n_results    : number of results to return
        affiliate_id : optional filter to a single affiliate

        Returns a list of dicts with keys: id, document, metadata, distance
        """
        embedding = self.embed(query)
        where = {"affiliate_id": affiliate_id} if affiliate_id else None
        raw = vector_store.search_communications(
            query_embedding=embedding,
            n_results=n_results,
            where=where,
        )
        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "id": raw["ids"][0][i],
                "document": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "distance": raw["distances"][0][i],
            })
        return results

    def search_similar_affiliates(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Find affiliates with profiles semantically similar to the query."""
        embedding = self.embed(query)
        raw = vector_store.find_similar_affiliates(
            query_embedding=embedding,
            n_results=n_results,
        )
        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "id": raw["ids"][0][i],
                "document": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "distance": raw["distances"][0][i],
            })
        return results


# Module-level singleton
_generator: Optional[EmbeddingGenerator] = None


def get_generator() -> EmbeddingGenerator:
    """Return (or lazily create) the module-level EmbeddingGenerator."""
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator()
    return _generator
