"""
ChromaDB client wrapper.

Collections
-----------
communications_embeddings  — one document per Communication row
affiliate_profiles         — one document per Affiliate (summary embedding)

Both collections use sentence-transformers/all-MiniLM-L6-v2 (384 dims).
"""

import os
from typing import Optional

import chromadb
from dotenv import load_dotenv

from src.core.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)

CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8001"))
CHROMA_TOKEN: str = os.getenv("CHROMA_TOKEN", "chroma_secret_token")

COMM_COLLECTION = "communications_embeddings"
PROFILE_COLLECTION = "affiliate_profiles"


def _get_client() -> chromadb.HttpClient:
    """Return a ChromaDB HTTP client."""
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


class VectorStore:
    """
    Thin wrapper around ChromaDB with helper methods for both collections.
    Instantiate once and reuse (lazy connection).
    """

    def __init__(self) -> None:
        self._client: Optional[chromadb.HttpClient] = None
        self._comms_col = None
        self._profiles_col = None

    @property
    def client(self) -> chromadb.HttpClient:
        if self._client is None:
            self._client = _get_client()
        return self._client

    @property
    def comms(self):
        """Return (or create) the communications_embeddings collection."""
        if self._comms_col is None:
            self._comms_col = self.client.get_or_create_collection(
                name=COMM_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._comms_col

    @property
    def profiles(self):
        """Return (or create) the affiliate_profiles collection."""
        if self._profiles_col is None:
            self._profiles_col = self.client.get_or_create_collection(
                name=PROFILE_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._profiles_col

    # ── Communications ────────────────────────────────────────────────────────

    def upsert_communication(
        self,
        comm_id: str,
        embedding: list[float],
        document: str,
        metadata: dict,
    ) -> None:
        """
        Upsert a communication embedding.

        Parameters
        ----------
        comm_id   : unique ID — typically f"comm_{uuid}"
        embedding : 384-dim float list from sentence-transformers
        document  : raw text content
        metadata  : dict with keys affiliate_id, channel, direction,
                    sentiment_label, tags (pipe-joined str), occurred_at
        """
        self.comms.upsert(
            ids=[comm_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    def search_communications(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Semantic search over communications.

        Parameters
        ----------
        query_embedding : embedding of the query string
        n_results       : number of results to return
        where           : optional ChromaDB metadata filter
                          e.g. {"affiliate_id": "aff-003"}

        Returns dict with keys: ids, documents, metadatas, distances
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.comms.query(**kwargs)

    def get_communication(self, comm_id: str) -> Optional[dict]:
        """Retrieve a single communication document by ID."""
        result = self.comms.get(ids=[comm_id], include=["documents", "metadatas"])
        if result["ids"]:
            return {
                "id": result["ids"][0],
                "document": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
        return None

    # ── Affiliate Profiles ────────────────────────────────────────────────────

    def upsert_affiliate_profile(
        self,
        affiliate_id: str,
        embedding: list[float],
        document: str,
        metadata: dict,
    ) -> None:
        """
        Upsert an affiliate profile embedding.

        Parameters
        ----------
        affiliate_id : used as the ChromaDB ID (f"aff_{uuid}")
        embedding    : 384-dim float list
        document     : profile summary string
        metadata     : dict with tier, niche, churn_risk_score,
                        growth_potential_score, health_score
        """
        self.profiles.upsert(
            ids=[f"aff_{affiliate_id}"],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    def find_similar_affiliates(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """Return affiliates with similar profiles to the query."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.profiles.query(**kwargs)

    # ── High-level helpers (used by embedding_generator) ──────────────────────

    def add_document(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        affiliate_id: str,
        affiliate_name: str,
        source: str,
        tags: list[str],
        occurred_at: str,
    ) -> None:
        """
        Store one communication chunk with full metadata.

        Tags are stored two ways:
        - ``tags`` field: pipe-delimited string for display ("|tag1|tag2|")
        - ``tag_{name}`` boolean fields: one per tag, used for ChromaDB
          ``$eq`` filtering (chromadb 1.x does not support string $contains
          on metadata fields).
        """
        metadata: dict = {
            "affiliate_id": affiliate_id,
            "affiliate_name": affiliate_name,
            "source": source,
            "tags": "|" + "|".join(tags) + "|" if tags else "",
            "occurred_at": occurred_at,
        }
        for tag in tags:
            metadata[f"tag_{tag}"] = True
        self.comms.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def search_similar(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        affiliate_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Semantic search over the communications_embeddings collection.

        Parameters
        ----------
        query_embedding : 384-dim float list
        n_results       : max results to return
        affiliate_id    : optional filter to one affiliate UUID
        tags            : optional list of tag names — all must be present
                          (uses per-tag boolean metadata fields)

        Returns
        -------
        List of dicts with keys: id, text, metadata, distance
        """
        conditions: list[dict] = []
        if affiliate_id:
            conditions.append({"affiliate_id": {"$eq": affiliate_id}})
        if tags:
            # Each tag is stored as tag_{name}=True — use $eq filter
            for tag in tags:
                conditions.append({f"tag_{tag}": {"$eq": True}})

        where: Optional[dict] = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        # Guard: ChromaDB raises if n_results > collection size
        try:
            count = self.comms.count()
        except Exception:
            count = n_results
        actual_n = min(n_results, max(1, count))

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": actual_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            raw = self.comms.query(**kwargs)
        except Exception:
            return []

        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "id": raw["ids"][0][i],
                "text": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "distance": raw["distances"][0][i],
            })
        return results

    # ── Utility ───────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if ChromaDB is reachable."""
        try:
            self.client.list_collections()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("ChromaDB health check failed", extra={"error": str(exc)})
            return False

    def collection_stats(self) -> dict:
        """Return document counts for both collections."""
        return {
            "communications_embeddings": self.comms.count(),
            "affiliate_profiles": self.profiles.count(),
        }


# Module-level singleton — import and reuse across the app
vector_store = VectorStore()
