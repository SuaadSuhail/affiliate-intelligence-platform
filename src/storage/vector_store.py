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

load_dotenv()

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

    # ── Utility ───────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if ChromaDB is reachable."""
        try:
            self.client.heartbeat()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[vector_store] ChromaDB health check failed: {exc}")
            return False

    def collection_stats(self) -> dict:
        """Return document counts for both collections."""
        return {
            "communications_embeddings": self.comms.count(),
            "affiliate_profiles": self.profiles.count(),
        }


# Module-level singleton — import and reuse across the app
vector_store = VectorStore()
