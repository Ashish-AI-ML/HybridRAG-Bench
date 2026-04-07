"""
retrieval/dense_retriever.py — Qdrant-backed dense vector retrieval.

Replaces the in-memory FAISS index with a production-grade Qdrant vector
database that supports:
  - Persistent storage (local file-based or cloud)
  - Metadata payload filtering
  - Incremental upserts (no full rebuild required)
  - gRPC + REST API access

Qdrant operates in two modes controlled by configs/default.yaml:
  - mode: "local"  → in-process Qdrant (no server, stored to disk)
  - mode: "cloud"  → remote Qdrant cluster via URL + API key
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from src.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    Embeds chunks with a bi-encoder and stores them in Qdrant.

    Parameters
    ----------
    embedding_model : str
        HuggingFace model name for the bi-encoder.
    collection_name : str
        Qdrant collection to use.
    qdrant_cfg : dict
        Sub-dict from config: mode, local_path, url, api_key.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "hybridrag",
        qdrant_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.collection_name = collection_name
        qdrant_cfg = qdrant_cfg or {}

        logger.info("Loading embedding model: %s", embedding_model)
        self.encoder = SentenceTransformer(embedding_model)
        self.dim = self.encoder.get_sentence_embedding_dimension()

        # ── Connect to Qdrant ────────────────────────────────────────────
        mode = qdrant_cfg.get("mode", "local")
        if mode == "cloud":
            url = qdrant_cfg["url"]
            api_key = qdrant_cfg.get("api_key")
            logger.info("Connecting to Qdrant Cloud: %s", url)
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            local_path = qdrant_cfg.get("local_path", "./qdrant_storage")
            logger.info("Using local Qdrant storage at: %s", local_path)
            self.client = QdrantClient(path=local_path)

        self._ensure_collection()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", self.collection_name, self.dim)
        else:
            count = self.client.count(self.collection_name).count
            logger.info(
                "Qdrant collection '%s' exists (%d vectors)",
                self.collection_name,
                count,
            )

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # ── Public API ──────────────────────────────────────────────────────────

    def is_populated(self) -> bool:
        """Return True if the collection already has vectors."""
        return self.client.count(self.collection_name).count > 0

    def add_chunks(self, chunks: list[dict[str, Any]], batch_size: int = 64) -> None:
        """Embed and upsert chunks into Qdrant."""
        if not chunks:
            logger.warning("add_chunks called with empty list — skipping")
            return

        logger.info("Embedding %d chunks (batch_size=%d)...", len(chunks), batch_size)
        t0 = time.perf_counter()

        points: list[PointStruct] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self._embed(texts)

            for chunk, vec in zip(batch, embeddings):
                # Payload = all chunk metadata (searchable/filterable in Qdrant)
                payload = {k: v for k, v in chunk.items() if k != "text"}
                payload["text"] = chunk["text"]  # also store text for retrieval
                points.append(
                    PointStruct(
                        id=abs(hash(chunk["chunk_id"])) % (2**31),
                        vector=vec.tolist(),
                        payload=payload,
                    )
                )

        self.client.upsert(collection_name=self.collection_name, points=points)
        elapsed = time.perf_counter() - t0
        logger.info("Upserted %d vectors in %.2fs", len(points), elapsed)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-K chunks by cosine similarity.

        Parameters
        ----------
        query : str
        top_k : int
        filter_payload : dict | None
            Optional Qdrant payload filter, e.g.::

                {"must": [{"key": "doc_id", "match": {"value": "DOC-3"}}]}

        Returns
        -------
        list of chunk dicts with an added ``dense_score`` field.
        """
        query_vec = self._embed([query])[0].tolist()

        # qdrant-client >= 1.7 uses query_points() instead of search()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )

        output: list[dict[str, Any]] = []
        for r in response.points:
            chunk = dict(r.payload)
            chunk["dense_score"] = float(r.score)
            output.append(chunk)

        return output

    def reset_collection(self) -> None:
        """Delete and recreate the collection (for testing/reindexing)."""
        self.client.delete_collection(self.collection_name)
        logger.warning("Collection '%s' deleted", self.collection_name)
        self._ensure_collection()
