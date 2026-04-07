"""
retrieval/reranker.py — Cross-encoder reranking stage.

Takes the merged candidate pool from BM25 + dense retrieval and re-scores
each candidate against the query using a cross-encoder model that jointly
encodes (query, passage) pairs — far more accurate than the bi-encoder
used in the initial retrieval.

Why this matters
----------------
Bi-encoders encode query and passage *independently*, so their dot product
is an approximation of relevance.  Cross-encoders encode them *together*,
giving attention across the full pair — at the cost of not being scalable
to large corpora.  The two-stage design (fast bi-encoder for broad recall,
slow cross-encoder for precision) is the industry standard.

Typical improvement: 10–25% gain in NDCG over reranker-less pipelines.
"""

from __future__ import annotations

import time
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Reranks a candidate chunk list using a cross-encoder model.

    Parameters
    ----------
    model_name : str
        HuggingFace cross-encoder model. Recommended:
        ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (fast, good quality)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self._model = None  # lazy-loaded on first use

    def _load_model(self) -> None:
        """Lazy-load the cross-encoder to avoid slow startup when reranker is disabled."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Cross-encoder loaded successfully")
        except Exception as exc:
            logger.error("Failed to load cross-encoder '%s': %s", self.model_name, exc)
            raise

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Score each candidate against the query and return top-K reranked results.

        Parameters
        ----------
        query : str
        candidates : list[dict]
            Merged candidates from BM25 + dense retrieval.
        top_k : int
            Number of results to return after reranking.

        Returns
        -------
        list of chunk dicts sorted by ``reranker_score`` descending,
        each also containing their pre-reranking ``bm25_score`` and
        ``dense_score`` for ablation analysis.
        """
        if not candidates:
            return []

        self._load_model()

        t0 = time.perf_counter()
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        elapsed = time.perf_counter() - t0

        # Attach scores and sort
        scored = []
        for chunk, score in zip(candidates, scores):
            c = dict(chunk)
            c["reranker_score"] = float(score)
            scored.append(c)

        scored.sort(key=lambda x: x["reranker_score"], reverse=True)

        logger.debug(
            "Reranked %d candidates → top %d in %.3fs",
            len(candidates),
            top_k,
            elapsed,
        )

        # Add final rank position
        result = scored[:top_k]
        for i, c in enumerate(result):
            c["final_rank"] = i + 1

        return result
