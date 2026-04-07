"""
retrieval/hybrid_retriever.py — Hybrid BM25 + Dense retrieval with fusion and reranking.

This is the core architectural upgrade of HybridRAG Bench.

Pipeline
--------
1. BM25 retrieval    → top_k_bm25 candidates (lexical precision)
2. Dense retrieval   → top_k_dense candidates (semantic recall)
3. RRF fusion        → merged + de-duplicated candidate pool
4. Reranking         → cross-encoder rescoring → final top_k results

Fusion Strategy: Reciprocal Rank Fusion (RRF)
---------------------------------------------
RRF is preferred over linear score combination because:
  - BM25 and cosine scores are not comparable scales
  - RRF uses rank positions rather than scores, making it robust
    to score distribution differences between the two retrievers
  - Formula: RRF(d) = Σ 1 / (k + rank_i(d))
    where k=60 is a smoothing constant (standard value from the paper)

Ablation modes
--------------
We expose ``mode`` parameter to enable clean ablation comparison:
  - "dense_only"  : skip BM25
  - "bm25_only"   : skip dense retrieval
  - "hybrid"      : BM25 + dense, no reranker
  - "hybrid_rerank" (default): full pipeline
"""

from __future__ import annotations

import time
from typing import Any, Literal

from src.logger import get_logger
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker

logger = get_logger(__name__)

AblationMode = Literal["dense_only", "bm25_only", "hybrid", "hybrid_rerank"]


def _rrf_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion.

    Returns a merged+de-duplicated list sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunks[cid] = chunk

    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunks:
            chunks[cid] = chunk

    merged = sorted(chunks.values(), key=lambda c: scores[c["chunk_id"]], reverse=True)
    for c in merged:
        c["rrf_score"] = scores[c["chunk_id"]]

    return merged


class HybridRetriever:
    """
    Orchestrates BM25 + Dense retrieval with optional reranking.

    Parameters
    ----------
    bm25 : BM25Retriever
    dense : DenseRetriever
    reranker : CrossEncoderReranker | None
    cfg : dict
        retrieval section of the pipeline config.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: CrossEncoderReranker | None,
        cfg: dict[str, Any],
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.cfg = cfg

    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: AblationMode = "hybrid_rerank",
    ) -> dict[str, Any]:
        """
        Execute the hybrid retrieval pipeline.

        Returns
        -------
        dict with keys:
          - ``chunks``       : final ranked chunk list
          - ``mode``         : ablation mode used
          - ``latency``      : per-stage timing in ms
          - ``candidate_count`` : number of candidates before reranking
        """
        final_k = top_k or self.cfg.get("final_top_k", 5)
        k_dense = self.cfg.get("top_k_dense", 10)
        k_bm25 = self.cfg.get("top_k_bm25", 10)
        rrf_k = self.cfg.get("rrf_k", 60)
        enable_reranker = self.cfg.get("enable_reranker", True)

        latency: dict[str, float] = {}

        # ── Stage 1: BM25 ────────────────────────────────────────────────
        bm25_results: list[dict] = []
        if mode in ("bm25_only", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            bm25_results = self.bm25.search(query, top_k=k_bm25)
            latency["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            logger.debug("BM25 returned %d results (%.1fms)", len(bm25_results), latency["bm25_ms"])

        # ── Stage 2: Dense ───────────────────────────────────────────────
        dense_results: list[dict] = []
        if mode in ("dense_only", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            dense_results = self.dense.search(query, top_k=k_dense)
            latency["dense_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            logger.debug("Dense returned %d results (%.1fms)", len(dense_results), latency["dense_ms"])

        # ── Stage 3: Fusion ──────────────────────────────────────────────
        t0 = time.perf_counter()
        if mode == "bm25_only":
            candidates = bm25_results[:final_k]
        elif mode == "dense_only":
            candidates = dense_results[:final_k]
        else:
            candidates = _rrf_fusion(bm25_results, dense_results, k=rrf_k)
        latency["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        candidate_count = len(candidates)

        # ── Stage 4: Reranking ───────────────────────────────────────────
        if mode == "hybrid_rerank" and enable_reranker and self.reranker and len(candidates) > 1:
            t0 = time.perf_counter()
            final_chunks = self.reranker.rerank(query, candidates, top_k=final_k)
            latency["reranker_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            logger.debug(
                "Reranker processed %d → %d (%.1fms)",
                candidate_count, len(final_chunks), latency["reranker_ms"],
            )
        else:
            final_chunks = candidates[:final_k]
            for i, c in enumerate(final_chunks):
                c.setdefault("final_rank", i + 1)

        return {
            "chunks": final_chunks,
            "mode": mode,
            "latency": latency,
            "candidate_count": candidate_count,
        }
