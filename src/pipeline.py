"""
pipeline.py — End-to-end HybridRAG pipeline orchestrator.

Wires together: Chunking → BM25 index → Qdrant dense index → HybridRetriever
→ Generator, with full latency profiling and confidence scoring on every query.

Usage
-----
    from src.pipeline import HybridRAGPipeline
    pipe = HybridRAGPipeline()
    pipe.build_index()
    result = pipe.query("What is Shor's algorithm?")
    print(result["answer"])
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.chunking.sentence_chunker import SentenceChunker
from src.config import get_config
from src.generation.generator import Generator
from src.logger import get_logger
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, AblationMode
from src.retrieval.reranker import CrossEncoderReranker

logger = get_logger(__name__)

# Confidence threshold: if reranker_score / rrf_score below this, flag low confidence
_CONFIDENCE_THRESHOLD = 0.3


class HybridRAGPipeline:
    """
    Full hybrid RAG pipeline: BM25 + Qdrant + Reranker + LLM.

    Parameters
    ----------
    cfg : dict | None
        Pipeline config dict.  Defaults to ``get_config()`` (loads from YAML + env).
    force_reindex : bool
        If True, clears and rebuilds the Qdrant collection even if it exists.
    """

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        force_reindex: bool = False,
    ) -> None:
        self.cfg = cfg or get_config()
        self.force_reindex = force_reindex
        self._index_built = False
        self._chunks: list[dict[str, Any]] = []

        # ── Chunker ──────────────────────────────────────────────────────
        chunk_cfg = self.cfg.get("chunking", {})
        self.chunker = SentenceChunker(
            chunk_size=chunk_cfg.get("chunk_size", 3),
            overlap=chunk_cfg.get("overlap", 1),
        )

        # ── Retrievers ───────────────────────────────────────────────────
        ret_cfg = self.cfg.get("retrieval", {})
        qdrant_cfg = self.cfg.get("qdrant", {})

        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(
            embedding_model=ret_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
            collection_name=self.cfg.get("pipeline", {}).get("collection_name", "hybridrag"),
            qdrant_cfg=qdrant_cfg,
        )

        reranker = None
        if ret_cfg.get("enable_reranker", True):
            reranker = CrossEncoderReranker(
                model_name=ret_cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            )

        self.retriever = HybridRetriever(
            bm25=self.bm25,
            dense=self.dense,
            reranker=reranker,
            cfg=ret_cfg,
        )

        # ── Generator ────────────────────────────────────────────────────
        gen_cfg = self.cfg.get("generation", {})
        secrets = self.cfg.get("secrets", {})
        self.generator = Generator(cfg=gen_cfg, secrets=secrets)

    # ── Index management ────────────────────────────────────────────────────

    def build_index(self, data_dir: str | None = None) -> int:
        """
        Chunk documents and populate both BM25 and Qdrant indexes.

        Parameters
        ----------
        data_dir : str | None
            Directory containing source documents.  Defaults to config value.

        Returns
        -------
        int
            Number of chunks indexed.
        """
        dir_path = data_dir or self.cfg.get("pipeline", {}).get("data_dir", "data/docs")
        logger.info("Building index from '%s'", dir_path)

        t0 = time.perf_counter()
        self._chunks = self.chunker.chunk_directory(dir_path)

        if not self._chunks:
            raise RuntimeError(f"No chunks generated from '{dir_path}'. Check docs directory.")

        # BM25 always rebuilt (in-memory, fast)
        self.bm25.build_index(self._chunks)

        # Qdrant: skip upsert if populated and not force-reindex
        if self.force_reindex:
            self.dense.reset_collection()

        if not self.dense.is_populated() or self.force_reindex:
            self.dense.add_chunks(self._chunks)
        else:
            logger.info("Qdrant collection already populated — skipping upsert")

        self._index_built = True
        elapsed = time.perf_counter() - t0
        logger.info(
            "Index ready: %d chunks across both stores (%.2fs total)",
            len(self._chunks),
            elapsed,
        )
        return len(self._chunks)

    # ── Query ────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int | None = None,
        mode: AblationMode = "hybrid_rerank",
        strict: bool | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full end-to-end RAG pipeline for a single question.

        Returns
        -------
        dict with:
          - question          : original question
          - answer            : LLM-generated answer
          - retrieved_chunks  : final ranked chunks
          - retrieval_mode    : ablation mode used
          - confidence_score  : 0–1 based on top chunk score
          - is_insufficient   : True if the model returned "I don't know"
          - latency           : per-stage timing dict (ms)
          - token_usage       : input/output token counts
          - source_docs       : unique doc_ids of retrieved chunks
        """
        if not self._index_built:
            logger.warning("Index not built — auto-building from default data dir")
            self.build_index()

        pipeline_latency: dict[str, float] = {}

        # ── Retrieval ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        retrieval_result = self.retriever.search(question, top_k=top_k, mode=mode)
        pipeline_latency["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        pipeline_latency.update(retrieval_result["latency"])

        chunks = retrieval_result["chunks"]

        # ── Confidence scoring ───────────────────────────────────────────
        confidence = self._compute_confidence(chunks)

        if not chunks:
            logger.warning("No chunks retrieved for query: '%s'", question)
            return {
                "question": question,
                "answer": "INSUFFICIENT CONTEXT: No relevant documents found.",
                "retrieved_chunks": [],
                "retrieval_mode": mode,
                "confidence_score": 0.0,
                "is_insufficient": True,
                "latency": pipeline_latency,
                "token_usage": {"input": 0, "output": 0},
                "source_docs": [],
            }

        # ── Generation ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        gen_result = self.generator.generate(question, chunks, strict=strict)
        pipeline_latency["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        pipeline_latency["total_ms"] = sum(pipeline_latency.values())

        source_docs = list(dict.fromkeys(c["doc_id"] for c in chunks))

        return {
            "question": question,
            "answer": gen_result["answer"],
            "retrieved_chunks": chunks,
            "retrieval_mode": mode,
            "confidence_score": confidence,
            "is_insufficient": gen_result["is_insufficient"],
            "latency": pipeline_latency,
            "token_usage": {
                "input": gen_result["input_tokens"],
                "output": gen_result["output_tokens"],
            },
            "source_docs": source_docs,
            "prompt_used": gen_result["prompt_used"],
        }

    def _compute_confidence(self, chunks: list[dict]) -> float:
        """
        Compute a 0–1 confidence score from the top chunk's retrieval signal.

        Logic: Use reranker_score if available, else dense_score, else rrf_score.
        Normalise to [0, 1] with a simple sigmoid-like clamp.
        """
        if not chunks:
            return 0.0
        top = chunks[0]
        raw_score = (
            top.get("reranker_score")
            or top.get("dense_score")
            or top.get("rrf_score")
            or 0.0
        )
        # Reranker scores can be negative (logit scale) — shift and clamp
        if raw_score < 0:
            # Sigmoid normalisation for cross-encoder logits
            import math
            raw_score = 1 / (1 + math.exp(-raw_score))
        return round(min(1.0, max(0.0, float(raw_score))), 4)


if __name__ == "__main__":
    import json

    pipeline = HybridRAGPipeline()
    pipeline.build_index()

    test_query = "When did Peter Shor propose his algorithm and what threat did it pose to RSA?"
    result = pipeline.query(test_query)

    print("\n" + "=" * 70)
    print("QUERY:", result["question"])
    print("=" * 70)
    print("\nANSWER:")
    print(result["answer"])
    print(f"\nCONFIDENCE: {result['confidence_score']:.3f}")
    print(f"INSUFFICIENT: {result['is_insufficient']}")
    print(f"SOURCES: {result['source_docs']}")
    print("\nLATENCY:")
    for k, v in result["latency"].items():
        print(f"  {k}: {v}ms")
    print("\nTOKEN USAGE:", result["token_usage"])
    print("\nTOP RETRIEVED CHUNK:")
    top = result["retrieved_chunks"][0]
    print(f"  [{top['doc_id']}] {top['text'][:120]}...")
