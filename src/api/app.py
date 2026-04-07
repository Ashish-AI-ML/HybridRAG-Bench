"""
api/app.py — FastAPI application for HybridRAG Bench.

Endpoints:
  GET  /health          : system health check
  POST /v1/query        : full RAG query with traceability
  POST /v1/evaluate     : single-query evaluation against a reference answer
  GET  /v1/index/stats  : corpus statistics

The pipeline is initialised once on startup and shared across requests.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    ChunkResult,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
)
from src.config import get_config
from src.evaluation.generation_metrics import GenerationEvaluator
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.logger import get_logger
from src.pipeline import HybridRAGPipeline

logger = get_logger(__name__)

# ── Global pipeline singleton ───────────────────────────────────────────────
_pipeline: HybridRAGPipeline | None = None
_gen_evaluator: GenerationEvaluator | None = None
_chunk_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the index on startup — shared across all requests."""
    global _pipeline, _gen_evaluator, _chunk_count
    logger.info("Starting HybridRAG Bench API...")
    cfg = get_config()
    _pipeline = HybridRAGPipeline(cfg=cfg)
    _chunk_count = _pipeline.build_index()
    _gen_evaluator = GenerationEvaluator(
        semantic_model=cfg["retrieval"].get("embedding_model", "all-MiniLM-L6-v2")
    )
    logger.info("API ready — %d chunks indexed", _chunk_count)
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="HybridRAG Bench API",
    description=(
        "Production-grade Hybrid RAG system with BM25 + Qdrant + Reranker. "
        "Includes multi-dimensional evaluation, answer traceability, and latency profiling."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request tracing middleware ───────────────────────────────────────────────

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.  Returns 200 if the pipeline is ready to serve queries.
    Verifies: index built, generator provider configured.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return HealthResponse(
        status="healthy",
        index_built=_pipeline._index_built,
        chunk_count=_chunk_count,
        provider=_pipeline.generator.provider,
    )


@app.post("/v1/query", response_model=QueryResponse, tags=["RAG"])
async def query(req: QueryRequest):
    """
    Execute a full hybrid RAG query.

    Returns the generated answer, retrieved source chunks with scores,
    confidence signal, latency breakdown per stage, and token usage.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        result = _pipeline.query(
            question=req.question,
            top_k=req.top_k,
            mode=req.mode,
            strict=req.strict_grounding,
        )
    except Exception as exc:
        logger.error("Query error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    chunks = [
        ChunkResult(
            chunk_id=c.get("chunk_id", ""),
            doc_id=c.get("doc_id", ""),
            text=c.get("text", ""),
            final_rank=c.get("final_rank"),
            dense_score=c.get("dense_score"),
            bm25_score=c.get("bm25_score"),
            rrf_score=c.get("rrf_score"),
            reranker_score=c.get("reranker_score"),
        )
        for c in result["retrieved_chunks"]
    ]

    lat = result.get("latency", {})
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        source_docs=result["source_docs"],
        confidence_score=result["confidence_score"],
        is_insufficient=result["is_insufficient"],
        retrieval_mode=result["retrieval_mode"],
        retrieved_chunks=chunks,
        latency=LatencyBreakdown(**{k: v for k, v in lat.items() if hasattr(LatencyBreakdown, k) or True}),
        token_usage=result.get("token_usage", {}),
    )


@app.post("/v1/evaluate", response_model=EvalResponse, tags=["Evaluation"])
async def evaluate_single(req: EvalRequest):
    """
    Evaluate a single question against a reference answer.

    Computes retrieval metrics (P@K, MRR, NDCG), generation metrics
    (ROUGE, semantic similarity, faithfulness), and latency profiling.
    """
    if _pipeline is None or _gen_evaluator is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    result = _pipeline.query(req.question, top_k=req.top_k, mode=req.mode)

    # Retrieval metrics (no ground truth doc IDs here — just placeholder)
    retr_metrics = compute_retrieval_metrics(result["retrieved_chunks"], [], k_values=[1, 3, 5])

    gen_metrics = _gen_evaluator.evaluate(
        reference=req.expected_answer,
        generated=result["answer"],
        retrieved_chunks=result["retrieved_chunks"],
    )

    lat = result.get("latency", {})
    return EvalResponse(
        question=result["question"],
        generated_answer=result["answer"],
        expected_answer=req.expected_answer,
        retrieval_metrics=retr_metrics,
        generation_metrics=gen_metrics,
        latency=LatencyBreakdown(**{k: v for k, v in lat.items()}),
    )


@app.get("/v1/index/stats", tags=["System"])
async def index_stats():
    """Return statistics about the indexed corpus."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    chunks = _pipeline._chunks
    doc_ids = list(dict.fromkeys(c["doc_id"] for c in chunks))

    return {
        "total_chunks": len(chunks),
        "total_documents": len(doc_ids),
        "documents": doc_ids,
        "embedding_model": _pipeline.dense.encoder.get_sentence_embedding_dimension(),
        "collection_name": _pipeline.dense.collection_name,
    }
