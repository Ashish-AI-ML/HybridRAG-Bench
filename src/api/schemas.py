"""
api/schemas.py — Pydantic request/response models for the FastAPI service.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Request models ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="The question to answer")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    mode: str = Field(
        "hybrid_rerank",
        description="Retrieval mode: dense_only | bm25_only | hybrid | hybrid_rerank",
    )
    strict_grounding: Optional[bool] = Field(True, description="Enforce strict context-only answering")

    model_config = {"json_schema_extra": {
        "example": {
            "question": "Why did Shor's algorithm threaten RSA encryption?",
            "top_k": 5,
            "mode": "hybrid_rerank",
            "strict_grounding": True,
        }
    }}


class EvalRequest(BaseModel):
    question: str
    expected_answer: str
    mode: str = "hybrid_rerank"
    top_k: int = 5


# ── Response models ─────────────────────────────────────────────────────────

class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    final_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    reranker_score: Optional[float] = None


class LatencyBreakdown(BaseModel):
    bm25_ms: Optional[float] = None
    dense_ms: Optional[float] = None
    fusion_ms: Optional[float] = None
    reranker_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    total_ms: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_docs: list[str]
    confidence_score: float
    is_insufficient: bool
    retrieval_mode: str
    retrieved_chunks: list[ChunkResult]
    latency: LatencyBreakdown
    token_usage: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    index_built: bool
    chunk_count: int
    provider: str
    version: str = "2.0.0"


class EvalResponse(BaseModel):
    question: str
    generated_answer: str
    expected_answer: str
    retrieval_metrics: dict[str, float]
    generation_metrics: dict[str, Any]
    latency: LatencyBreakdown
