"""
tests/integration/test_pipeline.py — Integration tests for the full RAG pipeline.

Runs the complete pipeline in mock mode (no LLM API calls, local Qdrant).
Tests end-to-end data flow: chunking → indexing → retrieval → generation → output shape.
"""

from __future__ import annotations

import pytest
from src.config import reload_config
from src.pipeline import HybridRAGPipeline


@pytest.fixture(scope="module")
def pipeline():
    """Build a dev-mode pipeline (mock LLM, no reranker, local Qdrant)."""
    cfg = reload_config("dev")
    # Force a unique test collection to avoid state collision
    cfg["pipeline"]["collection_name"] = "hybridrag_test"
    cfg["qdrant"]["local_path"] = "./qdrant_storage_test"
    pipe = HybridRAGPipeline(cfg=cfg, force_reindex=True)
    pipe.build_index("data/docs")
    return pipe


class TestPipelineBuild:
    def test_index_builds_successfully(self, pipeline):
        assert pipeline._index_built is True

    def test_chunks_populated(self, pipeline):
        assert len(pipeline._chunks) > 0

    def test_bm25_populated(self, pipeline):
        assert pipeline.bm25.is_populated()

    def test_qdrant_populated(self, pipeline):
        assert pipeline.dense.is_populated()


class TestPipelineQuery:
    def test_query_returns_dict(self, pipeline):
        result = pipeline.query("Who is Feynman?", mode="dense_only")
        assert isinstance(result, dict)

    def test_required_keys_present(self, pipeline):
        result = pipeline.query("What is Shor's algorithm?", mode="bm25_only")
        required = [
            "question", "answer", "retrieved_chunks", "retrieval_mode",
            "confidence_score", "is_insufficient", "latency", "token_usage", "source_docs"
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_retrieved_chunks_have_doc_id(self, pipeline):
        result = pipeline.query("Grover algorithm speedup", mode="hybrid")
        for chunk in result["retrieved_chunks"]:
            assert "doc_id" in chunk
            assert "text" in chunk

    def test_confidence_between_0_and_1(self, pipeline):
        result = pipeline.query("Deutsch quantum Turing machine 1985", mode="dense_only")
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_latency_keys_populated(self, pipeline):
        result = pipeline.query("RSA encryption quantum threat", mode="bm25_only")
        assert "total_ms" in result["latency"]

    def test_ablation_modes_work(self, pipeline):
        for mode in ["dense_only", "bm25_only", "hybrid"]:
            result = pipeline.query("quantum supremacy Sycamore 2019", mode=mode)
            assert result["retrieval_mode"] == mode
            assert len(result["retrieved_chunks"]) > 0

    def test_source_docs_list(self, pipeline):
        result = pipeline.query("What is the no-cloning theorem?", mode="dense_only")
        assert isinstance(result["source_docs"], list)
        assert all(isinstance(d, str) for d in result["source_docs"])


class TestPipelineRobustness:
    def test_out_of_domain_returns_low_confidence_or_insufficient(self, pipeline):
        result = pipeline.query("What is the capital of France?", mode="dense_only")
        # Either low confidence or explicitly insufficient
        assert result["confidence_score"] < 0.95 or result["is_insufficient"]

    def test_empty_ish_query_does_not_crash(self, pipeline):
        result = pipeline.query("?", mode="dense_only")
        assert "answer" in result
