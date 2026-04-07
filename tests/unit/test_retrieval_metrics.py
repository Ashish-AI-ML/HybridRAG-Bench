"""
tests/unit/test_retrieval_metrics.py — Unit tests for retrieval quality metrics.
"""

import pytest
from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    hit_rate_at_k,
    ndcg_at_k,
    compute_retrieval_metrics,
)


class TestPrecisionAtK:
    def test_perfect_retrieval(self):
        assert precision_at_k(["DOC-1", "DOC-2", "DOC-3"], ["DOC-1"], k=1) == 1.0

    def test_miss_at_1(self):
        assert precision_at_k(["DOC-2", "DOC-1", "DOC-3"], ["DOC-1"], k=1) == 0.0

    def test_partial_at_3(self):
        result = precision_at_k(["DOC-1", "DOC-2", "DOC-3"], ["DOC-1", "DOC-3"], k=3)
        assert abs(result - 2/3) < 1e-6

    def test_empty_retrieved(self):
        assert precision_at_k([], ["DOC-1"], k=3) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k(["DOC-1", "DOC-2"], ["DOC-1"], k=3) == 1.0

    def test_partial_recall(self):
        result = recall_at_k(["DOC-1", "DOC-3"], ["DOC-1", "DOC-2"], k=2)
        assert result == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["DOC-1"], [], k=3) == 0.0


class TestMRR:
    def test_first_rank(self):
        assert mrr(["DOC-1", "DOC-2"], ["DOC-1"]) == 1.0

    def test_second_rank(self):
        assert mrr(["DOC-2", "DOC-1"], ["DOC-1"]) == 0.5

    def test_not_found(self):
        assert mrr(["DOC-2", "DOC-3"], ["DOC-1"]) == 0.0


class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k(["DOC-1", "DOC-2"], ["DOC-1"], k=3) == 1.0

    def test_miss_outside_k(self):
        assert hit_rate_at_k(["DOC-2", "DOC-3", "DOC-1"], ["DOC-1"], k=2) == 0.0


class TestComputeMetrics:
    def test_returns_all_keys(self):
        chunks = [{"doc_id": "DOC-1"}, {"doc_id": "DOC-2"}, {"doc_id": "DOC-3"}]
        metrics = compute_retrieval_metrics(chunks, ["DOC-1"], k_values=[1, 3])
        assert "precision_at_1" in metrics
        assert "recall_at_3" in metrics
        assert "mrr" in metrics
        assert "ndcg_at_3" in metrics

    def test_perfect_metrics(self):
        chunks = [{"doc_id": "DOC-1"}]
        metrics = compute_retrieval_metrics(chunks, ["DOC-1"], k_values=[1])
        assert metrics["precision_at_1"] == 1.0
        assert metrics["mrr"] == 1.0
