"""
evaluation/retrieval_metrics.py — Retrieval quality evaluation.

Computes: Precision@K, Recall@K, MRR, Hit Rate, NDCG@K
Accepts results from any retrieval mode (BM25-only, dense-only, hybrid, hybrid+rerank)
for clean ablation comparison.
"""

from __future__ import annotations

import math
from typing import Any


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    return 1.0 if any(d in relevant_ids for d in retrieved_ids[:k]) else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    Binary relevance: 1 if doc in relevant_ids, else 0.
    """
    def dcg(ids: list[str], rel_set: set[str], n: int) -> float:
        return sum(
            1.0 / math.log2(rank + 1)
            for rank, did in enumerate(ids[:n], start=1)
            if did in rel_set
        )

    rel_set = set(relevant_ids)
    actual_dcg = dcg(retrieved_ids, rel_set, k)
    # Ideal DCG: all relevant docs at top positions
    ideal_order = relevant_ids + [d for d in retrieved_ids if d not in rel_set]
    ideal_dcg = dcg(ideal_order, rel_set, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_chunks: list[dict[str, Any]],
    relevant_doc_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute all retrieval metrics for a single query result.

    Parameters
    ----------
    retrieved_chunks : list of chunk dicts (must contain 'doc_id')
    relevant_doc_ids : ground-truth doc IDs for this query
    k_values : list of K values to evaluate (default: [1, 3, 5])
    """
    k_values = k_values or [1, 3, 5]
    retrieved_ids = [c["doc_id"] for c in retrieved_chunks]

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"precision_at_{k}"] = round(precision_at_k(retrieved_ids, relevant_doc_ids, k), 4)
        metrics[f"recall_at_{k}"] = round(recall_at_k(retrieved_ids, relevant_doc_ids, k), 4)
        metrics[f"hit_rate_at_{k}"] = round(hit_rate_at_k(retrieved_ids, relevant_doc_ids, k), 4)
        metrics[f"ndcg_at_{k}"] = round(ndcg_at_k(retrieved_ids, relevant_doc_ids, k), 4)

    metrics["mrr"] = round(mrr(retrieved_ids, relevant_doc_ids), 4)
    return metrics
