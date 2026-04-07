"""
evaluation/suite_runner.py — Automated multi-dimensional evaluation suite.

Single command runs the complete evaluation across all test tiers:
  - Retrieval quality (P@K, Recall@K, MRR, HitRate, NDCG)
  - Answer correctness (ROUGE, semantic similarity, entity recall)
  - Faithfulness (heuristic groundedness evaluation)
  - Latency profiling (per-stage wall-clock time)
  - Cost estimation (token usage and API cost projection)
  - Ablation comparison (BM25-only vs Dense-only vs Hybrid vs Hybrid+Reranker)

Results are saved as timestamped JSON for regression tracking.

Usage
-----
    python -m src.evaluation.suite_runner
    python -m src.evaluation.suite_runner --ablation --top-k 5
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import get_config
from src.evaluation.cost_estimator import CostAccumulator
from src.evaluation.generation_metrics import GenerationEvaluator
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.logger import get_logger
from src.pipeline import HybridRAGPipeline

logger = get_logger(__name__)


def _load_ground_truth(path: str) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_results(results: dict[str, Any], results_dir: str) -> Path:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results saved → %s", out_path)
    return out_path


def _aggregate_metrics(per_query: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of every numeric metric across all queries."""
    if not per_query:
        return {}
    keys = [k for k, v in per_query[0].items() if isinstance(v, (int, float))]
    agg: dict[str, float] = {}
    for k in keys:
        vals = [q[k] for q in per_query if isinstance(q.get(k), (int, float))]
        agg[f"mean_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return agg


def run_evaluation(
    pipeline: HybridRAGPipeline,
    ground_truth: list[dict[str, Any]],
    cfg: dict[str, Any],
    mode: str = "hybrid_rerank",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Run the full evaluation suite for one retrieval mode.

    Returns a results dict with per-query breakdown and aggregate metrics.
    """
    gen_evaluator = GenerationEvaluator(
        semantic_model=cfg["retrieval"].get("embedding_model", "all-MiniLM-L6-v2")
    )
    cost_acc = CostAccumulator(provider=cfg["generation"].get("provider", "mock"))

    per_query_results: list[dict[str, Any]] = []

    logger.info("=" * 60)
    logger.info("EVALUATION MODE: %s  |  top_k=%d  |  n=%d queries", mode, top_k, len(ground_truth))
    logger.info("=" * 60)

    for i, item in enumerate(ground_truth, start=1):
        q_id = item.get("id", f"Q{i:02d}")
        question = item["question"]
        expected_answer = item["expected_answer"]
        relevant_docs = item.get("source_docs", [])
        q_type = item.get("type", "unknown")

        logger.info("[%d/%d] %s — %s", i, len(ground_truth), q_id, question[:50] + "...")

        try:
            result = pipeline.query(question, top_k=top_k, mode=mode)
        except Exception as exc:
            logger.error("Query failed for %s: %s", q_id, exc)
            continue

        # ── Retrieval metrics ────────────────────────────────────────────
        retr_metrics = compute_retrieval_metrics(
            result["retrieved_chunks"], relevant_docs, k_values=[1, 3, 5]
        )

        # ── Generation metrics ───────────────────────────────────────────
        gen_metrics = gen_evaluator.evaluate(
            reference=expected_answer,
            generated=result["answer"],
            retrieved_chunks=result["retrieved_chunks"],
        )

        # ── Cost tracking ────────────────────────────────────────────────
        tu = result.get("token_usage", {})
        cost_acc.record(tu.get("input", 0), tu.get("output", 0))

        per_query_results.append(
            {
                "q_id": q_id,
                "q_type": q_type,
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": result["answer"],
                "source_docs": result["source_docs"],
                "confidence_score": result["confidence_score"],
                "is_insufficient": result["is_insufficient"],
                "latency_ms": result["latency"].get("total_ms", 0.0),
                "retrieval_latency_ms": result["latency"].get("retrieval_ms", 0.0),
                "generation_latency_ms": result["latency"].get("generation_ms", 0.0),
                **retr_metrics,
                **gen_metrics,
            }
        )

    aggregate = _aggregate_metrics(per_query_results)
    cost_summary = cost_acc.summary()

    # ── Latency stats ────────────────────────────────────────────────────
    latencies = [q["latency_ms"] for q in per_query_results if q["latency_ms"] > 0]
    if latencies:
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        latency_stats = {
            "p50_ms": sorted_lat[n // 2],
            "p90_ms": sorted_lat[int(n * 0.9)],
            "p99_ms": sorted_lat[int(n * 0.99)],
            "mean_ms": round(sum(sorted_lat) / n, 2),
        }
    else:
        latency_stats = {}

    return {
        "run_metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "top_k": top_k,
            "n_queries": len(per_query_results),
            "provider": cfg["generation"].get("provider", "mock"),
        },
        "aggregate_metrics": aggregate,
        "latency_stats": latency_stats,
        "cost_summary": cost_summary,
        "per_query_results": per_query_results,
    }


def run_ablation(
    pipeline: HybridRAGPipeline,
    ground_truth: list[dict[str, Any]],
    cfg: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Run evaluation across all 4 retrieval modes for ablation comparison.

    dense_only → bm25_only → hybrid → hybrid_rerank
    """
    modes = ["dense_only", "bm25_only", "hybrid", "hybrid_rerank"]
    ablation_results: dict[str, Any] = {}

    for mode in modes:
        logger.info("\n>>> ABLATION: %s", mode.upper())
        results = run_evaluation(pipeline, ground_truth, cfg, mode=mode, top_k=top_k)
        ablation_results[mode] = results["aggregate_metrics"]

    return ablation_results


def _print_summary(results: dict[str, Any]) -> None:
    """Pretty-print the evaluation summary to console."""
    agg = results.get("aggregate_metrics", {})
    meta = results.get("run_metadata", {})
    lat = results.get("latency_stats", {})
    cost = results.get("cost_summary", {})

    print("\n" + "=" * 70)
    print(f"  EVALUATION SUMMARY | mode={meta.get('mode')} | n={meta.get('n_queries')}")
    print("=" * 70)

    print("\n── RETRIEVAL METRICS ──────────────────────────────")
    for k in ["mean_precision_at_1", "mean_precision_at_3", "mean_precision_at_5",
              "mean_recall_at_5", "mean_mrr", "mean_ndcg_at_5", "mean_hit_rate_at_5"]:
        if k in agg:
            print(f"  {k:<35} {agg[k]:.4f}")

    print("\n── GENERATION METRICS ─────────────────────────────")
    for k in ["mean_rouge1_f", "mean_rougeL_f", "mean_semantic_similarity",
              "mean_entity_recall", "mean_faithfulness_score"]:
        if k in agg:
            print(f"  {k:<35} {agg[k]:.4f}")

    print("\n── LATENCY ────────────────────────────────────────")
    for k, v in lat.items():
        print(f"  {k:<35} {v:.1f}ms")

    print("\n── COST ESTIMATION ────────────────────────────────")
    print(f"  provider             : {cost.get('provider', 'N/A')}")
    print(f"  avg input tokens     : {cost.get('avg_input_tokens', 0):.0f}")
    print(f"  avg output tokens    : {cost.get('avg_output_tokens', 0):.0f}")
    print(f"  total eval cost (USD): ${cost.get('total_cost_usd', 0):.6f}")
    proj = cost.get("projections", {})
    for vol, details in proj.items():
        if details:
            print(f"  {vol}: ${details.get('projected_cost_usd', 0):.4f}/month")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the HybridRAG evaluation suite")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation across all modes")
    parser.add_argument("--mode", default="hybrid_rerank",
                        choices=["dense_only", "bm25_only", "hybrid", "hybrid_rerank"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force-reindex", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    gt_path = cfg["evaluation"]["ground_truth_path"]
    results_dir = cfg["evaluation"]["results_dir"]

    ground_truth = _load_ground_truth(gt_path)
    logger.info("Loaded %d ground truth items from '%s'", len(ground_truth), gt_path)

    pipeline = HybridRAGPipeline(cfg=cfg, force_reindex=args.force_reindex)
    pipeline.build_index()

    if args.ablation:
        logger.info("\n>>> FULL ABLATION STUDY STARTED")
        ablation = run_ablation(pipeline, ground_truth, cfg, top_k=args.top_k)
        print("\n\n── ABLATION COMPARISON ─────────────────────────────────────")
        header = f"{'Metric':<35}" + "".join(f"{m:<18}" for m in ablation.keys())
        print(header)
        print("-" * len(header))
        all_keys = sorted({k for m_res in ablation.values() for k in m_res})
        for metric in all_keys:
            if "precision" in metric or "recall" in metric or "mrr" in metric or \
               "ndcg" in metric or "semantic" in metric or "faithfulness" in metric:
                row = f"{metric:<35}"
                for mode_res in ablation.values():
                    val = mode_res.get(metric, float("nan"))
                    row += f"{val:<18.4f}"
                print(row)
        save_path = _save_results({"ablation": ablation}, results_dir)
    else:
        results = run_evaluation(pipeline, ground_truth, cfg, mode=args.mode, top_k=args.top_k)
        _print_summary(results)
        save_path = _save_results(results, results_dir)

    logger.info("Full results saved → %s", save_path)
