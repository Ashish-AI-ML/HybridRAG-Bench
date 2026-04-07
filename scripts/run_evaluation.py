"""
scripts/run_evaluation.py — CLI entry point for the evaluation suite.

Equivalent to: python -m src.evaluation.suite_runner
Provided as a convenience script at the project root.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --ablation
    python scripts/run_evaluation.py --mode hybrid_rerank --top-k 5
    python scripts/run_evaluation.py --suite tier3_adversarial
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

from src.config import get_config
from src.evaluation.suite_runner import (
    run_evaluation, run_ablation, _load_ground_truth,
    _save_results, _print_summary
)
from src.pipeline import HybridRAGPipeline
from src.logger import get_logger

logger = get_logger(__name__)

SUITE_PATHS = {
    "main":              "data/ground_truth.json",
    "tier1_direct":      "data/test_suites/tier1_direct.json",
    "tier3_adversarial": "data/test_suites/tier3_adversarial.json",
    "tier4_robustness":  "data/test_suites/tier4_robustness.json",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HybridRAG Bench evaluation suite")
    parser.add_argument(
        "--suite",
        default="main",
        choices=list(SUITE_PATHS.keys()),
        help="Which test suite to evaluate against",
    )
    parser.add_argument(
        "--mode",
        default="hybrid_rerank",
        choices=["dense_only", "bm25_only", "hybrid", "hybrid_rerank"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study across all retrieval modes")
    parser.add_argument("--force-reindex", action="store_true",
                        help="Force rebuild the vector index")
    args = parser.parse_args()

    cfg = get_config()
    gt_path = SUITE_PATHS[args.suite]
    results_dir = cfg["evaluation"]["results_dir"]

    logger.info("Loading test suite: %s (%s)", args.suite, gt_path)
    ground_truth = _load_ground_truth(gt_path)
    logger.info("Loaded %d test items", len(ground_truth))

    pipeline = HybridRAGPipeline(cfg=cfg, force_reindex=args.force_reindex)
    pipeline.build_index()

    if args.ablation:
        logger.info("Running full ablation study across all retrieval modes...")
        ablation_results = run_ablation(pipeline, ground_truth, cfg, top_k=args.top_k)

        print("\n\n" + "=" * 80)
        print("  ABLATION STUDY RESULTS")
        print("=" * 80)
        modes = list(ablation_results.keys())
        key_metrics = [
            "mean_precision_at_1", "mean_precision_at_3", "mean_mrr",
            "mean_ndcg_at_5", "mean_semantic_similarity", "mean_faithfulness_score",
        ]
        header = f"{'Metric':<40}" + "".join(f"{m:<20}" for m in modes)
        print(header)
        print("-" * len(header))
        for metric in key_metrics:
            row = f"{metric:<40}"
            for mode in modes:
                val = ablation_results[mode].get(metric, float("nan"))
                row += f"{val:<20.4f}"
            print(row)

        save_path = _save_results(
            {"ablation": ablation_results, "suite": args.suite},
            results_dir
        )
    else:
        results = run_evaluation(
            pipeline, ground_truth, cfg,
            mode=args.mode, top_k=args.top_k
        )
        _print_summary(results)
        save_path = _save_results(results, results_dir)

    logger.info("Results → %s", save_path)
