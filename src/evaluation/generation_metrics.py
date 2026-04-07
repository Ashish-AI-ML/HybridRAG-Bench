"""
evaluation/generation_metrics.py — Answer quality evaluation.

Covers:
  1. ROUGE-1 / ROUGE-L        (token overlap, lexical similarity)
  2. Semantic cosine similarity (paraphrase-aware similarity)
  3. Exact entity match        (factual keyword coverage)
  4. Faithfulness (heuristic)  (does the answer stay within the retrieved context?)

The faithfulness metric is the most critical for production RAG systems.
It detects *hallucination*: claims in the answer that are NOT in the context.

LLM-as-judge faithfulness is available when a judge model is configured
(see evaluation/faithfulness_judge.py). This module implements the fast
heuristic version for dev/offline evaluation.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from src.logger import get_logger

logger = get_logger(__name__)

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_STOPWORDS = {
    "The", "This", "That", "These", "Those", "With", "From", "Into",
    "They", "Their", "When", "Where", "Who", "What", "How", "Why",
    "For", "And", "But", "Yet", "Nor", "Its", "Has", "Had", "Based",
}


def _extract_factual_entities(text: str) -> set[str]:
    years = set(_YEAR_RE.findall(text))
    nouns = {w for w in _PROPER_NOUN_RE.findall(text) if w not in _STOPWORDS}
    return years | nouns


class GenerationEvaluator:
    """
    Multi-metric answer quality evaluator.

    Share one instance across all evaluations to avoid reloading the
    sentence-transformer model on each call.
    """

    def __init__(self, semantic_model: str = "all-MiniLM-L6-v2") -> None:
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        logger.info("Loading semantic evaluator model: %s", semantic_model)
        self.sem_model = SentenceTransformer(semantic_model)

    # ── Individual metrics ──────────────────────────────────────────────────

    def rouge_scores(self, reference: str, generated: str) -> dict[str, float]:
        scores = self.rouge.score(reference, generated)
        return {
            "rouge1_f": round(scores["rouge1"].fmeasure, 4),
            "rougeL_f": round(scores["rougeL"].fmeasure, 4),
        }

    def semantic_similarity(self, reference: str, generated: str) -> float:
        if not generated.strip():
            return 0.0
        emb_ref = self.sem_model.encode(reference, convert_to_numpy=True)
        emb_gen = self.sem_model.encode(generated, convert_to_numpy=True)
        sim = float(util.cos_sim(emb_ref, emb_gen))
        return round(max(0.0, sim), 4)

    def entity_recall(self, reference: str, generated: str) -> dict[str, Any]:
        """
        Checks what fraction of key factual entities in *reference* are
        present in *generated*.  High entity recall = low hallucination risk.
        """
        expected_entities = _extract_factual_entities(reference)
        if not expected_entities:
            return {"entity_recall": 1.0, "missing_entities": [], "total_entities": 0}

        gen_lower = generated.lower()
        missing = [e for e in expected_entities if e.lower() not in gen_lower]
        recall = round(1.0 - len(missing) / len(expected_entities), 4)

        return {
            "entity_recall": recall,
            "missing_entities": sorted(missing),
            "total_entities": len(expected_entities),
        }

    def heuristic_faithfulness(
        self, generated: str, retrieved_chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Heuristic faithfulness: checks what fraction of sentences in the
        generated answer are semantically supported by any retrieved chunk.

        This is a fast proxy — use LLM-as-judge for authoritative faithfulness.

        Returns
        -------
        dict with:
          - faithfulness_score : 0–1 (fraction of sentences supported)
          - unsupported_sentences : list of generated sentences with no chunk support
        """
        if not generated.strip() or not retrieved_chunks:
            return {"faithfulness_score": 0.0, "unsupported_sentences": []}

        # Split generated answer into sentences
        import nltk
        try:
            sentences = nltk.sent_tokenize(generated)
        except Exception:
            sentences = [s.strip() for s in generated.split(".") if s.strip()]

        if not sentences:
            return {"faithfulness_score": 1.0, "unsupported_sentences": []}

        context_texts = [c["text"] for c in retrieved_chunks]
        context_combined = " ".join(context_texts).lower()

        supported = 0
        unsupported: list[str] = []

        for sent in sentences:
            sent_emb = self.sem_model.encode(sent, convert_to_numpy=True)
            # Check if any chunk has cosine > 0.5 with this sentence
            is_supported = False
            for ctx_text in context_texts:
                ctx_emb = self.sem_model.encode(ctx_text, convert_to_numpy=True)
                if float(util.cos_sim(sent_emb, ctx_emb)) > 0.45:
                    is_supported = True
                    break
            if is_supported:
                supported += 1
            else:
                unsupported.append(sent)

        score = round(supported / len(sentences), 4)
        return {
            "faithfulness_score": score,
            "unsupported_sentences": unsupported,
        }

    # ── Combined scorer ─────────────────────────────────────────────────────

    def evaluate(
        self,
        reference: str,
        generated: str,
        retrieved_chunks: list[dict[str, Any]] | None = None,
        question: str = "",
    ) -> dict[str, Any]:
        """
        Run all generation metrics and return a unified result dict.

        Parameters
        ----------
        reference : str       Expected answer (ground truth)
        generated : str       LLM-generated answer
        retrieved_chunks : list | None
            Provide for faithfulness evaluation.
        """
        results: dict[str, Any] = {}

        # Shortcut: detect "I don't know" responses
        if "INSUFFICIENT CONTEXT" in generated.upper():
            results["is_abstention"] = True
            results["rouge1_f"] = 0.0
            results["rougeL_f"] = 0.0
            results["semantic_similarity"] = 0.0
            results["entity_recall"] = 0.0
            results["missing_entities"] = []
            results["faithfulness_score"] = 1.0  # abstention is technically faithful
            results["unsupported_sentences"] = []
            return results

        results["is_abstention"] = False
        results.update(self.rouge_scores(reference, generated))
        results["semantic_similarity"] = self.semantic_similarity(reference, generated)

        entity_res = self.entity_recall(reference, generated)
        results["entity_recall"] = entity_res["entity_recall"]
        results["missing_entities"] = entity_res["missing_entities"]

        if retrieved_chunks:
            faith_res = self.heuristic_faithfulness(generated, retrieved_chunks)
            results["faithfulness_score"] = faith_res["faithfulness_score"]
            results["unsupported_sentences"] = faith_res["unsupported_sentences"]
        else:
            results["faithfulness_score"] = None
            results["unsupported_sentences"] = []

        return results
