"""
retrieval/bm25_retriever.py — BM25 lexical retrieval component.

Provides the sparse (keyword-based) half of the hybrid retrieval system.
BM25 excels at matching:
  - Proper nouns and named entities (Feynman, Sycamore, UCSB)
  - Exact dates and years (1981, 2019)
  - Acronyms and technical terms (RSA, NMR, qubit)
  - Any query where users know the exact terminology

This complements dense retrieval, which is strong at semantic similarity
but weak on precise keyword matching.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.logger import get_logger

logger = get_logger(__name__)

# Simple tokenizer — lowercase, strip punctuation
_PUNCT_RE = re.compile(r"[^\w\s]")


def _tokenize(text: str) -> list[str]:
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return [tok for tok in cleaned.split() if len(tok) > 1]


class BM25Retriever:
    """
    In-memory BM25 retriever backed by rank_bm25.

    The corpus is built once from the chunk list and held in memory.
    For the project scale (~50–200 chunks), this is appropriate.
    For larger corpora, this would be replaced with Elasticsearch.

    Parameters
    ----------
    chunks : list[dict]
        Pre-built chunk dicts (the same format produced by SentenceChunker).
    """

    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    # ── Public API ──────────────────────────────────────────────────────────

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build the BM25 index from a list of chunk dicts."""
        if not chunks:
            logger.warning("build_index called with empty chunk list")
            return

        t0 = time.perf_counter()
        self._chunks = chunks
        tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        elapsed = time.perf_counter() - t0
        logger.info(
            "BM25 index built: %d chunks in %.3fs", len(chunks), elapsed
        )

    def is_populated(self) -> bool:
        return self._bm25 is not None and len(self._chunks) > 0

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Score all chunks against *query* using BM25 and return top-K results.

        Returns
        -------
        list of chunk dicts with an added ``bm25_score`` field.
        """
        if not self.is_populated():
            raise RuntimeError("BM25 index is empty. Call build_index() first.")

        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning("Query tokenized to empty list — returning empty BM25 results")
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair chunks with their scores, sort descending
        ranked = sorted(
            zip(scores, self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for score, chunk in ranked[:top_k]:
            c = dict(chunk)
            c["bm25_score"] = float(score)
            results.append(c)

        return results

    def save_index(self, path: str | Path) -> None:
        """Persist the chunk list so the index can be rebuilt without re-chunking."""
        Path(path).write_text(
            json.dumps(self._chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("BM25 corpus saved to '%s'", path)

    def load_corpus(self, path: str | Path) -> None:
        """Load a previously saved corpus and rebuild the BM25 index."""
        chunks = json.loads(Path(path).read_text(encoding="utf-8"))
        self.build_index(chunks)
        logger.info("BM25 corpus loaded from '%s' (%d chunks)", path, len(chunks))
