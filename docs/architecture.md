# Technical Architecture

## Overview

HybridRAG Bench is built around a three-stage retrieval architecture designed to address the two core failure modes of naive RAG systems:

1. **Dense-only retrieval** misses lexically precise queries (exact names, dates, acronyms)
2. **Most RAG systems have no systematic evaluation** — they cannot prove they're working or detect regressions

The system combines lexical precision (BM25) with semantic recall (Qdrant dense vectors) and optimizes the final ranking with a cross-encoder reranker.

---

## Pipeline Architecture

### Stage 1: BM25 Lexical Retrieval

**What it does**: Scores all corpus chunks against the query using the BM25Okapi algorithm (term frequency with document length normalization).

**Why it matters**: BM25 retrieves documents containing the *exact words* the user typed. For a history of quantum computing corpus, this is critical for:
- Proper names: `Feynman`, `Martinis`, `Sycamore`, `Wootters`
- Specific dates: `1981`, `1994`, `2019`
- Acronyms: `RSA`, `NMR`, `UCSB`, `NISQ`
- Exact technical terms: `no-cloning theorem`, `amplitude amplification`

Dense embeddings trained on general text often underrepresent rare domain-specific terms in their learned space, causing silent retrieval failures on these exact-match queries.

**Implementation**: `rank_bm25` (BM25Okapi), in-memory, tokenized with lowercase + punctuation stripping.

---

### Stage 2: Dense Vector Retrieval (Qdrant)

**What it does**: Encodes the query with `all-MiniLM-L6-v2` (384-dimensional) and performs cosine similarity search against the Qdrant vector store.

**Why it matters**: Dense retrieval finds documents that are *semantically related* even when they share no vocabulary with the query:
- "How does quantum factoring threaten public-key encryption?" → retrieves Shor's algorithm document even if the exact phrase "public-key" doesn't appear
- Handles paraphrases, synonyms, and conceptual reformulations

**Why Qdrant over FAISS**:
| Capability | FAISS | Qdrant |
|---|---|---|
| Persistence | None (in-memory) | Full disk-backed WAL |
| Metadata filtering | None | Native payload filtering |
| Incremental updates | Full rebuild | Insert/delete in place |
| API surface | Python-only | REST + gRPC |

---

### Stage 3: Reciprocal Rank Fusion (RRF)

**What it does**: Merges the BM25 and dense candidate lists into a single unified ranking using rank positions (not raw scores, which are incomparable).

**Formula**: `RRF(d) = Σ 1 / (k + rank_i(d))` where k=60 (standard smoothing constant)

**Why RRF over linear combination**: BM25 scores are term-frequency based integers; cosine similarities are floats in [-1, 1]. These scales are incompatible for direct addition. RRF uses only the rank position of each document in each list, making it scale-invariant and robust.

---

### Stage 4: Cross-Encoder Reranking

**What it does**: Takes the top-N candidates from Stage 3 (default: 20) and scores each with a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that jointly encodes the query and passage.

**Why cross-encoders are more accurate**: Bi-encoders (used in Stage 2) encode query and passage *independently* — the relevance signal is limited to their dot product. Cross-encoders see the query and passage *together*, with full attention across both, producing far more accurate relevance scores.

**The tradeoff**: Cross-encoders are O(n) — too slow to score the entire corpus. The two-stage design (fast bi-encoder for broad recall, slow cross-encoder for final precision) solves this.

**Typical improvement**: 10–25% NDCG gain over reranker-less pipelines in standard benchmarks.

---

## Evaluation Framework

### Retrieval Quality (5 metrics)

| Metric | Definition |
|---|---|
| Precision@K | Fraction of top-K retrieved documents that are relevant |
| Recall@K | Fraction of all relevant documents found in top-K |
| MRR | 1/rank of the first relevant document |
| Hit Rate@K | Binary: did any relevant document appear in top-K? |
| NDCG@K | Rank-weighted relevance gain (accounts for position in ranking) |

### Generation Quality (4 metrics)

| Metric | Definition |
|---|---|
| ROUGE-1/L | Unigram and longest-subsequence token overlap with reference |
| Semantic Similarity | Cosine similarity of answer embeddings (paraphrase-aware) |
| Entity Recall | Fraction of key named entities from reference found in generated answer |
| Faithfulness | Fraction of generated sentences semantically supported by retrieved context |

### Operational Metrics

| Metric | Definition |
|---|---|
| Latency (p50/p90/p99) | Per-stage wall-clock time across all pipeline stages |
| Token Usage | Input/output token counts per query |
| Cost Projection | Estimated monthly API cost at 100/1K/10K queries/day |

---

## Configuration System

All pipeline parameters are externalized to `configs/default.yaml`. No values are hardcoded in source. Profile overrides (`configs/dev.yaml`) allow environment-specific settings without modifying source code.

Secrets (API keys) are loaded exclusively from environment variables via `.env`.

## Test Suite Structure

| Tier | Purpose | Examples |
|---|---|---|
| Tier 1: Direct Lookup | Basic factual retrieval | "What year did Feynman propose quantum simulation?" |
| Tier 2: Synthesis | Multi-doc cross-reference | "How did Deutsch's work build on Feynman's ideas?" |
| Tier 3: Adversarial | Failure mode stress testing | Trick questions, out-of-domain, conflation traps |
| Tier 4: Robustness | Query variation tolerance | Paraphrases, typos, verbose/minimal queries |
