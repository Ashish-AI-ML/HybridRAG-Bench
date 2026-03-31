# History of Quantum Computing: Custom RAG & Evaluation Framework

## 1. Project Overview & Chosen Domain
This project implements a custom Retrieval-Augmented Generation (RAG) system dedicated entirely to a highly specialized niche: **The History of Quantum Computing**. 

**Why this domain?** 
The history of quantum computing is an exceptional proving ground for RAG architecture. 
1. **Factual Density**: The core narrative relies on highly specific names (Shor, Feynman, Deutsch) and years (1981, 1994, 2019), enabling extremely rigorous exact-match measurement.
2. **Dense Terminology**: Base LLMs typically hallucinate heavily when discussing quantum mechanics out of context (e.g., confusing annealing with universal gate models). Supplying accurate historical grounding immediately stabilizes generative outputs.

## 2. Dataset Description
The dataset was painstakingly constructed directly by the researcher to ensure absolute systemic integrity. It features **strict subtopic isolation with zero content overlap**. If a fact exists in one document, it absolutely does not exist anywhere else. 
- **Documents**: 8 independently constructed `.txt` documents.
- **Size**: Approximately 440–480 words per document.
- **Domain Scope**: Feynman's 1981 Proposal, Deutsch's Turing Machine, Shor's Factoring, Grover's Search, Early Hardware (NMR/D-Wave), Error Correction Foundations, Superconducting Milestones, and Google's "Sycamore" Supremacy. 
- **Ground Truth**: A robust `ground_truth.json` file dictates 12 complex Q&A pairs carefully mapped back to target arrays. 
    - *Q01-Q08*: Single-document evaluations.
    - *Q09-Q12*: Advanced synthesis questions demanding multi-hop RAG retrievals spanning exactly 2 separate documents.
    - *Metadata*: Document bounds, mappings, and exact-matches are rigorously detailed in `data/dataset_metadata.txt`.

## 3. RAG Pipeline Design
The system prioritizes extreme computational transparency over massive abstract libraries. All logic operates strictly offline and incurs zero API costs.

- **Chunking Method (`src/chunker.py`)**: Uses precision sentence-level boundaries. Paragraph-level chunking consistently scoops up too much irrelevant historical padding, diluting the embeddings. We utilize 3-sentence blocks with a 1-sentence overlap. The overlap ensures mid-paragraph context transitions aren't violently decapitated by hard index limits.
- **Embedding Model**: **`sentence-transformers/all-MiniLM-L6-v2`**. This 384-dimensional dense encoder runs natively on CPU hardware extremely quickly while retaining highly accurate semantic similarity mappings across technical terminology.
- **Vector Store**: **`FAISS` (Facebook AI Similarity Search)**. Specifically utilizing normalized vectors and `faiss.IndexFlatIP` to calculate Cosine Similarity. Because the entire corpus maps into roughly 52 chunks, in-memory FAISS indexes operate instantaneously.
- **LLM Prompt Strategy**: The prompt explicitly sandboxes the context away from the user query. The fundamental directive rigidly instructs the generator to solely exploit the provided Context array, ordering it to respond with "I don't know" rather than supplementing external pre-training weights regarding physics.

## 4. Evaluation Framework
A comprehensive, multi-tiered framework mathematically validates the retrieval and generation phases individually.

**Execution Metrics (`src/evaluation/metrics.py`)**
1. **Token Overlap**: `rouge-score` yields classic ROUGE-1 and ROUGE-L ratios measuring identical unigram crossover, catching basic structural integrity.
2. **Semantic Similarity**: Passes the expected and generated outputs back through the `all-MiniLM-L6-v2` embedding engine and returns the Cosine distance to calculate true conceptual overlap, bypassing paraphrasing penalties.
3. **Exact-Match Retrieval**: Extracts capitalized nouns (e.g., "Sycamore") and dates ("2019") from the expected text, verifying that 100% of the critical trivia successfully survived the generator's summarization.

### Retrieval Results
When tested against the custom cross-document dataset, the custom pipeline retrieved:
- **Mean Precision@1**: `1.00`
- **Mean Precision@3**: `0.97`
- **Mean Reciprocal Rank (MRR)**: `1.00`
*(The FAISS array effectively ranked the absolute target document in the #1 position across all 12 queries, even the mathematically complex `DOC-3, DOC-4` multi-hops).*

### Qualitative Terminal Framework
A completely custom terminal UI (`eval_cli.py`) powers the grading of Coherence, Completeness, Factual Accuracy, and Grounding. The UX loop rigorously prevents corrupted user skips by rejecting empty strings, rendering inputs strictly into the JSON logging table:
```text
  +-----------------------+-------+
  | Factual Correctness   |   1   |
  | Completeness          |   2   |
  | Coherence             |   3   |
  | Grounding             |   1   |
  +-----------------------+-------+
```

## 5. Challenges and Lessons Learned
Building the full-stack architecture from empty Python files up to the metrics dashboard revealed three massive developmental lessons:

1. **The "Trivia Mismatch" Evaluation Flaw**: The most insidious problem encountered was attempting to grade a generative LLM's "Grounding" (did it stick exclusively to the text?) against overly specific trivia questions (e.g. *"What specific chemical compound state was utilized... "*). The model was successfully retrieving the target document, but because the question only had one fundamental factual answer, evaluating "hallucinations vs. grounded knowledge" became impossible for the human grader. **Lesson:** *Questions driving RAG quality rubrics must fundamentally demand thematic synthesis rather than strict trivia lookup.*
2. **Qualitative UX Data Corruption**: When initially testing the CLI tool, using standard `<Enter>` logic to skip score columns accidentally polluted the final telemetry JSON with `""` empty strings, silently breaking math pipeline averages. **Lesson:** *Human-in-the-loop scoring arrays must aggressively enforce strict input data-typing (e.g., rejecting all values outside exactly `1, 2, 3, n`) prior to appending the file logic.*
3. **Cross-Document Semantic Bleeding**: During development, historically overlapping terms (like "quantum entanglement") occasionally confused basic term-frequency retrievers natively. **Lesson:** *Strictly bounding datasets (e.g. isolating Shor's 1994 logic strictly from Shor's 1995 logic) paired with dense dense embeddings (over standard TF-IDF models) is fundamentally required to successfully drive Precision@1 ratios.*

## 6. How to Run This Project

### Prerequisites
Before running any code, ensure you have Python 3.9+ installed and create a virtual environment to isolate the heavy ML dependencies:
```bash
# Create and activate a Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install all dependencies (PyTorch, FAISS, Sentence-Transformers, etc.)
pip install -r requirements.txt
```

### 1. Evaluate Retrieval Performance
To test the core FAISS vector embeddings against the custom ground truth dataset and calculate Precision@K and Mean Reciprocal Rank:
```bash
python -m src.evaluation.retrieval_eval
```

### 2. Run the Qualitative Evaluation CLI
To launch the interactive, terminal-based human grading UI (which strictly logs Factual Correctness, Completeness, Coherence, and Grounding):
```bash
python -m src.evaluation.eval_cli
```

### 3. Test the Full Pipeline
To test an arbitrary query directly against the generator and retriever logic:
```bash
python -m src.pipeline
```
