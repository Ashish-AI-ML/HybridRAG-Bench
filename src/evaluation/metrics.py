import re
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import numpy as np

class TextMetricsEvaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the quantitative metrics evaluator.
        """
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        # We reuse the same lightweight model as retrieval to compare semantics
        print(f"Loading Evaluator Model ({model_name})...")
        self.semantic_model = SentenceTransformer(model_name)
        
    def score_rouge(self, reference: str, generated: str) -> dict:
        """
        Computes ROUGE-1 (unigram token overlap) and ROUGE-L (longest common subsequence).
        Helpful for catching structural overlap, though it completely misses
        semantic paraphrasing.
        """
        scores = self.rouge.score(reference, generated)
        return {
            "ROUGE-1_fmeasure": scores['rouge1'].fmeasure,
            "ROUGE-L_fmeasure": scores['rougeL'].fmeasure
        }
        
    def score_semantic_similarity(self, reference: str, generated: str) -> float:
        """
        Computes cosine similarity between the embeddings of the expected vs. 
        generated answers. This catches semantic matches where ROUGE fails (e.g. 
        synonyms and rephrasings). Thresholds around 0.6+ generally indicate 'good' matches.
        """
        if not generated.strip():
            return 0.0
            
        ref_emb = self.semantic_model.encode(reference, convert_to_numpy=True)
        gen_emb = self.semantic_model.encode(generated, convert_to_numpy=True)
        
        # Calculate cosine similarity using sentence_transformers util
        cos_sim = util.cos_sim(ref_emb, gen_emb).item()
        return max(0.0, cos_sim) # bound at 0
        
    def _extract_keywords(self, text: str) -> set:
        """
        Simple heuristic logic to pull key names (capitalized words generally), 
        years (4 digits), or specific acronyms out of the expected text.
        """
        # Find 4 digit years
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        
        # Find heavily capitalized specific nouns / algorithms 
        # e.g., "Shor's", "Grover's", "Feynman", "Turing", "Sycamore"
        # Since standard NLP pipelining (spacy NER) might be too heavy here, 
        # we will use regex for Capitalized Words (excluding start of sentence heuristically)
        # But for robustness, we specifically look for specific domains in the text.
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Filter out extremely common stop words that might be capitalized at start of sentences
        stop_words = {"In", "The", "A", "An", "What", "How", "Why", "Instead", "It", "They", "But", "And", "Because"}
        entities = [w for w in capitalized if w not in stop_words]
        
        keywords = set(years + entities)
        return keywords
        
    def score_exact_match(self, reference: str, generated: str) -> dict:
        """
        Extracts key factual entities (names/years) from the expected answer and checks 
        if they are strictly present in the generated answer. This catches hallucinatory 
        failure modes where the semantic similarity is high but the specific factual year 
        or name is completely wrong.
        Outputs 1.0 (Pass) if all highly critical keywords are present, else 0.0.
        """
        if not generated.strip():
            return {"exact_match_score": 0.0, "missing": ["All"]}
            
        expected_keywords = self._extract_keywords(reference)
        if not expected_keywords:
            return {"exact_match_score": 1.0, "missing": []} # Vacuous truth if no strict named entities
            
        generated_lower = generated.lower()
        missing = []
        for kw in expected_keywords:
            if kw.lower() not in generated_lower:
                missing.append(kw)
                
        # We define exact match mathematically as 1 if all found, else 0. 
        # Realistically, a fractional recall is better. Let's provide fractional.
        score = 1.0 - (len(missing) / len(expected_keywords))
        
        return {
            "exact_match_score": float(np.round(score, 2)),
            "missing_keywords": missing,
            "target_keywords": list(expected_keywords)
        }

    def evaluate_all(self, reference: str, generated: str) -> dict:
        results = {}
        results.update(self.score_rouge(reference, generated))
        results["Semantic_Cosine"] = self.score_semantic_similarity(reference, generated)
        em_res = self.score_exact_match(reference, generated)
        results["Exact_Match"] = em_res["exact_match_score"]
        results["Missing_Keywords"] = em_res["missing_keywords"]
        return results

if __name__ == "__main__":
    evaluator = TextMetricsEvaluator()
    ref = "In 1994, Peter Shor invented Shor's algorithm, proving massive speedups."
    gen1 = "Peter Shor created his algorithm in 1994, which demonstrated huge speed advantages."
    gen2 = "In 1994, Lov Grover invented a search algorithm."
    
    print("Good Generation:")
    print(evaluator.evaluate_all(ref, gen1))
    
    print("\nBad Generation (hallucination):")
    print(evaluator.evaluate_all(ref, gen2))
