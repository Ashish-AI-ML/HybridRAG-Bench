import json
import logging
import pandas as pd

def calculate_precision_at_k(retrieved_doc_ids: list[str], ground_truth_doc_ids: list[str], k: int) -> float:
    """
    Precision@K measures what proportion of the top K retrieved documents 
    are actually relevant (exist in the ground truth list for that question).
    """
    top_k_retrieved = retrieved_doc_ids[:k]
    # In some datasets, multiple ground truth docs exist. We just check if they are in the set.
    relevant_retrieved = [doc_id for doc_id in top_k_retrieved if doc_id in ground_truth_doc_ids]
    
    # Precision is (Relevant Docs / K)
    # Some definitions of P@K for classification use (Relevant Retrieved / min(K, Total Relevant))
    # Standard P@K is simply Relevant/K.
    if not top_k_retrieved:
        return 0.0
    return len(relevant_retrieved) / k

def calculate_mrr(retrieved_doc_ids: list[str], ground_truth_doc_ids: list[str]) -> float:
    """
    Mean Reciprocal Rank (MRR) finds the rank of the *first* relevant document 
    retrieved and returns 1/rank. If no relevant document is retrieved, it is 0.
    """
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in ground_truth_doc_ids:
            return 1.0 / rank
    return 0.0

class RetrievalEvaluator:
    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)
            
    def evaluate_retrieval(self, pipeline, top_k: int = 3):
        """
        Run the provided RAG pipeline against all ground truth queries,
        specifically evaluating only the chunks retrieved.
        """
        results = []
        
        for idx, item in enumerate(self.ground_truth):
            question = item["question"]
            expected_docs = item["source_docs"]
            
            # Execute retrieval
            retrieved_chunks = pipeline.retriever.search(question, top_k=top_k)
            retrieved_doc_ids = [chunk["doc_id"] for chunk in retrieved_chunks]
            
            p_at_1 = calculate_precision_at_k(retrieved_doc_ids, expected_docs, k=1)
            p_at_3 = calculate_precision_at_k(retrieved_doc_ids, expected_docs, k=3)
            mrr = calculate_mrr(retrieved_doc_ids, expected_docs)
            
            # Log exact position
            correct_rank = "Not Found"
            for r, d_id in enumerate(retrieved_doc_ids, start=1):
                if d_id in expected_docs:
                    correct_rank = str(r)
                    break
            
            results.append({
                "Q_ID": idx + 1,
                "Question": question[:40] + "...",
                "Expected_Docs": ", ".join(expected_docs),
                "Top_3_Retrieved": ", ".join(retrieved_doc_ids),
                "Correct_Doc_Rank": correct_rank,
                "P@1": round(p_at_1, 2),
                "P@3": round(p_at_3, 2),
                "MRR": round(mrr, 2)
            })
            
        self._display_report(results)
        return results
        
    def _display_report(self, results: list[dict]):
        print("\n" + "="*80)
        print(" RETRIEVAL PERFORMANCE REPORT")
        print("="*80)
        
        # Display as tabular cleanly
        try:
            # Simple tabular display using Pandas if available
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            print("-" * 80)
            print(f"Mean P@1: {df['P@1'].mean():.2f}")
            print(f"Mean P@3: {df['P@3'].mean():.2f}")
            print(f"Mean Reciprocal Rank (MRR): {df['MRR'].mean():.2f}")
        except ImportError:
            # Fallback format
            for r in results:
                print(f"Q{r['Q_ID']} | Expected: {r['Expected_Docs']} | Retrieved: {r['Top_3_Retrieved']} | MRR: {r['MRR']} | P@1: {r['P@1']}")

if __name__ == "__main__":
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline(data_dir="data/docs")
    pipeline.build_index()
    
    evaluator = RetrievalEvaluator("data/ground_truth.json")
    evaluator.evaluate_retrieval(pipeline, top_k=3)
