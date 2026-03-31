import faiss
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer

# Suppress some noisy warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning)

class FaissRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model and FAISS vector store.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # We need the dimension of the embedding space
        self.d = self.model.get_sentence_embedding_dimension()
        
        # FAISS index for L2 distance. Cosine similarity can be used if vectors are normalized.
        # all-MiniLM-L6-v2 outputs are typically normalized, but we'll enforce it for Inner Product.
        self.index = faiss.IndexFlatIP(self.d)
        
        # Track inserted chunks to map indices back to content
        self.chunks_mapping = []

    def add_chunks(self, chunks: list[dict]):
        """
        Embed and add the chunks to the FAISS index.
        """
        print(f"Embedding {len(chunks)} chunks...")
        texts = [chunk["text"] for chunk in chunks]
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Normalize vectors for Inner Product to behave like Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.chunks_mapping.extend(chunks)
        print(f"Index size: {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the top_k chunks for a given query.
        Returns the mapped chunks along with their similarity score.
        """
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        # Return top K mappings
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue # FAISS returns -1 if there are fewer vectors than k
            chunk = self.chunks_mapping[idx].copy()
            chunk["score"] = float(score)
            chunk["rank"] = rank + 1
            results.append(chunk)
            
        return results

if __name__ == "__main__":
    from src.chunker import SentenceChunker
    import os
    
    # Simple test
    chunker = SentenceChunker(chunk_size=3, overlap=1)
    chunks = chunker.chunk_directory(os.path.join("data", "docs"))
    
    retriever = FaissRetriever()
    retriever.add_chunks(chunks)
    
    res = retriever.search("What is Shor's algorithm?", top_k=2)
    print("Search Results:")
    for r in res:
        print(f"[{r['doc_id']}] Score: {r['score']:.4f}\n{r['text']}\n")
