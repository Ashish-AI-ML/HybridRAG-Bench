import os
import json
from src.chunker import SentenceChunker
from src.retriever import FaissRetriever
from src.generator import Generator

class RAGPipeline:
    def __init__(self, data_dir: str = "data/docs", use_mock_gen: bool = True):
        self.chunker = SentenceChunker(chunk_size=3, overlap=1)
        self.retriever = FaissRetriever(model_name="all-MiniLM-L6-v2")
        self.generator = Generator(use_mock=use_mock_gen)
        self.data_dir = data_dir
        
    def build_index(self):
        """Chunk all documents and load them into FAISS."""
        print(f"Reading docs from {self.data_dir}")
        chunks = self.chunker.chunk_directory(self.data_dir)
        self.retriever.add_chunks(chunks)
        print("Index successfully built.")
        
    def query(self, question: str, top_k: int = 3, strict: bool = True) -> dict:
        """Run the end-to-end RAG process."""
        print(f"Retrieving for query: '{question}'")
        retrieved_chunks = self.retriever.search(question, top_k=top_k)
        
        prompt = self.generator.format_prompt(question, retrieved_chunks, strict=strict)
        
        print("Generating answer...")
        answer = self.generator.generate(prompt)
        
        return {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "prompt_used": prompt,
            "generated_answer": answer
        }

if __name__ == "__main__":
    pipeline = RAGPipeline(use_mock_gen=True) # Set to False if downloading flan-t5
    pipeline.build_index()
    
    res = pipeline.query("When did Peter Shor propose his algorithm and what limit did it bypass?", top_k=2)
    print("\n--- Answer ---")
    print(res["generated_answer"])
    print("\n--- Retrieved Sources ---")
    for ch in res["retrieved_chunks"]:
        print(f"[{ch['rank']}] {ch['doc_id']}: {ch['text'][:100]}...")
