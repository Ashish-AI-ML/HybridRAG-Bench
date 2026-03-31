import os
import getpass

def _get_api_key():
    # Helper to load a GOOGLE_API_KEY from environment 
    # Or just mock the API if none is provided. Since this is an offline-capable
    # pipeline, we can use a local transformers model, or standard API.
    pass

class Generator:
    def __init__(self, use_mock=True):
        """
        Initialize the generation component. 
        For true local operation, we could use transformers pipeline here, 
        but to avoid downloading massive models during development, a precise
        prompt injection engine with mock generation is perfectly suited, or we can use 
        an actual local model like google/flan-t5-small.
        
        We will implement a small local flan-t5 inference engine for zero-cost API.
        """
        self.use_mock = use_mock
        if not use_mock:
            from transformers import pipeline
            print("Loading local generation model (google/flan-t5-small)...")
            # For 2-4 factual sentences, flan-t5 is decent. 
            self.pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=100)
    
    def format_prompt(self, question: str, context_chunks: list[dict], strict: bool = True) -> str:
        """
        Injects context chunks into the prompt.
        Task 2C: Prompt template must separately inject context and question, 
        instruct the model to answer ONLY from context (strict mode default), 
        keep it 2-4 sentences, and say "I don't know" if missing.
        """
        
        context_text = "\n\n".join([f"Source {c['doc_id']}:\n{c['text']}" for c in context_chunks])
        
        if strict:
            instruction = (
                "You are an expert encyclopedic assistant. Your task is to accurately answer the question "
                "using ONLY the specific information provided in the Context below. "
                "Do NOT use any outside knowledge. If the answer is not explicitly stated in the context, "
                "you must reply exactly with: 'I don't know.'\n"
                "Provide a concise, factual answer of exactly 2 to 4 sentences."
            )
        else:
            instruction = (
                "You are an expert encyclopedic assistant. Your task is to accurately answer the question. "
                "Please prioritize using the Context provided below. If the context is insufficient, "
                "you may supplement with your own high-quality knowledge, but ensure the answer is factual.\n"
                "Provide a concise, factual answer of exactly 2 to 4 sentences."
            )
            
        prompt = (
            f"{instruction}\n\n"
            f"=== CONTEXT ===\n"
            f"{context_text}\n"
            f"===============\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return prompt

    def generate(self, prompt: str) -> str:
        """Execute generation over the local model."""
        if self.use_mock:
            return "This is a mocked answer generated from the context. It effectively simulates local deterministic RAG output for prototyping."
        
        res = self.pipe(prompt)
        return res[0]["generated_text"].strip()
