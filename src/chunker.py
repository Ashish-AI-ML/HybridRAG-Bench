import os
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    # also try punkt_tab for some newer versions
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass


class SentenceChunker:
    def __init__(self, chunk_size: int = 3, overlap: int = 1):
        """
        Sentence-level chunker logic.
        :param chunk_size: number of sentences per chunk.
        :param overlap: number of overlapping sentences between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Validation
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be less than chunk size.")

    def chunk_text(self, text: str, doc_id: str) -> list[dict]:
        """
        Split a document's text into overlapping chunks of sentences.
        Returns a list of chunk dictionaries.
        """
        sentences = nltk.sent_tokenize(text)
        chunks = []
        
        step = self.chunk_size - self.overlap
        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_sentences)
            
            # Optionally stop if a trailing chunk is entirely redundant (i.e. we've exhausted sentences)
            if not chunk_sentences:
                break
                
            chunk_id = f"{doc_id}_chunk_{len(chunks)}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text
            })
            
            # If we've reached the end of the sentences, break early
            if i + self.chunk_size >= len(sentences):
                break
                
        return chunks

    def chunk_directory(self, dir_path: str) -> list[dict]:
        """
        Read all txt files in a directory and chunk them.
        """
        all_chunks = []
        for filename in sorted(os.listdir(dir_path)):
            if filename.endswith(".txt"):
                doc_id = filename.replace(".txt", "")
                filepath = os.path.join(dir_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    
                doc_chunks = self.chunk_text(text, doc_id)
                all_chunks.extend(doc_chunks)
                
        return all_chunks

if __name__ == "__main__":
    chunker = SentenceChunker(chunk_size=3, overlap=1)
    chunks = chunker.chunk_directory(os.path.join("data", "docs"))
    print(f"Generated {len(chunks)} chunks across all documents.")
    print("Example chunk:", chunks[0])
