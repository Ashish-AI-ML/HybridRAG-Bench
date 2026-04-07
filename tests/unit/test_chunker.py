"""
tests/unit/test_chunker.py — Unit tests for the SentenceChunker.
"""

import pytest
from src.chunking.sentence_chunker import SentenceChunker


SAMPLE_TEXT = (
    "Richard Feynman proposed quantum simulation in 1981. "
    "He argued classical computers cannot simulate quantum systems efficiently. "
    "His work was foundational. "
    "David Deutsch later formalized the quantum Turing machine. "
    "He published in the Proceedings of the Royal Society in 1985."
)


class TestSentenceChunker:
    def test_basic_chunking(self):
        chunker = SentenceChunker(chunk_size=3, overlap=1)
        chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="TEST")
        assert len(chunks) > 0
        assert all("text" in c for c in chunks)
        assert all("chunk_id" in c for c in chunks)
        assert all("doc_id" in c for c in chunks)

    def test_chunk_ids_unique(self):
        chunker = SentenceChunker(chunk_size=2, overlap=1)
        chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="DOC-1")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_doc_id_propagated(self):
        chunker = SentenceChunker(chunk_size=3, overlap=1)
        chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="MY_DOC")
        assert all(c["doc_id"] == "MY_DOC" for c in chunks)

    def test_metadata_enriched(self):
        chunker = SentenceChunker(chunk_size=3, overlap=1)
        chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="TEST")
        # At least some chunks should have year_range metadata
        with_years = [c for c in chunks if c.get("year_range")]
        assert len(with_years) > 0

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            SentenceChunker(chunk_size=3, overlap=3)

    def test_overlap_less_than_chunk_size(self):
        with pytest.raises(ValueError):
            SentenceChunker(chunk_size=2, overlap=2)

    def test_single_sentence(self):
        chunker = SentenceChunker(chunk_size=3, overlap=1)
        chunks = chunker.chunk_text("Just one sentence.", doc_id="DOC-X")
        assert len(chunks) == 1

    def test_source_path_stored(self):
        chunker = SentenceChunker(chunk_size=3, overlap=1)
        chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="DOC-1", source_path="/some/path.txt")
        assert all(c["source_path"] == "/some/path.txt" for c in chunks)
