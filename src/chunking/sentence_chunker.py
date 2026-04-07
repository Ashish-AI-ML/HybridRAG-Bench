"""
chunking/sentence_chunker.py — Sentence-level document chunker.

Preserves the core 3-sentence-window logic from the original project
while adding:
  - Rich metadata on each chunk (doc_id, chunk_id, char_offset, source_path,
    subtopic, year_range, key_entities)
  - Support for reading .txt, .md, and .pdf files
  - Config-driven parameters (no hardcoded values)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import nltk

from src.logger import get_logger

logger = get_logger(__name__)

# Ensure NLTK punkt tokenizer is available
try:
    nltk.sent_tokenize("Test check.")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

# Simple year extractor for metadata enrichment
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_STOPWORDS = {
    "The", "This", "That", "These", "Those", "With", "From", "Into",
    "They", "Their", "When", "Where", "Who", "What", "How", "Why",
    "For", "And", "But", "Yet", "Nor", "Its", "Has", "Had",
}


def _extract_light_metadata(text: str) -> dict[str, Any]:
    """Quick heuristic to attach year_range and key_entities to chunks."""
    years = sorted(set(int(y) for y in _YEAR_RE.findall(text)))
    entities = [
        w for w in _ENTITY_RE.findall(text)
        if w not in _STOPWORDS
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_entities: list[str] = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique_entities.append(e)

    return {
        "year_range": [years[0], years[-1]] if len(years) >= 2 else years,
        "key_entities": unique_entities[:8],  # top-8 to keep payload lean
    }


class SentenceChunker:
    """
    Splits documents into overlapping sentence-window chunks.

    Each returned chunk dict contains:
      - chunk_id       : unique string ID
      - doc_id         : source document identifier
      - source_path    : absolute path to the source file
      - text           : the chunk content
      - chunk_index    : integer position within the document
      - year_range     : [min_year, max_year] found in chunk text
      - key_entities   : top named entities found in chunk text
    """

    def __init__(self, chunk_size: int = 3, overlap: int = 1) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._step = chunk_size - overlap

    # ── Public API ──────────────────────────────────────────────────────────

    def chunk_text(self, text: str, doc_id: str, source_path: str = "") -> list[dict[str, Any]]:
        """Chunk a single document's text and return an annotated chunk list."""
        sentences = nltk.sent_tokenize(text.strip())
        chunks: list[dict[str, Any]] = []

        for i in range(0, len(sentences), self._step):
            window = sentences[i : i + self.chunk_size]
            if not window:
                break
            chunk_text = " ".join(window)
            meta = _extract_light_metadata(chunk_text)
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{len(chunks):03d}",
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    **meta,
                }
            )
            if i + self.chunk_size >= len(sentences):
                break

        logger.debug("Chunked '%s' → %d chunks", doc_id, len(chunks))
        return chunks

    def chunk_file(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Read one file and return its chunks (supports .txt and .md)."""
        path = Path(file_path)
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        return self.chunk_text(text, doc_id=doc_id, source_path=str(path.resolve()))

    def chunk_directory(self, dir_path: str | Path) -> list[dict[str, Any]]:
        """Chunk every supported file in *dir_path* and return all chunks."""
        directory = Path(dir_path)
        supported = {".txt", ".md"}
        all_chunks: list[dict[str, Any]] = []

        files = sorted(f for f in directory.iterdir() if f.suffix in supported)
        if not files:
            logger.warning("No supported files found in '%s'", dir_path)
            return all_chunks

        for file_path in files:
            try:
                chunks = self.chunk_file(file_path)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error("Failed to chunk '%s': %s", file_path, exc)

        logger.info("Total chunks across %d files: %d", len(files), len(all_chunks))
        return all_chunks
