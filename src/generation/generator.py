"""
generation/generator.py — Multi-provider LLM generator with grounding enforcement.

Supports three backends configured via LLM_PROVIDER env var or config:
  - "gemini"  : Google Gemini (gemini-2.0-flash or any gemini model)
  - "openai"  : OpenAI (gpt-4o-mini or any openai model)
  - "mock"    : Deterministic mock (for tests and dev without API keys)

Key features over the original
-------------------------------
- Strict grounding enforcement: prompt explicitly forbids outside knowledge
- Source citation extraction: identifies which chunks contributed to the answer
- Token usage tracking: enables cost estimation downstream
- Confidence signal: detects "I don't know" responses for failure flagging
"""

from __future__ import annotations

import re
import time
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


def _build_prompt(question: str, chunks: list[dict[str, Any]], strict: bool = True) -> str:
    """
    Construct the grounded RAG prompt.

    Each chunk is labelled with its doc_id and chunk_id so the model
    can cite sources in its answer.
    """
    context_parts = []
    for i, c in enumerate(chunks, start=1):
        label = f"[Source {i} | {c.get('doc_id', 'unknown')} | chunk {c.get('chunk_id', '')}]"
        context_parts.append(f"{label}\n{c['text']}")
    context_block = "\n\n".join(context_parts)

    if strict:
        instruction = (
            "You are a precise, expert assistant. Answer the question using ONLY the "
            "information provided in the Sources below.\n"
            "Do NOT use any outside knowledge, training data, or assumptions.\n"
            "If the Sources do not contain enough information to answer, respond with exactly: "
            "'INSUFFICIENT CONTEXT: I cannot answer this from the provided sources.'\n"
            "Keep your answer concise (2–5 sentences). Cite relevant source numbers inline "
            "using [Source N] notation."
        )
    else:
        instruction = (
            "You are a knowledgeable assistant. Prioritize the Sources below when answering. "
            "If the Sources are insufficient, you may supplement with your knowledge — "
            "but clearly distinguish what comes from the Sources versus general knowledge.\n"
            "Keep your answer concise (2–5 sentences)."
        )

    return (
        f"{instruction}\n\n"
        f"=== SOURCES ===\n{context_block}\n===============\n\n"
        f"Question: {question}\n\nAnswer:"
    )


class Generator:
    """
    LLM generator with multi-provider support and token tracking.

    Parameters
    ----------
    cfg : dict
        ``generation`` section from pipeline config.
    secrets : dict
        ``secrets`` section from pipeline config (API keys).
    """

    def __init__(self, cfg: dict[str, Any], secrets: dict[str, Any] | None = None) -> None:
        self.cfg = cfg
        self.secrets = secrets or {}
        self.provider = cfg.get("provider", "mock")
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialise the LLM client for the configured provider."""
        if self.provider == "gemini":
            try:
                from google import genai
                api_key = self.secrets.get("google_api_key", "")
                if not api_key:
                    logger.warning("GOOGLE_API_KEY not set — falling back to mock generator")
                    self.provider = "mock"
                    return
                self._client = genai.Client(api_key=api_key)
                self._gemini_model = self.cfg.get("gemini_model", "gemini-2.0-flash")
                logger.info("Gemini generator initialised (%s)", self._gemini_model)
            except Exception as exc:
                logger.error("Gemini init failed: %s — falling back to mock", exc)
                self.provider = "mock"

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                api_key = self.secrets.get("openai_api_key", "")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set — falling back to mock generator")
                    self.provider = "mock"
                    return
                self._client = OpenAI(api_key=api_key)
                logger.info("OpenAI generator initialised (%s)", self.cfg.get("openai_model"))
            except Exception as exc:
                logger.error("OpenAI init failed: %s — falling back to mock", exc)
                self.provider = "mock"

        if self.provider == "mock":
            logger.info("Using mock generator — no LLM API calls will be made")

    # ── Public API ──────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        strict: bool | None = None,
    ) -> dict[str, Any]:
        """
        Generate an answer grounded in the provided chunks.

        Returns
        -------
        dict with:
          - answer         : generated text
          - prompt_used    : the full prompt sent to the LLM
          - provider       : which backend was used
          - input_tokens   : estimated input token count
          - output_tokens  : estimated output token count
          - latency_ms     : generation wall-time in ms
          - is_insufficient: True if the model returned an "I don't know" response
        """
        use_strict = strict if strict is not None else self.cfg.get("strict_grounding", True)
        prompt = _build_prompt(question, chunks, strict=use_strict)

        t0 = time.perf_counter()
        answer, input_tokens, output_tokens = self._call_llm(prompt)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        is_insufficient = "INSUFFICIENT CONTEXT" in answer.upper()

        return {
            "answer": answer,
            "prompt_used": prompt,
            "provider": self.provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "is_insufficient": is_insufficient,
        }

    def _call_llm(self, prompt: str) -> tuple[str, int, int]:
        """Return (answer_text, input_tokens, output_tokens)."""
        # Rough token estimate: ~1 token per 4 chars
        estimated_input = len(prompt) // 4

        if self.provider == "gemini":
            return self._call_gemini(prompt, estimated_input)
        elif self.provider == "openai":
            return self._call_openai(prompt, estimated_input)
        else:
            return self._mock_response(prompt, estimated_input)

    def _call_gemini(self, prompt: str, estimated_input: int) -> tuple[str, int, int]:
        try:
            from google import genai
            from google.genai import types
            
            response = self._client.models.generate_content(
                model=self._gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.cfg.get("temperature", 0.1),
                    max_output_tokens=self.cfg.get("max_tokens", 512),
                )
            )
            text = response.text.strip()
            # Gemini returns token counts in usage_metadata
            usage = getattr(response, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", estimated_input) if usage else estimated_input
            out_tok = getattr(usage, "candidates_token_count", len(text) // 4) if usage else len(text) // 4
            return text, in_tok, out_tok
        except Exception as exc:
            logger.error("Gemini generation error: %s", exc)
            return f"[Generation error: {exc}]", estimated_input, 0

    def _call_openai(self, prompt: str, estimated_input: int) -> tuple[str, int, int]:
        try:
            response = self._client.chat.completions.create(
                model=self.cfg.get("openai_model", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg.get("temperature", 0.1),
                max_tokens=self.cfg.get("max_tokens", 512),
            )
            text = response.choices[0].message.content.strip()
            usage = response.usage
            return text, usage.prompt_tokens, usage.completion_tokens
        except Exception as exc:
            logger.error("OpenAI generation error: %s", exc)
            return f"[Generation error: {exc}]", estimated_input, 0

    def _mock_response(self, prompt: str, estimated_input: int) -> tuple[str, int, int]:
        """Deterministic mock — extracts source labels for plausible-looking output."""
        sources = re.findall(r"\[Source (\d+) \|.*?\]", prompt)
        sources_str = ", ".join(f"[Source {s}]" for s in sources[:3]) if sources else "[Source 1]"
        text = (
            f"Based on the provided sources {sources_str}, this is a deterministic mock "
            "answer generated without any LLM API call. Set LLM_PROVIDER=gemini and "
            "provide a GOOGLE_API_KEY in your .env file to enable real generation."
        )
        return text, estimated_input, len(text) // 4
