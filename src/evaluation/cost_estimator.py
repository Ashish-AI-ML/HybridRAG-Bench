"""
evaluation/cost_estimator.py — LLM cost and token tracking.

Tracks cumulative token usage across evaluation runs and projects
monthly costs at various query volumes. Makes RAG evaluation
financially aware — an important production consideration.

Pricing (approximate, as of 2025 — update as needed):
  gemini-2.0-flash : $0.075 / 1M input, $0.30 / 1M output
  gpt-4o-mini      : $0.15  / 1M input, $0.60 / 1M output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Pricing per 1M tokens (USD)
_PRICING: dict[str, dict[str, float]] = {
    "gemini": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "openai": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "mock": {"input": 0.0, "output": 0.0},
}


@dataclass
class CostAccumulator:
    provider: str = "mock"
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    query_count: int = 0
    _records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.query_count += 1
        self._records.append({"input": input_tokens, "output": output_tokens})

    @property
    def _prices(self) -> dict[str, float]:
        key = self.provider.lower()
        return _PRICING.get(key, _PRICING["mock"])

    @property
    def total_cost_usd(self) -> float:
        p = self._prices
        return (self.total_input_tokens * p["input"] + self.total_output_tokens * p["output"]) / 1_000_000

    @property
    def avg_input_tokens(self) -> float:
        return self.total_input_tokens / self.query_count if self.query_count else 0.0

    @property
    def avg_output_tokens(self) -> float:
        return self.total_output_tokens / self.query_count if self.query_count else 0.0

    def project_monthly(self, queries_per_day: int) -> dict[str, float]:
        """Project total monthly cost at a given daily query volume."""
        if self.query_count == 0:
            return {}
        avg_in = self.avg_input_tokens
        avg_out = self.avg_output_tokens
        p = self._prices
        monthly_queries = queries_per_day * 30
        monthly_in = avg_in * monthly_queries
        monthly_out = avg_out * monthly_queries
        cost = (monthly_in * p["input"] + monthly_out * p["output"]) / 1_000_000
        return {
            "queries_per_day": queries_per_day,
            "monthly_queries": monthly_queries,
            "projected_cost_usd": round(cost, 4),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "query_count": self.query_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_input_tokens": round(self.avg_input_tokens, 1),
            "avg_output_tokens": round(self.avg_output_tokens, 1),
            "projections": {
                "100_queries_day": self.project_monthly(100),
                "1000_queries_day": self.project_monthly(1000),
                "10000_queries_day": self.project_monthly(10000),
            },
        }
