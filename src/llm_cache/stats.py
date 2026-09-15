"""In-memory cache statistics.

Deliberately not persisted: these counters describe the lifetime of a single
:class:`llm_cache.LLMCache` instance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CacheStats:
    """Running hit/miss counters for one cache instance."""

    total_requests: int = 0
    hits: int = 0
    misses: int = 0

    def record_hit(self) -> None:
        self.total_requests += 1
        self.hits += 1

    def record_miss(self) -> None:
        self.total_requests += 1
        self.misses += 1

    @property
    def hit_rate(self) -> float:
        """Fraction of requests served from cache, ``0.0`` when there are none."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def llm_calls_avoided(self) -> int:
        """Every hit is one LLM call that did not happen."""
        return self.hits

    def estimated_savings(self, cost_per_call: float) -> float:
        """Approximate cost avoided, given the caller's own per-call price."""
        return self.llm_calls_avoided * cost_per_call

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_requests": self.total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "llm_calls_avoided": self.llm_calls_avoided,
        }
