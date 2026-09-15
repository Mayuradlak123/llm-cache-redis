"""Data models used across the library."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class CacheResult:
    """The outcome of a :meth:`llm_cache.LLMCache.get_or_call` call."""

    response: str
    cache_hit: bool
    similarity: float | None = None
    latency_ms: float = 0.0
    cached_at: datetime | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        state = "HIT" if self.cache_hit else "MISS"
        sim = "n/a" if self.similarity is None else f"{self.similarity:.3f}"
        return f"[{state}] similarity={sim} latency={self.latency_ms:.1f}ms :: {self.response}"


@dataclass(slots=True)
class CacheEntry:
    """A single cached prompt/response pair plus its embedding and metadata."""

    id: str
    prompt: str
    embedding: list[float]
    response: str
    model: str | None = None
    temperature: float | None = None
    system_prompt_hash: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ttl: int | None = None
