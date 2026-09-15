"""The public entry point: :class:`LLMCache`."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from .connection import ReplicationConfig, SentinelConfig
from .embeddings import EmbeddingProvider, SentenceTransformerEmbeddings
from .models import CacheEntry, CacheResult
from .redis_store import (
    RedisStore,
    RedisStoreError,
    ReplicationError,
    hash_system_prompt,
    scope_id,
)
from .similarity import best_match
from .stats import CacheStats

logger = logging.getLogger("llm_cache")

LLMCallable = Callable[[str], str]


class LLMCache:
    """Semantic cache around any ``prompt -> response`` callable.

    A prompt is embedded and compared against previously cached prompts sharing
    the same generation configuration. When the closest match scores at or above
    ``similarity_threshold`` the cached response is returned and the LLM is never
    called.

    Caching is an optimization, never a hard dependency: if Redis or the
    embedding model fails, the LLM is called normally and the error is logged.
    That holds for a failover too: while Sentinel promotes a new master, lookups
    fail, fall through to the LLM, and recover on their own.

    Pass ``sentinel=`` to resolve the master through Redis Sentinel instead of a
    fixed address, and ``replication=`` to require replica acknowledgement of
    writes. Both default to off; see :mod:`llm_cache.connection`.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        *,
        similarity_threshold: float = 0.90,
        ttl: int | None = 3600,
        namespace: str = "default",
        embeddings: EmbeddingProvider | None = None,
        store: RedisStore | None = None,
        sentinel: SentinelConfig | None = None,
        replication: ReplicationConfig | None = None,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.namespace = namespace
        self.embeddings: EmbeddingProvider = embeddings or SentenceTransformerEmbeddings()
        self.store = store or RedisStore(
            redis_url=redis_url,
            namespace=namespace,
            sentinel=sentinel,
            replication=replication,
        )
        self._stats = CacheStats()

    def get_or_call(
        self,
        prompt: str,
        llm: LLMCallable,
        *,
        model: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        ttl: int | None = None,
    ) -> CacheResult:
        """Return a cached response for ``prompt``, or call ``llm`` and cache the result.

        ``model``, ``temperature`` and ``system_prompt`` form part of the cache
        identity: entries written under one configuration are never returned for
        another. ``ttl`` overrides the instance TTL for this request only.
        """
        started = time.perf_counter()
        scope = scope_id(model, temperature, system_prompt)

        embedding = self._embed(prompt)

        if embedding is not None:
            hit = self._lookup(embedding, scope)
            if hit is not None:
                entry, similarity = hit
                self._stats.record_hit()
                return CacheResult(
                    response=entry.response,
                    cache_hit=True,
                    similarity=similarity,
                    latency_ms=_elapsed_ms(started),
                    cached_at=entry.created_at,
                )

        response = llm(prompt)
        self._stats.record_miss()

        if embedding is not None:
            self._store(
                prompt=prompt,
                embedding=embedding,
                response=response,
                scope=scope,
                model=model,
                temperature=temperature,
                system_prompt=system_prompt,
                ttl=self.ttl if ttl is None else ttl,
            )

        return CacheResult(
            response=response,
            cache_hit=False,
            similarity=None,
            latency_ms=_elapsed_ms(started),
            cached_at=None,
        )

    def stats(self, cost_per_call: float | None = None) -> dict[str, float | int]:
        """Return this instance's hit/miss counters.

        Pass ``cost_per_call`` (the caller's own price for one LLM call) to get an
        approximate ``estimated_savings`` figure alongside them.
        """
        data = self._stats.as_dict()
        if cost_per_call is not None:
            data["estimated_savings"] = round(self._stats.estimated_savings(cost_per_call), 6)
        return data

    def replication_status(self) -> dict[str, object]:
        """Report the replication state of the node being written to.

        Returns ``{"available": False, "error": ...}`` rather than raising when
        Redis cannot be reached, so it is safe to call from a health endpoint.
        """
        try:
            return dict(self.store.replication_status())
        except RedisStoreError as exc:
            return {"available": False, "error": str(exc)}

    def reset_stats(self) -> None:
        """Zero the in-memory counters. Cached entries are untouched."""
        self._stats = CacheStats()

    def clear(self) -> int:
        """Delete every cached entry in this namespace. Returns the number removed."""
        try:
            return self.store.clear()
        except RedisStoreError as exc:
            logger.warning("cache clear failed: %s", exc)
            return 0

    def _embed(self, prompt: str) -> list[float] | None:
        """Embed ``prompt``, or return ``None`` if the embedding layer fails."""
        try:
            return self.embeddings.embed(prompt)
        except Exception as exc:
            logger.warning("embedding failed, bypassing cache: %s", exc)
            return None

    def _lookup(self, embedding: list[float], scope: str) -> tuple[CacheEntry, float] | None:
        """Return the best match above the threshold, or ``None``."""
        try:
            candidates = [(entry, entry.embedding) for entry in self.store.candidates(scope)]
        except RedisStoreError as exc:
            logger.warning("cache lookup failed, falling back to LLM: %s", exc)
            return None

        match = best_match(embedding, candidates)
        if match is None:
            return None
        entry, similarity = match
        if similarity >= self.similarity_threshold:
            return entry, similarity
        return None

    def _store(
        self,
        *,
        prompt: str,
        embedding: list[float],
        response: str,
        scope: str,
        model: str | None,
        temperature: float | None,
        system_prompt: str | None,
        ttl: int | None,
    ) -> None:
        entry = CacheEntry(
            id=uuid.uuid4().hex,
            prompt=prompt,
            embedding=embedding,
            response=response,
            model=model,
            temperature=temperature,
            system_prompt_hash=hash_system_prompt(system_prompt),
            created_at=datetime.now(UTC),
            ttl=ttl,
        )
        try:
            self.store.add(entry, scope)
        except ReplicationError as exc:
            # The entry is on the master; only its survival of a failover is in doubt.
            logger.warning("cache entry written but under-replicated: %s", exc)
        except RedisStoreError as exc:
            logger.warning("cache write failed, response not cached: %s", exc)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
