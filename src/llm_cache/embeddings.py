"""Embedding layer.

Phase 1 ships exactly one provider (sentence-transformers, running locally).
The :class:`EmbeddingProvider` protocol exists so it can be swapped later
without touching the cache logic.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

DEFAULT_MODEL = "all-MiniLM-L6-v2"

logger = logging.getLogger("llm_cache")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that can turn text into a vector."""

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        ...


class SentenceTransformerEmbeddings:
    """Local embeddings via `sentence-transformers`.

    The model is loaded lazily on first use so that importing ``llm_cache`` stays
    cheap and so tests can run without downloading model weights.

    Once the weights are cached, loading works without a network. Set
    ``offline=True`` to skip the Hugging Face hub entirely.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, *, offline: bool = False) -> None:
        self.model_name = model_name
        self.offline = offline
        self._model: object | None = None

    def _load(self) -> object:
        """Load the model, falling back to the local cache if the hub is unreachable.

        sentence-transformers probes the hub for optional files (adapter and
        processor configs) even when the weights are already cached, so a DNS
        failure or a blocked hub would otherwise make a perfectly usable local
        model fail to load.
        """
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        if self.offline:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
            return self._model

        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            logger.warning("could not reach the model hub (%s); retrying from the local cache", exc)
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception as offline_exc:
                raise RuntimeError(
                    f"could not load embedding model {self.model_name!r}: the hub is "
                    f"unreachable ({exc}) and it is not in the local cache "
                    f"({offline_exc}). Connect once to download it, or pass a model "
                    f"you already have."
                ) from offline_exc
            logger.info("loaded %s from the local cache", self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load()
        vector = model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
        return [float(x) for x in vector]
