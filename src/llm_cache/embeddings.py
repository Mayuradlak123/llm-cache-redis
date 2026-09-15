"""Embedding layer.

Phase 1 ships exactly one provider (sentence-transformers, running locally).
The :class:`EmbeddingProvider` protocol exists so it can be swapped later
without touching the cache logic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

DEFAULT_MODEL = "all-MiniLM-L6-v2"


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
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load()
        vector = model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
        return [float(x) for x in vector]
