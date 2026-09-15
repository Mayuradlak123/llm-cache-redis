"""Cosine similarity helpers.

Kept deliberately tiny: the only thing Phase 1 needs is "which stored embedding
is closest to this one, and how close is it".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, in ``[-1.0, 1.0]``.

    Returns ``0.0`` when either vector is all zeros or the lengths differ.
    """
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.shape != vb.shape or va.size == 0:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def best_match[T](
    query: Sequence[float],
    candidates: Sequence[tuple[T, Sequence[float]]],
) -> tuple[T, float] | None:
    """Return the ``(item, similarity)`` pair with the highest cosine similarity.

    ``None`` is returned when there are no candidates.
    """
    best: tuple[T, float] | None = None
    for item, embedding in candidates:
        score = cosine_similarity(query, embedding)
        if best is None or score > best[1]:
            best = (item, score)
    return best
