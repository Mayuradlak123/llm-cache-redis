"""Cosine similarity, and the configurable similarity threshold."""

from __future__ import annotations

import math

import pytest
from conftest import CountingLLM, FakeEmbeddings, FakeRedis, build_cache

from llm_cache import cosine_similarity

# Unit vectors at known angles from [1.0, 0.0], so similarity is exactly the x component.
BASE = [1.0, 0.0]
NEAR = [0.95, math.sqrt(1 - 0.95**2)]  # cosine similarity 0.95
FAR = [0.72, math.sqrt(1 - 0.72**2)]  # cosine similarity 0.72


def test_identical_vectors_score_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_magnitude_does_not_matter() -> None:
    assert cosine_similarity([1.0, 1.0], [10.0, 10.0]) == pytest.approx(1.0)


def test_zero_vector_scores_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_mismatched_dimensions_score_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_known_angles() -> None:
    assert cosine_similarity(BASE, NEAR) == pytest.approx(0.95, abs=1e-4)
    assert cosine_similarity(BASE, FAR) == pytest.approx(0.72, abs=1e-4)


@pytest.fixture
def angled_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(overrides={"base prompt": BASE, "near prompt": NEAR, "far prompt": FAR})


def test_threshold_admits_scores_at_or_above_it(
    client: FakeRedis, llm: CountingLLM, angled_embeddings: FakeEmbeddings
) -> None:
    cache = build_cache(client, similarity_threshold=0.90, embeddings=angled_embeddings)
    cache.get_or_call("base prompt", llm)

    near = cache.get_or_call("near prompt", llm)
    far = cache.get_or_call("far prompt", llm)

    assert near.cache_hit is True
    assert near.similarity == pytest.approx(0.95, abs=1e-4)
    assert far.cache_hit is False


def test_raising_the_threshold_turns_a_hit_into_a_miss(
    client: FakeRedis, llm: CountingLLM, angled_embeddings: FakeEmbeddings
) -> None:
    strict = build_cache(client, similarity_threshold=0.99, embeddings=angled_embeddings)
    strict.get_or_call("base prompt", llm)

    assert strict.get_or_call("near prompt", llm).cache_hit is False


def test_lowering_the_threshold_turns_a_miss_into_a_hit(
    client: FakeRedis, llm: CountingLLM, angled_embeddings: FakeEmbeddings
) -> None:
    loose = build_cache(client, similarity_threshold=0.70, embeddings=angled_embeddings)
    loose.get_or_call("base prompt", llm)

    result = loose.get_or_call("far prompt", llm)

    assert result.cache_hit is True
    assert result.similarity == pytest.approx(0.72, abs=1e-4)


def test_closest_entry_wins(
    client: FakeRedis, llm: CountingLLM, angled_embeddings: FakeEmbeddings
) -> None:
    # 0.94 sits above the near/far similarity (~0.90), so both prompts get cached,
    # and below the base/near similarity (0.95), so the nearer entry can win.
    cache = build_cache(client, similarity_threshold=0.94, embeddings=angled_embeddings)
    cache.get_or_call("far prompt", llm)
    cache.get_or_call("near prompt", llm)

    result = cache.get_or_call("base prompt", llm)

    assert result.cache_hit is True
    assert result.response == "answer: near prompt"
