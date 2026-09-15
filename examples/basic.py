"""Minimal end-to-end demo: MISS -> LLM call, then semantic HIT -> no LLM call.

Run Redis first:

    docker compose up -d

Then:

    uv run python examples/basic.py
"""

from __future__ import annotations

from llm_cache import LLMCache

cache = LLMCache(
    redis_url="redis://localhost:6379",
    similarity_threshold=0.90,
    ttl=3600,
    namespace="example",
)


def fake_llm(prompt: str) -> str:
    """Stands in for a real provider call. Prints so you can see when it runs."""
    print(f"  -> Calling LLM for: {prompt!r}")
    return f"Generated answer for: {prompt}"


def main() -> None:
    cache.clear()  # start from a clean slate so the demo is repeatable

    print("1. First request (expect MISS, LLM is called)")
    result1 = cache.get_or_call(prompt="What is Docker?", llm=fake_llm)
    print(f"   {result1}\n")

    print("2. Semantically similar request (expect HIT, LLM is NOT called)")
    result2 = cache.get_or_call(prompt="Can you explain Docker?", llm=fake_llm)
    print(f"   {result2}\n")

    print("3. Unrelated request (expect MISS, LLM is called)")
    result3 = cache.get_or_call(prompt="How do I make cold brew coffee?", llm=fake_llm)
    print(f"   {result3}\n")

    print("4. Same prompt, different model (expect MISS: model is part of cache identity)")
    result4 = cache.get_or_call(prompt="What is Docker?", llm=fake_llm, model="gpt-4o-mini")
    print(f"   {result4}\n")

    print("Stats:", cache.stats(cost_per_call=0.002))


if __name__ == "__main__":
    main()
