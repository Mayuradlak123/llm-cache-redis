Build a small production-quality Python library called `llm-cache`.

The goal is to create a lightweight semantic caching library for LLM applications.

IMPORTANT:
This is PHASE 1 MVP only.

Do NOT add LangGraph, LangChain, agents, Kafka, Celery, ARQ, background workers, FastAPI server, frontend, dashboards, authentication, databases other than Redis, multi-agent workflows, observability platforms, or unnecessary abstractions.

The library must solve one clear problem:

"Can we avoid repeated LLM calls when a new user prompt is semantically similar to a previous prompt?"

==================================================

1. # CORE IDEA

The library should provide semantic caching.

Example:

Prompt 1:
"What is Docker?"

Prompt 2:
"Can you explain Docker to me?"

Prompt 3:
"Explain Docker in simple words."

These prompts are different strings but can have similar semantic meaning.

The library should:

1. Receive a prompt.
2. Generate an embedding for the prompt.
3. Search previously cached prompts in Redis.
4. Calculate semantic similarity.
5. If similarity >= configured threshold:
   - return cached response
   - mark request as cache HIT
   - do NOT call the LLM.

6. Otherwise:
   - call the provided LLM function
   - store prompt, embedding, response and metadata in Redis
   - return the new response
   - mark request as cache MISS.

================================================== 2. TECH STACK
=============

Use:

- Python 3.12+
- uv for dependency management
- Redis
- Redis Stack if vector search functionality is required
- sentence-transformers for local embeddings
- NumPy only if required
- pytest for tests
- ruff for linting
- mypy if practical
- pyproject.toml
- Docker Compose only for running Redis locally

Keep dependencies minimal.

The package must be installable as a normal Python package.

================================================== 3. PROJECT STRUCTURE
====================

Create:

llm-cache/
│
├── src/
│ └── llm_cache/
│ ├── **init**.py
│ ├── cache.py
│ ├── embeddings.py
│ ├── redis_store.py
│ ├── similarity.py
│ ├── models.py
│ └── stats.py
│
├── tests/
│ ├── test_cache.py
│ ├── test_similarity.py
│ ├── test_ttl.py
│ └── test_stats.py
│
├── examples/
│ └── basic.py
│
├── docker-compose.yml
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore

Keep the architecture simple.

================================================== 4. PUBLIC API
=============

The primary API should be easy to understand.

Example:

from llm_cache import LLMCache

cache = LLMCache(
redis_url="redis://localhost:6379",
similarity_threshold=0.90,
ttl=3600
)

result = cache.get_or_call(
prompt="Explain Docker in simple words",
llm=my_llm
)

The library should return a structured result.

Example:

result.response
result.cache_hit
result.similarity
result.latency_ms

Expected behavior:

First request:

cache_hit = False

Second semantically similar request:

cache_hit = True

================================================== 5. LLM FUNCTION INTERFACE
=========================

Do NOT tightly couple the library to OpenAI, Anthropic, Gemini, Groq, DeepSeek, etc.

The user should provide the LLM callable.

For example:

def my_llm(prompt: str) -> str:
return "LLM response"

Then:

result = cache.get_or_call(
prompt="Explain Docker",
llm=my_llm
)

The library only manages caching.

Do not implement provider-specific SDK integrations in Phase 1.

================================================== 6. EMBEDDING LAYER
==================

Create a small embedding abstraction.

Example:

class EmbeddingProvider:
def embed(self, text: str) -> list[float]:
...

Implement one default provider using sentence-transformers.

Use a small local model suitable for development.

The embedding provider should be replaceable later.

Do not create multiple embedding providers in Phase 1.

================================================== 7. REDIS STORAGE
================

Use Redis for cache storage.

Each cache entry should contain at least:

- cache ID
- prompt
- embedding
- response
- model identifier if supplied
- temperature if supplied
- system prompt hash if supplied
- created_at
- TTL metadata

Use Redis TTL so expired cache entries are automatically removed.

Use a clear Redis key namespace such as:

llm-cache:<namespace>:<id>

Do not create a complicated database abstraction.

================================================== 8. SEMANTIC SEARCH
==================

The library must support semantic similarity lookup.

For Phase 1:

- generate embedding for incoming prompt
- search stored embeddings
- calculate cosine similarity
- select the highest similarity result
- compare it against similarity_threshold

Example:

similarity_threshold = 0.90

If:

similarity = 0.95

=> HIT

If:

similarity = 0.72

=> MISS

Make the threshold configurable.

Do not hard-code the threshold.

If Redis Stack vector search is used, keep the implementation straightforward and documented.

If vector search becomes unnecessarily complicated, implement a simple MVP-compatible approach while keeping the storage interface replaceable.

================================================== 9. CACHE IDENTITY
=================

Do not use only the raw prompt as the cache identity.

Relevant generation configuration must be considered.

Support optional values such as:

- model
- temperature
- system_prompt

Conceptually:

cache identity =
prompt + model + temperature + system_prompt

At minimum, ensure that changing the model or important generation configuration does not incorrectly return an incompatible cached result.

Keep this implementation simple.

================================================== 10. CACHE RESULT
================

Create a result model such as:

CacheResult

with:

response: str
cache_hit: bool
similarity: float | None
latency_ms: float
cached_at: datetime | None

The exact implementation can use dataclasses or Pydantic, but avoid unnecessary dependencies.

================================================== 11. STATISTICS
==============

Track basic in-memory statistics:

- total_requests
- cache_hits
- cache_misses
- hit_rate

Example:

cache.stats()

should return something like:

{
"total_requests": 100,
"hits": 67,
"misses": 33,
"hit_rate": 0.67
}

Do not build a persistent analytics database.

================================================== 12. COST / TOKEN SAVINGS
========================

Phase 1 should provide a simple way to estimate avoided LLM calls.

Track:

- total requests
- cache hits
- LLM calls avoided

If token/cost metadata is supplied by the caller, optionally calculate estimated savings.

Do not implement complicated provider-specific pricing.

The README should explain that cost calculation is approximate unless the caller provides accurate model pricing.

================================================== 13. TTL
=======

Support:

cache = LLMCache(
ttl=3600
)

A cached result should expire automatically after the configured TTL.

Allow TTL configuration per instance.

If practical, allow per-request TTL override, but do not over-engineer this.

================================================== 14. NAMESPACE
=============

Support a simple namespace:

cache = LLMCache(
redis_url="redis://localhost:6379",
namespace="my-app"
)

This prevents unrelated applications from sharing cache entries accidentally.

Do not implement multi-tenancy/authentication in Phase 1.

================================================== 15. ERROR HANDLING
==================

Caching should never break the application.

This is very important.

If Redis is unavailable:

The application should still be able to call the LLM.

Example:

Redis unavailable
↓
cache lookup fails
↓
call LLM normally
↓
return response

Similarly, if embedding generation fails:

Do not crash the entire LLM application unnecessarily.

Fail gracefully and provide a clear error/log.

Caching should be an optimization, not a hard dependency for application correctness.

================================================== 16. TESTING
===========

Create pytest tests for:

1. First request is MISS.
2. Second exact request is HIT.
3. Semantically similar request is HIT.
4. Semantically unrelated request is MISS.
5. Similarity threshold works.
6. TTL expires cache entry.
7. Different namespace does not collide.
8. Different model configuration does not incorrectly reuse cache.
9. Redis failure does not prevent LLM invocation.
10. Statistics correctly track hits/misses.
11. LLM is called only on MISS.
12. LLM is not called on HIT.

Use mocks/fakes where appropriate.

Do not require a real external LLM API for tests.

================================================== 17. EXAMPLE
===========

Create:

examples/basic.py

Demonstrate:

from llm_cache import LLMCache

cache = LLMCache(
redis_url="redis://localhost:6379",
similarity_threshold=0.90,
ttl=3600
)

def fake_llm(prompt: str) -> str:
print("Calling LLM...")
return f"Generated answer for: {prompt}"

result1 = cache.get_or_call(
prompt="What is Docker?",
llm=fake_llm
)

print(result1)

result2 = cache.get_or_call(
prompt="Can you explain Docker?",
llm=fake_llm
)

print(result2)

print(cache.stats())

The example should clearly demonstrate:

MISS → LLM call

then:

semantic HIT → no LLM call

================================================== 18. CLI
=======

A CLI is optional for Phase 1.

If implemented, keep it tiny.

Example:

llm-cache stats

Output:

Requests: 100
Hits: 67
Misses: 33
Hit rate: 67%
LLM calls avoided: 67

Do not build a dashboard.

================================================== 19. README
==========

README must clearly explain:

1. What problem this library solves.
2. Why normal caching is insufficient for LLM applications.
3. What semantic caching means.
4. Architecture diagram.
5. Installation.
6. Redis setup.
7. Basic usage.
8. Semantic cache example.
9. Configuration.
10. TTL.
11. Similarity threshold.
12. Cache statistics.
13. Failure behavior.
14. Limitations.
15. Roadmap.

Use a simple example:

"What is Docker?"

vs

"Can you explain Docker?"

Explain that these may produce a semantic cache HIT even though the strings are different.

================================================== 20. README ARCHITECTURE
=======================

Include this conceptual architecture:

Application
|
v
llm-cache
|
v
Generate embedding
|
v
Redis semantic search
|
+---- HIT ----> Cached response
|
+---- MISS ---> LLM
|
v
Redis
|
v
Response

================================================== 21. NON-GOALS
=============

Explicitly document that Phase 1 does NOT include:

- LangChain integration
- LangGraph integration
- Agent memory
- Multi-agent systems
- Kafka
- Celery
- ARQ
- background workers
- FastAPI server
- frontend
- authentication
- dashboards
- distributed tracing
- provider-specific billing
- multiple embedding providers
- advanced cache invalidation
- streaming response caching
- multimodal caching

These can be future roadmap items.

================================================== 22. QUALITY REQUIREMENTS
========================

The code should be:

- typed
- modular
- readable
- small
- testable
- documented
- installable using uv
- compatible with Python 3.12+
- free of unnecessary abstractions

Prefer simple functions/classes over framework-style architecture.

Do not create abstractions unless they solve a real problem.

================================================== 23. DEFINITION OF DONE
======================

Phase 1 is complete when:

1. Redis starts using docker-compose.
2. Package installs using uv.
3. A developer can import LLMCache.
4. First request calls the LLM.
5. Result is stored in Redis.
6. A semantically similar second request returns the cached response.
7. LLM is not called on cache HIT.
8. Similarity threshold is configurable.
9. TTL works.
10. Basic statistics work.
11. Redis failure gracefully falls back to LLM.
12. Tests pass.
13. README contains setup and usage.
14. Example demonstrates MISS → HIT.
15. Package can be built successfully.

Do NOT add additional features after these requirements are complete.

Focus on making the core semantic-cache experience reliable and easy to understand.
