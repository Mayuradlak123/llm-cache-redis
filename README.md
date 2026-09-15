# llm-cache

Lightweight **semantic caching** for LLM applications, backed by Redis.

`llm-cache` answers one question:

> Can we avoid a repeated LLM call when a new prompt means the same thing as a previous one?

```python
from llm_cache import LLMCache

cache = LLMCache(redis_url="redis://localhost:6379", similarity_threshold=0.90, ttl=3600)

result = cache.get_or_call(prompt="Explain Docker in simple words", llm=my_llm)
result.response  # the answer
result.cache_hit  # True when it came from cache
result.similarity  # how close the matched prompt was
```

---

## 1. The problem

LLM calls are the slowest and most expensive part of most LLM applications, and users ask the
same thing over and over in different words. Every one of those repeats costs a full round trip
and full token billing.

## 2. Why normal caching is not enough

A conventional cache keys on an exact string (or its hash):

```
"What is Docker?"          -> hash A
"Can you explain Docker?"  -> hash B
"Explain Docker simply"    -> hash C
```

Three different keys, three cache misses, three LLM calls — even though a single answer would
have served all three. Exact-match caching only helps when users repeat themselves character for
character, which they rarely do.

## 3. What semantic caching means

Instead of comparing strings, compare *meaning*. Each prompt is turned into an embedding (a
vector), and a new prompt is matched against previously cached prompts by **cosine similarity**.
If the closest match scores at or above a configurable threshold, its cached response is
returned and the LLM is never called.

Measured against `"What is Docker?"` with the default model (`all-MiniLM-L6-v2`):

| prompt | similarity | at threshold 0.90 |
| --- | --- | --- |
| `"Tell me about Docker"` | 0.9238 | HIT |
| `"Can you explain Docker?"` | 0.9182 | HIT |
| `"Can you explain Docker to me?"` | 0.9017 | HIT |
| `"Explain Docker in simple words."` | 0.8860 | MISS |
| `"How do I make cold brew coffee?"` | 0.0839 | MISS |

Note the gap: unrelated prompts score near zero, while paraphrases cluster tightly in the
0.88–0.93 band. The threshold is the dial you turn inside that band, and it is genuinely a
judgement call — at `0.90`, `"Explain Docker in simple words."` misses by 0.014.

Coverage also improves as the cache fills, because new entries become match targets themselves.
Once `"Can you explain Docker?"` is cached, `"Explain Docker in simple words."` matches *it* at
0.9644 and hits — even though it missed the original entry.

## 4. Architecture

```
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
    +---- HIT  ----> Cached response
    |
    +---- MISS ----> LLM
                      |
                      v
                    Redis
                      |
                      v
                   Response
```

## 5. Installation

```bash
uv venv
uv pip install -e ".[dev]"
```

Requires Python 3.12+. The first run downloads the local embedding model
(`all-MiniLM-L6-v2`, ~90 MB) via `sentence-transformers`.

## 6. Redis setup

```bash
docker compose up -d
```

That starts plain Redis 7 on `localhost:6379`. Redis Stack is **not** required — see
[Limitations](#14-limitations) for why.

For a replicated, Sentinel-managed setup, see
[High availability](#135-high-availability-replicas-sentinel-and-wait).

## 7. Basic usage

Your LLM is just a callable. The library never talks to a provider itself:

```python
from llm_cache import LLMCache

cache = LLMCache(
    redis_url="redis://localhost:6379",
    similarity_threshold=0.90,
    ttl=3600,
    namespace="my-app",
)


def my_llm(prompt: str) -> str:
    # call OpenAI / Anthropic / Gemini / a local model — whatever you use
    return "..."


result = cache.get_or_call(prompt="What is Docker?", llm=my_llm)
print(result.response, result.cache_hit, result.similarity, result.latency_ms)
```

`get_or_call` returns a `CacheResult`:

| field | type | meaning |
| --- | --- | --- |
| `response` | `str` | the answer, cached or freshly generated |
| `cache_hit` | `bool` | `True` when no LLM call was made |
| `similarity` | `float \| None` | score of the matched entry; `None` on a miss |
| `latency_ms` | `float` | wall-clock time for the whole call |
| `cached_at` | `datetime \| None` | when the returned entry was stored; `None` on a miss |

## 8. Semantic cache example

```python
result1 = cache.get_or_call(prompt="What is Docker?", llm=my_llm)
# cache_hit=False  -> LLM is called, answer stored in Redis

result2 = cache.get_or_call(prompt="Can you explain Docker?", llm=my_llm)
# cache_hit=True   -> different string, same meaning, no LLM call
```

A runnable version lives in [`examples/basic.py`](examples/basic.py).

## 9. Configuration

```python
LLMCache(
    redis_url="redis://localhost:6379",
    similarity_threshold=0.90,  # cosine similarity required for a HIT
    ttl=3600,  # seconds; None disables expiry
    namespace="my-app",  # isolates this app's entries
    embeddings=None,  # any object with .embed(text) -> list[float]
    store=None,  # inject a pre-built RedisStore (e.g. with your own client)
)
```

### Cache identity

The cache key is **not** the prompt alone. `model`, `temperature` and `system_prompt` are hashed
into a scope segment of the Redis key:

```
llm-cache:<namespace>:<config-hash>:<id>
```

So switching models or system prompts can never return an incompatible cached answer:

```python
cache.get_or_call("What is Docker?", my_llm, model="gpt-4o-mini", temperature=0.0)
cache.get_or_call("What is Docker?", my_llm, model="claude-sonnet-5")  # MISS — different model
```

### Namespace

Two applications pointed at the same Redis instance stay isolated:

```python
support = LLMCache(namespace="support-bot")
docs = LLMCache(namespace="docs-bot")
```

## 10. TTL

Entries expire via native Redis TTL, so there is no sweeper process to run:

```python
cache = LLMCache(ttl=3600)  # instance default: 1 hour
cache.get_or_call("What is Docker?", my_llm, ttl=60)  # per-request override
cache = LLMCache(ttl=None)  # never expire
```

## 11. Similarity threshold

The threshold is configurable and never hard-coded.

| threshold | behaviour |
| --- | --- |
| `0.95+` | strict — near-paraphrases only |
| `0.90` | balanced default |
| `0.80` | loose — more hits, higher risk of serving a subtly wrong answer |

With `similarity_threshold=0.90`: a match at `0.95` is a **HIT**, a match at `0.72` is a **MISS**.

**Tune it against your own prompts.** The default of `0.90` is a starting point, not a
recommendation — as the table in [§3](#3-what-semantic-caching-means) shows, real paraphrases of
one question land anywhere from 0.886 to 0.924, so a shift of 0.02 flips real traffic between hit
and miss. Log `result.similarity` on your own prompts for a while before settling on a value.
Unrelated prompts score far lower (~0.08), so the risk is never "coffee answers a Docker
question" — it is two genuinely close prompts that nonetheless want different answers.

## 12. Cache statistics

```python
cache.stats()
# {'total_requests': 100, 'hits': 67, 'misses': 33, 'hit_rate': 0.67, 'llm_calls_avoided': 67}

cache.stats(cost_per_call=0.002)
# {..., 'estimated_savings': 0.134}
```

Counters are **in-memory and per-instance** — they reset when your process restarts, and there is
no analytics database. `cache.reset_stats()` zeroes them without touching cached entries.

**Cost savings are approximate.** `estimated_savings` is simply
`llm_calls_avoided * cost_per_call`, using the price *you* supply. The library does no
provider-specific pricing or token accounting, so the figure is only as accurate as your input.

There is also a tiny CLI for the Redis side:

```bash
llm-cache stats --namespace my-app
llm-cache clear --namespace my-app
```

## 13. Failure behavior

**Caching is an optimization, never a hard dependency.** If the cache layer fails, your
application still gets its answer:

```
Redis unavailable  ->  lookup fails    ->  LLM called normally  ->  response returned
Embedding fails    ->  cache bypassed  ->  LLM called normally  ->  response returned
```

Both paths log a warning on the `llm_cache` logger and return `cache_hit=False`. Writes that fail
are logged and skipped — the response is still returned. No cache failure raises out of
`get_or_call`.

## 13.5 High availability: replicas, Sentinel and WAIT

All of this is **opt-in and off by default**. A cache is regenerable by definition, so single-node
Redis is a perfectly respectable choice — reach for this when a cache miss is genuinely expensive
(a slow or costly model) or when a cold cache after a restart would hurt.

Start the topology — 1 master, 2 replicas, 3 Sentinels:

```bash
docker compose -f docker-compose.ha.yml up -d
```

### Connecting through Sentinel

Point the client at the *Sentinels*, not at a Redis address. It asks them who the master is on
every connection, so a failover needs no restart and no config change:

```python
from llm_cache import LLMCache, ReplicationConfig, SentinelConfig

cache = LLMCache(
    namespace="my-app",
    sentinel=SentinelConfig(
        sentinels=[("127.0.0.1", 26379), ("127.0.0.1", 26380), ("127.0.0.1", 26381)],
        service_name="llm-cache-master",
        read_from_replicas=True,  # send the candidate SCAN to a replica
    ),
    replication=ReplicationConfig(wait_replicas=1, wait_timeout_ms=50),
)
```

`read_from_replicas=True` takes the read half of the workload off the master. The replica may be
a few milliseconds behind, but for a cache a stale read is just a **cache miss** — the worst case
is one extra LLM call, never a wrong answer.

### The two quorums

These get conflated constantly. They are unrelated:

| | Sentinel quorum | Write quorum (`WAIT`) |
| --- | --- | --- |
| Configured in | `sentinel monitor <name> <host> <port> 2` (server) | `ReplicationConfig(wait_replicas=1)` (client) |
| Answers | "do enough of us agree the master is down?" | "did enough replicas receive this write?" |
| Affects | whether a failover starts | how much a failover can lose |

A third rule sits underneath both: **a failover also needs a majority of Sentinels alive** to
elect the one that performs it. That is why the compose file runs three, not two — with two,
losing one leaves no majority and nothing can be promoted. Always run an odd number ≥ 3.

### What WAIT actually does

`WAIT n timeout` blocks until `n` replicas acknowledge every write issued so far on that
connection, then returns how many actually did.

It is **not** a transaction and **not** a rollback. The write is already on the master either
way. All `WAIT` buys you is *knowing* whether a failover at that instant would lose the entry:

```python
ReplicationConfig(wait_replicas=1, wait_timeout_ms=50)  # warn on shortfall
ReplicationConfig(wait_replicas=2, wait_timeout_ms=100, require_acks=True)  # raise on shortfall
```

On a shortfall the default logs a warning and moves on. `require_acks=True` raises
`ReplicationError` instead — which the cache layer catches, so **the caller still gets their LLM
response**. Nothing about replication is allowed to fail a request.

The honest trade: `wait_replicas` adds a network round trip to every cache *miss*, in exchange
for durability of data you could regenerate by calling the LLM again. Measure before enabling it.

### Checking health

```bash
llm-cache health --sentinel 127.0.0.1:26379,127.0.0.1:26380,127.0.0.1:26381 \
                 --service-name llm-cache-master --wait-replicas 1
```

```
Sentinel service: llm-cache-master
Sentinels up:     3/3
  127.0.0.1:26379        master=172.19.0.2:6379
  ...
Endpoint:         sentinel[llm-cache-master] via 127.0.0.1:26379, ...
Role:             master
Replicas:         2
  172.19.0.4:6379        state=online     lag=0 bytes
Reads from:       replica
Write quorum:     WAIT 1 replicas / 100ms
```

Exit code is `1` when the topology looks unhealthy — no Sentinel majority, Sentinels disagreeing
about the master, fewer replicas than the write quorum needs, or a write endpoint that is not a
master. `cache.replication_status()` returns the same data as a dict for your own health endpoint.

### During a failover

Nothing special is required of you. While Sentinel promotes a replica, lookups fail, fall through
to the LLM exactly like any other Redis outage, and start hitting cache again once the new master
is elected. Try it:

```bash
docker stop llm-cache-master
```

A runnable walkthrough is in [`examples/high_availability.py`](examples/high_availability.py).

## 14. Limitations

- **Linear search.** Candidates are gathered with a Redis `SCAN` over the namespace and compared
  in-process. This is fine for thousands of entries; it is not a vector database. Redis Stack
  vector search was deliberately skipped in Phase 1 — an index and schema is a lot of machinery
  for a handful of dot products at this scale. `RedisStore` exposes a small surface
  (`add`, `candidates`, `ping`, `clear`) so a vector-search backend can replace it later without
  touching the cache logic.
- **Similarity is a heuristic.** Two prompts can be close in embedding space and still want
  different answers ("How do I *start* a container?" vs "How do I *stop* a container?"). Choose
  the threshold conservatively.
- **Stale answers.** A cached response is returned as-is until its TTL expires. There is no
  invalidation beyond TTL and `cache.clear()`.
- **One embedding model.** Local `sentence-transformers` only; the first call loads the model.
- **Synchronous only.** No async API in Phase 1. `WAIT` blocks the calling thread for up to
  `wait_timeout_ms` on every cache miss when enabled.
- **HA is untested against real Docker.** The replication/Sentinel code is unit-tested against
  fakes, but `docker-compose.ha.yml` has not been run on a live cluster (no Docker on the
  development machine). Verify a failover yourself before relying on it.
- **No streaming.** Only complete string responses are cached.

## 15. Non-goals (Phase 1)

Explicitly out of scope: LangChain and LangGraph integration, agent memory, multi-agent systems,
Kafka, Celery, ARQ, background workers, a FastAPI server, a frontend, authentication, dashboards,
distributed tracing, provider-specific billing, multiple embedding providers, advanced cache
invalidation, streaming response caching, and multimodal caching.

## 16. Roadmap

- Redis Stack vector search (`FT.SEARCH` with a KNN index) for large caches
- Redis Cluster (sharding) alongside the existing replication support
- Async API (`await cache.get_or_call(...)`)
- Pluggable embedding providers (API-based and local)
- Streaming response caching
- Smarter invalidation than TTL
- Optional framework integrations

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
uv build
```

There is no test suite in the repository: it was removed by request. The suite that existed
(79 tests covering hit/miss behaviour, TTL, namespaces, cache identity, failure fallback,
replication and the demo adapter) is recoverable from git history at commit `9996072`.

## License

MIT — see [LICENSE](LICENSE).
