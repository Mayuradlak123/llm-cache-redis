"""Running llm-cache against a replicated, Sentinel-managed Redis.

Start the HA topology first:

    docker compose -f docker-compose.ha.yml up -d

Then (see the host-access caveat at the top of that file):

    uv run python examples/high_availability.py
"""

from __future__ import annotations

import json

from llm_cache import LLMCache, ReplicationConfig, SentinelConfig

# Point at the Sentinels, not at a Redis address. The client asks them who the
# master is on every connection, so a failover needs no restart on your side.
sentinel = SentinelConfig(
    sentinels=[("127.0.0.1", 26379), ("127.0.0.1", 26380), ("127.0.0.1", 26381)],
    service_name="llm-cache-master",
    # Send the candidate SCAN to a replica and keep the master for writes.
    # A slightly stale replica read can only ever cause a cache miss.
    read_from_replicas=True,
)

# Require one replica to acknowledge each write, waiting at most 50ms.
# Off by default: only worth the latency when regenerating a miss is expensive.
replication = ReplicationConfig(wait_replicas=1, wait_timeout_ms=50)

cache = LLMCache(
    similarity_threshold=0.90,
    ttl=3600,
    namespace="ha-example",
    sentinel=sentinel,
    replication=replication,
)


def fake_llm(prompt: str) -> str:
    print(f"  -> Calling LLM for: {prompt!r}")
    return f"Generated answer for: {prompt}"


def main() -> None:
    print("Replication status before any writes:")
    print(json.dumps(cache.replication_status(), indent=2), "\n")

    print("1. First request (MISS -> LLM call -> write, then WAIT for 1 replica)")
    print(f"   {cache.get_or_call('What is Docker?', fake_llm)}\n")

    print("2. Semantically similar request (HIT, read served by a replica)")
    print(f"   {cache.get_or_call('Can you explain Docker?', fake_llm)}\n")

    print("Replication status after writing:")
    print(json.dumps(cache.replication_status(), indent=2), "\n")

    print("Stats:", cache.stats())
    print()
    print("Now try a failover and watch it recover on its own:")
    print("  docker stop llm-cache-master")
    print("  # requests fall through to the LLM while Sentinel promotes a replica,")
    print("  # then start hitting cache again once the new master is elected")
    print("  llm-cache health --sentinel 127.0.0.1:26379 --service-name llm-cache-master")


if __name__ == "__main__":
    main()
