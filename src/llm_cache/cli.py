"""A tiny CLI for inspecting the Redis side of the cache.

Request/hit counters live in memory inside a running :class:`LLMCache`, so they
are not visible from another process. What this CLI can report is what Redis
actually holds, and how its replication is doing.
"""

from __future__ import annotations

import argparse
import sys

from .connection import DEFAULT_SERVICE_NAME, ReplicationConfig, SentinelConfig, discover
from .redis_store import KEY_PREFIX, RedisStore, RedisStoreError


def _parse_sentinels(raw: str | None) -> list[tuple[str, int]]:
    """Parse ``host:port,host:port`` into address tuples."""
    if not raw:
        return []
    addresses: list[tuple[str, int]] = []
    for part in raw.split(","):
        host, _, port = part.strip().rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"invalid sentinel address {part.strip()!r}, expected host:port")
        addresses.append((host, int(port)))
    return addresses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-cache", description="Inspect an llm-cache namespace."
    )
    parser.add_argument("command", choices=["stats", "clear", "health"])
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--namespace", default="default")
    parser.add_argument(
        "--sentinel",
        metavar="HOST:PORT,...",
        help="resolve the master through these Sentinels instead of --redis-url",
    )
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument(
        "--wait-replicas",
        type=int,
        default=0,
        help="replicas that should acknowledge a write (reported by 'health')",
    )
    args = parser.parse_args(argv)

    try:
        addresses = _parse_sentinels(args.sentinel)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    sentinel = (
        SentinelConfig(sentinels=addresses, service_name=args.service_name) if addresses else None
    )
    store = RedisStore(
        redis_url=args.redis_url,
        namespace=args.namespace,
        sentinel=sentinel,
        replication=ReplicationConfig(wait_replicas=args.wait_replicas),
    )

    if args.command == "health":
        return _health(store, sentinel)

    if not store.ping():
        target = args.sentinel or args.redis_url
        print(f"Redis unreachable at {target}", file=sys.stderr)
        return 1

    try:
        if args.command == "clear":
            removed = store.clear()
            print(f"Removed {removed} cached entries from namespace '{args.namespace}'")
            return 0

        entries = sum(1 for _ in store.candidates("*"))
        print(f"Namespace:      {args.namespace}")
        print(f"Key pattern:    {KEY_PREFIX}:{args.namespace}:<config>:<id>")
        print(f"Cached entries: {entries}")
        print("Hit/miss counters are per-process; read them with LLMCache.stats().")
        return 0
    except RedisStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _health(store: RedisStore, sentinel: SentinelConfig | None) -> int:
    """Print replication and Sentinel state. Exit 1 when the topology looks unhealthy."""
    healthy = True

    if sentinel is not None:
        view = discover(sentinel)
        reachable = view["reachable_sentinels"]
        total = view["total_sentinels"]
        print(f"Sentinel service: {view['service_name']}")
        print(f"Sentinels up:     {reachable}/{total}")
        for entry in view["sentinels"]:
            if entry["reachable"]:
                print(f"  {entry['sentinel']:22} master={entry['master']}")
            else:
                print(f"  {entry['sentinel']:22} UNREACHABLE ({entry['error']})")
        if view["split_view"]:
            print("  WARNING: Sentinels disagree about the master")
            healthy = False
        # A Sentinel quorum needs a majority of Sentinels alive to elect at all.
        if reachable <= total // 2:
            print(f"  WARNING: only {reachable}/{total} Sentinels reachable; no failover possible")
            healthy = False
        print()

    try:
        status = store.replication_status()
    except RedisStoreError as exc:
        print(f"Replication:      unavailable ({exc})", file=sys.stderr)
        return 1

    print(f"Endpoint:         {status['endpoint']}")
    print(f"Role:             {status['role']}")
    print(f"Replicas:         {status['connected_replicas']}")
    for replica in status["replicas"]:
        print(
            f"  {replica['address']:22} state={replica['state']:10} "
            f"lag={replica['lag_bytes']} bytes"
        )
    print(f"Reads from:       {'replica' if status['reads_from_replica'] else 'master'}")

    required = int(status["wait_replicas"])
    if required:
        print(f"Write quorum:     WAIT {required} replicas / {status['wait_timeout_ms']}ms")
        if int(status["connected_replicas"]) < required:
            print("  WARNING: fewer replicas connected than the write quorum requires")
            healthy = False
    else:
        print("Write quorum:     disabled (WAIT not used)")

    if status["role"] != "master":
        print("  WARNING: the write endpoint is not a master")
        healthy = False

    return 0 if healthy else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
