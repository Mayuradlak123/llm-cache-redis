"""How the store reaches Redis: a single node, or a Sentinel-managed replica set.

Three separate concerns live here, and they are worth keeping apart in your head:

* **Replication** — a master with N replicas. Gives you copies of the data.
* **Sentinel** — a quorum of observers that agree a master is down and promote a
  replica. Gives you automatic failover, and is what makes the *client* able to
  find the new master without a config change.
* **WAIT** — a per-write acknowledgement that N replicas have the write. Gives
  you a bound on how much a failover can lose. See :class:`ReplicationConfig`.

There are two quorums in play and they are unrelated:

* the **Sentinel quorum** (server side, ``sentinel monitor ... <quorum>``) — how
  many Sentinels must agree the master is down before a failover starts;
* the **write quorum** (client side, :attr:`ReplicationConfig.wait_replicas`) —
  how many replicas must confirm a write before it is considered durable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import redis

DEFAULT_SERVICE_NAME = "mymaster"


@dataclass(frozen=True, slots=True)
class SentinelConfig:
    """Connect through Redis Sentinel instead of to a fixed address.

    The client asks the Sentinels who the master is on every connection, so a
    failover needs no restart and no config change on your side.

    ``read_from_replicas`` sends the read half of the workload (the ``SCAN`` for
    candidates) to a replica. That takes load off the master, at the cost of
    reading data that may be a few milliseconds stale — for a cache, a slightly
    stale read is just a cache miss, which is safe.
    """

    sentinels: Sequence[tuple[str, int]]
    service_name: str = DEFAULT_SERVICE_NAME
    read_from_replicas: bool = False
    socket_timeout: float = 0.5
    db: int = 0
    password: str | None = None
    sentinel_password: str | None = None

    def __post_init__(self) -> None:
        if not self.sentinels:
            raise ValueError("SentinelConfig requires at least one sentinel address")


@dataclass(frozen=True, slots=True)
class ReplicationConfig:
    """Per-write replication acknowledgement via the Redis ``WAIT`` command.

    ``WAIT n timeout`` blocks until ``n`` replicas have acknowledged every write
    issued so far on this connection, or the timeout elapses, and returns how
    many actually acknowledged. It is **not** a transaction: the write is already
    on the master either way. All it buys is knowing whether a failover right now
    would lose that entry.

    Defaults are off (``wait_replicas=0``), which is the right default for a
    cache: entries are regenerable, so paying latency on every miss to protect
    them is usually a bad trade. Turn it on when a cache miss is genuinely
    expensive — a slow or costly model — and you would rather spend a few
    milliseconds than risk regenerating after a failover.

    ``require_acks=True`` escalates an under-replicated write from a logged
    warning to a :class:`llm_cache.redis_store.ReplicationError`. Even then the
    caller still gets their LLM response: the cache layer catches it.
    """

    wait_replicas: int = 0
    wait_timeout_ms: int = 100
    require_acks: bool = False

    def __post_init__(self) -> None:
        if self.wait_replicas < 0:
            raise ValueError("wait_replicas cannot be negative")
        if self.wait_timeout_ms < 0:
            raise ValueError("wait_timeout_ms cannot be negative")

    @property
    def enabled(self) -> bool:
        return self.wait_replicas > 0


@dataclass(slots=True)
class Endpoints:
    """The client(s) the store should use.

    ``writer`` always points at the master. ``reader`` is either the same client
    or a replica-backed one; it is never used for writes.
    """

    writer: redis.Redis
    reader: redis.Redis
    description: str = "redis"
    sentinel: Any | None = field(default=None, repr=False)

    @property
    def reads_from_replica(self) -> bool:
        return self.reader is not self.writer


def build_endpoints(
    redis_url: str = "redis://localhost:6379",
    sentinel: SentinelConfig | None = None,
) -> Endpoints:
    """Build the read/write clients, via Sentinel when configured."""
    if sentinel is None:
        client = redis.Redis.from_url(redis_url)
        return Endpoints(writer=client, reader=client, description=redis_url)
    return _sentinel_endpoints(sentinel)


def _sentinel_endpoints(cfg: SentinelConfig) -> Endpoints:
    from redis.sentinel import Sentinel

    kwargs: dict[str, Any] = {
        "socket_timeout": cfg.socket_timeout,
        "db": cfg.db,
        "password": cfg.password,
    }
    manager = Sentinel(  # type: ignore[no-untyped-call]
        list(cfg.sentinels),
        socket_timeout=cfg.socket_timeout,
        sentinel_kwargs={"password": cfg.sentinel_password},
    )
    # master_for/slave_for re-resolve the address on every connection, so a
    # failover is picked up without restarting the application.
    writer = manager.master_for(cfg.service_name, **kwargs)  # type: ignore[no-untyped-call]
    reader = (
        manager.slave_for(cfg.service_name, **kwargs)  # type: ignore[no-untyped-call]
        if cfg.read_from_replicas
        else writer
    )
    hosts = ", ".join(f"{h}:{p}" for h, p in cfg.sentinels)
    return Endpoints(
        writer=writer,
        reader=reader,
        description=f"sentinel[{cfg.service_name}] via {hosts}",
        sentinel=manager,
    )


def discover(cfg: SentinelConfig) -> dict[str, Any]:
    """Ask the Sentinels about the monitored master. Used by ``llm-cache health``.

    Reports each Sentinel's view, so you can see a split opinion (one Sentinel
    naming a different master) rather than only the first answer.
    """
    from redis.sentinel import Sentinel

    views: list[dict[str, Any]] = []
    for host, port in cfg.sentinels:
        view: dict[str, Any] = {"sentinel": f"{host}:{port}"}
        try:
            manager = Sentinel(  # type: ignore[no-untyped-call]
                [(host, port)],
                socket_timeout=cfg.socket_timeout,
                sentinel_kwargs={"password": cfg.sentinel_password},
            )
            host_, port_ = manager.discover_master(cfg.service_name)  # type: ignore[no-untyped-call]
            view["master"] = f"{host_}:{port_}"
            replicas = manager.discover_slaves(cfg.service_name)  # type: ignore[no-untyped-call]
            view["replicas"] = [f"{h}:{p}" for h, p in replicas]
            view["reachable"] = True
        except Exception as exc:
            view["reachable"] = False
            view["error"] = str(exc)
        views.append(view)

    masters = {v["master"] for v in views if v.get("master")}
    return {
        "service_name": cfg.service_name,
        "sentinels": views,
        "reachable_sentinels": sum(1 for v in views if v["reachable"]),
        "total_sentinels": len(views),
        "agreed_master": masters.pop() if len(masters) == 1 else None,
        "split_view": len(masters) > 1,
    }
