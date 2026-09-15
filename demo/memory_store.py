"""An in-process stand-in for Redis, so the demo runs with no infrastructure.

``RedisStore`` accepts any object implementing the handful of commands it uses,
so this drops straight in where a ``redis.Redis`` client would go. It exists
because Docker is not always available; **it is not part of the library and not
something to use in production** — the data dies with the process and is never
shared between workers.

The demo prefers real Redis and only falls back to this when the configured
``REDIS_URL`` cannot be reached.
"""

from __future__ import annotations

import fnmatch
import threading
import time
from collections.abc import Iterator
from typing import Any


class MemoryRedis:
    """The subset of the Redis command surface that :class:`RedisStore` calls."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._expiry: dict[str, float] = {}
        self._lock = threading.Lock()

    # -- housekeeping -------------------------------------------------------

    def _sweep(self) -> None:
        """Drop expired keys, mimicking Redis TTL."""
        now = time.time()
        for key in [k for k, deadline in self._expiry.items() if now >= deadline]:
            self._data.pop(key, None)
            self._expiry.pop(key, None)

    # -- commands -----------------------------------------------------------

    def ping(self) -> bool:
        return True

    def hset(self, key: str, mapping: dict[str, Any]) -> int:
        with self._lock:
            self._data.setdefault(key, {}).update(mapping)
            return len(mapping)

    def expire(self, key: str, seconds: int) -> bool:
        with self._lock:
            if key not in self._data:
                return False
            self._expiry[key] = time.time() + seconds
            return True

    def hgetall(self, key: str) -> dict[str, Any]:
        with self._lock:
            self._sweep()
            return dict(self._data.get(key, {}))

    def scan_iter(self, match: str = "*", count: int = 10) -> Iterator[str]:
        with self._lock:
            self._sweep()
            keys = [k for k in self._data if fnmatch.fnmatch(k, match)]
        yield from keys

    def delete(self, *keys: str) -> int:
        with self._lock:
            removed = 0
            for key in keys:
                if self._data.pop(key, None) is not None:
                    removed += 1
                self._expiry.pop(key, None)
            return removed

    def wait(self, num_replicas: int, timeout: int) -> int:
        """No replicas exist in-process, so nothing can ever acknowledge."""
        return 0

    def info(self, section: str = "all") -> dict[str, Any]:
        return {"role": "master", "master_repl_offset": 0, "connected_slaves": 0}
