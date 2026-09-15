"""Demo configuration, read from a ``.env`` file.

A ~30 line parser rather than a `python-dotenv` dependency — the project keeps
its dependency list short, and this is a demo.

Precedence: real environment variables win over ``.env``, so you can override a
single value for one run without editing the file:

    NAMESPACE=scratch uv run python demo/server.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=value`` pairs from a ``.env`` file into ``os.environ``.

    Missing file is not an error — the defaults below then apply. Existing
    environment variables are left alone unless ``override`` is set.
    """
    env_path = path or ENV_FILE
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _int_or_none(name: str, default: int | None) -> int | None:
    """Parse an int, treating empty/``none``/``0`` as "no expiry"."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if raw.lower() in {"none", "null", "off"}:
        return None
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer or 'none', got {raw!r}") from exc
    return None if parsed <= 0 else parsed


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the demo reads from the environment."""

    redis_url: str = "redis://localhost:6379"
    similarity_threshold: float = 0.90
    ttl: int | None = 3600
    namespace: str = "example"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"
    temperature: float = 0.2
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def has_api_key(self) -> bool:
        return bool(self.groq_api_key)

    def describe(self) -> dict[str, object]:
        """Config summary for the UI. Never includes the API key itself."""
        return {
            "redis_url": self.redis_url,
            "similarity_threshold": self.similarity_threshold,
            "ttl": self.ttl,
            "namespace": self.namespace,
            "groq_model": self.groq_model,
            "temperature": self.temperature,
            "has_api_key": self.has_api_key,
        }


def load_settings(path: Path | None = None) -> Settings:
    """Read ``.env`` (if present) and build the :class:`Settings`."""
    load_env(path)
    settings = Settings(
        redis_url=_str("REDIS_URL", "redis://localhost:6379"),
        similarity_threshold=_float("SIMILARITY_THRESHOLD", 0.90),
        ttl=_int_or_none("TTL", 3600),
        namespace=_str("NAMESPACE", "example"),
        groq_api_key=os.environ.get("GROQ_API_KEY") or None,
        groq_model=_str("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=_float("TEMPERATURE", 0.2),
        host=_str("HOST", "127.0.0.1"),
        port=int(_float("PORT", 8000)),
    )
    if not 0.0 <= settings.similarity_threshold <= 1.0:
        raise ValueError(
            f"SIMILARITY_THRESHOLD must be between 0.0 and 1.0, got {settings.similarity_threshold}"
        )
    return settings
