"""llm-cache: lightweight semantic caching for LLM applications."""

from .cache import LLMCache
from .connection import ReplicationConfig, SentinelConfig, discover
from .embeddings import EmbeddingProvider, SentenceTransformerEmbeddings
from .models import CacheEntry, CacheResult
from .redis_store import RedisStore, RedisStoreError, ReplicationError
from .similarity import cosine_similarity
from .stats import CacheStats

__version__ = "0.1.0"

__all__ = [
    "CacheEntry",
    "CacheResult",
    "CacheStats",
    "EmbeddingProvider",
    "LLMCache",
    "RedisStore",
    "RedisStoreError",
    "ReplicationConfig",
    "ReplicationError",
    "SentinelConfig",
    "SentenceTransformerEmbeddings",
    "cosine_similarity",
    "discover",
]
