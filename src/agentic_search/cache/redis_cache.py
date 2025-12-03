"""Redis-based semantic cache implementation."""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from agentic_search.core.interfaces import BaseCache, BaseEmbedding
from agentic_search.core.models import CacheStats
from agentic_search.core.exceptions import CacheError

logger = logging.getLogger(__name__)


class RedisSemanticCache(BaseCache):
    """Redis-based semantic cache for query-response pairs.

    Uses Redis for persistent storage and FAISS for in-memory similarity search.
    Supports TTL (time-to-live) for automatic cache expiration.

    Example:
        embedding = SentenceTransformerEmbedding()
        cache = RedisSemanticCache(
            embedding=embedding,
            similarity_threshold=0.78,
            redis_url="redis://localhost:6379",
            ttl_seconds=86400,  # 24 hours
        )

        # First query - cache miss
        response, hit, _ = cache.lookup("What is Lyft's revenue?")

        # Store the response
        cache.store("What is Lyft's revenue?", "Lyft's revenue was...")

        # Similar query - cache hit
        response, hit, _ = cache.lookup("What are Lyft's revenues?")
    """

    ESTIMATED_RAG_TIME_MS = 2000  # Baseline for time savings calculation
    CACHE_KEY_PREFIX = "semantic_cache:"
    INDEX_KEY = "semantic_cache:index"
    STATS_KEY = "semantic_cache:stats"
    COUNTER_KEY = "semantic_cache:counter"

    def __init__(
        self,
        embedding: BaseEmbedding,
        similarity_threshold: float = 0.78,
        max_size: int = 10000,
        redis_url: str = "redis://localhost:6379",
        redis_db: int = 0,
        ttl_seconds: Optional[int] = None,
        key_prefix: str = "semantic_cache:",
    ):
        """Initialize Redis semantic cache.

        Args:
            embedding: Embedding provider for encoding queries.
            similarity_threshold: Minimum cosine similarity for cache hit (0-1).
            max_size: Maximum number of entries before eviction.
            redis_url: Redis connection URL.
            redis_db: Redis database number.
            ttl_seconds: Time-to-live for cache entries (None = no expiration).
            key_prefix: Prefix for Redis keys.
        """
        self._embedding = embedding
        self._similarity_threshold = similarity_threshold
        self._max_size = max_size
        self._redis_url = redis_url
        self._redis_db = redis_db
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix

        # Convert cosine similarity threshold to euclidean distance
        self._distance_threshold = np.sqrt(2 * (1 - similarity_threshold))

        # Statistics (in-memory, synced to Redis periodically)
        self._stats = CacheStats()

        # FAISS index (in-memory, rebuilt from Redis on startup)
        self._index = None
        self._index_to_key: List[str] = []  # Maps FAISS index to Redis key

        # Redis client (lazy initialized)
        self._redis = None

        # Initialize connection and load existing data
        self._init_redis()
        self._load_from_redis()

    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            import redis

            self._redis = redis.from_url(
                self._redis_url,
                db=self._redis_db,
                decode_responses=False,  # We need bytes for embeddings
            )
            # Test connection
            self._redis.ping()
            logger.info(f"Connected to Redis at {self._redis_url}")
        except ImportError:
            raise CacheError("redis package not installed. Run: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheError(f"Redis connection failed: {e}")

    def _init_index(self, dimension: int):
        """Initialize or reinitialize FAISS index."""
        import faiss

        self._index = faiss.IndexFlatL2(dimension)
        self._index_to_key = []

    def _load_from_redis(self):
        """Load existing cache entries from Redis into FAISS index."""
        try:
            # Get all cache keys
            pattern = f"{self._key_prefix}entry:*"
            keys = list(self._redis.scan_iter(match=pattern))

            if not keys:
                logger.info("No existing cache entries in Redis")
                return

            # Initialize FAISS index
            dimension = self._embedding.dimension
            self._init_index(dimension)

            # Load embeddings and build index
            embeddings = []
            valid_keys = []

            for key in keys:
                try:
                    data = self._redis.hgetall(key)
                    if b"embedding" in data:
                        embedding = np.frombuffer(data[b"embedding"], dtype=np.float32)
                        embeddings.append(embedding)
                        valid_keys.append(key.decode() if isinstance(key, bytes) else key)
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {key}: {e}")

            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self._index.add(embeddings_array)
                self._index_to_key = valid_keys

            # Load stats
            stats_data = self._redis.hgetall(self.STATS_KEY)
            if stats_data:
                self._stats = CacheStats(
                    total_queries=int(stats_data.get(b"total_queries", 0)),
                    cache_hits=int(stats_data.get(b"cache_hits", 0)),
                    cache_misses=int(stats_data.get(b"cache_misses", 0)),
                    total_time_saved_ms=float(stats_data.get(b"total_time_saved_ms", 0)),
                    cache_size=len(valid_keys),
                )

            logger.info(f"Loaded {len(valid_keys)} cache entries from Redis")

        except Exception as e:
            logger.warning(f"Failed to load cache from Redis: {e}")

    @property
    def name(self) -> str:
        return "semantic_redis"

    def lookup(
        self,
        question: str,
        **kwargs,
    ) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """Look up a question in the cache.

        Args:
            question: The question to look up.
            **kwargs: Additional parameters.

        Returns:
            Tuple of (cached_response, is_hit, metadata).
        """
        self._stats.total_queries += 1

        if self._index is None or self._index.ntotal == 0:
            self._stats.cache_misses += 1
            return None, False, {}

        try:
            # Encode query
            query_embedding = self._embedding.encode_single(question)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

            # Search FAISS index
            distances, indices = self._index.search(query_embedding, 1)
            distance = distances[0][0]
            idx = indices[0][0]

            if idx >= 0 and distance < self._distance_threshold:
                # Get the Redis key for this index
                redis_key = self._index_to_key[idx]

                # Fetch from Redis
                data = self._redis.hgetall(redis_key)
                if not data:
                    # Entry expired or deleted
                    self._stats.cache_misses += 1
                    return None, False, {}

                # Cache hit!
                self._stats.cache_hits += 1
                self._stats.total_time_saved_ms += self.ESTIMATED_RAG_TIME_MS

                cached_question = data[b"question"].decode()
                cached_response = data[b"response"].decode()
                metadata = json.loads(data.get(b"metadata", b"{}").decode())

                # Calculate similarity from distance
                similarity = 1 - (distance ** 2) / 2

                logger.info(
                    f"Cache HIT: similarity={similarity:.2%}, "
                    f"cached_question='{cached_question[:50]}...'"
                )

                # Update stats in Redis periodically
                self._sync_stats()

                return cached_response, True, {
                    "cached_question": cached_question,
                    "similarity_score": float(similarity),
                    "distance": float(distance),
                    "redis_key": redis_key,
                }

            self._stats.cache_misses += 1
            return None, False, {}

        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            self._stats.cache_misses += 1
            return None, False, {}

    def store(
        self,
        question: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Store a question-response pair in the cache.

        Args:
            question: The question.
            response: The response to cache.
            metadata: Optional metadata to store.
            **kwargs: Additional parameters.
        """
        try:
            # Encode question
            embedding = self._embedding.encode_single(question)
            embedding = embedding.astype(np.float32)

            # Check if cache needs eviction
            current_size = self._redis.scard(f"{self._key_prefix}keys") or 0
            if current_size >= self._max_size:
                self._evict_oldest(int(self._max_size * 0.1))

            # Generate unique key
            entry_id = self._redis.incr(self.COUNTER_KEY)
            redis_key = f"{self._key_prefix}entry:{entry_id}"

            # Store in Redis
            entry_data = {
                "question": question.encode(),
                "response": response.encode(),
                "embedding": embedding.tobytes(),
                "metadata": json.dumps(metadata or {}).encode(),
                "created_at": datetime.now().isoformat().encode(),
            }
            self._redis.hset(redis_key, mapping=entry_data)

            # Set TTL if configured
            if self._ttl_seconds:
                self._redis.expire(redis_key, self._ttl_seconds)

            # Track keys for eviction
            self._redis.sadd(f"{self._key_prefix}keys", redis_key)

            # Update FAISS index
            if self._index is None:
                self._init_index(len(embedding))

            self._index.add(embedding.reshape(1, -1))
            self._index_to_key.append(redis_key)

            logger.debug(f"Cached response for: '{question[:50]}...'")

        except Exception as e:
            logger.warning(f"Failed to store in cache: {e}")

    def _evict_oldest(self, count: int):
        """Evict the oldest entries from the cache."""
        if count <= 0:
            return

        logger.info(f"Evicting {count} oldest cache entries")

        try:
            # Get oldest keys (by entry ID which is sequential)
            all_keys = list(self._redis.smembers(f"{self._key_prefix}keys"))
            if not all_keys:
                return

            # Sort by entry ID (extract number from key)
            def get_entry_id(key):
                key_str = key.decode() if isinstance(key, bytes) else key
                try:
                    return int(key_str.split(":")[-1])
                except:
                    return 0

            all_keys.sort(key=get_entry_id)
            keys_to_delete = all_keys[:count]

            # Delete from Redis
            for key in keys_to_delete:
                self._redis.delete(key)
                self._redis.srem(f"{self._key_prefix}keys", key)

            # Rebuild FAISS index
            self._load_from_redis()

        except Exception as e:
            logger.warning(f"Failed to evict cache entries: {e}")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.cache_size = len(self._index_to_key) if self._index_to_key else 0
        return self._stats

    def _sync_stats(self):
        """Sync statistics to Redis."""
        try:
            self._redis.hset(
                self.STATS_KEY,
                mapping={
                    "total_queries": str(self._stats.total_queries),
                    "cache_hits": str(self._stats.cache_hits),
                    "cache_misses": str(self._stats.cache_misses),
                    "total_time_saved_ms": str(self._stats.total_time_saved_ms),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to sync stats to Redis: {e}")

    def clear(self) -> None:
        """Clear all entries from the cache."""
        try:
            # Get all cache keys
            pattern = f"{self._key_prefix}*"
            keys = list(self._redis.scan_iter(match=pattern))

            if keys:
                self._redis.delete(*keys)

            self._index = None
            self._index_to_key = []
            self._stats = CacheStats()

            logger.info("Redis cache cleared")

        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        return {
            "name": self.name,
            "redis_url": self._redis_url,
            "redis_db": self._redis_db,
            "ttl_seconds": self._ttl_seconds,
            "max_size": self._max_size,
            "current_size": len(self._index_to_key) if self._index_to_key else 0,
            "similarity_threshold": self._similarity_threshold,
            "stats": self.get_stats().__dict__,
        }
