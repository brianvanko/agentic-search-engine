"""FAISS-based semantic cache implementation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from agentic_search.core.interfaces import BaseCache, BaseEmbedding
from agentic_search.core.models import CacheStats
from agentic_search.core.exceptions import CacheError

logger = logging.getLogger(__name__)


class SemanticCache(BaseCache):
    """FAISS-based semantic cache for query-response pairs.

    Caches responses based on semantic similarity of queries.
    Similar questions return cached responses without calling the LLM.

    Example:
        embedding = SentenceTransformerEmbedding()
        cache = SemanticCache(
            embedding=embedding,
            similarity_threshold=0.78,
            cache_file="./cache.json",
        )

        # First query - cache miss
        response, hit, _ = cache.lookup("What is Lyft's revenue?")

        # Store the response
        cache.store("What is Lyft's revenue?", "Lyft's revenue was...")

        # Similar query - cache hit
        response, hit, _ = cache.lookup("What are Lyft's revenues?")
    """

    ESTIMATED_RAG_TIME_MS = 2000  # Baseline for time savings calculation

    def __init__(
        self,
        embedding: BaseEmbedding,
        similarity_threshold: float = 0.78,
        max_size: int = 10000,
        cache_file: Optional[str] = None,
        persist: bool = True,
    ):
        """Initialize semantic cache.

        Args:
            embedding: Embedding provider for encoding queries.
            similarity_threshold: Minimum cosine similarity for cache hit (0-1).
            max_size: Maximum number of entries before eviction.
            cache_file: Path to persist cache (JSON file).
            persist: Whether to persist cache to disk.
        """
        self._embedding = embedding
        self._similarity_threshold = similarity_threshold
        self._max_size = max_size
        self._cache_file = Path(cache_file) if cache_file else None
        self._persist = persist

        # Convert cosine similarity threshold to euclidean distance
        # For normalized vectors: euclidean_dist = sqrt(2 * (1 - cosine_sim))
        self._distance_threshold = np.sqrt(2 * (1 - similarity_threshold))

        # Cache data structures
        self._cache: Dict[str, Any] = {
            "questions": [],
            "responses": [],
            "embeddings": [],
            "metadata": [],
            "created_at": [],
        }

        # Statistics
        self._stats = CacheStats()

        # FAISS index (lazy loaded)
        self._index = None

        # Load existing cache
        if self._cache_file and self._cache_file.exists():
            self._load_cache()

    @property
    def name(self) -> str:
        return "semantic_faiss"

    def _init_index(self):
        """Initialize or reinitialize FAISS index."""
        import faiss

        dimension = self._embedding.dimension
        self._index = faiss.IndexFlatL2(dimension)

        # Add existing embeddings to index
        if self._cache["embeddings"]:
            embeddings = np.array(self._cache["embeddings"], dtype=np.float32)
            self._index.add(embeddings)

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

        if not self._cache["questions"]:
            self._stats.cache_misses += 1
            return None, False, {}

        # Ensure index is initialized
        if self._index is None:
            self._init_index()

        try:
            # Encode query
            query_embedding = self._embedding.encode_single(question)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

            # Search FAISS index
            distances, indices = self._index.search(query_embedding, 1)
            distance = distances[0][0]
            idx = indices[0][0]

            if idx >= 0 and distance < self._distance_threshold:
                # Cache hit!
                self._stats.cache_hits += 1
                self._stats.total_time_saved_ms += self.ESTIMATED_RAG_TIME_MS

                cached_question = self._cache["questions"][idx]
                cached_response = self._cache["responses"][idx]
                metadata = self._cache["metadata"][idx] if idx < len(self._cache["metadata"]) else {}

                # Calculate similarity from distance
                similarity = 1 - (distance ** 2) / 2

                logger.info(
                    f"Cache HIT: similarity={similarity:.2%}, "
                    f"cached_question='{cached_question[:50]}...'"
                )

                return cached_response, True, {
                    "cached_question": cached_question,
                    "similarity_score": float(similarity),
                    "distance": float(distance),
                    "index": int(idx),
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
            if len(self._cache["questions"]) >= self._max_size:
                self._evict_oldest(int(self._max_size * 0.1))

            # Add to cache
            self._cache["questions"].append(question)
            self._cache["responses"].append(response)
            self._cache["embeddings"].append(embedding.tolist())
            self._cache["metadata"].append(metadata or {})
            self._cache["created_at"].append(datetime.now().isoformat())

            # Add to FAISS index
            if self._index is None:
                self._init_index()
            self._index.add(embedding.reshape(1, -1))

            # Persist to disk
            if self._persist and self._cache_file:
                self._save_cache()

            logger.debug(f"Cached response for: '{question[:50]}...'")

        except Exception as e:
            logger.warning(f"Failed to store in cache: {e}")

    def _evict_oldest(self, count: int):
        """Evict the oldest entries from the cache."""
        if count <= 0:
            return

        logger.info(f"Evicting {count} oldest cache entries")

        # Remove oldest entries
        for key in ["questions", "responses", "embeddings", "metadata", "created_at"]:
            if key in self._cache:
                self._cache[key] = self._cache[key][count:]

        # Rebuild FAISS index
        self._init_index()

        # Save updated cache
        if self._persist and self._cache_file:
            self._save_cache()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.cache_size = len(self._cache["questions"])
        return self._stats

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache = {
            "questions": [],
            "responses": [],
            "embeddings": [],
            "metadata": [],
            "created_at": [],
        }
        self._index = None
        self._stats = CacheStats()

        if self._persist and self._cache_file:
            self._save_cache()

        logger.info("Cache cleared")

    def _load_cache(self):
        """Load cache from disk."""
        try:
            with open(self._cache_file, "r") as f:
                data = json.load(f)

            self._cache = {
                "questions": data.get("questions", []),
                "responses": data.get("responses", []),
                "embeddings": data.get("embeddings", []),
                "metadata": data.get("metadata", []),
                "created_at": data.get("created_at", []),
            }

            # Load statistics
            stats_data = data.get("stats", {})
            self._stats = CacheStats(
                total_queries=stats_data.get("total_queries", 0),
                cache_hits=stats_data.get("cache_hits", 0),
                cache_misses=stats_data.get("cache_misses", 0),
                total_time_saved_ms=stats_data.get("total_time_saved_ms", 0.0),
                cache_size=len(self._cache["questions"]),
            )

            # Initialize FAISS index with loaded data
            if self._cache["embeddings"]:
                self._init_index()

            logger.info(
                f"Loaded cache from {self._cache_file}: "
                f"{len(self._cache['questions'])} entries"
            )

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save cache to disk."""
        try:
            data = {
                "questions": self._cache["questions"],
                "responses": self._cache["responses"],
                "embeddings": self._cache["embeddings"],
                "metadata": self._cache["metadata"],
                "created_at": self._cache["created_at"],
                "stats": {
                    "total_queries": self._stats.total_queries,
                    "cache_hits": self._stats.cache_hits,
                    "cache_misses": self._stats.cache_misses,
                    "total_time_saved_ms": self._stats.total_time_saved_ms,
                },
            }

            with open(self._cache_file, "w") as f:
                json.dump(data, f)

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
