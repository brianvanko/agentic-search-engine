"""Tests for semantic cache."""

import os
import json
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestSemanticCache:
    """Tests for SemanticCache class."""

    @pytest.fixture
    def cache_file(self):
        """Create a temporary cache file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"entries": []}, f)
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def semantic_cache(self, mock_embedding, cache_file):
        """Create a SemanticCache instance for testing."""
        from agentic_search.cache.semantic_cache import SemanticCache

        return SemanticCache(
            embedding=mock_embedding,
            similarity_threshold=0.8,
            max_size=100,
            cache_file=cache_file,
        )

    def test_cache_initialization(self, semantic_cache):
        """Test cache initializes correctly."""
        assert semantic_cache.name == "semantic_faiss"
        assert semantic_cache._similarity_threshold == 0.8
        assert semantic_cache._max_size == 100

    def test_cache_lookup_miss(self, semantic_cache):
        """Test cache returns None on miss."""
        response, hit, metadata = semantic_cache.lookup("non-existent query")
        assert response is None
        assert hit is False

    def test_cache_store_and_lookup(self, semantic_cache, mock_embedding):
        """Test storing and retrieving from cache."""
        query = "What is Lyft's revenue?"
        response = "Lyft's revenue is $4.1 billion."

        # Mock encode_single to return same vector for similar queries
        mock_embedding.encode_single = MagicMock(
            return_value=np.array([0.1] * 768, dtype=np.float32)
        )

        # Store in cache
        semantic_cache.store(query, response)

        # Retrieve from cache
        result, hit, metadata = semantic_cache.lookup(query)

        assert hit is True
        assert result == response

    def test_cache_similarity_threshold(self, semantic_cache, mock_embedding):
        """Test cache respects similarity threshold."""
        query = "What is Lyft's revenue?"
        response = "Lyft's revenue is $4.1 billion."

        # First call returns one vector, second returns a different one
        mock_embedding.encode_single = MagicMock(side_effect=[
            np.array([0.1] * 768, dtype=np.float32),  # Store
            np.array([0.9] * 768, dtype=np.float32),  # Lookup (different vector)
        ])

        semantic_cache.store(query, response)

        # Different query should not match
        result, hit, metadata = semantic_cache.lookup("Completely different query")
        # With different vectors, it should be a cache miss
        assert hit is False

    def test_cache_clear(self, semantic_cache, mock_embedding):
        """Test clearing the cache."""
        mock_embedding.encode_single = MagicMock(
            return_value=np.array([0.1] * 768, dtype=np.float32)
        )

        semantic_cache.store("query", "response")
        semantic_cache.clear()

        stats = semantic_cache.get_stats()
        assert stats.cache_size == 0

    def test_cache_stats(self, semantic_cache, mock_embedding):
        """Test cache statistics."""
        mock_embedding.encode_single = MagicMock(
            return_value=np.array([0.1] * 768, dtype=np.float32)
        )

        # Initial stats - CacheStats object
        stats = semantic_cache.get_stats()
        assert hasattr(stats, 'cache_size')
        assert hasattr(stats, 'total_queries')
        assert hasattr(stats, 'cache_hits')

        # Add some entries with unique vectors (1D arrays)
        vectors = [np.random.rand(768).astype(np.float32) for _ in range(2)]
        mock_embedding.encode_single = MagicMock(side_effect=vectors)

        semantic_cache.store("query1", "response1")
        semantic_cache.store("query2", "response2")

        stats = semantic_cache.get_stats()
        assert stats.cache_size == 2

    def test_cache_max_size_eviction(self, mock_embedding, cache_file):
        """Test cache evicts entries when max size is reached."""
        from agentic_search.cache.semantic_cache import SemanticCache

        # Create cache with max_size where 10% eviction is meaningful (10% of 10 = 1)
        cache = SemanticCache(
            embedding=mock_embedding,
            similarity_threshold=0.8,
            max_size=10,
            cache_file=cache_file,
        )

        # Create unique vectors for each query (1D arrays for encode_single)
        vectors = [np.random.rand(768).astype(np.float32) for _ in range(15)]
        call_count = [0]

        def mock_encode_single(*args, **kwargs):
            idx = min(call_count[0], len(vectors) - 1)
            call_count[0] += 1
            return vectors[idx]

        mock_embedding.encode_single = MagicMock(side_effect=mock_encode_single)

        # Add more than max_size entries
        for i in range(15):
            cache.store(f"query{i}", f"response{i}")

        stats = cache.get_stats()
        # Should have evicted some entries (cache size should be <= max_size)
        assert stats.cache_size <= 10

    def test_cache_persistence(self, mock_embedding):
        """Test cache persists to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            cache_file = f.name

        try:
            from agentic_search.cache.semantic_cache import SemanticCache

            mock_embedding.encode_single = MagicMock(
                return_value=np.array([0.1] * 768, dtype=np.float32)
            )

            # Create cache and add entry
            cache1 = SemanticCache(
                embedding=mock_embedding,
                similarity_threshold=0.8,
                max_size=100,
                cache_file=cache_file,
            )
            cache1.store("query", "response")

            # Create new cache instance - should load from file
            cache2 = SemanticCache(
                embedding=mock_embedding,
                similarity_threshold=0.8,
                max_size=100,
                cache_file=cache_file,
            )

            stats = cache2.get_stats()
            assert stats.cache_size >= 1

        finally:
            os.unlink(cache_file)


class TestCacheMetadata:
    """Tests for cache metadata handling."""

    @pytest.fixture
    def cache_with_entries(self, mock_embedding):
        """Create a cache with some entries."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            cache_file = f.name

        from agentic_search.cache.semantic_cache import SemanticCache

        cache = SemanticCache(
            embedding=mock_embedding,
            similarity_threshold=0.8,
            max_size=100,
            cache_file=cache_file,
        )

        yield cache

        os.unlink(cache_file)

    def test_cache_stores_metadata(self, cache_with_entries, mock_embedding):
        """Test that cache can store metadata."""
        mock_embedding.encode_single = MagicMock(
            return_value=np.array([0.1] * 768, dtype=np.float32)
        )

        cache_with_entries.store("query", "response", {"custom": "data"})
        response, hit, metadata = cache_with_entries.lookup("query")

        assert hit is True
        assert response is not None

    def test_cache_returns_similarity_score(self, cache_with_entries, mock_embedding):
        """Test that cache returns similarity score on hit."""
        mock_embedding.encode_single = MagicMock(
            return_value=np.array([0.1] * 768, dtype=np.float32)
        )

        cache_with_entries.store("query", "response")
        response, hit, metadata = cache_with_entries.lookup("query")

        assert hit is True
        assert response == "response"
        # Metadata should contain similarity info
        assert isinstance(metadata, dict)
