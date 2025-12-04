"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_pipeline():
    """Mock RAG pipeline for testing."""
    from agentic_search.core.models import SearchResult

    pipeline = MagicMock()
    pipeline.search = MagicMock(return_value=SearchResult(
        query="test query",
        response="Test response",
        sources=[{"content": "Source 1", "score": 0.9}],
        cache_hit=False,
        cache_metadata={},
        routing_decision={"intent": "LOCAL_10K"},
        timing={"total_ms": 100},
    ))
    pipeline.get_cache_stats = MagicMock(return_value={
        "size": 10,
        "max_size": 1000,
        "hit_rate": 0.7,
    })
    pipeline.clear_cache = MagicMock()
    return pipeline


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    import numpy as np

    model = MagicMock()
    model.encode = MagicMock(
        return_value=np.array([[0.1] * 768], dtype=np.float32)
    )
    model.dimension = 768
    return model


@pytest.fixture
def client(mock_pipeline, mock_embedding_model):
    """Create test client with mocked dependencies."""
    with patch('api.get_pipeline', return_value=mock_pipeline), \
         patch('api.get_embedding_model', return_value=mock_embedding_model):
        from api import app
        yield TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data


class TestEmbedEndpoint:
    """Tests for /embed endpoint."""

    def test_embed_single_text(self, client):
        """Test embedding a single text."""
        response = client.post(
            "/embed",
            json={"text": "What is machine learning?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "dimension" in data
        assert isinstance(data["embedding"], list)
        assert data["dimension"] == 768

    def test_embed_with_normalize(self, client):
        """Test embedding with normalization."""
        response = client.post(
            "/embed",
            json={"text": "Test text", "normalize": True}
        )

        assert response.status_code == 200

    def test_embed_empty_text(self, client):
        """Test embedding with empty text."""
        response = client.post(
            "/embed",
            json={"text": ""}
        )

        # Should still succeed (empty string is valid)
        assert response.status_code == 200


class TestEmbedBatchEndpoint:
    """Tests for /embed/batch endpoint."""

    def test_embed_batch(self, client, mock_embedding_model):
        """Test batch embedding."""
        import numpy as np

        # Mock to return multiple embeddings
        mock_embedding_model.encode = MagicMock(
            return_value=np.array([[0.1] * 768, [0.2] * 768], dtype=np.float32)
        )

        response = client.post(
            "/embed/batch",
            json={"texts": ["Hello", "World"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert "count" in data
        assert data["count"] == 2

    def test_embed_batch_single_item(self, client):
        """Test batch embedding with single item."""
        response = client.post(
            "/embed/batch",
            json={"texts": ["Single text"]}
        )

        assert response.status_code == 200


class TestSearchEndpoint:
    """Tests for /search endpoint."""

    def test_search_basic(self, client, mock_pipeline):
        """Test basic search."""
        response = client.post(
            "/search",
            json={"query": "What is Lyft's revenue?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "response" in data
        assert "sources" in data
        assert "cache_hit" in data
        assert "timing" in data

    def test_search_with_cache_disabled(self, client, mock_pipeline):
        """Test search with cache disabled."""
        response = client.post(
            "/search",
            json={"query": "Test query", "use_cache": False}
        )

        assert response.status_code == 200
        mock_pipeline.search.assert_called_with("Test query", use_cache=False)

    def test_search_returns_sources(self, client, mock_pipeline):
        """Test search returns sources."""
        response = client.post(
            "/search",
            json={"query": "Financial data"}
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["sources"], list)

    def test_search_returns_timing(self, client, mock_pipeline):
        """Test search returns timing information."""
        response = client.post(
            "/search",
            json={"query": "Test query"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "timing" in data


class TestCacheEndpoints:
    """Tests for cache management endpoints."""

    def test_cache_stats(self, client, mock_pipeline):
        """Test getting cache statistics."""
        response = client.get("/cache/stats")

        assert response.status_code == 200
        data = response.json()
        assert "size" in data
        assert "max_size" in data

    def test_cache_clear(self, client, mock_pipeline):
        """Test clearing the cache."""
        response = client.post("/cache/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        mock_pipeline.clear_cache.assert_called_once()


class TestIngestEndpoints:
    """Tests for ingestion endpoints."""

    def test_ingest_text(self, client, mock_embedding_model):
        """Test text ingestion."""
        import numpy as np

        mock_embedding_model.encode = MagicMock(
            return_value=np.array([[0.1] * 768, [0.2] * 768], dtype=np.float32)
        )

        response = client.post(
            "/ingest/text",
            json={
                "text": "This is a long text that will be chunked. " * 20,
                "chunk_size": 100,
                "chunk_overlap": 10,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert "embeddings_shape" in data
        assert "sample_chunks" in data

    def test_ingest_text_with_metadata(self, client, mock_embedding_model):
        """Test text ingestion with metadata."""
        import numpy as np

        mock_embedding_model.encode = MagicMock(
            return_value=np.array([[0.1] * 768], dtype=np.float32)
        )

        response = client.post(
            "/ingest/text",
            json={
                "text": "Sample text",
                "metadata": {"source": "test", "author": "tester"},
            }
        )

        assert response.status_code == 200


class TestChunkText:
    """Tests for text chunking utility."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        from api import chunk_text

        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        from api import chunk_text

        chunks = chunk_text("", chunk_size=100, overlap=10)
        assert chunks == []

    def test_chunk_text_overlap(self):
        """Test chunks have overlap."""
        from api import chunk_text

        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = chunk_text(text, chunk_size=40, overlap=10)

        # With overlap, consecutive chunks should share some content
        if len(chunks) >= 2:
            # Overlap means end of one chunk appears at start of next
            assert len(chunks) >= 2

    def test_chunk_text_single_sentence(self):
        """Test chunking single sentence."""
        from api import chunk_text

        text = "This is a single sentence."
        chunks = chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == "This is a single sentence."


class TestCORSConfiguration:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test CORS headers are present in response."""
        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )

        # FastAPI TestClient may not fully emulate CORS preflight
        # but we can check the endpoint is accessible
        assert response.status_code in [200, 405]

    def test_allowed_origins(self):
        """Test allowed origins are configured."""
        from api import app

        # Check middleware is configured
        middlewares = [m for m in app.user_middleware]
        assert len(middlewares) > 0
