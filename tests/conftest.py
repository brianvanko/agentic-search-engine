"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def mock_llm():
    """Mock LLM provider for testing."""
    llm = MagicMock()
    llm.name = "mock_llm"
    llm.model = "mock-model"
    return llm


@pytest.fixture
def mock_embedding():
    """Mock embedding provider for testing."""
    import numpy as np

    embedding = MagicMock()
    embedding.name = "mock_embedding"
    embedding.dimension = 768
    # encode returns 2D array (batch_size, dimension)
    embedding.encode = MagicMock(return_value=np.random.rand(1, 768).astype(np.float32))
    # encode_single returns 1D array (dimension,) - directly mock this for SemanticCache
    embedding.encode_single = MagicMock(return_value=np.random.rand(768).astype(np.float32))
    return embedding


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing."""
    retriever = MagicMock()
    retriever.name = "mock_retriever"
    retriever.retrieve = MagicMock(return_value=[])
    return retriever


@pytest.fixture
def sample_queries():
    """Sample queries for testing router classification."""
    return {
        "openai": [
            "How do I use the OpenAI API?",
            "What is an AI agent?",
            "Explain embeddings in machine learning",
            "How does GPT work?",
        ],
        "10k": [
            "What is Lyft's revenue?",
            "What are the risk factors in the 10-K?",
            "Company financial performance",
            "What is the operating income?",
        ],
        "web": [
            "Latest Tesla stock price",
            "Recent news about AI",
            "What happened today in tech?",
            "Current market trends",
        ],
        "hybrid": [
            "Compare Lyft's 10-K revenue to their latest quarterly report",
            "How has Lyft changed since their last filing?",
        ],
    }


@pytest.fixture
def mock_search_result():
    """Mock search result for testing."""
    from agentic_search.core.models import SearchResult

    return SearchResult(
        query="test query",
        response="This is a test response.",
        sources=[{"content": "Source 1", "score": 0.9}],
        cache_hit=False,
        cache_metadata={},
        routing_decision={},
        timing={"total_ms": 100},
    )
