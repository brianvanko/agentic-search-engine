"""Core interfaces and models for the agentic search engine."""

from agentic_search.core.interfaces import (
    BaseLLM,
    BaseEmbedding,
    BaseRetriever,
    BaseRouter,
    BaseCache,
)
from agentic_search.core.models import (
    SearchResult,
    RoutingDecision,
    QueryIntent,
    Document,
    CacheEntry,
    CacheStats,
)
from agentic_search.core.exceptions import (
    AgenticSearchError,
    LLMError,
    RetrieverError,
    RouterError,
    CacheError,
    ConfigurationError,
)

__all__ = [
    # Interfaces
    "BaseLLM",
    "BaseEmbedding",
    "BaseRetriever",
    "BaseRouter",
    "BaseCache",
    # Models
    "SearchResult",
    "RoutingDecision",
    "QueryIntent",
    "Document",
    "CacheEntry",
    "CacheStats",
    # Exceptions
    "AgenticSearchError",
    "LLMError",
    "RetrieverError",
    "RouterError",
    "CacheError",
    "ConfigurationError",
]
