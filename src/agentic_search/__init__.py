"""
Agentic Search Engine - A pluggable RAG pipeline.

This package provides a modular, extensible search engine with:
- Swappable LLM providers (OpenAI, Anthropic, local)
- Pluggable document retrievers (Qdrant, web search, custom)
- Customizable query routers
- Semantic caching with configurable backends
"""

from agentic_search.core.models import SearchResult, RoutingDecision, QueryIntent
from agentic_search.pipeline.rag_pipeline import RAGPipeline
from agentic_search.config.factory import create_pipeline

__version__ = "2.0.0"

__all__ = [
    "RAGPipeline",
    "SearchResult",
    "RoutingDecision",
    "QueryIntent",
    "create_pipeline",
]
