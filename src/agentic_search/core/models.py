"""Data models for the agentic search engine."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
import numpy as np


class QueryIntent(Enum):
    """Types of query intents for routing."""
    LOCAL_10K = "local_10k"
    LOCAL_OPENAI = "local_openai"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"
    CUSTOM = "custom"


@dataclass
class Document:
    """A retrieved document with metadata."""
    content: str
    source: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class RoutingDecision:
    """Result of query routing analysis."""
    intent: QueryIntent
    confidence: float
    reason: str
    search_query: str
    requires_web: bool = False
    keywords: List[str] = field(default_factory=list)
    target_retrievers: List[str] = field(default_factory=list)
    web_search_query: Optional[str] = None  # Targeted query for web search in HYBRID cases

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "search_query": self.search_query,
            "requires_web": self.requires_web,
            "keywords": self.keywords,
            "target_retrievers": self.target_retrievers,
            "web_search_query": self.web_search_query,
        }


@dataclass
class SearchResult:
    """Complete search result with all metadata."""
    query: str
    response: str
    sources: List[Dict[str, Any]]
    routing_decision: Dict[str, Any]
    cache_hit: bool
    cache_metadata: Dict[str, Any]
    timing: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "sources": self.sources,
            "routing_decision": self.routing_decision,
            "cache_hit": self.cache_hit,
            "cache_metadata": self.cache_metadata,
            "timing": self.timing,
            "timestamp": self.timestamp,
        }


@dataclass
class CacheEntry:
    """A single cache entry with embedding."""
    question: str
    response: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "response": self.response,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class CacheStats:
    """Statistics about cache performance."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_time_saved_ms: float = 0.0
    cache_size: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    @property
    def hit_rate_percent(self) -> float:
        return self.hit_rate * 100

    @property
    def avg_time_saved_ms(self) -> float:
        if self.cache_hits == 0:
            return 0.0
        return self.total_time_saved_ms / self.cache_hits

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": self.hit_rate_percent,
            "total_time_saved_ms": self.total_time_saved_ms,
            "avg_time_saved_ms": self.avg_time_saved_ms,
            "cache_size": self.cache_size,
        }


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
        }
