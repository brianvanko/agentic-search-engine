"""Web search retriever using DuckDuckGo."""

import logging
from typing import List, Dict, Any

from agentic_search.core.interfaces import BaseRetriever
from agentic_search.core.models import Document
from agentic_search.core.exceptions import RetrieverError

logger = logging.getLogger(__name__)


class WebSearchRetriever(BaseRetriever):
    """DuckDuckGo-based web search retriever.

    Retrieves documents from the internet using DuckDuckGo search.
    Automatically detects news vs. general search queries.

    Example:
        retriever = WebSearchRetriever(max_results=5, timeout=10)
        docs = retriever.retrieve("Latest Nvidia earnings", top_k=5)
    """

    # Keywords that indicate a news/current events query
    NEWS_KEYWORDS = [
        "latest", "recent", "news", "today", "yesterday", "this week",
        "this month", "2024", "2025", "current", "now", "breaking",
        "announced", "report", "update", "top", "best", "box office",
        "earnings", "stock", "price",
    ]

    def __init__(
        self,
        max_results: int = 5,
        timeout: int = 10,
        region: str = "wt-wt",
        safesearch: str = "moderate",
    ):
        """Initialize web search retriever.

        Args:
            max_results: Maximum results per search.
            timeout: Timeout for searches in seconds.
            region: DuckDuckGo region code.
            safesearch: Safe search level ('off', 'moderate', 'strict').
        """
        self._max_results = max_results
        self._timeout = timeout
        self._region = region
        self._safesearch = safesearch

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def source_type(self) -> str:
        return "web_search"

    def _is_news_query(self, query: str) -> bool:
        """Check if the query is asking for news/current events."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.NEWS_KEYWORDS)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        force_news: bool = False,
        **kwargs,
    ) -> List[Document]:
        """Retrieve documents from web search.

        Args:
            query: The search query.
            top_k: Maximum number of documents to return.
            force_news: Force news search regardless of query content.
            **kwargs: Additional search parameters.

        Returns:
            List of Document objects sorted by relevance.

        Raises:
            RetrieverError: If retrieval fails.
        """
        try:
            from ddgs import DDGS

            max_results = min(top_k, self._max_results)
            use_news = force_news or self._is_news_query(query)

            with DDGS(timeout=self._timeout) as ddgs:
                if use_news:
                    logger.info(f"Using NEWS search for: {query}")
                    results = list(
                        ddgs.news(
                            query,
                            max_results=max_results,
                            region=self._region,
                            safesearch=self._safesearch,
                        )
                    )
                else:
                    logger.info(f"Using TEXT search for: {query}")
                    results = list(
                        ddgs.text(
                            query,
                            max_results=max_results,
                            region=self._region,
                            safesearch=self._safesearch,
                        )
                    )

            # Convert to Document objects
            documents = []
            for i, result in enumerate(results):
                title = result.get("title", "No title")
                body = result.get("body", result.get("excerpt", "No description"))
                url = result.get("url", result.get("href", ""))

                # Format content with title and body
                content = f"**{title}**\n\n{body}\n\nSource URL: {url}"

                # Decreasing relevance score based on position
                score = 1.0 - (i * 0.1)

                documents.append(
                    Document(
                        content=content,
                        source=self.source_type,
                        score=score,
                        metadata={
                            "url": url,
                            "title": title,
                            "search_type": "news" if use_news else "text",
                        },
                    )
                )

            logger.debug(f"Retrieved {len(documents)} web results")
            return documents

        except Exception as e:
            logger.warning(f"Web search failed (returning empty results): {e}")
            # Graceful degradation - return empty list instead of raising
            return []

    def get_info(self) -> Dict[str, Any]:
        """Get information about this retriever."""
        return {
            "name": self.name,
            "source_type": self.source_type,
            "max_results": self._max_results,
            "timeout": self._timeout,
            "region": self._region,
        }
