"""Rule-based query router implementation."""

import logging
import re
from typing import List, Optional, Dict, Set

from agentic_search.core.interfaces import BaseRouter
from agentic_search.core.models import RoutingDecision, QueryIntent

logger = logging.getLogger(__name__)


class RuleBasedRouter(BaseRouter):
    """Fast rule-based query router using keyword matching.

    Routes queries based on configurable keyword patterns. This is faster
    than LLM-based routing and handles obvious cases efficiently.

    Example:
        router = RuleBasedRouter(
            local_companies=["lyft", "uber"],
            financial_keywords=["revenue", "earnings", "10-k"],
        )
        decision = router.route("What is Lyft's revenue?")
        # -> RoutingDecision(intent=LOCAL_10K, ...)
    """

    # Default keyword sets
    DEFAULT_FINANCIAL_KEYWORDS = [
        "10-k", "10k", "annual report", "revenue", "earnings",
        "profit", "loss", "financial", "quarterly", "fiscal",
        "balance sheet", "income statement", "cash flow",
        "sec filing", "risk factor", "shareholder", "dividend",
        "eps", "ebitda", "gross margin", "operating income",
    ]

    DEFAULT_OPENAI_KEYWORDS = [
        "openai", "gpt", "chatgpt", "agent", "embedding",
        "api key", "fine-tun", "prompt", "token", "model",
        "dall-e", "whisper", "assistant", "function call",
    ]

    DEFAULT_WEB_INDICATORS = [
        "latest", "current", "today", "yesterday", "this week",
        "this month", "2024", "2025", "recent", "now", "live",
        "stock price", "breaking", "news", "just announced",
    ]

    DEFAULT_COMPARISON_KEYWORDS = [
        "compare", "compared", "comparison", "versus", "vs", "vs.",
        "difference", "between", "against", "better than", "worse than",
        " and ", " or ", " to ",
    ]

    # Words that shouldn't be treated as company names
    NON_COMPANY_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "what", "which", "who", "where", "when", "why", "how",
        "this", "that", "these", "those", "i", "you", "we", "they",
        "annual", "report", "revenue", "earnings", "profit", "loss",
        "financial", "fiscal", "quarterly", "stock", "price", "market",
        "company", "business", "year", "month", "day", "week",
        "compare", "compared", "comparing", "versus", "difference",
    }

    def __init__(
        self,
        local_companies: Optional[List[str]] = None,
        financial_keywords: Optional[List[str]] = None,
        openai_keywords: Optional[List[str]] = None,
        web_indicators: Optional[List[str]] = None,
        comparison_keywords: Optional[List[str]] = None,
        retriever_mapping: Optional[Dict[QueryIntent, List[str]]] = None,
    ):
        """Initialize rule-based router.

        Args:
            local_companies: List of companies in local data (lowercase).
            financial_keywords: Keywords indicating financial queries.
            openai_keywords: Keywords indicating OpenAI doc queries.
            web_indicators: Keywords indicating current events.
            comparison_keywords: Keywords indicating comparison queries.
            retriever_mapping: Maps intents to retriever names.
        """
        self._local_companies = set(local_companies or ["lyft"])
        self._financial_keywords = financial_keywords or self.DEFAULT_FINANCIAL_KEYWORDS
        self._openai_keywords = openai_keywords or self.DEFAULT_OPENAI_KEYWORDS
        self._web_indicators = web_indicators or self.DEFAULT_WEB_INDICATORS
        self._comparison_keywords = comparison_keywords or self.DEFAULT_COMPARISON_KEYWORDS

        # Default retriever mapping
        self._retriever_mapping = retriever_mapping or {
            QueryIntent.LOCAL_10K: ["qdrant_10k_data"],
            QueryIntent.LOCAL_OPENAI: ["qdrant_opnai_data"],
            QueryIntent.WEB_SEARCH: ["web_search"],
            QueryIntent.HYBRID: ["qdrant_10k_data", "web_search"],
        }

    @property
    def name(self) -> str:
        return "rule_based"

    def _has_local_company(self, query: str) -> bool:
        """Check if query mentions a company in local data."""
        query_lower = query.lower()
        return any(company in query_lower for company in self._local_companies)

    def _extract_potential_companies(self, query: str) -> List[str]:
        """Extract potential company names from query."""
        potential = []

        # Find capitalized words (potential proper nouns)
        words = re.findall(r'\b[A-Z][a-zA-Z]*\b', query)
        for word in words:
            word_lower = word.lower()
            if word_lower not in self.NON_COMPANY_WORDS and word_lower not in self._local_companies:
                potential.append(word_lower)

        # Check for company patterns like "X Inc", "X Corp"
        patterns = re.findall(
            r'\b([A-Z][a-zA-Z]*)\s+(?:Inc|Corp|LLC|Ltd|Corporation|Company)\b',
            query, re.IGNORECASE
        )
        for match in patterns:
            if match.lower() not in self._local_companies:
                potential.append(match.lower())

        return list(set(potential))

    def _has_external_company(self, query: str) -> bool:
        """Check if query mentions a company NOT in local data."""
        return len(self._extract_potential_companies(query)) > 0

    def _is_comparison_query(self, query: str) -> bool:
        """Check if query is comparing multiple entities."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self._comparison_keywords)

    def _contains_keywords(self, query: str, keywords: List[str]) -> List[str]:
        """Find which keywords are present in the query."""
        query_lower = query.lower()
        return [kw for kw in keywords if kw in query_lower]

    def route(
        self,
        query: str,
        available_retrievers: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[RoutingDecision]:
        """Route a query using rule-based matching.

        Args:
            query: The user's query.
            available_retrievers: List of available retriever names.
            **kwargs: Additional parameters.

        Returns:
            RoutingDecision if a rule matches, None if LLM routing needed.
        """
        query_lower = query.lower()

        # Check for web search indicators first
        web_matches = self._contains_keywords(query, self._web_indicators)
        if web_matches:
            financial_matches = self._contains_keywords(query, self._financial_keywords)
            if financial_matches and self._has_local_company(query):
                return RoutingDecision(
                    intent=QueryIntent.HYBRID,
                    confidence=0.85,
                    reason="Query mentions recent data for a company in local filings",
                    search_query=query,
                    requires_web=True,
                    keywords=web_matches + financial_matches,
                    target_retrievers=self._retriever_mapping.get(QueryIntent.HYBRID, []),
                )
            return RoutingDecision(
                intent=QueryIntent.WEB_SEARCH,
                confidence=0.9,
                reason="Query asks for current/recent information",
                search_query=query,
                requires_web=True,
                keywords=web_matches,
                target_retrievers=self._retriever_mapping.get(QueryIntent.WEB_SEARCH, []),
            )

        # Check for OpenAI-specific queries
        openai_matches = self._contains_keywords(query, self._openai_keywords)
        if openai_matches:
            return RoutingDecision(
                intent=QueryIntent.LOCAL_OPENAI,
                confidence=0.9,
                reason="Query relates to OpenAI/GPT documentation",
                search_query=query,
                requires_web=False,
                keywords=openai_matches,
                target_retrievers=self._retriever_mapping.get(QueryIntent.LOCAL_OPENAI, []),
            )

        # Check for financial queries
        financial_matches = self._contains_keywords(query, self._financial_keywords)
        if financial_matches:
            has_local = self._has_local_company(query)
            has_external = self._has_external_company(query)
            is_comparison = self._is_comparison_query(query)

            logger.debug(
                f"Financial query: local={has_local}, "
                f"external={has_external}, comparison={is_comparison}"
            )

            if has_local and (has_external or is_comparison):
                return RoutingDecision(
                    intent=QueryIntent.HYBRID,
                    confidence=0.9,
                    reason="Query compares local company with external company",
                    search_query=query,
                    requires_web=True,
                    keywords=financial_matches,
                    target_retrievers=self._retriever_mapping.get(QueryIntent.HYBRID, []),
                )

            if has_local and not has_external:
                return RoutingDecision(
                    intent=QueryIntent.LOCAL_10K,
                    confidence=0.85,
                    reason="Query relates to financial data for company in local filings",
                    search_query=query,
                    requires_web=False,
                    keywords=financial_matches,
                    target_retrievers=self._retriever_mapping.get(QueryIntent.LOCAL_10K, []),
                )

            if has_external and not has_local:
                return RoutingDecision(
                    intent=QueryIntent.WEB_SEARCH,
                    confidence=0.9,
                    reason="Query about external company not in local data",
                    search_query=query,
                    requires_web=True,
                    keywords=financial_matches,
                    target_retrievers=self._retriever_mapping.get(QueryIntent.WEB_SEARCH, []),
                )

        # No clear match - return None to indicate LLM routing needed
        return None
