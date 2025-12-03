"""LLM-based query router implementation."""

import logging
from typing import List, Optional, Dict

from agentic_search.core.interfaces import BaseRouter, BaseLLM
from agentic_search.core.models import RoutingDecision, QueryIntent
from agentic_search.core.exceptions import RouterError

logger = logging.getLogger(__name__)


class LLMRouter(BaseRouter):
    """LLM-based query router for complex routing decisions.

    Uses an LLM to classify queries that cannot be handled by rule-based
    routing. More flexible but slower and more expensive.

    Example:
        llm = OpenAILLM(api_key="sk-...")
        router = LLMRouter(llm=llm)
        decision = router.route("Should I invest in tech stocks?")
    """

    ROUTING_PROMPT = """You are a query router for a search engine. Analyze the query and determine the best data source.

Available data sources:
1. LOCAL_10K: Annual 10-K financial filings (financial statements, risk factors, business descriptions)
2. LOCAL_OPENAI: OpenAI documentation (agents, APIs, models, embeddings, AI development)
3. WEB_SEARCH: Live internet search (current events, news, real-time data)
4. HYBRID: Both local search and web search for comprehensive answers

Respond with JSON:
{{
    "intent": "LOCAL_10K" | "LOCAL_OPENAI" | "WEB_SEARCH" | "HYBRID",
    "confidence": 0.0-1.0,
    "reason": "Brief explanation",
    "search_query": "Optimized search query",
    "requires_web": true/false,
    "keywords": ["key", "terms"]
}}

Guidelines:
- LOCAL_10K: Company financials, revenue, earnings, risk factors, SEC filings
- LOCAL_OPENAI: OpenAI, GPT models, agents, embeddings, API usage
- WEB_SEARCH: Current events, latest news, real-time prices, recent announcements
- HYBRID: Questions needing both historical data AND current information

User Query: {query}

JSON Response:"""

    def __init__(
        self,
        llm: BaseLLM,
        retriever_mapping: Optional[Dict[QueryIntent, List[str]]] = None,
    ):
        """Initialize LLM router.

        Args:
            llm: LLM provider for classification.
            retriever_mapping: Maps intents to retriever names.
        """
        self._llm = llm
        self._retriever_mapping = retriever_mapping or {
            QueryIntent.LOCAL_10K: ["qdrant_10k_data"],
            QueryIntent.LOCAL_OPENAI: ["qdrant_opnai_data"],
            QueryIntent.WEB_SEARCH: ["web_search"],
            QueryIntent.HYBRID: ["qdrant_10k_data", "web_search"],
        }

    @property
    def name(self) -> str:
        return "llm"

    def route(
        self,
        query: str,
        available_retrievers: Optional[List[str]] = None,
        **kwargs,
    ) -> RoutingDecision:
        """Route a query using LLM classification.

        Args:
            query: The user's query.
            available_retrievers: List of available retriever names.
            **kwargs: Additional parameters.

        Returns:
            RoutingDecision with routing information.
        """
        try:
            prompt = self.ROUTING_PROMPT.format(query=query)
            result = self._llm.generate_json(prompt, temperature=0.1)

            intent_map = {
                "LOCAL_10K": QueryIntent.LOCAL_10K,
                "LOCAL_OPENAI": QueryIntent.LOCAL_OPENAI,
                "WEB_SEARCH": QueryIntent.WEB_SEARCH,
                "HYBRID": QueryIntent.HYBRID,
            }

            intent = intent_map.get(result.get("intent", "LOCAL_10K"), QueryIntent.LOCAL_10K)

            return RoutingDecision(
                intent=intent,
                confidence=result.get("confidence", 0.7),
                reason=result.get("reason", "LLM classification"),
                search_query=result.get("search_query", query),
                requires_web=result.get("requires_web", False),
                keywords=result.get("keywords", []),
                target_retrievers=self._retriever_mapping.get(intent, []),
            )

        except Exception as e:
            logger.warning(f"LLM routing failed, using default: {e}")
            # Default to local 10-K search on failure
            return RoutingDecision(
                intent=QueryIntent.LOCAL_10K,
                confidence=0.5,
                reason="Default routing (LLM unavailable or error)",
                search_query=query,
                requires_web=False,
                keywords=[],
                target_retrievers=self._retriever_mapping.get(QueryIntent.LOCAL_10K, []),
            )
