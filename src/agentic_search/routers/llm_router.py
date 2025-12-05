"""LLM-based query router implementation."""

import logging
from typing import List, Optional, Dict

from agentic_search.core.interfaces import BaseRouter, BaseLLM
from agentic_search.core.models import RoutingDecision, QueryIntent
from agentic_search.core.exceptions import RouterError

logger = logging.getLogger(__name__)


class LLMRouter(BaseRouter):
    """LLM-based query router for intelligent query classification.

    Uses an LLM to classify queries and determine the best data source.
    This is the primary router - no keyword matching, pure LLM intelligence.

    Example:
        llm = OpenAILLM(api_key="sk-...")
        router = LLMRouter(llm=llm)
        decision = router.route("What is Lyft's revenue?")
    """

    ROUTING_PROMPT = """You are a query classification assistant. Your task is to analyze the user's query and determine the most appropriate action.

Based on the query, classify it into ONE of the following categories:

1. OPENAI_QUERY: Questions about OpenAI, GPT models, AI agents, embeddings, APIs, or AI development topics.
   Examples: "How do I use the OpenAI API?", "What is an AI agent?", "Explain embeddings"

2. 10K_DOCUMENT_QUERY: Questions about company financials, SEC 10-K filings, revenue, earnings, risk factors, or business operations from annual reports.
   Note: Our local database only contains 10-K filings for Lyft.
   Examples: "What is Lyft's revenue?", "What are the risk factors in the 10-K?", "Lyft financial performance"

3. INTERNET_QUERY: Questions requiring current/real-time information, news, recent events, or data not in our local documents.
   Examples: "Latest Tesla stock price", "Recent news about AI", "What happened today in tech?"

4. HYBRID: Questions that need BOTH local document data AND current internet information for a complete answer.
   Use this when comparing Lyft data (local) with other companies (web search needed).
   Examples: "Compare Lyft's 10-K revenue to their latest quarterly report", "Lyft vs Apple earnings"

Respond with JSON only:
{{
    "action": "OPENAI_QUERY" | "10K_DOCUMENT_QUERY" | "INTERNET_QUERY" | "HYBRID",
    "reason": "Brief explanation of why this classification was chosen",
    "confidence": 0.0-1.0,
    "web_search_query": "For HYBRID queries only: a focused search query to find the information NOT available locally. Be specific with company names (e.g., 'Apple Inc AAPL 2019 fiscal year revenue net income'). Null for non-HYBRID."
}}

User Query: {query}

JSON Response:"""

    # Map from LLM response action to QueryIntent
    ACTION_TO_INTENT = {
        "OPENAI_QUERY": QueryIntent.LOCAL_OPENAI,
        "10K_DOCUMENT_QUERY": QueryIntent.LOCAL_10K,
        "INTERNET_QUERY": QueryIntent.WEB_SEARCH,
        "HYBRID": QueryIntent.HYBRID,
    }

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

            # Get the action from LLM response and map to QueryIntent
            action = result.get("action", "10K_DOCUMENT_QUERY")
            intent = self.ACTION_TO_INTENT.get(action, QueryIntent.LOCAL_10K)

            # Determine if web search is needed based on intent
            requires_web = intent in (QueryIntent.WEB_SEARCH, QueryIntent.HYBRID)

            # Get targeted web search query for HYBRID cases
            web_search_query = result.get("web_search_query")
            if web_search_query and web_search_query.lower() in ("null", "none", ""):
                web_search_query = None

            logger.info(f"LLM classified query as: {action} -> {intent.value}")
            if web_search_query:
                logger.info(f"Targeted web search query: {web_search_query}")

            return RoutingDecision(
                intent=intent,
                confidence=result.get("confidence", 0.7),
                reason=result.get("reason", "LLM classification"),
                search_query=query,
                requires_web=requires_web,
                keywords=[],
                target_retrievers=self._retriever_mapping.get(intent, []),
                web_search_query=web_search_query,
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
