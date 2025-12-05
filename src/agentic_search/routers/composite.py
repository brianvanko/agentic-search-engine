"""Composite router that combines LLM and rule-based routing."""

import logging
from typing import List, Optional

from agentic_search.core.interfaces import BaseRouter
from agentic_search.core.models import RoutingDecision

logger = logging.getLogger(__name__)


class CompositeRouter(BaseRouter):
    """Composite router that tries LLM routing first, with rule-based fallback.

    This is the recommended router for production use. It provides:
    - Intelligent routing for all queries (LLM)
    - Graceful fallback to rules if LLM fails (API down, timeout, error)
    - Deterministic backup ensures queries are always routed

    Example:
        rule_router = RuleBasedRouter(local_companies=["lyft"])
        llm_router = LLMRouter(llm=OpenAILLM(...))
        router = CompositeRouter(
            llm_router=llm_router,
            rule_router=rule_router,
        )
        decision = router.route("What is Lyft's revenue?")
    """

    def __init__(
        self,
        llm_router: Optional[BaseRouter] = None,
        rule_router: Optional[BaseRouter] = None,
    ):
        """Initialize composite router.

        Args:
            llm_router: LLM router for intelligent routing (primary).
            rule_router: Rule-based router for fallback.
        """
        self._llm_router = llm_router
        self._rule_router = rule_router

    @property
    def name(self) -> str:
        return "composite"

    def route(
        self,
        query: str,
        available_retrievers: Optional[List[str]] = None,
        **kwargs,
    ) -> RoutingDecision:
        """Route a query using LLM first, then rule-based fallback.

        Args:
            query: The user's query.
            available_retrievers: List of available retriever names.
            **kwargs: Additional parameters.

        Returns:
            RoutingDecision with routing information.
        """
        # Try LLM routing first
        if self._llm_router is not None:
            try:
                decision = self._llm_router.route(query, available_retrievers, **kwargs)
                logger.info(
                    f"LLM routing: {decision.intent.value} "
                    f"(confidence: {decision.confidence:.0%})"
                )
                return decision
            except Exception as e:
                logger.warning(f"LLM routing failed: {e}, falling back to rule-based")

        # Fall back to rule-based routing
        if self._rule_router is not None:
            decision = self._rule_router.route(query, available_retrievers, **kwargs)
            if decision is not None:
                logger.info(
                    f"Rule-based fallback: {decision.intent.value} "
                    f"(confidence: {decision.confidence:.0%})"
                )
                return decision

        # No routers available or rule-based returned None - return default
        from agentic_search.core.models import QueryIntent

        logger.warning("All routers failed, using default routing")
        return RoutingDecision(
            intent=QueryIntent.LOCAL_10K,
            confidence=0.5,
            reason="Default routing (all routers failed or unavailable)",
            search_query=query,
            requires_web=False,
            keywords=[],
            target_retrievers=["qdrant_10k_data"],
        )
