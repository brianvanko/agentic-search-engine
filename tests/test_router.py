"""Tests for LLM-based query router."""

import pytest
from unittest.mock import MagicMock, patch

from agentic_search.routers.llm_router import LLMRouter
from agentic_search.core.models import QueryIntent, RoutingDecision


class TestLLMRouter:
    """Tests for LLMRouter class."""

    def test_router_initialization(self, mock_llm):
        """Test router initializes correctly."""
        router = LLMRouter(llm=mock_llm)

        assert router.name == "llm"
        assert router._llm == mock_llm
        assert QueryIntent.LOCAL_10K in router._retriever_mapping
        assert QueryIntent.LOCAL_OPENAI in router._retriever_mapping
        assert QueryIntent.WEB_SEARCH in router._retriever_mapping
        assert QueryIntent.HYBRID in router._retriever_mapping

    def test_router_with_custom_mapping(self, mock_llm):
        """Test router with custom retriever mapping."""
        custom_mapping = {
            QueryIntent.LOCAL_10K: ["custom_retriever"],
        }
        router = LLMRouter(llm=mock_llm, retriever_mapping=custom_mapping)

        assert router._retriever_mapping == custom_mapping

    def test_route_openai_query(self, mock_llm):
        """Test routing an OpenAI-related query."""
        mock_llm.generate_json = MagicMock(return_value={
            "action": "OPENAI_QUERY",
            "reason": "Query is about OpenAI API usage",
            "confidence": 0.95,
        })

        router = LLMRouter(llm=mock_llm)
        decision = router.route("How do I use the OpenAI API?")

        assert isinstance(decision, RoutingDecision)
        assert decision.intent == QueryIntent.LOCAL_OPENAI
        assert decision.confidence == 0.95
        assert decision.requires_web is False
        assert "qdrant_opnai_data" in decision.target_retrievers

    def test_route_10k_query(self, mock_llm):
        """Test routing a 10-K document query."""
        mock_llm.generate_json = MagicMock(return_value={
            "action": "10K_DOCUMENT_QUERY",
            "reason": "Query is about company financials from 10-K",
            "confidence": 0.9,
        })

        router = LLMRouter(llm=mock_llm)
        decision = router.route("What is Lyft's revenue?")

        assert decision.intent == QueryIntent.LOCAL_10K
        assert decision.requires_web is False
        assert "qdrant_10k_data" in decision.target_retrievers

    def test_route_web_query(self, mock_llm):
        """Test routing an internet query."""
        mock_llm.generate_json = MagicMock(return_value={
            "action": "INTERNET_QUERY",
            "reason": "Query requires current information",
            "confidence": 0.85,
        })

        router = LLMRouter(llm=mock_llm)
        decision = router.route("Latest Tesla stock price")

        assert decision.intent == QueryIntent.WEB_SEARCH
        assert decision.requires_web is True
        assert "web_search" in decision.target_retrievers

    def test_route_hybrid_query(self, mock_llm):
        """Test routing a hybrid query."""
        mock_llm.generate_json = MagicMock(return_value={
            "action": "HYBRID",
            "reason": "Query needs both local and web data",
            "confidence": 0.8,
        })

        router = LLMRouter(llm=mock_llm)
        decision = router.route("Compare Lyft's 10-K to their latest report")

        assert decision.intent == QueryIntent.HYBRID
        assert decision.requires_web is True
        assert "qdrant_10k_data" in decision.target_retrievers
        assert "web_search" in decision.target_retrievers

    def test_route_fallback_on_error(self, mock_llm):
        """Test router falls back to LOCAL_10K on error."""
        mock_llm.generate_json = MagicMock(side_effect=Exception("LLM error"))

        router = LLMRouter(llm=mock_llm)
        decision = router.route("Any query")

        assert decision.intent == QueryIntent.LOCAL_10K
        assert decision.confidence == 0.5
        assert "Default routing" in decision.reason

    def test_route_with_unknown_action(self, mock_llm):
        """Test router handles unknown action gracefully."""
        mock_llm.generate_json = MagicMock(return_value={
            "action": "UNKNOWN_ACTION",
            "reason": "Unknown",
            "confidence": 0.5,
        })

        router = LLMRouter(llm=mock_llm)
        decision = router.route("Some query")

        # Should default to LOCAL_10K
        assert decision.intent == QueryIntent.LOCAL_10K

    def test_route_preserves_search_query(self, mock_llm):
        """Test that search_query is preserved in routing decision."""
        mock_llm.generate_json = MagicMock(return_value={
            "action": "10K_DOCUMENT_QUERY",
            "reason": "Test",
            "confidence": 0.9,
        })

        router = LLMRouter(llm=mock_llm)
        query = "What is the company's revenue breakdown?"
        decision = router.route(query)

        assert decision.search_query == query

    def test_action_to_intent_mapping(self):
        """Test ACTION_TO_INTENT mapping is complete."""
        expected_mappings = {
            "OPENAI_QUERY": QueryIntent.LOCAL_OPENAI,
            "10K_DOCUMENT_QUERY": QueryIntent.LOCAL_10K,
            "INTERNET_QUERY": QueryIntent.WEB_SEARCH,
            "HYBRID": QueryIntent.HYBRID,
        }

        assert LLMRouter.ACTION_TO_INTENT == expected_mappings


class TestRouterPrompt:
    """Tests for router prompt template."""

    def test_prompt_contains_all_categories(self):
        """Test prompt includes all classification categories."""
        prompt = LLMRouter.ROUTING_PROMPT

        assert "OPENAI_QUERY" in prompt
        assert "10K_DOCUMENT_QUERY" in prompt
        assert "INTERNET_QUERY" in prompt
        assert "HYBRID" in prompt

    def test_prompt_has_json_format(self):
        """Test prompt specifies JSON response format."""
        prompt = LLMRouter.ROUTING_PROMPT

        assert '"action"' in prompt
        assert '"reason"' in prompt
        assert '"confidence"' in prompt

    def test_prompt_format_with_query(self):
        """Test prompt can be formatted with a query."""
        query = "Test query"
        formatted = LLMRouter.ROUTING_PROMPT.format(query=query)

        assert query in formatted
        assert "{query}" not in formatted
