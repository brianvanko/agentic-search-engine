"""Anthropic LLM provider implementation."""

import json
import logging
import re
from typing import Dict, Any

from agentic_search.core.interfaces import BaseLLM
from agentic_search.core.models import LLMResponse
from agentic_search.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    """Anthropic LLM provider.

    Supports Claude models including Claude 3.5 Sonnet, Claude 3 Opus, etc.

    Example:
        llm = AnthropicLLM(api_key="sk-ant-...", model="claude-3-5-sonnet-20241022")
        response = llm.generate("What is the capital of France?")
        print(response.content)  # "The capital of France is Paris."

    Note:
        Requires the anthropic package: pip install anthropic
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        **kwargs,
    ):
        """Initialize Anthropic LLM provider.

        Args:
            api_key: Anthropic API key.
            model: Model identifier (default: claude-3-5-sonnet-20241022).
            **kwargs: Additional client options.
        """
        if not api_key:
            raise LLMError("Anthropic API key is required")

        try:
            import anthropic
        except ImportError:
            raise LLMError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        self._client = anthropic.Anthropic(api_key=api_key, **kwargs)
        self._model = model
        logger.info(f"Initialized Anthropic LLM with model: {model}")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response using Anthropic's messages API.

        Args:
            prompt: The input prompt.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in response.
            **kwargs: Additional API parameters.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMError: If API call fails.
        """
        try:
            import anthropic

            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **kwargs,
            )

            content = response.content[0].text if response.content else ""
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=response.stop_reason or "end_turn",
            )

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise LLMError(f"Anthropic API call failed: {e}") from e

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a JSON response using Anthropic's messages API.

        Args:
            prompt: The input prompt expecting JSON output.
            temperature: Sampling temperature (lower for more deterministic).
            **kwargs: Additional API parameters.

        Returns:
            Parsed JSON dictionary.

        Raises:
            LLMError: If API call fails or JSON parsing fails.
        """
        try:
            response = self.generate(
                prompt=prompt,
                temperature=temperature,
                **kwargs,
            )

            content = response.content

            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # If no JSON found, try parsing the entire content
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            raise LLMError(f"Failed to parse JSON response: {e}") from e

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response with a system prompt.

        Args:
            system_prompt: The system instructions.
            user_prompt: The user's input.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional API parameters.

        Returns:
            LLMResponse with generated content.
        """
        try:
            import anthropic

            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                **kwargs,
            )

            content = response.content[0].text if response.content else ""
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=response.stop_reason or "end_turn",
            )

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise LLMError(f"Anthropic API call failed: {e}") from e
