"""OpenAI LLM provider implementation."""

import json
import logging
import re
from typing import Dict, Any

from openai import OpenAI, OpenAIError

from agentic_search.core.interfaces import BaseLLM
from agentic_search.core.models import LLMResponse
from agentic_search.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider.

    Supports all OpenAI chat models including GPT-4, GPT-4o, GPT-3.5-turbo.

    Example:
        llm = OpenAILLM(api_key="sk-...", model="gpt-4o-mini")
        response = llm.generate("What is the capital of France?")
        print(response.content)  # "The capital of France is Paris."
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        **kwargs,
    ):
        """Initialize OpenAI LLM provider.

        Args:
            api_key: OpenAI API key.
            model: Model identifier (default: gpt-4o-mini).
            **kwargs: Additional client options.
        """
        if not api_key:
            raise LLMError("OpenAI API key is required")

        self._client = OpenAI(api_key=api_key, **kwargs)
        self._model = model
        logger.info(f"Initialized OpenAI LLM with model: {model}")

    @property
    def name(self) -> str:
        return "openai"

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
        """Generate a response using OpenAI's chat API.

        Args:
            prompt: The input prompt.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens in response.
            **kwargs: Additional API parameters.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMError: If API call fails.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            choice = response.choices[0]
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else {}

            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason or "stop",
            )

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMError(f"OpenAI API call failed: {e}") from e

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a JSON response using OpenAI's chat API.

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
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            choice = response.choices[0]
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else {}

            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason or "stop",
            )

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMError(f"OpenAI API call failed: {e}") from e
