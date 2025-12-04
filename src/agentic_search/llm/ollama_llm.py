"""Ollama LLM provider implementation for local models."""

import json
import logging
import re
from typing import Dict, Any, Optional

import requests

from agentic_search.core.interfaces import BaseLLM
from agentic_search.core.models import LLMResponse
from agentic_search.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama LLM provider for local models.

    Connects to a locally running Ollama server to use models like
    Llama 3, Mistral, Mixtral, CodeLlama, etc.

    Example:
        # Using default Ollama server at localhost:11434
        llm = OllamaLLM(model="llama3.2")
        response = llm.generate("What is the capital of France?")
        print(response.content)

        # Using a custom server
        llm = OllamaLLM(model="mistral", base_url="http://192.168.1.100:11434")

    Note:
        Requires Ollama to be installed and running locally.
        Install from: https://ollama.ai
        Pull models with: ollama pull llama3.2
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        **kwargs,
    ):
        """Initialize Ollama LLM provider.

        Args:
            model: Model name (e.g., "llama3.2", "mistral", "mixtral").
            base_url: Ollama server URL (default: http://localhost:11434).
            timeout: Request timeout in seconds.
            **kwargs: Additional options.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._options = kwargs

        # Verify connection
        self._verify_connection()
        logger.info(f"Initialized Ollama LLM with model: {model} at {base_url}")

    def _verify_connection(self) -> None:
        """Verify Ollama server is accessible."""
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise LLMError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise LLMError(f"Timeout connecting to Ollama at {self._base_url}")

    @property
    def name(self) -> str:
        return "ollama"

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
        """Generate a response using Ollama's generate API.

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
            payload = {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **self._options,
                    **kwargs.get("options", {}),
                },
            }

            response = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Extract usage info if available
            usage = {}
            if "prompt_eval_count" in data:
                usage["prompt_tokens"] = data.get("prompt_eval_count", 0)
            if "eval_count" in data:
                usage["completion_tokens"] = data.get("eval_count", 0)
            if usage:
                usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

            return LLMResponse(
                content=data.get("response", ""),
                model=data.get("model", self._model),
                usage=usage,
                finish_reason="stop" if data.get("done", False) else "length",
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise LLMError(f"Ollama API call failed: {e}") from e

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a JSON response using Ollama.

        Args:
            prompt: The input prompt expecting JSON output.
            temperature: Sampling temperature (lower for more deterministic).
            **kwargs: Additional API parameters.

        Returns:
            Parsed JSON dictionary.

        Raises:
            LLMError: If API call fails or JSON parsing fails.
        """
        # Add JSON format hint to options
        kwargs.setdefault("options", {})
        kwargs["options"]["format"] = "json"

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
        """Generate a response with a system prompt using chat API.

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
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **self._options,
                    **kwargs.get("options", {}),
                },
            }

            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Extract content from message
            content = ""
            if "message" in data and "content" in data["message"]:
                content = data["message"]["content"]

            # Extract usage info
            usage = {}
            if "prompt_eval_count" in data:
                usage["prompt_tokens"] = data.get("prompt_eval_count", 0)
            if "eval_count" in data:
                usage["completion_tokens"] = data.get("eval_count", 0)
            if usage:
                usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

            return LLMResponse(
                content=content,
                model=data.get("model", self._model),
                usage=usage,
                finish_reason="stop" if data.get("done", False) else "length",
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise LLMError(f"Ollama API call failed: {e}") from e

    def list_models(self) -> list:
        """List available models on the Ollama server.

        Returns:
            List of available model names.
        """
        try:
            response = requests.get(
                f"{self._base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
