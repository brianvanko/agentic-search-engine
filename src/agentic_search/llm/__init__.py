"""LLM provider implementations.

Supported providers:
- OpenAI: GPT-4, GPT-4o, GPT-3.5-turbo (requires openai package)
- Anthropic: Claude 3.5, Claude 3 (requires anthropic package)
- Ollama: Local models like Llama 3, Mistral (requires running Ollama server)
"""

from agentic_search.llm.openai_llm import OpenAILLM

# Lazy imports for optional providers
def get_anthropic_llm():
    """Get AnthropicLLM class (lazy import)."""
    from agentic_search.llm.anthropic_llm import AnthropicLLM
    return AnthropicLLM

def get_ollama_llm():
    """Get OllamaLLM class (lazy import)."""
    from agentic_search.llm.ollama_llm import OllamaLLM
    return OllamaLLM

# Direct imports for type hints
try:
    from agentic_search.llm.anthropic_llm import AnthropicLLM
except ImportError:
    AnthropicLLM = None  # anthropic package not installed

try:
    from agentic_search.llm.ollama_llm import OllamaLLM
except ImportError:
    OllamaLLM = None  # requests package should always be available

__all__ = ["OpenAILLM", "AnthropicLLM", "OllamaLLM", "get_anthropic_llm", "get_ollama_llm"]
