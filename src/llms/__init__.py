from typing import Dict, Any, Optional

from .interface import LLM
from .mistral import MistralLLM
from .gemini import GeminiLLM
from .groq import GroqLLM
from .utils import ConfigLoader

def get_llm(provider: str = None, config_path: str = "config.yaml") -> LLM:
    """Factory to create LLM instance based on provider name."""
    config = ConfigLoader.load_config(config_path)
    
    # Determined default provider if None
    if not provider:
         provider = config.get("defaults", {}).get("model", "mistral")
    
    # Normalize provider string (e.g. "Mistral" -> "mistral")
    provider = provider.lower()

    if "mistral" in provider:
        return MistralLLM(config)
    elif "gemini" in provider:
        return GeminiLLM(config)
    elif "groq" in provider:
        return GroqLLM(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

__all__ = ["LLM", "MistralLLM", "GeminiLLM", "GroqLLM", "get_llm", "ConfigLoader"]
