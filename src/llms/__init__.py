from typing import Dict, Any, Optional

from .base import LLM
from .mistral import MistralLLM
from .gemini import GeminiLLM
from .groq import GroqLLM
from .utils import ConfigLoader

def get_llm(llm_type: str = None, llm_model: str = None, config: Dict[str, Any] = None) -> LLM:
    """
    Factory to create LLM instance.
    Prioritizes explicit arguments over config values.
    """
    config = config or {}
    
    # Determine Provider: Arg > Config Agent > Config Root > Default
    provider = llm_type or config.get("llm_type", "mistral")
    provider = provider.lower()
    
    if "mistral" in provider:
        return MistralLLM(config, model_name=llm_model)
    elif "gemini" in provider:
        return GeminiLLM(config, model_name=llm_model)
    elif "groq" in provider:
        return GroqLLM(config, model_name=llm_model)
    else:
        raise ValueError(f"Unknown LLM type: {provider}")

__all__ = ["LLM", "MistralLLM", "GeminiLLM", "GroqLLM", "get_llm", "ConfigLoader"]
