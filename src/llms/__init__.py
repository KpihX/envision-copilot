from typing import Dict, Any, Optional

from .base import LLM
from .mistral import MistralLLM
from .gemini import GeminiLLM
from .groq import GroqLLM
from .ollama import OllamaLLM
from .qwen import QwenLLM
from .utils import ConfigLoader

def get_llm(llm_type: str = None, llm_model: str = None, config: Dict[str, Any] = None) -> LLM:
    """
    Factory to create LLM instance.
    
    Priority: explicit args > external config (copilot) > internal config (llms/config.yaml)
    """
    # 1. Load internal config as base/fallback
    internal_config = ConfigLoader.load_config()
    external_config = config or {}
    
    # 2. Determine Provider: Arg > External agent.llm_type > Internal defaults.model > "mistral"
    provider = llm_type or \
               external_config.get("agent", {}).get("llm_type") or \
               external_config.get("llm_type") or \
               internal_config.get("defaults", {}).get("model", "mistral")
    provider = provider.lower()
    
    # 3. Determine Model: Arg > External agent.llm_model > Internal providers.X.model_name
    model = llm_model or \
            external_config.get("agent", {}).get("llm_model") or \
            external_config.get("llm_model") or \
            internal_config.get("providers", {}).get(provider, {}).get("model_name")
    
    # 4. Build merged config for provider (internal as base, external overrides)
    merged_config = {
        "defaults": internal_config.get("defaults", {}),
        "providers": internal_config.get("providers", {}),
    }
    # Override with external config if present
    if external_config.get("agent", {}).get("temperature") is not None:
        merged_config["defaults"]["temperature"] = external_config["agent"]["temperature"]
    if external_config.get("agent", {}).get("max_tokens") is not None:
        merged_config["defaults"]["max_tokens"] = external_config["agent"]["max_tokens"]
    
    if "mistral" in provider and "ollama" not in provider:
        return MistralLLM(merged_config, model_name=model)
    elif "gemini" in provider:
        return GeminiLLM(merged_config, model_name=model)
    elif "groq" in provider:
        return GroqLLM(merged_config, model_name=model)
    elif provider == "qwen":
        return QwenLLM(merged_config, model_name=model)
    elif provider in ("ollama", "llama", "codellama", "deepseek"):
        return OllamaLLM(merged_config, model_name=model)
    else:
        raise ValueError(f"Unknown LLM type: {provider}. Supported: mistral, gemini, groq, qwen, ollama/llama")

__all__ = ["LLM", "MistralLLM", "GeminiLLM", "GroqLLM", "OllamaLLM", "QwenLLM", "get_llm", "ConfigLoader"]
