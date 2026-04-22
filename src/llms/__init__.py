from typing import Dict, Any, Optional
from .base import LLM
# Specialized providers are imported dynamically in get_llm to reduce startup time and dependencies footprint
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
               internal_config.get("defaults", {}).get("model", "deepseek")
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
        from .mistral import MistralLLM
        return MistralLLM(merged_config, model_name=model)
    elif "gemini" in provider:
        from .gemini import GeminiLLM
        return GeminiLLM(merged_config, model_name=model)
    elif "groq" in provider:
        from .groq import GroqLLM
        return GroqLLM(merged_config, model_name=model)
    elif provider == "qwen":
        from .qwen import QwenLLM
        return QwenLLM(merged_config, model_name=model)
    elif provider == "deepseek":
        from .deepseek import DeepSeekLLM
        return DeepSeekLLM(merged_config, model_name=model)
    elif provider in ("ollama", "llama", "codellama"):
        from .ollama import OllamaLLM
        return OllamaLLM(merged_config, model_name=model)
    else:
        raise ValueError(f"Unknown LLM type: {provider}. Supported: mistral, gemini, groq, qwen, deepseek, ollama/llama")

__all__ = ["LLM", "get_llm", "ConfigLoader"]
