"""
Ollama LLM Provider (Local models: Qwen, Llama, etc.)
"""
import os
from typing import Optional, Dict, Any
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from .base import LLM


class OllamaLLM(LLM):
    """LLM provider for local Ollama models (Qwen, Llama, Mistral, etc.)."""
    
    def __init__(self, config: Dict[str, Any] = None, model_name: str = None):
        config = config or {}
        super().__init__(config)
        
        load_dotenv(override=True)
        
        # Model name priority: Arg > Config agent.llm_model > Config providers.ollama > Default
        self._model_name = model_name or \
                           config.get("agent", {}).get("llm_model") or \
                           config.get("llm_model") or \
                           config.get("providers", {}).get("ollama", {}).get("model_name", "qwen2.5-coder:latest")
        
        # Ollama base URL (default: local)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Get temperature/max_tokens from config
        temperature = config.get("agent", {}).get("temperature") or \
                      config.get("defaults", {}).get("temperature", 0.0)
        max_tokens = config.get("agent", {}).get("max_tokens") or \
                     config.get("defaults", {}).get("max_tokens", 4096)
        
        self.client = ChatOllama(
            model=self._model_name,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,  # Ollama uses num_predict instead of max_tokens
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))
        
        response = self.client.invoke(messages)
        return response.content

    @property
    def model_name(self) -> str:
        return self._model_name
