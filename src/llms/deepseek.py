"""
DeepSeek LLM Provider (OpenAI-compatible API)
"""
import os
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from .base import LLM


class DeepSeekLLM(LLM):
    """LLM provider for DeepSeek models (OpenAI-compatible)."""
    
    def __init__(self, config: Dict[str, Any] = None, model_name: str = None):
        config = config or {}
        super().__init__(config)
        
        load_dotenv(override=True)
        
        # Model name priority: Arg > Config agent.llm_model > Config providers.deepseek > Default
        self._model_name = model_name or \
                           config.get("agent", {}).get("llm_model") or \
                           config.get("llm_model") or \
                           config.get("providers", {}).get("deepseek", {}).get("model_name", "deepseek-chat")
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")
        
        base_url = config.get("providers", {}).get("deepseek", {}).get("api_base", "https://api.deepseek.com/v1")
        
        # Temperature: agent.temperature > defaults.temperature > 0.0
        self.temperature = config.get("agent", {}).get("temperature") or \
                           config.get("defaults", {}).get("temperature", 0.0)
        self.max_tokens = config.get("agent", {}).get("max_tokens") or \
                          config.get("defaults", {}).get("max_tokens", 4096)
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        completion = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content

    @property
    def model_name(self) -> str:
        return self._model_name
