"""
Qwen LLM Provider (via Dashscope/Alibaba Cloud OpenAI-compatible API)
"""
import os
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from .base import LLM


class QwenLLM(LLM):
    """LLM provider for Qwen models via Dashscope API (OpenAI-compatible)."""
    
    # Region base URLs
    REGIONS = {
        "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        "china": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    
    def __init__(self, config: Dict[str, Any] = None, model_name: str = None):
        config = config or {}
        super().__init__(config)
        
        load_dotenv(override=True)
        
        # Model name priority: Arg > Config agent.llm_model > Config providers.qwen > Default
        self._model_name = model_name or \
                           config.get("agent", {}).get("llm_model") or \
                           config.get("llm_model") or \
                           config.get("providers", {}).get("qwen", {}).get("model_name", "qwen-plus")
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables.")
        
        # Region: default to Singapore (international)
        region = config.get("providers", {}).get("qwen", {}).get("region", "singapore")
        base_url = self.REGIONS.get(region, self.REGIONS["singapore"])
        
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
