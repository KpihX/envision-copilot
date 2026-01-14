import os
from typing import Optional, Dict, Any
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from .interface import LLM

class GroqLLM(LLM):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        load_dotenv(override=True)
        
        provider_config = config.get("providers", {}).get("groq", {})
        self._model_name = provider_config.get("model_name", "llama3-70b-8192")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.client = ChatGroq(
            model_name=self._model_name,
            groq_api_key=api_key,
            temperature=config.get("defaults", {}).get("temperature", 0.0),
            max_tokens=config.get("defaults", {}).get("max_tokens", 1024)
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
