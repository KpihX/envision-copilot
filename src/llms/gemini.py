import os
from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from .interface import LLM

class GeminiLLM(LLM):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        load_dotenv(override=True)
        
        provider_config = config.get("providers", {}).get("gemini", {})
        self._model_name = provider_config.get("model_name", "gemini-1.5-pro-latest")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        self.client = ChatGoogleGenerativeAI(
            model=self._model_name,
            google_api_key=api_key,
            temperature=config.get("defaults", {}).get("temperature", 0.0),
            max_output_tokens=config.get("defaults", {}).get("max_tokens", 1024)
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
