from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

class LLM(ABC):
    """Abstract Base Class for LLM Providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates a text response for the given prompt."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
