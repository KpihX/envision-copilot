"""
Base Vector Engine - Abstract base classes for vector search components.

Architecture:
- BaseEmbedder: Abstract base class for indexers (builds vector store)
- BaseRetriever: Abstract base class for retrievers (queries vector store)

Implementations:
- sentence_engine.py: Standard implementation using FAISS + Sentence Transformers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseEmbedder(ABC):
    """
    Abstract base class for embedders (indexers).
    Responsible for:
    1. Loading data
    2. Chunking
    3. Computing embeddings
    4. Building vector index
    """
    
    def __init__(self, engine_type: str, config: Dict[str, Any] = None):
        self.engine_type = engine_type
        self.config = config or {}
    
    @abstractmethod
    def build(self) -> None:
        """Build the vector index."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.engine_type})"


class BaseRetriever(ABC):
    """
    Abstract base class for retrievers.
    Responsible for querying the vector index.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        pass

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant chunks."""
        pass
