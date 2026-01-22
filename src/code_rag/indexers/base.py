"""
Base class for indexers.

Indexers build vector indices from graph data for semantic search.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseIndexer(ABC):
    """
    Abstract base class for all indexers.
    
    Indexers are responsible for:
    1. Loading graph data from network files
    2. Chunking nodes using a chunker
    3. Computing embeddings using an embedding model
    4. Building and saving a vector index (FAISS)
    """
    
    def __init__(self, indexer_type: str, config: Dict[str, Any] = None):
        """
        Initialize base indexer.
        
        Args:
            indexer_type: Identifier for this indexer type
            config: Full config dict from config.yaml
        """
        self.indexer_type = indexer_type
        self.config = config or {}
    
    @abstractmethod
    def build(self) -> None:
        """
        Build the vector index.
        
        This method should:
        1. Load graph data
        2. Generate chunks
        3. Compute embeddings
        4. Build and save the FAISS index
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.indexer_type})"
