"""
Base class for chunkers.

Chunkers transform graph nodes into vector-ready chunks for embedding.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseChunker(ABC):
    """
    Abstract base class for all chunkers.
    
    Chunkers are responsible for:
    1. Breaking down graph nodes into smaller pieces for embedding
    2. Adding contextual information (graph awareness)
    3. Managing chunk overlap and size constraints
    """
    
    def __init__(self, chunker_type: str):
        """
        Initialize base chunker.
        
        Args:
            chunker_type: Identifier for this chunker type
        """
        self.chunker_type = chunker_type
    
    @abstractmethod
    def chunk_node(self, node: Dict[str, Any], neighborhood: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Produce chunks for a single graph node.
        
        Args:
            node: The graph node to chunk (dict with id, type, content, etc.)
            neighborhood: Dict with 'incoming' and 'outgoing' edge lists
            
        Returns:
            List of chunk dicts, each containing:
            - id: Unique chunk identifier
            - source_id: ID of the source node
            - source: Display path for the chunk
            - text: Full text for embedding (with context)
            - content: Raw content without context
            - context: Contextual header/metadata
            - lines: Line range (e.g., "1-50")
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.chunker_type})"
