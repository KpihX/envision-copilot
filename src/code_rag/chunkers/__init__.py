"""
Chunkers package - Transform graph nodes into embeddable chunks.

Available chunkers:
- graph: Graph-aware chunker with dependency context (default)

Usage:
    from code_rag.chunkers import get_chunker
    
    chunker = get_chunker("graph", config)
    chunks = chunker.chunk_node(node, neighborhood)
"""

from typing import Dict, Any

from .base import BaseChunker
from .graph_chunker import GraphChunker


# Registry of available chunkers
CHUNKERS = {
    "graph": GraphChunker,
}


def get_chunker(chunker_type: str = "graph", config: Dict[str, Any] = None) -> BaseChunker:
    """
    Factory function to get a chunker instance.
    
    Args:
        chunker_type: Type of chunker ("graph")
        config: Config dict (indexing section from config.yaml)
    
    Returns:
        Initialized chunker instance
        
    Example:
        >>> chunker = get_chunker("graph", config["indexing"])
        >>> chunks = chunker.chunk_node(node, neighborhood)
    """
    if chunker_type not in CHUNKERS:
        raise ValueError(f"Unknown chunker type: {chunker_type}. Available: {list(CHUNKERS.keys())}")
    
    chunker_class = CHUNKERS[chunker_type]
    return chunker_class(config=config)


__all__ = [
    "BaseChunker",
    "GraphChunker",
    "get_chunker",
    "CHUNKERS",
]
