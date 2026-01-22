"""
Indexers package - Build vector indices from graph data.

Available indexers:
- graph: Graph-aware indexer with FAISS (default)

Usage:
    from code_rag.indexers import get_indexer
    
    indexer = get_indexer("graph", config)
    indexer.build()
"""

from typing import Dict, Any

from .base import BaseIndexer
from .graph_indexer import GraphIndexer


# Registry of available indexers
INDEXERS = {
    "graph": GraphIndexer,
}


def get_indexer(indexer_type: str = "graph", config: Dict[str, Any] = None) -> BaseIndexer:
    """
    Factory function to get an indexer instance.
    
    Args:
        indexer_type: Type of indexer ("graph")
        config: Full config dict from config.yaml
    
    Returns:
        Initialized indexer instance
        
    Example:
        >>> indexer = get_indexer("graph")
        >>> indexer.build()
    """
    if indexer_type not in INDEXERS:
        raise ValueError(f"Unknown indexer type: {indexer_type}. Available: {list(INDEXERS.keys())}")
    
    indexer_class = INDEXERS[indexer_type]
    return indexer_class(config=config)


__all__ = [
    "BaseIndexer",
    "GraphIndexer",
    "get_indexer",
    "INDEXERS",
]
