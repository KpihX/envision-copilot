"""
Vector Engine - Modular vector store and retrieval implementations.

Architecture:
- base.py: Abstract base class for all engines
- sentence_engine.py: SentenceTransformers/BERT embeddings
- qwen_engine.py: Qwen text-embedding-v4 via Dashscope API

Available Engine Types:
- sentence-transformers: FAISS + SentenceBERT models (local)
- qwen: FAISS + Qwen embedding API (cloud)

Usage:
    from code_rag.vector_engines import get_retriever, get_embedder

    indexer = get_embedder()
    indexer.build()

    retriever = get_retriever()
    results = retriever.query("my query")
"""

from typing import Dict, Any, Optional
from .base import BaseEmbedder, BaseRetriever
from .sentence_engine import SentenceEmbedder, SentenceRetriever
from .qwen_engine import QwenEmbedder, QwenRetriever
from ..utils import ConfigLoader

# Registry of available engines
EMBEDDERS = {
    "sentence-transformers": SentenceEmbedder,
    "qwen": QwenEmbedder,
}

RETRIEVERS = {
    "sentence-transformers": SentenceRetriever,
    "qwen": QwenRetriever,
}

def get_embedder(config: Dict[str, Any] = None) -> BaseEmbedder:
    """
    Factory to get an Embedder (Indexer) instance.
    
    Args:
        config: Optional config dict. If None, loads from config.yaml.
        
    Returns:
        Initialized Embedder instance.
    """
    if config is None:
        config = ConfigLoader.load_config()
        
    # Read engine type from config (no hardcoding)
    engine_type = config.get("indexing", {}).get("engine_type", "sentence-transformers")
    
    if engine_type not in EMBEDDERS:
        raise ValueError(f"Unknown embedder type: {engine_type}. Available: {list(EMBEDDERS.keys())}")
        
    return EMBEDDERS[engine_type](config)

def get_retriever(config_path: str = "src/code_rag/config.yaml", verbose: bool = True) -> BaseRetriever:
    """
    Factory to get a Retriever instance.
    
    Args:
        config_path: Path to config.yaml
        verbose: Enable verbose logging
        
    Returns:
        Initialized Retriever instance.
    """
    # Load config to determine type
    config = ConfigLoader.load_config(config_path)
    engine_type = config.get("indexing", {}).get("engine_type", "sentence-transformers")
    
    if engine_type not in RETRIEVERS:
        raise ValueError(f"Unknown retriever type: {engine_type}. Available: {list(RETRIEVERS.keys())}")
        
    return RETRIEVERS[engine_type](config_path, verbose)

__all__ = [
    "BaseEmbedder", 
    "BaseRetriever",
    "SentenceEmbedder", 
    "SentenceRetriever",
    "QwenEmbedder",
    "QwenRetriever",
    "get_embedder",
    "get_retriever",
    "EMBEDDERS",
    "RETRIEVERS"
]
