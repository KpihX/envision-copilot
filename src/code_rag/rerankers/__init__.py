"""
Reranker Package - Modular reranking implementations.

Architecture:
- base.py: Abstract base class for all rerankers (BaseReranker)
- sentence_reranker.py: Unified implementation for sentence-transformers models
- heuristic_reranker.py: Domain-specific heuristics for Envision code

Available Reranker Types:
- cross-encoder: MS-MARCO cross-encoder models (fast, general purpose)
- bge: BAAI/bge-reranker models (robust for technical/code terms)
- mxbai: mixedbread-ai/mxbai-reranker (best speed/accuracy tradeoff)
- answerai: answerdotai optimized reranker (Q&A scenarios)
- heuristic: Domain-specific heuristics (no ML model, fast)

Usage:
    from code_rag.rerankers import get_reranker
    
    # Using type shorthand (recommended)
    reranker = get_reranker("heuristic")  # Fast, no model loading
    reranker = get_reranker("bge")        # ML-based
    
    # Using explicit model name (ML rerankers only)
    reranker = get_reranker("cross-encoder", "cross-encoder/ms-marco-MiniLM-L-12-v2")

Extensibility:
    To add a new reranker family:
    1. Create a new file (e.g., my_reranker.py)
    2. Inherit from BaseReranker and implement load() and rank()
    3. Register it in RERANKERS dict below
"""

from .base import BaseReranker
from .sentence_reranker import SentenceReranker, DEFAULT_MODELS, MODEL_FAMILY_NAMES
from .heuristic_reranker import HeuristicReranker

# Registry of available rerankers
RERANKERS = {
    # ML-based (sentence-transformers CrossEncoder)
    "cross-encoder": SentenceReranker,
    "bge": SentenceReranker,
    "mxbai": SentenceReranker,
    "answerai": SentenceReranker,
    # Heuristic-based (no ML model)
    "heuristic": HeuristicReranker,
}


def get_reranker(reranker_type: str = "cross-encoder", model_name: str = None, config: dict = None) -> BaseReranker:
    """
    Factory function to get a reranker instance.
    
    Args:
        reranker_type: Type of reranker ("cross-encoder", "bge", "mxbai", "answerai", "heuristic")
        model_name: Optional specific model name (ignored for heuristic)
        config: Optional config dict (loads from file if None)
    
    Returns:
        Initialized reranker instance
        
    Example:
        >>> reranker = get_reranker("heuristic")  # Fast, domain-specific
        >>> reranker = get_reranker("bge")        # ML-based
        >>> results = reranker.rank(query, candidates, top_k=10)
    """
    if reranker_type not in RERANKERS:
        raise ValueError(f"Unknown reranker type: {reranker_type}. Available: {list(RERANKERS.keys())}")
    
    reranker_class = RERANKERS[reranker_type]
    
    # Handle different reranker types
    if reranker_class == HeuristicReranker:
        # Load full heuristic config from config.yaml
        if config is None:
            from ..utils import ConfigLoader
            config = ConfigLoader.load_config()
        
        # Pass full heuristic_reranking section (weights, ttb_scores, pds_params, etc.)
        heuristic_config = config.get("heuristic_reranking", {})
        return HeuristicReranker(config=heuristic_config)
    elif reranker_class == SentenceReranker:
        return SentenceReranker.from_type(reranker_type, model_name)
    else:
        # Generic fallback
        if model_name is None:
            model_name = DEFAULT_MODELS.get(reranker_type)
        return reranker_class(model_name)


__all__ = [
    "BaseReranker",
    "SentenceReranker",
    "HeuristicReranker",
    "get_reranker",
    "RERANKERS",
    "DEFAULT_MODELS",
    "MODEL_FAMILY_NAMES",
]

