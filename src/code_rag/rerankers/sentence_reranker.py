"""
Sentence-Transformers Reranker - All models using sentence-transformers CrossEncoder.

Supports multiple model families:
- cross-encoder: cross-encoder/ms-marco-* models (general purpose)
- bge: BAAI/bge-reranker-* models (robust for technical terms)
- mxbai: mixedbread-ai/mxbai-rerank-* models (best speed/accuracy)
- answerai: answerdotai/* models (Q&A optimized)

All these models use the same sentence-transformers CrossEncoder interface,
so they are unified in this single implementation.
"""

import sys
from typing import List
from sentence_transformers import CrossEncoder
from .base import BaseReranker


# Default models for each reranker type
# Note: Model names must match exactly what's on HuggingFace
DEFAULT_MODELS = {
    "cross-encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
    "mxbai": "mixedbread-ai/mxbai-rerank-base-v1",  # Note: "rerank" not "reranker"
    "answerai": "tomaarsen/reranker-modernbert-base-msmarco-bce",  # ModernBERT-based
}

# Human-readable names for logging
MODEL_FAMILY_NAMES = {
    "cross-encoder": "CrossEncoder",
    "bge": "BGE Reranker",
    "mxbai": "Mxbai Reranker",
    "answerai": "AnswerAI Reranker",
}


class RerankerLoadError(Exception):
    """Raised when a reranker model fails to load."""
    pass


class SentenceReranker(BaseReranker):
    """
    Unified reranker for all sentence-transformers CrossEncoder models.
    
    This class handles all models that use the CrossEncoder interface:
    - MS-MARCO cross-encoders (general purpose, fast)
    - BGE rerankers (robust for technical/code terms)
    - Mxbai rerankers (best speed/accuracy tradeoff)
    - AnswerAI rerankers (optimized for Q&A)
    
    Usage:
        # Using type shorthand
        reranker = SentenceReranker.from_type("bge")
        
        # Using explicit model name
        reranker = SentenceReranker("BAAI/bge-reranker-large")
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__(model_name)
        self._family = self._detect_family(model_name)
    
    @classmethod
    def from_type(cls, reranker_type: str, model_name: str = None) -> "SentenceReranker":
        """
        Create a reranker from a type shorthand.
        
        Args:
            reranker_type: One of "cross-encoder", "bge", "mxbai", "answerai"
            model_name: Optional override of the default model for the type
            
        Returns:
            Configured SentenceReranker instance
        """
        if reranker_type not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown reranker type: {reranker_type}. "
                f"Available: {list(DEFAULT_MODELS.keys())}"
            )
        
        if model_name is None:
            model_name = DEFAULT_MODELS[reranker_type]
        
        instance = cls(model_name)
        instance._family = reranker_type
        return instance
    
    def _detect_family(self, model_name: str) -> str:
        """Auto-detect model family from model name."""
        model_lower = model_name.lower()
        if "bge-reranker" in model_lower:
            return "bge"
        elif "mxbai" in model_lower:
            return "mxbai"
        elif "answerdotai" in model_lower or "reranker-base-multilingual" in model_lower:
            return "answerai"
        else:
            return "cross-encoder"
    
    def load(self) -> None:
        """
        Load the CrossEncoder model.
        
        Raises:
            RerankerLoadError: If the model cannot be loaded (invalid name, auth required, etc.)
        """
        if not self._loaded:
            family_name = MODEL_FAMILY_NAMES.get(self._family, "Reranker")
            print(f"⚖️ Loading {family_name}: {self.model_name}")
            
            try:
                self.model = CrossEncoder(self.model_name)
                self._loaded = True
            except OSError as e:
                error_msg = str(e)
                
                # Provide helpful error messages
                if "404" in error_msg or "not a valid model identifier" in error_msg:
                    print(f"\n❌ [ERROR] Model not found: {self.model_name}")
                    print(f"   The model name may be incorrect or the repository doesn't exist.")
                    print(f"   Check the model name on: https://huggingface.co/models")
                    print(f"\n   Available reranker types and their default models:")
                    for rtype, model in DEFAULT_MODELS.items():
                        print(f"     - {rtype}: {model}")
                    
                elif "401" in error_msg or "Unauthorized" in error_msg:
                    print(f"\n❌ [ERROR] Authentication required for: {self.model_name}")
                    print(f"   This model requires HuggingFace authentication.")
                    print(f"   Run: huggingface-cli login")
                    print(f"   Then accept the model's license at: https://huggingface.co/{self.model_name}")
                    
                elif "gated" in error_msg.lower():
                    print(f"\n❌ [ERROR] Gated model: {self.model_name}")
                    print(f"   This model requires you to accept its license.")
                    print(f"   1. Login: huggingface-cli login")
                    print(f"   2. Accept license at: https://huggingface.co/{self.model_name}")
                
                else:
                    print(f"\n❌ [ERROR] Failed to load model: {self.model_name}")
                    print(f"   Error: {error_msg[:200]}")
                
                raise RerankerLoadError(
                    f"Failed to load reranker model '{self.model_name}'. "
                    f"Try using 'bge' or 'cross-encoder' which don't require authentication."
                ) from e
    
    def _predict(self, pairs: List[List[str]]) -> List[float]:
        """Predict relevance scores for query-candidate pairs."""
        scores = self.model.predict(pairs)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
