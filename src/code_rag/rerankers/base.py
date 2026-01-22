"""
Base Reranker - Abstract base class for all rerankers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """
    Abstract base class for rerankers.
    
    All rerankers must implement:
    - load(): Load the model into memory
    - rank(): Rerank candidates against a query
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load the reranker model into memory."""
        pass
    
    @abstractmethod
    def _predict(self, pairs: List[List[str]]) -> List[float]:
        """
        Internal prediction method.
        
        Args:
            pairs: List of [query, candidate] pairs
            
        Returns:
            List of scores for each pair
        """
        pass
    
    def rank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 5,
        use_contextual: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Reranks a list of candidate chunks against the query.
        
        Uses "Contextual Reranking" by default: enriches the text sent to
        the reranker with metadata (path, source) for better context.
        
        Args:
            query: The search query
            candidates: List of chunk dicts with 'text', 'source', etc.
            top_k: Number of top results to return
            use_contextual: If True, enrich candidates with metadata
            
        Returns:
            List of (original_index, score) sorted by score desc
        """
        self.load()
        
        # Build pairs with optional contextual enrichment
        pairs = []
        for cand in candidates:
            if isinstance(cand, str):
                # Already a string (original behavior)
                enriched = cand
            elif use_contextual:
                # Dict with contextual enrichment
                enriched = self._enrich_candidate(cand)
            else:
                # Dict but no contextual - just get text
                enriched = cand.get("text", "")
            pairs.append([query, enriched])
        
        # Get scores
        scores = self._predict(pairs)
        
        # Enumerate to keep track of original indices
        scored = list(enumerate(scores))
        
        # Sort by score desc
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
    
    def _enrich_candidate(self, candidate: Dict[str, Any]) -> str:
        """
        Enrich candidate text with metadata for contextual reranking.
        
        The Cross-Encoder is sensitive to formatting, so we structure
        the input to highlight key information.
        """
        parts = []
        
        # Add source path if available
        source = candidate.get("source")
        if source:
            parts.append(f"FILE: {source}")
        
        # Add script name if available
        name = candidate.get("name")
        if name:
            parts.append(f"SCRIPT: {name}")
        
        # Add context header if available (contains imports/reads)
        context = candidate.get("context")
        if context:
            parts.append(f"CONTEXT: {context}")
        
        # Add main content
        text = candidate.get("content", "")
        parts.append(f"CONTENT:\n{text}")
        
        return "\n".join(parts)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}', loaded={self._loaded})"
