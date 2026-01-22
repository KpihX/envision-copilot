"""
Oriented Reranker - Symbol-Aware Reranking Strategy.

This reranker extends the HeuristicReranker to allow "surgical" boosting of chunks
based on specific keywords or technical terms provided by the caller (e.g. Copilot).

It privilegies chunks that contain the highest number of these keys.
"""

from typing import List, Dict, Any, Tuple
from .heuristic_reranker import HeuristicReranker

class OrientedReranker(HeuristicReranker):
    """
    Reranker that accepts explicit targeting constraints (keywords/terms)
    to boost relevant chunks.
    
    If no targets are provided, it behaves exactly like HeuristicReranker.
    """
    
    # Configuration defaults (Fallback only - real values in config.yaml)
    ORIENTED_DEFAULTS = {
        "weights": {
            "keyword_match": 1.0,   
            "term_match": 1.0,      
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize OrientedReranker.
        
        Args:
            config: Full config dictionary. Expects 'oriented_reranking' key for specific settings.
        """
        # Initialize parent (Heuristic) with ITS specific config
        # This ensures HeuristicReranker loads its custom weights from config.yaml
        cfg = config or {}
        heuristic_conf = cfg.get("heuristic_reranking", {})
        super().__init__(heuristic_conf)
        
        # Load Oriented settings
        cfg = config or {}
        oriented_cfg = cfg.get("oriented_reranking", {})
        
        self.oriented_weights = {
            **self.ORIENTED_DEFAULTS["weights"], 
            **oriented_cfg.get("weights", {})
        }
        
    def rank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 5, 
        use_contextual: bool = True,
        **kwargs
    ) -> List[Tuple[int, float]]:
        """
        Rerank with optional orientation targets.
        
        Args:
            keywords (List[str]): General keywords to boost (e.g. "def", "show")
            terms (List[str]): Specific technical terms (e.g. "StockEvol", "Catalog")
        """
        keywords = kwargs.get("keywords", []) or []
        terms = kwargs.get("terms", []) or []
        
        # 1. Get Base Heuristic Scores
        # We ask for all candidates to be ranked so we can re-sort them
        base_ranked = super().rank(
            query, 
            candidates, 
            top_k=len(candidates), 
            use_contextual=use_contextual
        )
        
        # If no orientation provided, return base results
        if not keywords and not terms:
            return base_ranked[:top_k]
            
        # 2. Map indices to scores for easy updating
        score_map = {idx: score for idx, score in base_ranked}
        
        # 3. Apply Oriented Boost
        for idx, cand in enumerate(candidates):
            # Extract content
            if isinstance(cand, str):
                text = cand
            else:
                text = cand.get("text", "") or cand.get("content", "")
                
            text_lower = text.lower()
            boost = 0.0
            
            # Boost Keywords
            for kw in keywords:
                if kw.lower() in text_lower:
                    boost += self.oriented_weights["keyword_match"]
                    
            # Boost Terms (Technical Symbols)
            for term in terms:
                if term.lower() in text_lower:
                    boost += self.oriented_weights["term_match"]
            
            # Update Score
            if idx in score_map:
                score_map[idx] += boost
            else:
                score_map[idx] = boost
                
        # 4. Re-Sort
        final_ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        
        return final_ranked[:top_k]

    def __repr__(self) -> str:
        return f"OrientedReranker(weights={self.oriented_weights}, base={super().__repr__()})"
