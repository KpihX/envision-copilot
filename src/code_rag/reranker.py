from sentence_transformers import CrossEncoder
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None

    def load(self):
        if self.model is None:
            print(f"⚖️ Loading Reranker Model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)

    def rank(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Reranks a list of candidate texts against the query.
        Returns list of (original_index, score) sorted by score desc.
        """
        self.load()
        
        pairs = [[query, cand] for cand in candidates]
        scores = self.model.predict(pairs)
        
        # Enumerate to keep track of original indices
        scored = list(enumerate(scores))
        
        # Sort by score desc
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
