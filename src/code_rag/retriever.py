import logging
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer, CrossEncoder

from .utils import ConfigLoader

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.idx_config = self.config.get("indexing", {})
        self.output_config = self.config.get("output", {})
        
        self.store_dir = Path(self.output_config["store_dir"])
        self.index_path = self.store_dir / "faiss.index"
        self.meta_path = Path(self.output_config["metadata_file"])
        
        # Load Resources
        self._load_resources()

    def _load_resources(self):
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Index not built. Run 'uv run index --build'")
            
        # 1. Load FAISS
        self.index = faiss.read_index(str(self.index_path))
        
        # 2. Load Metadata
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            self.chunks = self.metadata.get("chunks", [])
            
        # 3. Load Embedder
        self.embedder = SentenceTransformer(self.idx_config.get("model_name", "all-MiniLM-L6-v2"))
        
        # 4. Load Reranker (Optional but powerful)
        # Using a lightweight cross-encoder for precision
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query: str, k: int = 20, rerank_top_k: int = 5) -> List[Dict[str, Any]]:
        # 1. Embed Query
        query_vec = self.embedder.encode([query])
        
        # 2. Dense Retrieval (FAISS)
        # Retrieve more candidates first (k*2) for reranking
        D, I = self.index.search(np.array(query_vec), k)
        
        candidates = []
        for i, idx in enumerate(I[0]):
            if idx == -1: continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(D[0][i])
            candidates.append(chunk)
            
        # 3. Reranking
        if not candidates:
            return []
            
        # Prepare pairs for cross-encoder
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])
            
        # Sort by rerank score
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        return ranked[:rerank_top_k]
