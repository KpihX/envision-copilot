import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import ConfigLoader
from .reranker import Reranker

logger = logging.getLogger(__name__)

class GraphRetriever:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.idx_config = self.config.get("indexing", {})
        self.ret_config = self.config.get("retrieval", {})
        self.output_config = self.config.get("output", {})
        
        self.index = None
        self.chunks = []
        self.bi_encoder = None
        self.reranker = None
        self._reranker_instance = None
        
        if self.ret_config.get("use_reranker", True):
            self._reranker_instance = Reranker(self.ret_config.get("reranker_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
            self.reranker = self._reranker_instance

    def set_reranking(self, enabled: bool):
        """Enable or disable reranking at runtime."""
        if enabled:
            self._reranker_instance = Reranker(self.ret_config.get("reranker_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
            self.reranker = self._reranker_instance
        else:
            self.reranker = None

    def _ensure_loaded(self):
        if self.index is not None:
            return

        index_file = Path(self.output_config.get("index_file", "data/vector_store/faiss.index"))
        meta_file = Path(self.output_config.get("metadata_file", "data/vector_store/metadata.json"))

        # Try project root path if relative not found (heuristic)
        if not index_file.exists():
             project_root = Path(__file__).parent.parent.parent
             index_file = project_root / self.output_config.get("index_file", "data/vector_store/faiss.index")
             
        if not meta_file.exists():
             project_root = Path(__file__).parent.parent.parent
             meta_file = project_root / self.output_config.get("metadata_file", "data/vector_store/metadata.json")

        if not index_file.exists():
             raise FileNotFoundError(f"Index not found at {index_file}. Run build first.")
             
        # Load FAISS
        self.index = faiss.read_index(str(index_file))
        
        # Load Metadata
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.chunks = data.get("chunks", [])

        # Load Bi-Encoder
        self.bi_encoder = SentenceTransformer(self.idx_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        
    def query(self, query_text: str, top_k: int = 5, recall_k: int = None) -> Dict[str, Any]:
        """
        Full Retrieval Pipeline: Recall -> Rerank.
        
        Args:
            query_text: The query string
            top_k: Number of final results to return
            recall_k: Number of candidates to retrieve from FAISS (default: from config)
        """
        self._ensure_loaded()
        
        # 1. Recall (Broad)
        if recall_k is None:
            recall_k = self.ret_config.get("top_k_recall", 50)
        print(f"[RAG] 🔍 Retrieving top {recall_k} candidates from vector index...")
        
        query_vec = self.bi_encoder.encode([query_text]).astype('float32')
        # FAISS search
        distances, indices = self.index.search(query_vec, recall_k)
        
        candidates = []
        candidate_indices = []
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            chunk = self.chunks[idx]
            candidates.append(chunk)
            candidate_indices.append(idx)

        stats = {
            "total_candidates": len(candidates),
            "reranked": False
        }
        
        print(f"[RAG] ✅ Found {len(candidates)} candidates.")

        final_results = []
        
        # 2. Rerank (Precision)
        if self.reranker:
            print(f"[RAG] ⚖️ Reranking {len(candidates)} candidates...")
            # Prepare texts: Context + Content
            cand_texts = [c["text"] for c in candidates]
            
            ranked_indices = self.reranker.rank(query_text, cand_texts, top_k=top_k)
            
            stats["reranked"] = True
            
            for original_idx_in_candidates, score in ranked_indices:
                chunk = candidates[original_idx_in_candidates]
                res = chunk.copy()
                res["score"] = float(score)
                res["rank_method"] = "cross_encoder"
                final_results.append(res)
        else:
            print("[RAG] ⚠️ Reranker disabled. Returning raw vector results.")
            # No Reranker, just return top K from FAISS
            for i in range(min(len(candidates), top_k)):
                chunk = candidates[i]
                res = chunk.copy()
                res["score"] = float(distances[0][i]) # L2 distance (lower is better usually, but check index type)
                # IndexFlatL2 returns squared Euclidean distance. 
                # Converting to similarity/score might be needed for consistency, but raw distance tells relative rank.
                res["rank_method"] = "dense_l2"
                final_results.append(res)
        
        print(f"[RAG] 🎯 Returning top {len(final_results)} results.")

        return {
            "stats": stats,
            "results": final_results
        }
