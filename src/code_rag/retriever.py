import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import ConfigLoader
from .rerankers import get_reranker, RERANKERS, DEFAULT_MODELS

logger = logging.getLogger(__name__)

class GraphRetriever:
    """
    RAG Retriever with modular reranking support.
    
    Supports multiple reranker types configured via config.yaml:
    - cross-encoder: sentence-transformers cross-encoder (default)
    - bge: BAAI/bge-reranker (robust for technical terms)
    - mxbai: mixedbread-ai/mxbai-reranker (best speed/accuracy)
    - answerai: answerdotai optimized reranker (for Q&A)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.idx_config = self.config.get("indexing", {})
        self.ret_config = self.config.get("retrieval", {})
        self.output_config = self.config.get("output", {})
        
        self.index = None
        self.chunks = []
        self.bi_encoder = None
        self.reranker = None
        self._use_reranker = self.ret_config.get("use_reranker", True)
        self._use_contextual = self.ret_config.get("contextual_reranking", True)
        
        if self._use_reranker:
            self._init_reranker()

    def _init_reranker(self):
        """Initialize the reranker based on config."""
        reranker_type = self.ret_config.get("reranker_type", "cross-encoder")
        reranker_name = self.ret_config.get("reranker_name")
        
        # If reranker_name looks like a full model path, use it directly
        if reranker_name and "/" in reranker_name:
            # Legacy config: full model name specified
            self.reranker = get_reranker(reranker_type, reranker_name)
        else:
            # New config: use type and default model
            self.reranker = get_reranker(reranker_type, reranker_name)

    def set_reranking(self, enabled: bool):
        """Enable or disable reranking at runtime."""
        self._use_reranker = enabled
        if enabled and self.reranker is None:
            self._init_reranker()
        elif not enabled:
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
        if self.reranker and self._use_reranker:
            reranker_type = self.ret_config.get("reranker_type", "cross-encoder")
            print(f"[RAG] ⚖️ Reranking {len(candidates)} candidates with {reranker_type}...")
            
            # Contextual: pass full chunk dicts with metadata
            # Non-contextual: pass only text strings (like original reranker)
            if self._use_contextual:
                ranked_indices = self.reranker.rank(
                    query_text, 
                    candidates, 
                    top_k=top_k,
                    use_contextual=True
                )
            else:
                # Original behavior: pass just the text strings
                cand_texts = [c["text"] for c in candidates]
                ranked_indices = self.reranker.rank(
                    query_text, 
                    cand_texts, 
                    top_k=top_k,
                    use_contextual=False
                )
            
            stats["reranked"] = True
            stats["reranker_type"] = reranker_type
            
            for original_idx_in_candidates, score in ranked_indices:
                chunk = candidates[original_idx_in_candidates]
                res = chunk.copy()
                res["score"] = float(score)
                res["rank_method"] = f"rerank_{reranker_type}"
                final_results.append(res)
        else:
            print("[RAG] ⚠️ Reranker disabled. Returning raw vector results.")
            # No Reranker, just return top K from FAISS
            for i in range(min(len(candidates), top_k)):
                chunk = candidates[i]
                res = chunk.copy()
                res["score"] = float(distances[0][i])
                res["rank_method"] = "dense_l2"
                final_results.append(res)
        
        print(f"[RAG] 🎯 Returning top {len(final_results)} results.")

        return {
            "stats": stats,
            "results": final_results
        }
