"""
Sentence Transformers Engine for RAG.

Contains:
- SetenceEmbedder: Builds FAISS index from graph using Sentence Transformers.
- SentenceRetriever: Retrieves and reranks chunks using FAISS + Sentence Transformers.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.progress import track

from .base import BaseEmbedder, BaseRetriever
from ..chunkers import get_chunker
from ..utils import ConfigLoader
from ..rerankers import get_reranker

logger = logging.getLogger(__name__)


# ============================================================================
# EMBEDDER (Indexer)
# ============================================================================

class SentenceEmbedder(BaseEmbedder):
    """
    Builds FAISS indices from graph nodes using Sentence Transformers.
    
    Features:
    - Modular chunking
    - Neighborhood context support
    - Sentence-BERT embeddings
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = ConfigLoader.load_config()
        
        super().__init__("sentence-transformers", config)
        
        self.idx_config = config.get("indexing", {})
        self.output_config = config.get("output", {})
        self.input_config = config.get("input", {})
        
        # Load embedding model
        model_name = self.idx_config.get("engine_name", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"📦 Loading Embedding Model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize chunker
        chunker_type = self.idx_config.get("chunker_type", "graph")
        self.chunker = get_chunker(chunker_type, self.idx_config)
    
    def build(self) -> None:
        """Build the vector index from graph data."""
        # 1. Load Network
        net_path = self._resolve_network_path()
        if net_path is None:
            return
        
        print(f"📂 Loading Graph from {net_path}...")
        with open(net_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data.get("nodes", {})
        edges = data.get("edges", [])
        
        # 2. Build Neighborhood Map
        print("🔗 Building Neighborhood Map...")
        neighborhoods = self._build_neighborhoods(nodes, edges)
        
        # 3. Generate Chunks
        print("🔄 Generating Graph-Aware Chunks...")
        all_chunks = self._generate_chunks(nodes, neighborhoods)
        
        print(f"📊 Generated {len(all_chunks)} chunks.")
        
        if not all_chunks:
            print("⚠️ No chunks generated. Exiting.")
            return
        
        # 4. Compute Embeddings
        # 4. Compute Embeddings
        print("🧠 Computing Embeddings (Normalized)...")
        texts = [c["text"] for c in all_chunks]
        # Normalize for Cosine Similarity
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
        
        # 5. Build FAISS Index (Inner Product)
        dimension = embeddings.shape[1]
        print(f"🗂️ Building FAISS Index (Dim: {dimension}, Metric: InnerProduct/Cosine)...")
        index = faiss.IndexFlatIP(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        # 6. Save
        self._save_index(index, all_chunks)
    
    def _resolve_network_path(self) -> Path:
        """Resolve the network file path."""
        # Default now points to datas/
        default_path = "datas/network/network.json"
        net_path = Path(self.input_config.get("network_file", default_path))
        
        if not net_path.exists():
            project_root = Path(__file__).parent.parent.parent.parent
            net_path = project_root / self.input_config.get("network_file", default_path)
        
        if not net_path.exists():
            print(f"❌ Network file not found at {net_path}. Run 'uv run network --build' first.")
            return None
        
        return net_path
    
    def _build_neighborhoods(self, nodes: Dict, edges: list) -> Dict[str, Dict]:
        """Build neighborhood map for each node."""
        neighborhoods = {nid: {"incoming": [], "outgoing": []} for nid in nodes}
        
        for edge in edges:
            src, tgt, etype = edge["source"], edge["target"], edge["type"]
            
            if src in neighborhoods:
                tgt_node = nodes.get(tgt)
                tgt_label = tgt_node.get("name") if tgt_node else tgt
                neighborhoods[src]["outgoing"].append({
                    "target_id": tgt, 
                    "target_label": tgt_label, 
                    "edge_type": etype
                })
            
            if tgt in neighborhoods:
                src_node = nodes.get(src)
                src_label = src_node.get("name") if src_node else src
                neighborhoods[tgt]["incoming"].append({
                    "source_id": src, 
                    "source_label": src_label, 
                    "edge_type": etype
                })
        
        return neighborhoods
    
    def _generate_chunks(self, nodes: Dict, neighborhoods: Dict) -> list:
        all_chunks = []
        script_nodes = [n for nid, n in nodes.items() if n["type"] == "script"]
        
        for node in track(script_nodes, description="Chunking..."):
            nid = node["id"]
            node_chunks = self.chunker.chunk_node(node, neighborhoods[nid])
            for c in node_chunks:
                c["source"] = node.get("path", nid)
            all_chunks.extend(node_chunks)
        return all_chunks
    
    def _save_index(self, index, chunks: list) -> None:
        index_file = Path(self.output_config.get("index_file", "datas/code_rag/index/faiss.index"))
        meta_file = Path(self.output_config.get("metadata_file", "datas/code_rag/index/metadata.json"))
        
        index_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(index_file))
        
        meta = {
            "generated_at": datetime.now().isoformat(),
            "engine_name": self.idx_config.get("engine_name"),
            "count": len(chunks),
            "chunks": chunks
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✅ Index saved.")
        print(f"   - Index: {index_file}")
        print(f"   - Metadata: {meta_file}")


# ============================================================================
# RETRIEVER
# ============================================================================

class SentenceRetriever(BaseRetriever):
    """
    Retrieves and reranks chunks using FAISS and Sentence Transformers.
    """
    
    def __init__(self, config_path: str = "config.yaml", verbose: bool = True):
        self.config = ConfigLoader.load_config(config_path)
        self.idx_config = self.config.get("indexing", {})
        self.ret_config = self.config.get("retrieval", {})
        self.output_config = self.config.get("output", {})
        self.verbose = verbose
        
        self.index = None
        self.chunks = []
        self.bi_encoder = None
        self.reranker = None
        self._use_reranker = self.ret_config.get("use_reranker", True)
        self._use_contextual = self.ret_config.get("contextual_reranking", True)
        
        if self._use_reranker:
            self._init_reranker()

    def _init_reranker(self):
        reranker_type = self.ret_config.get("reranker_type", "cross-encoder")
        reranker_name = self.ret_config.get("reranker_name")
        self.reranker = get_reranker(reranker_type, reranker_name)

    def set_reranking(self, enabled: bool):
        self._use_reranker = enabled
        if enabled and self.reranker is None:
            self._init_reranker()
        elif not enabled:
            self.reranker = None

    def _ensure_loaded(self):
        if self.index is not None:
            return

        default_idx = "datas/code_rag/index/faiss.index"
        default_meta = "datas/code_rag/index/metadata.json"
        
        index_file = Path(self.output_config.get("index_file", default_idx))
        meta_file = Path(self.output_config.get("metadata_file", default_meta))

        # Heuristic lookup for project root if relative path fails
        if not index_file.exists():
             project_root = Path(__file__).parent.parent.parent.parent
             index_file = project_root / self.output_config.get("index_file", default_idx)
             
        if not meta_file.exists():
             project_root = Path(__file__).parent.parent.parent.parent
             meta_file = project_root / self.output_config.get("metadata_file", default_meta)

        if not index_file.exists():
             raise FileNotFoundError(f"Index not found at {index_file}. Run 'uv run index --build' first.")
             
        # Load FAISS
        self.index = faiss.read_index(str(index_file))
        
        # Load Metadata
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.chunks = data.get("chunks", [])

        # Load Bi-Encoder (Same as Embedder)
        model_name = self.idx_config.get("engine_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.bi_encoder = SentenceTransformer(model_name)
        
    def query(self, query_text: str, top_k: int = 5, recall_k: int = None, targets: List[str] = None, keywords: List[str] = None, horizon: bool = False) -> Dict[str, Any]:
        self._ensure_loaded()
        
        # 1. Recall
        if recall_k is None:
            recall_k = self.ret_config.get("top_k_recall", 50)
            
        if self.verbose:
            print(f"[RAG] 🔍 Retrieving top {recall_k} candidates...")
        
        # Encode query normalized for Cosine Similarity
        query_vec = self.bi_encoder.encode([query_text], normalize_embeddings=True).astype('float32')
        distances, indices = self.index.search(query_vec, recall_k)
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(distances[0][i])
            candidates.append(chunk)

        if self.verbose:
            print(f"[RAG] ✅ Found {len(candidates)} candidates.")

        final_results = []
        
        # 2. Rerank
        if self.reranker and self._use_reranker:
            reranker_type = self.ret_config.get("reranker_type", "cross-encoder")
            if self.verbose:
                print(f"[RAG] ⚖️ Reranking candidates with {reranker_type}...")
            
            # If horizon is requested, we need ALL rankings to populate horizon list (K+1...)
            rerank_k = top_k
            if horizon:
                rerank_k = len(candidates)

            if self._use_contextual:
                ranked_indices = self.reranker.rank(query_text, candidates, top_k=rerank_k, use_contextual=True, targets=targets, keywords=keywords)
            else:
                texts = [c["text"] for c in candidates]
                ranked_indices = self.reranker.rank(query_text, texts, top_k=rerank_k, targets=targets, keywords=keywords)
            
            final_results = []
            # Only take top_k for main results
            for idx, score in ranked_indices[:top_k]:
                cand = candidates[idx].copy()
                cand["score"] = float(score)
                final_results.append(cand)
            
            # --- HORIZON LOGIC ---
            horizon_results = []
            if horizon:
                # Get candidates from rank K+1 up to the total recalled (usually recall_k)
                # We limit horizon to avoid overwhelming context (e.g. max 50 items)
                horizon_indices = ranked_indices[top_k:]
                
                for idx, score in horizon_indices:
                    cand = candidates[idx]
                    # Lightweight representation (No Content)
                    horizon_results.append({
                        "id": cand.get("id"),
                        "score": float(score),
                        "source": cand.get("source"),
                        "metadata": cand.get("metadata", {}), # Contains docs/symbols
                        # We explicitly exclude 'content' and 'text'
                    })
        else:
            final_results = candidates[:top_k]
            horizon_results = []
            if horizon and len(candidates) > top_k:
                # If no reranker, horizon is just the next items in recall list
                for cand in candidates[top_k:]:
                     horizon_results.append({
                        "id": cand.get("id"),
                        "score": 0.0, # No reranker score
                        "source": cand.get("source"),
                        "metadata": cand.get("metadata", {}),
                    })
            
        if self.verbose:
            print(f"[RAG] 🎯 Returning top {len(final_results)} results.")
            
        return {
            "query": query_text,
            "results": final_results,
            "horizon": horizon_results,
            "stats": {
                "total_candidates": len(candidates),
                "reranked": len(ranked_indices) if self.reranker and self._use_reranker else 0,
            },
            "total_found": len(candidates)
        }
