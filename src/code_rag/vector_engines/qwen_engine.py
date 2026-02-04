"""
Qwen Embedding Engine for RAG.

Uses Alibaba's text-embedding-v4 model via Dashscope API (OpenAI-compatible).

Contains:
- QwenEmbedder: Builds FAISS index using Qwen embeddings.
- QwenRetriever: Retrieves and reranks chunks using FAISS + Qwen embeddings.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from rich.progress import track

from .base import BaseEmbedder, BaseRetriever
from ..chunkers import get_chunker
from ..utils import ConfigLoader
from ..rerankers import get_reranker

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)


# ============================================================================
# EMBEDDER (Indexer)
# ============================================================================

class QwenEmbedder(BaseEmbedder):
    """
    Builds FAISS indices using Qwen text-embedding-v4 via Dashscope API.
    
    Features:
    - Modular chunking
    - Neighborhood context support
    - High-quality multilingual embeddings (1024 dim)
    """
    
    # Region base URLs
    REGIONS = {
        "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        "china": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = ConfigLoader.load_config()
        
        super().__init__("qwen", config)
        
        self.idx_config = config.get("indexing", {})
        self.output_config = config.get("output", {})
        self.input_config = config.get("input", {})
        
        # API setup
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables.")
        
        region = self.idx_config.get("qwen_region", "singapore")
        base_url = self.REGIONS.get(region, self.REGIONS["singapore"])
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = self.idx_config.get("engine_name", "text-embedding-v4")
        self.batch_size = self.idx_config.get("batch_size", 10)  # API limit: max 10
        
        print(f"📦 Using Qwen Embedding Model: {self.model_name} (Region: {region})")
        
        # Initialize chunker
        chunker_type = self.idx_config.get("chunker_type", "graph")
        self.chunker = get_chunker(chunker_type, self.idx_config)
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using Qwen API with batching."""
        all_embeddings = []
        
        for i in track(range(0, len(texts), self.batch_size), description="🧠 Embedding..."):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype='float32')
    
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
        
        # 4. Compute Embeddings via Qwen API
        texts = [c["text"] for c in all_chunks]
        embeddings = self._embed_texts(texts)
        
        # Normalize for Cosine Similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)
        
        # 5. Build FAISS Index (Inner Product)
        dimension = embeddings.shape[1]
        print(f"🗂️ Building FAISS Index (Dim: {dimension}, Metric: InnerProduct/Cosine)...")
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # 6. Save Index + Metadata
        self._save_index(index, all_chunks, dimension)
        print("✅ Index Build Complete!")
    
    def _resolve_network_path(self) -> Path | None:
        """Resolve path to network.json."""
        default_path = "datas/network/network.json"
        net_path = Path(self.input_config.get("network_file", default_path))
        
        if not net_path.exists():
            project_root = Path(__file__).parent.parent.parent.parent
            net_path = project_root / self.input_config.get("network_file", default_path)
        
        if not net_path.exists():
            print(f"❌ Network file not found: {net_path}. Run 'uv run network --build' first.")
            return None
        return net_path
    
    def _build_neighborhoods(self, nodes: Dict, edges: List) -> Dict[str, Dict]:
        """Build neighborhood map for each node (same format as SentenceEmbedder)."""
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
    
    def _generate_chunks(self, nodes: Dict, neighborhoods: Dict) -> List[Dict]:
        """Delegate to chunker - process each script node."""
        all_chunks = []
        script_nodes = [n for nid, n in nodes.items() if n["type"] == "script"]
        
        for node in track(script_nodes, description="Chunking..."):
            nid = node["id"]
            node_chunks = self.chunker.chunk_node(node, neighborhoods[nid])
            for c in node_chunks:
                c["source"] = node.get("path", nid)
            all_chunks.extend(node_chunks)
        return all_chunks
    
    def _save_index(self, index: faiss.Index, chunks: List[Dict], dimension: int):
        """Save FAISS index and metadata."""
        index_file = Path(self.output_config.get("index_file", "datas/code_rag/index/faiss.index"))
        meta_file = Path(self.output_config.get("metadata_file", "datas/code_rag/index/metadata.json"))
        
        index_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, str(index_file))
        
        # Save metadata
        metadata = {
            "engine_type": "qwen",
            "engine_name": self.model_name,
            "dimension": dimension,
            "count": len(chunks),
            "generated_at": datetime.now().isoformat(),
            "chunks": chunks
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Index saved to {index_file.parent}")


# ============================================================================
# RETRIEVER
# ============================================================================

class QwenRetriever(BaseRetriever):
    """
    Retrieves chunks using FAISS + Qwen embeddings.
    Compatible with SentenceRetriever interface for benchmarks.
    """
    
    REGIONS = QwenEmbedder.REGIONS
    
    def __init__(self, config_path: str = "src/code_rag/config.yaml", verbose: bool = True):
        self.config = ConfigLoader.load_config(config_path)
        self.idx_config = self.config.get("indexing", {})
        self.ret_config = self.config.get("retrieval", {})
        self.output_config = self.config.get("output", {})
        self.verbose = verbose
        
        # API setup
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables.")
        
        region = self.idx_config.get("qwen_region", "singapore")
        base_url = self.REGIONS.get(region, self.REGIONS["singapore"])
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = self.idx_config.get("engine_name", "text-embedding-v4")
        
        # Lazy loading
        self.index = None
        self.chunks = []
        self.dimension = 1024
        
        # Reranker
        self.reranker = None
        self._use_reranker = self.ret_config.get("use_reranker", True)
        if self._use_reranker:
            self._init_reranker()
    
    def _init_reranker(self):
        reranker_type = self.ret_config.get("reranker_type", "heuristic")
        reranker_name = self.ret_config.get("reranker_name")
        self.reranker = get_reranker(reranker_type, reranker_name)
    
    def set_reranking(self, enabled: bool):
        """Enable/disable reranking (benchmark compatibility)."""
        self._use_reranker = enabled
        if enabled and self.reranker is None:
            self._init_reranker()
        elif not enabled:
            self.reranker = None
    
    def _ensure_loaded(self):
        """Lazy load index on first query."""
        if self.index is not None:
            return
        
        default_idx = "datas/code_rag/index/faiss.index"
        default_meta = "datas/code_rag/index/metadata.json"
        
        index_file = Path(self.output_config.get("index_file", default_idx))
        meta_file = Path(self.output_config.get("metadata_file", default_meta))
        
        # Fallback to project root
        if not index_file.exists():
            project_root = Path(__file__).parent.parent.parent.parent
            index_file = project_root / self.output_config.get("index_file", default_idx)
        if not meta_file.exists():
            project_root = Path(__file__).parent.parent.parent.parent
            meta_file = project_root / self.output_config.get("metadata_file", default_meta)
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found at {index_file}. Run 'uv run index --build' first.")
        
        if self.verbose:
            print(f"📂 Loading Qwen Index from {index_file.parent}...")
        
        self.index = faiss.read_index(str(index_file))
        
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        self.chunks = meta.get("chunks", [])
        self.dimension = meta.get("dimension", 1024)
        
        if self.verbose:
            print(f"✅ Loaded {len(self.chunks)} chunks (dim={self.dimension})")
    
    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a single query via Qwen API."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype='float32')
        # Normalize for cosine similarity
        embedding = embedding / np.maximum(np.linalg.norm(embedding), 1e-10)
        return embedding.reshape(1, -1)
    
    def query(
        self, 
        query_text: str, 
        top_k: int = 5,
        recall_k: int = None,
        keywords: List[str] = None,
        targets: List[str] = None,
        horizon: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks (compatible with SentenceRetriever).
        """
        self._ensure_loaded()
        
        # 1. Recall
        if recall_k is None:
            recall_k = self.ret_config.get("top_k_recall", 50)
        
        if self.verbose:
            print(f"[RAG] 🔍 Retrieving top {recall_k} candidates...")
        
        # 2. Embed query
        query_vec = self._embed_query(query_text)
        distances, indices = self.index.search(query_vec, recall_k)
        
        # 3. Build candidates
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(distances[0][i])
            candidates.append(chunk)
        
        if self.verbose:
            print(f"[RAG] ✅ Found {len(candidates)} candidates.")
        
        # 4. Rerank
        final_results = []
        reranked_count = 0
        
        if self._use_reranker and self.reranker and candidates:
            if self.verbose:
                print(f"[RAG] 🔄 Reranking with {type(self.reranker).__name__}...")
            
            # Reranker returns List[(idx, score)]
            ranked_indices = self.reranker.rank(
                query_text, 
                candidates, 
                top_k=top_k, 
                targets=targets, 
                keywords=keywords
            )
            reranked_count = len(ranked_indices)
            
            # Rebuild dicts from indices
            for idx, score in ranked_indices[:top_k]:
                cand = candidates[idx].copy()
                cand["score"] = float(score)
                final_results.append(cand)
        else:
            final_results = candidates[:top_k]
        
        # 5. Stats
        stats = {
            "total_candidates": len(candidates),
            "reranked": reranked_count,
            "returned": len(final_results)
        }
        
        response = {"results": final_results, "stats": stats}
        
        # 6. Horizon (optional)
        if horizon and final_results:
            response["horizon"] = self._get_horizon(final_results)
        
        return response
    
    def _get_horizon(self, results: List[Dict]) -> List[Dict]:
        """Get neighboring chunks for context."""
        horizon_chunks = []
        seen_ids = {r.get("source_id") for r in results}
        
        for r in results[:3]:
            source_id = r.get("source_id")
            for chunk in self.chunks:
                if chunk.get("source_id") == source_id and chunk.get("text") != r.get("text"):
                    if chunk.get("source_id") not in seen_ids:
                        horizon_chunks.append({
                            "source_id": chunk.get("source_id"),
                            "source": chunk.get("source"),
                            "lines": chunk.get("lines"),
                            "score": 0.0,
                            "content": chunk.get("text", "")[:500]
                        })
        
        return horizon_chunks[:5]
