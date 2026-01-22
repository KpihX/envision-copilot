"""
Graph-aware indexer with FAISS vector store.

Builds a vector index from graph nodes, using chunkers for preprocessing
and sentence-transformers for embeddings.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.progress import track

from .base import BaseIndexer
from ..chunkers import get_chunker
from ..utils import ConfigLoader

logger = logging.getLogger(__name__)


class GraphIndexer(BaseIndexer):
    """
    Graph-aware indexer that builds FAISS indices from graph nodes.
    
    Features:
    - Uses modular chunkers for preprocessing
    - Supports multiple embedding model families
    - Builds neighborhood context for each node
    
    All parameters are loaded from config.yaml.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize graph indexer.
        
        Args:
            config: Full config dict from config.yaml
        """
        if config is None:
            config = ConfigLoader.load_config()
        
        super().__init__("graph", config)
        
        self.idx_config = config.get("indexing", {})
        self.output_config = config.get("output", {})
        self.input_config = config.get("input", {})
        
        # Load embedding model
        model_name = self.idx_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"📦 Loading Embedding Model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize chunker (modular)
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
        
        # 3. Generate Graph-Aware Chunks
        print("🔄 Generating Graph-Aware Chunks...")
        all_chunks = self._generate_chunks(nodes, neighborhoods)
        
        print(f"📊 Generated {len(all_chunks)} chunks.")
        
        if not all_chunks:
            print("⚠️ No chunks generated. Exiting.")
            return
        
        # 4. Compute Embeddings
        print("🧠 Computing Embeddings...")
        texts = [c["text"] for c in all_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # 5. Build FAISS Index
        dimension = embeddings.shape[1]
        print(f"🗂️ Building FAISS Index (Dim: {dimension})...")
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        # 6. Save
        self._save_index(index, all_chunks)
    
    def _resolve_network_path(self) -> Path:
        """Resolve the network file path."""
        net_path = Path(self.input_config.get("network_file", "data/network/network.json"))
        
        if not net_path.exists():
            project_root = Path(__file__).parent.parent.parent.parent
            net_path = project_root / self.input_config.get("network_file", "data/network/network.json")
        
        if not net_path.exists():
            print(f"❌ Network file not found at {net_path}. Run 'uv run network --build' first.")
            return None
        
        return net_path
    
    def _build_neighborhoods(self, nodes: Dict, edges: list) -> Dict[str, Dict]:
        """Build neighborhood map for each node."""
        neighborhoods = {nid: {"incoming": [], "outgoing": []} for nid in nodes}
        
        for edge in edges:
            src, tgt, etype = edge["source"], edge["target"], edge["type"]
            
            # Outgoing
            if src in neighborhoods:
                tgt_node = nodes.get(tgt)
                tgt_label = tgt_node.get("name") if tgt_node else tgt
                neighborhoods[src]["outgoing"].append({
                    "target_id": tgt,
                    "target_label": tgt_label,
                    "edge_type": etype
                })
            
            # Incoming
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
        """Generate chunks from all script nodes."""
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
        """Save FAISS index and metadata."""
        index_file = Path(self.output_config.get("index_file", "data/vector_store/faiss.index"))
        meta_file = Path(self.output_config.get("metadata_file", "data/vector_store/metadata.json"))
        
        index_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(index_file))
        
        meta = {
            "generated_at": datetime.now().isoformat(),
            "model": self.idx_config.get("model_name"),
            "count": len(chunks),
            "chunks": chunks
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✅ Index saved.")
        print(f"   - Index: {index_file}")
        print(f"   - Metadata: {meta_file}")
