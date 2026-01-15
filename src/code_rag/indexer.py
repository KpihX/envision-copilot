import os
import json
import logging
from pathlib import Path
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.progress import track

from .utils import ConfigLoader
from .chunker import GraphChunker

logger = logging.getLogger(__name__)

class GraphIndexer:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.idx_config = self.config.get("indexing", {})
        self.output_config = self.config.get("output", {})
        self.input_config = self.config.get("input", {})
        
        model_name = self.idx_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"📦 Loading Embedding Model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        self.chunker = GraphChunker(
            chunk_size=self.idx_config.get("chunk_size", 512),
            overlap=self.idx_config.get("overlap", 50),
            block_keywords=self.idx_config.get("block_keywords")
        )

    def build(self):
        # 1. Load Network
        net_path = Path(self.input_config.get("network_file", "data/network/network.json"))
        # Fix relative path if needed (though running via uv usually sets CWD root or we should use Project Root trick)
        # Using simple Path assuming execution from root as per standard
        
        if not net_path.exists():
            # Try finding it relative to project root heuristic
            project_root = Path(__file__).parent.parent.parent
            net_path = project_root / self.input_config.get("network_file", "data/network/network.json")
            
        if not net_path.exists():
            print(f"❌ Network file not found at {net_path}. Run 'uv run network --build' first.")
            return

        print(f"📂 Loading Graph from {net_path}...")
        with open(net_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        nodes = data.get("nodes", {})
        edges = data.get("edges", [])
        
        # 2. Build Neighborhood Map (Optimization)
        print("🔗 Building Neighborhood Map...")
        neighborhoods = {nid: {"incoming": [], "outgoing": []} for nid in nodes}
        
        nodes_lookup = nodes # ID -> Node
        
        for edge in edges:
            src = edge["source"]
            tgt = edge["target"]
            etype = edge["type"]
            
            # Outgoing
            if src in neighborhoods:
                tgt_node = nodes_lookup.get(tgt)
                tgt_label = tgt_node.get("name") if tgt_node else tgt
                neighborhoods[src]["outgoing"].append({
                    "target_id": tgt,
                    "target_label": tgt_label,
                    "edge_type": etype
                })
                
            # Incoming
            if tgt in neighborhoods:
                src_node = nodes_lookup.get(src)
                src_label = src_node.get("name") if src_node else src
                neighborhoods[tgt]["incoming"].append({
                    "source_id": src,
                    "source_label": src_label,
                    "edge_type": etype
                })
                
        # 3. Generate Graph-Aware Chunks
        print("🔄 Generating Graph-Aware Chunks...")
        all_chunks = []
        
        # Iterate only scripts
        script_nodes = [n for nid, n in nodes.items() if n["type"] == "script"]
        
        for node in track(script_nodes, description="Chunking..."):
            nid = node["id"]
            node_chunks = self.chunker.chunk_node(node, neighborhoods[nid])
            for c in node_chunks:
                c["source"] = node.get("path", nid)  # Add Source Path for RAG
            all_chunks.extend(node_chunks)
            
        print(f"📊 Generated {len(all_chunks)} chunks from {len(script_nodes)} scripts.")
        
        if not all_chunks:
            print("⚠️ No chunks generated. Exiting.")
            return

        # 4. Embed
        print("🧠 Computing Embeddings...")
        texts = [c["text"] for c in all_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # 5. Index (FAISS)
        dimension = embeddings.shape[1]
        print(f"🗂️ Building FAISS Index (Dim: {dimension})...")
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        # 6. Save
        self._save_index(index, all_chunks)

    def _save_index(self, index, chunks):
        index_file = Path(self.output_config.get("index_file", "data/vector_store/faiss.index"))
        meta_file = Path(self.output_config.get("metadata_file", "data/vector_store/metadata.json"))

        # Create parent directories if needed
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

if __name__ == "__main__":
    indexer = GraphIndexer()
    indexer.build()
