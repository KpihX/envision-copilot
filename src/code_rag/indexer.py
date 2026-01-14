import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import ConfigLoader

logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.idx_config = self.config.get("indexing", {})
        self.input_config = self.config.get("input", {})
        self.output_config = self.config.get("output", {})
        
        self.model = SentenceTransformer(self.idx_config.get("model_name", "all-MiniLM-L6-v2"))
        
    def build(self):
        # 1. Load Network
        net_path = Path(self.input_config["network_file"])
        if not net_path.exists():
            print(f"❌ Network file not found at {net_path}. Run 'uv run network --build' first.")
            return

        with open(net_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        nodes = data.get("nodes", {})
        
        # 2. Chunking (Script Content)
        print("🔄 Chunking scripts...")
        chunks = self._chunk_nodes(nodes)
        print(f"📊 Generated {len(chunks)} chunks.")
        
        # 3. Embedding
        print("🧠 Generating embeddings...")
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # 4. Indexing (FAISS)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        
        # 5. Saving
        self._save_index(index, chunks)
        
    def _chunk_nodes(self, nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        chunk_size = self.idx_config.get("chunk_size", 512)
        overlap = self.idx_config.get("overlap", 50)
        
        for node_id, node in nodes.items():
            if node["type"] == "script" and node.get("content"):
                lines = node["content"].splitlines()
                # Simple sliding window over lines (approx tokens)
                # Ideally use semantic chunker from old impl if possible
                # For now, strict line chunking for simplicity & standard output
                
                current_chunk = []
                current_len = 0
                
                for i, line in enumerate(lines):
                    line_len = len(line.split()) # Rough token count
                    current_chunk.append(line)
                    current_len += line_len
                    
                    if current_len >= chunk_size:
                        text = "\n".join(current_chunk)
                        chunks.append({
                            "text": text,
                            "source_id": node_id,
                            "lines": f"{max(1, i - len(current_chunk))}-{i+1}",
                            "type": "code_block"
                        })
                        # Overlap: Keep last N lines
                        # Quick approximation: keep last 5 lines (~50 tokens)
                        keep_lines = 5
                        current_chunk = current_chunk[-keep_lines:]
                        current_len = sum(len(l.split()) for l in current_chunk)

                if current_chunk:
                    chunks.append({
                        "text": "\n".join(current_chunk),
                        "source_id": node_id,
                        "lines": "tail",
                        "type": "code_block_tail"
                    })
                    
        return chunks

    def _save_index(self, index, chunks):
        out_dir = Path(self.output_config["store_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS
        faiss.write_index(index, str(out_dir / "faiss.index"))
        
        # Save Metadata (Chunks + Stats)
        meta = {
            "generated_at": datetime.now().isoformat(),
            "model": self.idx_config["model_name"],
            "count": len(chunks),
            "chunks": chunks
        }
        
        with open(Path(self.output_config["metadata_file"]), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
            
        print(f"✅ Index saved to {out_dir}")
