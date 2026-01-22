"""
Graph-aware chunker for Envision code.

Transforms Graph Nodes into vector-ready Chunks with Rich Context.
Leverages structural dependencies (Reads, Imports) to enrich embeddings.
"""

import re
import logging
from typing import List, Dict, Any

from .base import BaseChunker

logger = logging.getLogger(__name__)


class GraphChunker(BaseChunker):
    """
    Graph-aware chunker that enriches chunks with dependency context.
    
    Features:
    - Prepends graph context (imports, reads, writes) to each chunk
    - Uses sliding window with configurable overlap
    - Respects chunk size limits for embedding models
    
    All parameters are loaded from config.yaml.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        "chunk_size": 512,
        "overlap": 5,
        "max_deps": 7,
        "block_keywords": [
            "read", "write", "export", "def", "process", 
            "store", "import", "show", "table", "where", "keep", "when"
        ],
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize graph chunker.
        
        Args:
            config: Chunker config from config.yaml (indexing section)
        """
        super().__init__("graph")
        
        cfg = config or {}
        
        self.chunk_size = cfg.get("chunk_size", self.DEFAULT_CONFIG["chunk_size"])
        self.overlap = cfg.get("overlap", self.DEFAULT_CONFIG["overlap"])
        self.max_deps = cfg.get("max_deps", self.DEFAULT_CONFIG["max_deps"])
        
        block_keywords = cfg.get("block_keywords", self.DEFAULT_CONFIG["block_keywords"])
        
        # Build optimized regex for block detection
        pattern_str = r'^\s*(' + '|'.join(re.escape(k) for k in block_keywords) + r')\b'
        self.block_starters = re.compile(pattern_str, re.IGNORECASE)

    def chunk_node(self, node: Dict[str, Any], neighborhood: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Produce chunks for a single Script Node.
        
        Args:
            node: Node dict with id, type, content, path, etc.
            neighborhood: Dict with 'incoming' and 'outgoing' edge lists
            
        Returns:
            List of chunk dicts ready for embedding
        """
        if node["type"] != "script" or not node.get("content"):
            return []

        # 1. Build Context Header (Graph Awareness)
        context_lines = []
        node_id = node["id"]
        name = node.get("name")
        path = node.get("path") or node.get("metadata", {}).get("logical_path")
        
        context_lines.append(f"[Script ({node_id}): {name} | Path: {path}]")
        
        # Add dependency summary
        deps = set()
        for out in neighborhood.get("outgoing", []):
            label = out.get("target_label", out["target_id"])
            etype = out.get("edge_type", "uses")
            if etype in ["imports", "reads", "writes", "export"]:
                deps.add(f"{etype.title()}: {label}")
        
        # Limit context size
        sorted_deps = sorted(list(deps))
        if len(sorted_deps) > self.max_deps:
            context_lines.extend([f"[{d}]" for d in sorted_deps[:self.max_deps]])
            context_lines.append(f"[... and {len(sorted_deps)-self.max_deps} more dependencies]")
        else:
            context_lines.extend([f"[{d}]" for d in sorted_deps])

        header = "\n".join(context_lines)
        
        # 2. Split Content using sliding window
        raw_lines = node["content"].splitlines()
        chunks = []
        
        current_chunk = []
        current_cost = 0
        start_line = 0
        
        context_cost = len(header.split())
        
        for i, line in enumerate(raw_lines):
            line_cost = len(line.split())
            
            # Commit chunk if size exceeded
            if current_cost + line_cost + context_cost > self.chunk_size:
                if current_chunk:
                    self._commit(chunks, node_id, path, header, current_chunk, start_line, i)
                    
                    # Overlap: Keep last N lines
                    current_chunk = current_chunk[-self.overlap:]
                    current_cost = sum(len(l.split()) for l in current_chunk)
                    start_line = max(0, i - self.overlap)
            
            current_chunk.append(line)
            current_cost += line_cost
            
        if current_chunk:
            self._commit(chunks, node_id, path, header, current_chunk, start_line, len(raw_lines))

        return chunks

    def _commit(self, chunks: List, node_id: str, path: str, header: str, 
                lines: List[str], start: int, end: int) -> None:
        """Commit a chunk to the chunks list."""
        body = "\n".join(lines)
        full_text = f"{header}\n\n{body}"
        
        chunks.append({
            "id": f"{node_id}_{start}_{end}",
            "source_id": node_id,
            "source": path,
            "text": full_text,
            "content": body,
            "context": header,
            "lines": f"{start+1}-{end}"
        })
