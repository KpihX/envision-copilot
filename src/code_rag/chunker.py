import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GraphChunker:
    """
    Transforms Graph Nodes into vector-ready Chunks with Rich Context.
    Leverages structural dependencies (Reads, Imports) to enrich embeddings.
    """
    
    def __init__(self, chunk_size=512, overlap=50, block_keywords=None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Valid Default for Envision DSL
        if block_keywords is None:
            block_keywords = ["read", "write", "export", "def", "process", "store", "import", "show", "table", "where", "keep", "when"]
            
        # Build optimized regex: ^\s*(keyword1|keyword2|...)\b
        pattern_str = r'^\s*(' + '|'.join(re.escape(k) for k in block_keywords) + r')\b'
        self.block_starters = re.compile(pattern_str, re.IGNORECASE)

    def chunk_node(self, node: Dict[str, Any], neighborhood: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Produce chunks for a single Script Node.
        Neighborhood contains {'incoming': [], 'outgoing': []}.
        """
        if node["type"] != "script" or not node.get("content"):
            return []

        # 1. Build Context Header (The "Graph Awareness")
        context_lines = []
        name = node.get("name") or node["id"]
        path = node.get("path") or node.get("metadata", {}).get("logical_path")
        
        context_lines.append(f"[Script: {name} | Path: {path}]")
        
        # Add Imports/Reads summary to header
        # Using a set to avoid dupes and keep it concise
        deps = set()
        for out in neighborhood.get("outgoing", []):
            label = out.get("target_label", out["target_id"])
            etype = out.get("edge_type", "uses")
            if etype in ["imports", "reads", "writes", "export"]:
                deps.add(f"{etype.title()}: {label}")
        
        # Limit context size to avoid drowning the code
        sorted_deps = sorted(list(deps))
        if len(sorted_deps) > 5:
            context_lines.extend([f"[{d}]" for d in sorted_deps[:5]])
            context_lines.append(f"[... and {len(sorted_deps)-5} more dependencies]")
        else:
            context_lines.extend([f"[{d}]" for d in sorted_deps])

        header = "\n".join(context_lines)
        
        # 2. Split Content (Semantic Block-aware or Sliding Window)
        # For robustness, we'll use a sliding window over lines, 
        # but we prepend the header to EACH chunk.
        
        raw_lines = node["content"].splitlines()
        chunks = []
        
        current_chunk = []
        current_cost = 0 # Approximate token count (words)
        start_line = 0
        
        context_cost = len(header.split())
        
        for i, line in enumerate(raw_lines):
            line_cost = len(line.split())
            
            # If adding this line exceeds limit (minus context), commit chunk
            if current_cost + line_cost + context_cost > self.chunk_size:
                if current_chunk:
                    self._commit(chunks, node["id"], path, header, current_chunk, start_line, i)
                    
                    # Overlap: Keep last N words roughly matching overlap size
                    # Simplified: Keep last 5 lines
                    keep = 5
                    current_chunk = current_chunk[-keep:]
                    current_cost = sum(len(l.split()) for l in current_chunk)
                    start_line = max(0, i - keep)
            
            current_chunk.append(line)
            current_cost += line_cost
            
        if current_chunk:
             self._commit(chunks, node["id"], path, header, current_chunk, start_line, len(raw_lines))

        return chunks

    def _commit(self, chunks, node_id, path, header, lines, start, end):
        body = "\n".join(lines)
        # Full text for embedding
        full_text = f"{header}\n\n{body}"
        
        chunks.append({
            "id": f"{node_id}_{start}_{end}",
            "source_id": node_id,
            "source": path,        # Full Path for User Display
            "text": full_text,     # Used for Embedding
            "content": body,       # Used for Display
            "context": header,     # Metadata
            "lines": f"{start+1}-{end}"
        })
