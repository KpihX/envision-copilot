import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from .builder import NetworkBuilder
from .utils import ConfigLoader
from .typedefs import Network, Node, Edge, NodeType, EdgeType

logger = logging.getLogger(__name__)

class EnvisionGraphAPI:
    """
    Public API for interacting with the Envision Dependency Graph.
    Provides methods to Build, Inspect, and Query the graph programmatically.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.meta_path = Path(self.config.get("output", {}).get("metadata_file", "data/network/metadata.json"))
        self.net_path = Path(self.config.get("output", {}).get("network_file", "data/network/network.json"))
        self._graph_cache = None
        self._stats_cache = None

    def build_graph(self) -> Dict[str, Any]:
        """
        Triggers a full rebuild of the network from source scripts.
        Returns the stats of the build.
        """
        print("[Network] 🏗️ Building dependency graph from source...")
        builder = NetworkBuilder() # Config is loaded internally by Builder too, or pass it? Builder loads its own.
        # Ideally pass config to builder to avoid double load, but Builder init handles it.
        builder.build()
        self.clear_cache()
        return self.get_stats()

    def clear_cache(self):
        self._graph_cache = None
        self._stats_cache = None

    def _load_data(self):
        if self._graph_cache is not None:
            return
            
        if not self.net_path.exists():
            raise FileNotFoundError(f"Graph file not found at {self.net_path}. Run build() first.")
            
        with open(self.net_path, 'r', encoding='utf-8') as f:
            self._graph_cache = json.load(f) # Dict with "nodes" and "edges"
            
        if self.meta_path.exists():
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self._stats_cache = json.load(f)

    def get_stats(self) -> Dict[str, Any]:
        """Returns the metadata/stats of the current graph."""
        self._load_data()
        return self._stats_cache if self._stats_cache else {}

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Direct lookup of a node by ID. Returns Dict or None."""
        self._load_data()
        return self._graph_cache["nodes"].get(node_id)
        
    def get_edges(self, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns list of edges, optionally filtered by type."""
        self._load_data()
        edges = self._graph_cache.get("edges", [])
        if edge_type:
            return [e for e in edges if e["type"] == edge_type]
        return edges

    def get_nodes(self, node_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns list of nodes, optionally filtered by type."""
        self._load_data()
        nodes = []
        for nid, node in self._graph_cache.get("nodes", {}).items():
            if node_type:
                if node.get("type") == node_type:
                    nodes.append(node)
            else:
                nodes.append(node)
        return nodes

    def search_nodes(self, query: str) -> List[Dict[str, Any]]:
        """
        Broad search for nodes by ID, Name, or Logical Path.
        Returns detailed list of matching nodes.
        """
        self._load_data()
        matches = []
        q = query.lower()
        
        for nid, node in self._graph_cache["nodes"].items():
            # Match ID
            if q in nid.lower():
                matches.append(node)
                continue
            # Match Name (New field)
            name = node.get("name")
            if name and q in name.lower():
                matches.append(node)
                continue
            # Match Logical Path
            lpath = node.get("metadata", {}).get("logical_path") # Old metadata location
            # New location? We removed it from metadata. It is now node.path (which was logical path in builder)
            path = node.get("path")
            if path and q in path.lower():
                matches.append(node)
                
        return matches

    def resolve_node_id(self, query: str) -> Optional[str]:
        """
        Helper to resolve a query string (ID or Path) to a specific unique Node ID.
        Returns None if not found or ambiguous (simple exact match priority).
        """
        self._load_data()
        
        # 1. Exact ID
        if query in self._graph_cache["nodes"]:
            return query
            
        # 2. Exact Path
        for nid, node in self._graph_cache["nodes"].items():
            if node.get("metadata", {}).get("logical_path") == query:
                return nid
                
        return None

    def grep_nodes(self, pattern: str, node_type: Optional[str] = None, node_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for a regex pattern within the CONTENT of nodes.
        Filters:
          - node_type: Only search nodes of this type (e.g., 'script', 'function').
          - node_ids: Only search within this specific list of node IDs.
        """
        self._load_data()
        import re
        
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [{"error": f"Invalid Regex: {e}"}]

        matches = []
        
        # Decide which nodes to iterate
        candidates = self._graph_cache["nodes"].items()
        
        for nid, node in candidates:
            # 1. Filter by ID List
            if node_ids and nid not in node_ids:
                continue
                
            # 2. Filter by Type
            if node_type and node.get("type") != node_type:
                continue
                
            # 3. Check Content
            content = node.get("content", "")
            if not content:
                continue
                
            if regex.search(content):
                # Found a match!
                # Return snippet? For now just return the node info. Tool will format match lines.
                matches.append({
                    "id": nid,
                    "type": node.get("type"),
                    "path": node.get("path"),
                    "name": node.get("name")
                })
                
        return matches

    def get_neighbors(self, node_id: str, 
                      direction: str = "all", 
                      relation_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get connected nodes and edges.
        direction: 'incoming', 'outgoing', 'all'
        relation_type: Filter by EdgeType value (e.g. 'reads', 'imports')
        
        Returns:
        {
            "incoming": [{source: ..., type: ..., metadata: ...}, ...],
            "outgoing": [{target: ..., type: ..., metadata: ...}, ...]
        }
        """
        self._load_data()
        
        result = {"incoming": [], "outgoing": []}
        
        # Pre-filter edges slightly inefficient O(E) but E is small enough (<100k usually). 
        # For huge graphs, we'd index edges by source/target on load.
        # Given DSL size, simple iteration is fine (400 files ~ 5k edges).
        
        for edge in self._graph_cache["edges"]:
            # Filter by type if requested
            if relation_type and edge["type"] != relation_type:
                continue
                
            # Outgoing: source == node_id
            if direction in ["all", "outgoing"] and edge["source"] == node_id:
                # User wants "answers as json bien exploitable".
                # Providing target node minimal info (like logical path) is helpful.
                target_node = self._graph_cache["nodes"].get(edge["target"])
                # Provide raw fields for CLI flexibility
                target_name = target_node.get("name") if target_node else None
                target_path = target_node.get("path") if target_node else None
                # Default label (still useful for quick consumers)
                target_label = target_path or target_name or edge["target"]
                
                result["outgoing"].append({
                    "target_id": edge["target"],
                    "target_label": target_label,
                    "target_name": target_name,
                    "target_path": target_path,
                    "target_type": target_node["type"] if target_node else "unknown",
                    "edge_type": edge["type"],
                    "count": edge.get("metadata", {}).get("count", 1),
                    "metadata": edge.get("metadata", {})
                })
                
            # Incoming: target == node_id
            if direction in ["all", "incoming"] and edge["target"] == node_id:
                source_node = self._graph_cache["nodes"].get(edge["source"])
                source_name = source_node.get("name") if source_node else None
                source_path = source_node.get("path") if source_node else None
                source_label = source_path or source_name or edge["source"]

                result["incoming"].append({
                    "source_id": edge["source"],
                    "source_label": source_label,
                    "source_name": source_name,
                    "source_path": source_path,
                    "source_type": source_node["type"] if source_node else "unknown",
                    "source_type": source_node["type"] if source_node else "unknown",
                    "edge_type": edge["type"],
                    "count": edge.get("metadata", {}).get("count", 1),
                    "metadata": edge.get("metadata", {})
                })
                
        # Calculate Rich Statistics
        def compute_stats(items, key_id):
            if not items:
                return {"total": 0, "unique_nodes": 0, "by_type": {}}
            
            unique_nodes = set(item[key_id] for item in items)
            by_type = {}
            for item in items:
                etype = item["edge_type"]
                by_type[etype] = by_type.get(etype, 0) + 1
            
            return {
                "total": len(items),
                "unique_nodes": len(unique_nodes),
                "by_type": by_type
            }

        stats = {
            "incoming": compute_stats(result["incoming"], "source_id"),
            "outgoing": compute_stats(result["outgoing"], "target_id"),
            "filter_applied": relation_type
        }
        
        return {
            "stats": stats,
            "incoming": result["incoming"],
            "outgoing": result["outgoing"]
        }
