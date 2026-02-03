"""
Envision Graph API
==================

Public API for interacting with the Envision Dependency Graph.
Provides 4 domains of functionality:

1. **Navigation**: Tree traversal and folder hierarchy exploration
   - Supports TWO separate folder trees: SCRIPTS and DATA domains
2. **Stats**: Network statistics and overview
3. **Content**: Read file contents and search with grep
4. **Graph Exploration**: Node lookup, semantic search, neighbor discovery

All methods return structured data with stats first, then data.
No truncation is applied at API level (smart_truncate is applied by tools).

Author: Envision Copilot Team
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, Union
from dataclasses import dataclass
import logging

from .builder import NetworkBuilder
from .utils import ConfigLoader
from .typedefs import NodeType, EdgeType, TreeDomain

logger = logging.getLogger(__name__)


# Type aliases for clarity
DirectionType = Literal["incoming", "outgoing", "all", "siblings"]
TreeDomainType = Literal["scripts", "data", "both"]
NodeTypeFilter = Optional[List[str]]


class EnvisionGraphAPI:
    """
    Public API for interacting with the Envision Dependency Graph.
    
    The API is organized into 4 domains:
    
    **Navigation Domain:**
    - get_tree(): Hierarchical folder structure (scripts or data domain)
    
    **Stats Domain:**
    - get_stats(): Network overview and statistics (including per-domain)
    
    **Content Domain:**
    - read(): Read full content of a node (script, function, table)
    - grep(): Search patterns across node contents
    
    **Graph Exploration Domain:**
    - get_node(): Direct node lookup by ID
    - search(): Semantic/text search across nodes
    - get_neighbors(): Connected nodes exploration
    
    **Build Domain:**
    - build(): Rebuild the graph from source
    
    Example:
        >>> api = EnvisionGraphAPI()
        >>> stats = api.get_stats()
        >>> print(f"Graph has {stats['node_count']} nodes")
        
        >>> tree = api.get_tree("/project/module")
        >>> for item in tree["children"]:
        ...     print(f"  {item['name']} ({item['type']})")
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the API.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = ConfigLoader.load_config(config_path)
        self.output_config = self.config.get("output", {})
        self.search_config = self.config.get("search", {})
        self.grep_config = self.config.get("grep", {})
        
        self.meta_path = Path(self.output_config.get("metadata_file", "data/network/metadata.json"))
        self.net_path = Path(self.output_config.get("network_file", "data/network/network.json"))
        
        self._graph_cache: Optional[Dict[str, Any]] = None
        self._stats_cache: Optional[Dict[str, Any]] = None

    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def clear_cache(self):
        """Clear cached graph data to force reload on next access."""
        self._graph_cache = None
        self._stats_cache = None

    def _load_data(self):
        """Load graph data from disk if not cached."""
        if self._graph_cache is not None:
            return
            
        if not self.net_path.exists():
            raise FileNotFoundError(
                f"Graph file not found at {self.net_path}. "
                "Run build() first to generate the graph."
            )
            
        with open(self.net_path, 'r', encoding='utf-8') as f:
            self._graph_cache = json.load(f)
            
        if self.meta_path.exists():
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self._stats_cache = json.load(f)

    # =========================================================================
    # Build Domain
    # =========================================================================
    
    def build(self) -> Dict[str, Any]:
        """
        Rebuild the dependency graph from source scripts.
        
        Triggers a full re-scan of the scripts directory, creating:
        - Folder hierarchy nodes
        - Script, data_file, table, function nodes
        - All relationship edges (contains, reads, writes, imports, defines, sibling)
        
        Returns:
            Dict containing build statistics:
            {
                "status": "success",
                "stats": {
                    "node_count": 523,
                    "edge_count": 847,
                    "nodes_by_type": {"script": 60, "folder": 25, ...},
                    "edges_by_type": {"contains": 85, "reads": 234, ...},
                    "generated_at": "2024-01-15T10:30:00"
                }
            }
        """
        print("[Network] 🏗️ Building dependency graph from source...")
        builder = NetworkBuilder()
        builder.build()
        self.clear_cache()
        
        return {
            "status": "success",
            "stats": self.get_stats()
        }

    # =========================================================================
    # Stats Domain
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get network overview and statistics.
        
        Returns:
            Dict with comprehensive statistics:
            {
                "generated_at": "2024-01-15T10:30:00",
                "source_files": 60,
                "node_count": 523,
                "edge_count": 847,
                "nodes_by_type": {
                    "folder": 49,
                    "script": 60,
                    "data_file": 73,
                    "table": 96,
                    "function": 17
                },
                "edges_by_type": {
                    "contains": 181,
                    "reads": 270,
                    "writes": 63,
                    "imports": 55,
                    "defines": 113,
                    "sibling": 91
                },
                "domains": {
                    "scripts": {"folders": 15, "files": 60, "siblings": 59},
                    "data": {"folders": 34, "files": 73, "siblings": 32}
                },
                "resolutions": {
                    "globs": [...],
                    "placeholders": [...]
                }
            }
        """
        self._load_data()
        stats = self._stats_cache.copy() if self._stats_cache else {}
        
        # Compute per-domain stats
        if self._graph_cache:
            scripts_folders = 0
            data_folders = 0
            scripts_siblings = 0
            data_siblings = 0
            
            for nid, node in self._graph_cache["nodes"].items():
                if node.get("type") == NodeType.FOLDER.value:
                    domain = node.get("metadata", {}).get("domain", "")
                    if domain == "scripts":
                        scripts_folders += 1
                    elif domain == "data":
                        data_folders += 1
            
            for edge in self._graph_cache["edges"]:
                if edge.get("type") == EdgeType.SIBLING.value:
                    domain = edge.get("metadata", {}).get("domain", "")
                    if domain == "scripts":
                        scripts_siblings += 1
                    elif domain == "data":
                        data_siblings += 1
            
            stats["domains"] = {
                "scripts": {
                    "folders": scripts_folders,
                    "files": stats.get("nodes_by_type", {}).get("script", 0),
                    "siblings": scripts_siblings
                },
                "data": {
                    "folders": data_folders,
                    "files": stats.get("nodes_by_type", {}).get("data_file", 0),
                    "siblings": data_siblings
                }
            }
        
        return stats

    def get_nodes(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all nodes, optionally filtered by type.
        
        Args:
            node_type: Filter by node type (e.g., "script", "data_file", "folder", "table", "function")
                       If None, returns all nodes.
                       
        Returns:
            Dict with node statistics and full list:
            {
                "stats": {
                    "node_type": "script",  # or "all" if no filter
                    "total_nodes": 60,
                    "by_type": {"script": 60} if filtered, else full breakdown
                },
                "nodes": [
                    {
                        "id": "68010",
                        "name": "DataLoader",
                        "type": "script",
                        "path": "/1. utilities/DataLoader",
                        "metadata": {...}
                    },
                    ...
                ]
            }
        """
        self._load_data()
        
        nodes_list = []
        by_type = {}
        
        for nid, node in self._graph_cache["nodes"].items():
            ntype = node.get("type")
            
            # Count by type
            by_type[ntype] = by_type.get(ntype, 0) + 1
            
            # Filter if requested
            if node_type and ntype != node_type:
                continue
            
            # Build node info (exclude content to keep response light)
            node_info = {
                "id": nid,
                "name": node.get("name"),
                "type": ntype,
                "path": node.get("path")
            }
            
            # Add metadata if present (but not content)
            if node.get("metadata"):
                node_info["metadata"] = node["metadata"]
            
            # Add specific fields based on type
            if ntype == NodeType.SCRIPT.value:
                exec_order = node.get("metadata", {}).get("execution_order")
                if exec_order:
                    node_info["execution_order"] = exec_order
            elif ntype == NodeType.FOLDER.value:
                domain = node.get("metadata", {}).get("domain")
                if domain:
                    node_info["domain"] = domain
                    
            nodes_list.append(node_info)
        
        # Sort: by path, then by name
        nodes_list.sort(key=lambda x: (x.get("path") or "", x.get("name") or ""))
        
        return {
            "stats": {
                "node_type": node_type or "all",
                "total_nodes": len(nodes_list),
                "by_type": {node_type: len(nodes_list)} if node_type else by_type
            },
            "nodes": nodes_list
        }

    def get_edges(self, relation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all edges, optionally filtered by relation type.
        
        Args:
            relation_type: Filter by edge type (e.g., "reads", "writes", "imports", "defines", "sibling", "contains")
                          If None, returns all edges.
                          
        Returns:
            {
                "stats": {
                    "relation_type": "reads",  // or "all" if no filter
                    "total_edges": 270,
                    "by_type": {"reads": 270},  // if filtered, else full breakdown
                    "distinct_targets": [
                        {"id": "/Clean/Items.ion", "name": "Items.ion", "type": "data_file", "path": "...", "count": 28}
                    ]
                },
                "edges": [
                    {
                        "source": "68010",
                        "target": "/Clean/Items.ion",
                        "type": "reads",
                        "metadata": {...}
                    }
                ]
            }
        """
        self._load_data()
        
        edges_list = []
        by_type = {}
        target_counts = {}  # id -> count
        
        for edge in self._graph_cache["edges"]:
            etype = edge.get("type")
            
            # Count by type
            by_type[etype] = by_type.get(etype, 0) + 1
            
            # Filter if requested
            if relation_type and etype != relation_type:
                continue
            
            source_id = edge["source"]
            target_id = edge["target"]
            
            # Track target counts
            target_counts[target_id] = target_counts.get(target_id, 0) + 1
            
            # Build simple edge info
            edge_info = {
                "source": source_id,
                "target": target_id,
                "type": etype
            }
            
            if edge.get("metadata"):
                edge_info["metadata"] = edge["metadata"]
                
            edges_list.append(edge_info)
        
        # Build distinct targets (sorted by count desc)
        def build_distinct_list(counts_dict):
            sorted_items = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
            result = []
            for node_id, count in sorted_items:
                node = self._graph_cache["nodes"].get(node_id, {})
                result.append({
                    "id": node_id,
                    "name": node.get("name"),
                    "type": node.get("type"),
                    "path": node.get("path"),
                    "count": count
                })
            return result
        
        return {
            "stats": {
                "relation_type": relation_type or "all",
                "total_edges": len(edges_list),
                "by_type": {relation_type: len(edges_list)} if relation_type else by_type,
                "distinct_targets": build_distinct_list(target_counts)
            },
            "edges": edges_list
        }

    # =========================================================================
    # Navigation Domain
    # =========================================================================
    
    def get_tree(self, path: str = "/", domain: TreeDomainType = "scripts", max_depth: Optional[int] = 1) -> Dict[str, Any]:
        """
        Get hierarchical folder structure starting from a path.
        
        The graph has TWO separate folder trees:
        - "scripts": folders containing script files (.nvn) - includes execution_order
        - "data": folders containing data files (.ion, .csv, etc.)
        
        Use domain="both" to get both trees at once (for overview).
        
        Args:
            path: Folder path to start from (default: "/" for root)
            domain: Which folder tree - "scripts", "data", or "both"
            max_depth: Maximum depth to traverse (default: 1 = immediate children only)
                       Use None for unlimited depth (full tree)
            
        Returns:
            {
                "stats": {
                    "folder_count": 5,
                    "file_count": 12,
                    "total": 17,
                    "depth_max": 4  // actual max depth in tree
                },
                "domain": "scripts",
                "path": "/1. utilities",
                "children": [
                    {
                        "id": "folder::scripts::/1. utilities/Modules",
                        "name": "1. Modules",
                        "type": "folder",
                        "path": "/1. utilities/Modules",
                        "execution_order": 1,  // folder order in scripts domain
                        "child_count": 8,
                        "children": [...]  // if max_depth > 1 or None
                    },
                    {
                        "id": "67982",
                        "name": "1 - Populate tables",
                        "type": "script",
                        "path": "/1. utilities/1. populating dataset/1 - Populate tables",
                        "execution_order": 1  // script execution order in folder
                    }
                ]
            }
            
            For domain="both", returns:
            {
                "stats": {...},
                "trees": {
                    "scripts": {"path": "/", "children": [...]},
                    "data": {"path": "/", "children": [...]}
                }
            }
            
        Error (if path not found):
            {
                "stats": {"error": true},
                "error": "Folder not found in scripts domain: /invalid/path"
            }
        """
        self._load_data()
        
        # Handle "both" domain - return both trees
        if domain == "both":
            scripts_tree = self._get_tree_for_domain(path, "scripts", max_depth)
            data_tree = self._get_tree_for_domain(path, "data", max_depth)
            
            return {
                "stats": {
                    "scripts": scripts_tree["stats"],
                    "data": data_tree["stats"]
                },
                "trees": {
                    "scripts": scripts_tree,
                    "data": data_tree
                }
            }
        
        return self._get_tree_for_domain(path, domain, max_depth)
    
    def _get_tree_for_domain(self, path: str, domain: str, max_depth: Optional[int] = 1, current_depth: int = 0) -> Dict[str, Any]:
        """
        Get tree for a specific domain (scripts or data).
        
        Internal helper for get_tree().
        
        Args:
            path: Folder path to start from
            domain: "scripts" or "data"
            max_depth: Maximum depth to traverse (None for unlimited)
            current_depth: Current recursion depth (internal use)
        """
        # Normalize path
        path = path.rstrip('/')
        if not path:
            path = '/'
        
        folder_id = f"folder::{domain}::{path}"
        
        # Check folder exists - return error JSON instead of raising
        if folder_id not in self._graph_cache["nodes"] and path != '/':
            return {
                "stats": {"error": True},
                "error": f"Folder not found in {domain} domain: {path}"
            }
        
        # Find all children via CONTAINS edges
        children = []
        total_folder_count = 0
        total_file_count = 0
        actual_max_depth = current_depth
        
        # Track execution order for sorting (scripts domain)
        folder_order_counter = 0
        file_order_counter = 0
        
        for edge in self._graph_cache["edges"]:
            if edge["type"] != EdgeType.CONTAINS.value:
                continue
            if edge["source"] != folder_id:
                continue
                
            child_id = edge["target"]
            child_node = self._graph_cache["nodes"].get(child_id)
            
            if not child_node:
                continue
            
            child_type = child_node.get("type")
            child_info = {
                "id": child_id,
                "name": child_node.get("name", child_id),
                "type": child_type,
                "path": child_node.get("path")
            }
            
            # Add type-specific info
            if child_type == NodeType.FOLDER.value:
                total_folder_count += 1
                folder_order_counter += 1
                
                # Count direct children of this folder
                child_count = sum(
                    1 for e in self._graph_cache["edges"]
                    if e["source"] == child_id and e["type"] == EdgeType.CONTAINS.value
                )
                child_info["child_count"] = child_count
                
                # Add execution_order if present in stored data
                exec_order = child_node.get("metadata", {}).get("execution_order")
                if exec_order is not None:
                    child_info["execution_order"] = exec_order
                
                # Recurse if depth allows (None = unlimited)
                should_recurse = (max_depth is None) or (current_depth + 1 < max_depth)
                if should_recurse and child_count > 0:
                    child_path = child_node.get("path")
                    subtree = self._get_tree_for_domain(child_path, domain, max_depth, current_depth + 1)
                    if "children" in subtree:
                        child_info["children"] = subtree["children"]
                        # Accumulate nested counts
                        total_folder_count += subtree["stats"].get("folder_count", 0)
                        total_file_count += subtree["stats"].get("file_count", 0)
                        # Track actual max depth
                        subtree_depth = subtree["stats"].get("depth_max", current_depth + 1)
                        actual_max_depth = max(actual_max_depth, subtree_depth)
                    
            elif child_type == NodeType.SCRIPT.value:
                total_file_count += 1
                file_order_counter += 1
                # execution_order from stored data
                exec_order = child_node.get("metadata", {}).get("execution_order")
                if exec_order is not None:
                    child_info["execution_order"] = exec_order
            else:
                # data_file or other
                total_file_count += 1
                    
            children.append(child_info)
        
        # Update actual max depth
        if children:
            actual_max_depth = max(actual_max_depth, current_depth + 1)
        
        # Sort: folders first, then by execution_order/name
        children.sort(key=lambda x: (
            0 if x["type"] == NodeType.FOLDER.value else 1,
            x.get("execution_order") if x.get("execution_order") is not None else float('inf'),
            x["name"]
        ))
        
        result = {
            "stats": {
                "folder_count": total_folder_count,
                "file_count": total_file_count,
                "total": total_folder_count + total_file_count,
                "depth_max": actual_max_depth
            },
            "domain": domain,
            "path": path,
            "children": children
        }
        
        return result

    # =========================================================================
    # Content Domain
    # =========================================================================
    
    def read(self, node_id: str, start_line: Optional[int] = None, 
             end_line: Optional[int] = None) -> Dict[str, Any]:
        """
        Read the content of a node.
        
        Works for scripts, functions, and tables. Returns full content
        with optional line range filtering.
        
        Args:
            node_id: ID of the node to read (e.g., "68010", "67992::func::StockEvol")
            start_line: Optional 1-indexed start line (inclusive)
            end_line: Optional 1-indexed end line (inclusive, None = end of file)
            
        Returns:
            {
                "stats": {
                    "lines_total": 234,
                    "lines_returned": 100,
                    "range": [1, 100]
                },
                "node": {
                    "id": "68010",
                    "type": "script",
                    "name": "2 - Sales Analysis",
                    "path": "/3. Inspectors/2 - Sales Analysis",
                    "content": "/// input: /clean/\n...",
                    "metadata": {...}
                }
            }
        
        Error (if node not found or no content):
            {
                "stats": {"error": true},
                "error": "Node '/Clean/Items.ion' (data_file) has no readable content."
            }
        """
        self._load_data()
        
        node = self._graph_cache["nodes"].get(node_id)
        if not node:
            return {
                "stats": {"error": True},
                "error": f"Node not found: {node_id}"
            }
            
        content = node.get("content")
        if content is None:
            return {
                "stats": {"error": True},
                "error": f"Node '{node_id}' ({node.get('type')}) has no readable content."
            }
        
        lines = content.splitlines()
        total_lines = len(lines)
        
        # Apply line range
        if start_line is not None or end_line is not None:
            s = (start_line or 1) - 1  # Convert to 0-indexed
            e = end_line if end_line is not None else total_lines
            lines = lines[s:e]
            range_info = [s + 1, min(e, total_lines)]
        else:
            range_info = [1, total_lines]
        
        # Build node info with content included
        node_info = {
            "id": node_id,
            "type": node.get("type"),
            "name": node.get("name"),
            "path": node.get("path"),
            "content": "\n".join(lines)
        }
        
        # Add execution_order from stored metadata
        exec_order = node.get("metadata", {}).get("execution_order")
        if exec_order is not None:
            node_info["execution_order"] = exec_order
        
        return {
            "stats": {
                "lines_total": total_lines,
                "lines_returned": len(lines),
                "range": range_info
            },
            "node": node_info
        }
    
    def grep(self, pattern: str, node_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search for regex pattern across node contents.
        
        Scans all nodes of specified types for pattern matches.
        Returns match statistics with line previews.
        
        Args:
            pattern: Regex pattern (case-insensitive)
            node_types: Filter by node types (default: ["script"])
            
        Returns:
            {
                "stats": {
                    "pattern": "StockEvol",
                    "total_matches": 15,
                    "nodes_with_matches": 4
                },
                "results": [
                    {
                        "node_id": "68010",
                        "node_type": "script",
                        "node_path": "/3. Inspectors/2 - Sales Analysis",
                        "match_count": 3,
                        "lines": [12, 45, 89],
                        "previews": [
                            {"line": 12, "text": "Items.Stock = StockEvol(horizon, 7)"},
                            {"line": 45, "text": "  result = StockEvol(Items, 30)"},
                            {"line": 89, "text": "// using StockEvol function"}
                        ]
                    }
                ]
            }
        
        Error (if invalid regex):
            {
                "stats": {"error": true},
                "error": "Invalid regex pattern: ..."
            }
        """
        self._load_data()
        
        # Default node_types to just "script"
        if node_types is None:
            node_types = ["script"]
        
        # Compile pattern
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return {
                "stats": {"error": True},
                "error": f"Invalid regex pattern: {e}"
            }
        
        results = []
        total_matches = 0
        
        # Search nodes
        for nid, node in self._graph_cache["nodes"].items():
            if node.get("type") not in node_types:
                continue
                
            content = node.get("content")
            if not content:
                continue
                
            lines = content.splitlines()
            
            # Find all matches with line numbers
            match_lines = []
            previews = []
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    match_lines.append(i)
                    previews.append({"line": i, "text": line.strip()})
            
            if not match_lines:
                continue
            
            total_matches += len(match_lines)
            
            results.append({
                "node_id": nid,
                "node_type": node.get("type"),
                "node_path": node.get("path"),
                "match_count": len(match_lines),
                "lines": match_lines,
                "previews": previews
            })
        
        # Sort by match_count descending
        results.sort(key=lambda x: x["match_count"], reverse=True)
        
        return {
            "stats": {
                "pattern": pattern,
                "total_matches": total_matches,
                "nodes_with_matches": len(results)
            },
            "results": results
        }

    # =========================================================================
    # Graph Exploration Domain
    # =========================================================================
    
    def get_node(self, node_id: str) -> Dict[str, Any]:
        """
        Get a node by its exact ID.
        
        Returns full node data including metadata (without content).
        For content, use read() instead.
        
        Args:
            node_id: Exact node ID
            
        Returns:
            {
                "stats": {"found": true},
                "node": {<node data without content>}
            }
            
        Error:
            {
                "stats": {"error": true, "found": false},
                "error": "Node not found: ..."
            }
        """
        self._load_data()
        
        node = self._graph_cache["nodes"].get(node_id)
        if not node:
            return {
                "stats": {"error": True, "found": False},
                "error": f"Node not found: {node_id}"
            }
        
        # Return node without content (use read() for that)
        node_info = {k: v for k, v in node.items() if k != "content"}
        
        return {
            "stats": {"found": True},
            "node": node_info
        }
    
    def search(self, query: str, node_types: Optional[List[str]] = None,
               top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Search nodes by ID, name, or path.
        
        Performs substring matching across node identifiers.
        For content-based search, use grep() instead.
        
        Args:
            query: Search query (case-insensitive substring match)
            node_types: Filter by node types (default: all types)
            top_k: Maximum number of results (default from config)
            
        Returns:
            Dict with search results:
            {
                "stats": {
                    "query": "loader",
                    "total_matches": 8,
                    "by_type": {"script": 3, "function": 5}
                },
                "matches": [
                    {
                        "id": "67982",
                        "name": "DataLoader",
                        "type": "script",
                        "path": "/project/DataLoader.nvn",
                        "match_field": "name"
                    },
                    ...
                ]
            }
        """
        self._load_data()
        
        if top_k is None:
            top_k = self.search_config.get("top_k", 20)
            
        q = query.lower()
        matches = []
        
        for nid, node in self._graph_cache["nodes"].items():
            # Filter by type
            if node_types and node.get("type") not in node_types:
                continue
            
            match_field = None
            
            # Check ID
            if q in nid.lower():
                match_field = "id"
            # Check name
            elif node.get("name") and q in node["name"].lower():
                match_field = "name"
            # Check path
            elif node.get("path") and q in node["path"].lower():
                match_field = "path"
                
            if match_field:
                matches.append({
                    "id": nid,
                    "name": node.get("name"),
                    "type": node.get("type"),
                    "path": node.get("path"),
                    "match_field": match_field
                })
        
        # Compute stats by type
        by_type = {}
        for m in matches:
            t = m["type"]
            by_type[t] = by_type.get(t, 0) + 1
        
        # Apply top_k limit
        matches = matches[:top_k]
        
        return {
            "stats": {
                "query": query,
                "total_matches": len(matches),
                "by_type": by_type
            },
            "matches": matches
        }
    
    def get_neighbors(self, node_id: str, 
                      direction: DirectionType = "all",
                      relation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get neighboring nodes connected by edges.
        
        Explores the graph structure around a specific node.
        Excludes folder nodes from results (folder info is deducible from metadata).
        
        Args:
            node_id: ID of the center node
            direction: Which edges to follow:
                - "incoming": Nodes pointing TO this node
                - "outgoing": Nodes this node points TO
                - "all": Both incoming and outgoing (excludes siblings)
                - "siblings": Only sibling relationships (same folder)
            relation_type: Filter by edge type (e.g., "reads", "imports")
                Note: When direction="siblings", relation_type is automatically "sibling"
                
        Returns:
            {
                "stats": {
                    "incoming": {"total": 5, "by_type": {"imports": 3, "reads": 2}},
                    "outgoing": {"total": 12, "by_type": {"reads": 8, "writes": 4}},
                    "siblings": {"total": 3}  // only if direction includes siblings
                },
                "incoming": [<node objects with edge_type and metadata>],
                "outgoing": [...],
                "siblings": [...]  // only if direction="siblings" or "all"
            }
            
        Error:
            {
                "stats": {"error": true},
                "error": "Node not found: ..."
            }
        """
        self._load_data()
        
        # Validate node exists
        if node_id not in self._graph_cache["nodes"]:
            return {
                "stats": {"error": True},
                "error": f"Node not found: {node_id}"
            }
        
        # Validate direction/relation_type combo
        if direction == "siblings" and relation_type and relation_type != EdgeType.SIBLING.value:
            return {
                "stats": {"error": True},
                "error": f"Invalid combination: direction='siblings' requires relation_type='sibling' or None, got '{relation_type}'"
            }
        
        # Force sibling relation_type when direction is siblings
        if direction == "siblings":
            relation_type = EdgeType.SIBLING.value
        
        result = {
            "incoming": [],
            "outgoing": [],
            "siblings": []
        }
        
        for edge in self._graph_cache["edges"]:
            # Filter by type if requested
            if relation_type and edge["type"] != relation_type:
                continue
            
            edge_type = edge["type"]
            is_sibling = edge_type == EdgeType.SIBLING.value
            
            # Handle sibling edges (undirected)
            if is_sibling:
                other_id = None
                if edge["source"] == node_id:
                    other_id = edge["target"]
                elif edge["target"] == node_id:
                    other_id = edge["source"]
                    
                if other_id:
                    other_node = self._graph_cache["nodes"].get(other_id, {})
                    
                    # Skip folders
                    if other_node.get("type") == NodeType.FOLDER.value:
                        continue
                        
                    result["siblings"].append({
                        "node_id": other_id,
                        "node_name": other_node.get("name"),
                        "node_type": other_node.get("type"),
                        "node_path": other_node.get("path"),
                        "edge_type": edge_type,
                        "metadata": edge.get("metadata", {})
                    })
            else:
                # Regular directed edges
                # Outgoing: this node is source
                if edge["source"] == node_id:
                    target_node = self._graph_cache["nodes"].get(edge["target"], {})
                    
                    # Skip folders
                    if target_node.get("type") == NodeType.FOLDER.value:
                        continue
                        
                    result["outgoing"].append({
                        "node_id": edge["target"],
                        "node_name": target_node.get("name"),
                        "node_type": target_node.get("type"),
                        "node_path": target_node.get("path"),
                        "edge_type": edge_type,
                        "metadata": edge.get("metadata", {})
                    })
                    
                # Incoming: this node is target
                if edge["target"] == node_id:
                    source_node = self._graph_cache["nodes"].get(edge["source"], {})
                    
                    # Skip folders
                    if source_node.get("type") == NodeType.FOLDER.value:
                        continue
                        
                    result["incoming"].append({
                        "node_id": edge["source"],
                        "node_name": source_node.get("name"),
                        "node_type": source_node.get("type"),
                        "node_path": source_node.get("path"),
                        "edge_type": edge_type,
                        "metadata": edge.get("metadata", {})
                    })
        
        # Compute stats
        def compute_edge_stats(items):
            by_type = {}
            for item in items:
                t = item["edge_type"]
                by_type[t] = by_type.get(t, 0) + 1
            return {"total": len(items), "by_type": by_type}
        
        # Build response based on direction
        response: Dict[str, Any] = {"stats": {}}
        
        if direction in ["incoming", "all"]:
            response["incoming"] = result["incoming"]
            response["stats"]["incoming"] = compute_edge_stats(result["incoming"])
            
        if direction in ["outgoing", "all"]:
            response["outgoing"] = result["outgoing"]
            response["stats"]["outgoing"] = compute_edge_stats(result["outgoing"])
        
        if direction in ["siblings", "all"]:
            response["siblings"] = result["siblings"]
            response["stats"]["siblings"] = {"total": len(result["siblings"])}
        
        if relation_type:
            response["stats"]["filter"] = relation_type
            
        return response