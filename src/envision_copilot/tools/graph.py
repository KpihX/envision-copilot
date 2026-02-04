"""
Graph Tool - Network Navigation and Exploration
================================================

Provides tools for navigating and exploring the Envision dependency graph.
Wraps EnvisionGraphAPI methods with a unified interface.

Actions:
- tree: Hierarchical folder navigation (scripts/data domains)
- node: Get single node details
- neighbors: Find connected nodes
- edges: List edges with filters  
- search: Fuzzy search across nodes

Note: Scripts and folders have an `execution_order` field extracted from
their name prefix (e.g., "01 - Catalog" → 1, "3. Inspectors" → 3).
This indicates the execution sequence in the Envision workflow.
"""

from typing import Any, Dict, List, Optional, Literal

from .base import BaseTool, ToolResult


# Type aliases
ActionType = Literal["tree", "node", "neighbors", "edges", "search"]
DirectionType = Literal["incoming", "outgoing", "all", "siblings"]
DomainType = Literal["scripts", "data"]


class GraphTool(BaseTool):
    """
    Tool for navigating and exploring the Envision dependency graph.
    
    Provides access to network structure: scripts, data files, tables,
    functions, and their relationships (imports, reads, writes, etc.).
    
    Actions:
        tree: Browse folder hierarchy by domain (scripts/data)
        node: Get detailed info for a specific node
        neighbors: Find nodes connected to a given node
        edges: List edges with optional type/source/target filters
        search: Fuzzy search for nodes by name/path
    
    The graph has two separate folder trees:
    - scripts: Envision script files (.nvn)
    - data: Data files (.ion, .csv, etc.)
    
    Scripts are executed in order indicated by `execution_order`.
    """
    
    name = "graph"
    description = "Navigate and explore the Envision dependency graph (scripts, data, relationships)"
    
    def execute(self, action: ActionType, **kwargs) -> ToolResult:
        """
        Execute a graph action.
        
        Args:
            action: One of 'tree', 'node', 'neighbors', 'edges', 'search'
            **kwargs: Action-specific parameters
            
        Returns:
            ToolResult with graph data
        """
        if not self.api:
            return self._error("Graph API not initialized", action)
        
        handlers = {
            "tree": self._tree,
            "node": self._node,
            "neighbors": self._neighbors,
            "edges": self._edges,
            "search": self._search
        }
        
        handler = handlers.get(action)
        if not handler:
            return self._error(f"Unknown action: {action}. Valid: {list(handlers.keys())}", action)
        
        try:
            return handler(**kwargs)
        except Exception as e:
            return self._error(f"Execution failed: {str(e)}", action)
    
    def _tree(
        self,
        path: str = "/",
        domain: DomainType = "scripts",
        max_depth: Optional[int] = None,
        **_
    ) -> ToolResult:
        """
        Get hierarchical folder tree.
        
        Args:
            path: Folder path (default: "/" for root)
            domain: "scripts" or "data" tree
            max_depth: Max traversal depth (None = unlimited)
        """
        result = self.api.get_tree(path, domain=domain, max_depth=max_depth)
        
        if result.get("stats", {}).get("error"):
            return self._error(result.get("error", "Tree retrieval failed"), "tree")
        
        return self._success(result, action="tree")
    
    def _node(self, node_id: str, **_) -> ToolResult:
        """
        Get details for a single node.
        
        Args:
            node_id: ID of the node to retrieve
        """
        if not node_id:
            return self._error("node_id is required", "node")
        
        result = self.api.get_node(node_id)
        
        if not result.get("stats", {}).get("found"):
            return self._error(f"Node not found: {node_id}", "node")
        
        return self._success(result, action="node")
    
    def _neighbors(
        self,
        node_id: str,
        direction: DirectionType = "all",
        relation_type: Optional[str] = None,
        **_
    ) -> ToolResult:
        """
        Find nodes connected to a given node.
        
        Args:
            node_id: Source node ID
            direction: "incoming", "outgoing", "all", or "siblings"
            relation_type: Filter by edge type (reads, writes, imports, etc.)
        """
        if not node_id:
            return self._error("node_id is required", "neighbors")
        
        result = self.api.get_neighbors(node_id, direction=direction, relation_type=relation_type)
        
        if result.get("stats", {}).get("error"):
            return self._error(result.get("error", "Neighbors retrieval failed"), "neighbors")
        
        return self._success(result, action="neighbors")
    
    def _edges(
        self,
        relation_type: Optional[str] = None,
        **_
    ) -> ToolResult:
        """
        List edges with optional type filter.
        
        Args:
            relation_type: Filter by edge type (reads, writes, imports, defines, sibling, contains)
        """
        result = self.api.get_edges(relation_type=relation_type)
        
        return self._success(result, action="edges")
    
    def _search(
        self,
        query: str,
        node_types: Optional[List[str]] = None,
        **_
    ) -> ToolResult:
        """
        Fuzzy search for nodes by name/path.
        
        Args:
            query: Search query
            node_types: Filter by node types
        """
        if not query:
            return self._error("query is required", "search")
        
        result = self.api.search(query, node_types=node_types)
        
        return self._success(result, action="search")
