from typing import Dict, Any, Union, List
import logging
import json
from envision_preprocess.api import EnvisionGraphAPI
from envision_copilot.utils.utils import smart_truncate

from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box
from rich.markdown import Markdown

class StructuralResults:
    """
    Handles presentation logic for Structural Search results (String & Rich UI).
    """
    def __init__(self, result: Any, config: Dict[str, Any]):
        self.result = result
        self.config = config or {}
        self.max_items = self.config.get("presentation", {}).get("max_items", 20)
        self.max_lines = self.config.get("presentation", {}).get("max_lines", 50)

    def to_dict(self):
        return self.result if isinstance(self.result, (dict, list)) else {"error": self.result}

    def __str__(self) -> str:
        """Format result for LLM context."""
        if self.result is None:
             return "StructuralSearch Tool (Ready)"

        if isinstance(self.result, str):
            return self.result
        
        # Generic JSON Dump with Truncation
        truncated = smart_truncate(self.result, max_lines=self.max_lines, max_items=self.max_items)
        return json.dumps(truncated, indent=2, ensure_ascii=False) if isinstance(truncated, (dict, list)) else str(truncated)

    def print(self) -> Panel:
        """Format result for Rich UI."""
        if self.result is None:
             return Panel("StructuralSearch Tool (Ready)", title="🕸️ Structural", border_style="dim")

        if isinstance(self.result, str):
             # Try to parse if it looks like JSON string? No, assume error message or plain text.
             return Panel(self.result, title="🕸️ Structural Message", border_style="cyan")
            
        # Generic JSON Dump with Truncation wrapped in Markdown
        truncated = smart_truncate(self.result, max_lines=self.max_lines, max_items=self.max_items)
        truncated_json = json.dumps(truncated, indent=2, ensure_ascii=False) if isinstance(truncated, (dict, list)) else str(truncated)
        
        return Panel(Markdown(f"```json\n{truncated_json}\n```"), title="🕸️ Structural Result", border_style="cyan")


class StructuralSearch:
    """
    Wrapper around EnvisionGraphAPI to provide tool-friendly output.
    Delegates Actions -> API -> Returns StructuralResults.
    """
    def __init__(self, api = None, config: Dict[str, Any] = None, result: Any = None):
        self.config = config or {}
        # Result can be passed in constructor (for result container usage) or generated
        self.result = result 
        
        try:
            if api:
                self.api = api
            # Lazy init API only if we rely on this instance to RUN commands AND api was not provided
            elif result is None:
                self.api = EnvisionGraphAPI()
                if not self.api.net_path.exists():
                    logging.warning(f"Graph not found at {self.api.net_path}. Structural tools might fail.")
            else:
                self.api = None
        except Exception as e:
             logging.error(f"Failed to initialize StructuralSearch: {e}")
             self.api = None

    def explore(self, action: str = "stats", node_id: str = None, **kwargs) -> StructuralResults:
        """
        Main entry point for tool actions.
        """
        if not self.api:
            return StructuralResults("Error: Graph API not initialized.", self.config)

        # 1. stats
        if action == "stats":
            return StructuralResults(self.api.get_stats(), self.config)

        # 2. nodes
        if action == "nodes":
            node_type = kwargs.get("type")
            data = self.api.get_nodes(node_type=node_type) 
            # API returns {stats:..., nodes:[...]}
            
            # Apply Limit Logic Here ? Or keep it raw?
            # Existing logic capped output for UI/LLM. 
            # Let's keep raw data in 'result' but Presentation will truncate.
            # However, if the list is HUGE (10k nodes), passing it around is expensive.
            # Let's truncate the list IN THE RESULT object itself for performance?
            # Or rely on API to paginate? API doesn't paginate yet.
            # Let's simple truncate list in memory here.
            
            limit = self.config.get("tools", {}).get("structural", {}).get("limit", 50)
            nodes = data.get("nodes", [])
            if len(nodes) > limit:
                data["nodes"] = nodes[:limit]
                data["warning"] = f"ResultSet truncated (showing {limit}/{len(nodes)}). Use filters."
                
            return StructuralResults(data, self.config)

        # 3. edges
        if action == "edges":
            # Map 'type' or 'relation_type' to edge_type
            edge_type = kwargs.get("type") or kwargs.get("relation_type")
            return StructuralResults(self.api.get_edges(edge_type=edge_type), self.config)

        # 4. neighbors
        if action == "neighbors":
            if not node_id:
                return StructuralResults("Error: node_id is required for neighbors action.", self.config)
            try:
                result = self.api.get_neighbors(node_id, **kwargs)
                if result is None:
                    return StructuralResults(f"Error: Node '{node_id}' not found.", self.config)
                return StructuralResults(result, self.config)
            except Exception as e:
                return StructuralResults(f"Error exploring neighbors: {e}", self.config)

        # 5. [NEW] search_node
        if action == "search_node":
            query = kwargs.get("query")
            if not query:
                return StructuralResults("Error: query is required for search_node action.", self.config)
            
            data = self.api.search_nodes(query)
            # data = {stats:..., matches:[...]}
            return StructuralResults(data, self.config)

        # 6. [NEW] get_node
        if action == "get_node":
            if not node_id:
                 return StructuralResults("Error: node_id is required for get_node action.", self.config)
            
            node = self.api.get_node(node_id)
            if not node:
                return StructuralResults(f"Error: Node '{node_id}' not found via get_node.", self.config)
            
            # Smart Truncate Content if it's a script
            if node.get("type") == "script" and "content" in node:
                # We do NOT want to flood the context with full script
                # Rely on smart_truncate util when creating StructuralResults?
                # The user asked: "if node=script; the content will be truncated using smart_truncate"
                # Let's do it explicitly here or rely on result formatting.
                # StructuralResults.__str__ calls smart_truncate, but let's be safe.
                # Actually, modifying the 'node' dict in place might be misleading if we returned it to code.
                # But here we return a Result object for viewing.
                pass # StructuralResults will handle truncation via smart_truncate(max_lines)

            return StructuralResults(node, self.config)

        return StructuralResults(f"Error: Unknown action '{action}'", self.config)

    # Proxy methods for backward compatibility if needed (e.g. if code calls tooling.to_dict())
    # But strictly speaking, the tools usually return the object and the system calls to_dict or str.
    def to_dict(self):
         # If self was init with result (legacy container usage)
         if self.result is not None:
             return StructuralResults(self.result, self.config).to_dict()
         return {}
    
    def __str__(self):
        if self.result is not None:
             return str(StructuralResults(self.result, self.config))
        return "StructuralSearch Tool (Runner)"

