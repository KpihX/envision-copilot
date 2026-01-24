from typing import Dict, Any, Union, List
import logging
import json
from envision_preprocess.api import EnvisionGraphAPI
from envision_copilot.utils.utils import smart_truncate

from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box

from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box

class StructuralSearch:
    """
    Wrapper around EnvisionGraphAPI to provide tool-friendly output.
    Acts as both Runner and Result Container.
    """
    def __init__(self, config: Dict[str, Any] = None, result: Any = None):
        self.config = config or {}
        self.max_output = self.config.get("presentation", {}).get("max_output", 30)
        self.result = result
        try:
             if result is None:
                 self.api = EnvisionGraphAPI()
                 if not self.api.net_path.exists():
                     logging.warning(f"Graph not found at {self.api.net_path}. Structural tools might fail.")
             else:
                 self.api = None
        except Exception as e:
            logging.error(f"Failed to initialize StructuralSearch: {e}")
            self.api = None

    def explore(self, action: str = "stats", node_id: str = None, **kwargs) -> 'StructuralSearch':
        """
        Explores the graph structure.
        Returns a NEW instance of StructuralSearch containing the result.
        """
        if not self.api:
            return StructuralSearch(self.config, result="Error: Graph API not initialized (Run 'uv run network --build' first).")

        if action == "stats":
            return StructuralSearch(self.config, result=self.api.get_stats())

        if action == "nodes":
            node_type = kwargs.get("type")
            nodes = self.api.get_nodes(node_type=node_type)
            simplified = []
            for n in nodes:
                simplified.append({
                    "id": n.get("id", "unknown"),
                    "name": n.get("name"), 
                    "path": n.get("path"),
                    "type": n.get("type")
                })
            
            limit = self.config.get("tools", {}).get("structural", {}).get("limit", 50)
            if len(simplified) > limit:
                data = {
                    "total_count": len(simplified),
                    "showing_first": limit,
                    "nodes": simplified[:limit],
                    "warning": "ResultSet truncated. Use more specific filters."
                }
                return StructuralSearch(self.config, result=data)
            return StructuralSearch(self.config, result=simplified)
            
        if action == "edges":
            edge_type = kwargs.get("type")
            return StructuralSearch(self.config, result=self.api.get_edges(edge_type=edge_type))

        if action == "neighbors":
            if not node_id:
                return StructuralSearch(self.config, result="Error: node_id is required for neighbors action.")
            
            try:
                result = self.api.get_neighbors(node_id, **kwargs)
                if result is None:
                    return StructuralSearch(self.config, result=f"Error: Node '{node_id}' not found in graph.")
                return StructuralSearch(self.config, result=result)
            except Exception as e:
                return StructuralSearch(self.config, result=f"Error exploring neighbors: {e}")

        return StructuralSearch(self.config, result=f"Error: Unknown action '{action}'")

    def __str__(self) -> str:
        """Format result for LLM context."""
        if self.result is None:
             return "StructuralSearch Tool (Ready)"

        if isinstance(self.result, str):
            return self.result
        
        buffer = ["\n### 🕸️ Structural Exploration Results:"]
        
        result = self.result
        if isinstance(result, dict) and "total_count" in result:
            buffer.append(f"Total: {result['total_count']} (showing first {result['showing_first']})")
            buffer.append(f"⚠️ {result.get('warning', '')}")
            nodes = result.get("nodes", [])
            for node in nodes[:self.max_output]:
                buffer.append(f"  [{node.get('id')}] {node.get('name')} ({node.get('type')})")
        
        elif isinstance(result, list):
            nodes = result
            buffer.append(f"Found: {len(nodes)} nodes")
            for node in nodes[:self.max_output]:
                buffer.append(f"  [{node.get('id')}] {node.get('name')} ({node.get('type')})")
                
        elif isinstance(result, dict) and "node_count" in result:  # Stats
            buffer.append(f"Graph Stats:")
            for key, val in result.items():
                buffer.append(f"  {key}: {val}")

        else:
            # Neighbors or other dict
            limit = self.config.get("presentation", {}).get("max_output_lines", 100)
            return json.dumps(smart_truncate(result, max_lines=limit), indent=2, ensure_ascii=False)
        
        return "\n".join(buffer)

    def print(self) -> Panel:
        """Format result for Rich UI."""
        if self.result is None:
             return "StructuralSearch Tool (Ready)"

        if isinstance(self.result, str):
            from rich.text import Text
            return Panel(Text(self.result, style="red"), title="🕸️ Structural Error", border_style="red")
            
        result = self.result

        if isinstance(result, dict) and "node_count" in result: # Stats
            # Stats Table
            table = Table(show_header=False, box=box.SIMPLE)
            for k, v in result.items():
                table.add_row(str(k), str(v))
            return Panel(table, title="🕸️ Graph Statistics", border_style="cyan")
            
        # Nodes List (List or Paginated Dict)
        nodes = []
        if isinstance(result, list): nodes = result
        elif isinstance(result, dict) and "nodes" in result: nodes = result["nodes"]
        
        if nodes:
            tree = Tree(f"🕸️ [bold]Structural Nodes ({len(nodes)})[/bold]")
            for n in nodes[:self.max_output]:
                tree.add(f"[{n.get('id')}] [bold]{n.get('name')}[/bold] [dim]({n.get('type')})[/dim]")
            if len(nodes) > self.max_output:
                tree.add(f"[italic]... +{len(nodes)-self.max_output} more[/italic]")
            return Panel(tree, title="🕸️ Nodes", border_style="cyan")
            
        # Fallback (Full print as requested)
        return Panel(str(self.result), title="🕸️ Structural Result", border_style="cyan")

    def to_dict(self):
        return self.result if isinstance(self.result, (dict, list)) else {"error": self.result}
