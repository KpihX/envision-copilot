from typing import Dict, Any, Union
import logging
from envision_preprocess.api import EnvisionGraphAPI

class StructuralTools:
    """
    Wrapper around EnvisionGraphAPI to provide tool-friendly output.
    """
    def __init__(self, config: Dict[str, Any] = None):
        try:
            # Initialize with default or provided config
            self.api = EnvisionGraphAPI()
            # Cache check
            if not self.api.net_path.exists():
                logging.warning(f"Graph not found at {self.api.net_path}. Structural tools might fail.")
        except Exception as e:
            logging.error(f"Failed to initialize StructuralTools: {e}")
            self.api = None

    def explore(self, action: str = "stats", node_id: str = None, **kwargs) -> Union[Dict, str]:
        """
        Explores the graph structure.
        """
        if not self.api:
            return "Error: Graph API not initialized (Run 'uv run network --build' first)."

        if action == "stats":
            return self.api.get_stats()

        if action == "nodes":
            node_type = kwargs.get("type")
            nodes = self.api.get_nodes(node_type=node_type)
            # Simplify output: ID, Name, Path only. NO CONTENT.
            simplified = []
            for n in nodes:
                simplified.append({
                    "id": n.get("id", "unknown"),
                    "name": n.get("name"), 
                    "path": n.get("path"),
                    "type": n.get("type")
                })
            
            # Truncate if too many
            limit = 50
            if len(simplified) > limit:
                return {
                    "total_count": len(simplified),
                    "showing_first": limit,
                    "nodes": simplified[:limit],
                    "warning": "ResultSet truncated. Use more specific filters."
                }
            return simplified
            
        if action == "edges":
            edge_type = kwargs.get("type")
            return self.api.get_edges(edge_type=edge_type)

        if action == "neighbors":
            if not node_id:
                return "Error: node_id is required for neighbors action. Use 'nodes' or 'edges' for global queries."
            
            # Try exact first
            try:
                # Pass kwargs to support relation_type and direction filters
                result = self.api.get_neighbors(node_id, **kwargs)
                
                if result is None:
                    return f"Error: Node '{node_id}' not found in graph."
                
                return result
            except Exception as e:
                return f"Error exploring neighbors: {e}"

        return f"Error: Unknown action '{action}'"
