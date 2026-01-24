from typing import Dict, Any, Union, List
import logging
from envision_preprocess.api import EnvisionGraphAPI

class GrepTools:
    """
    Wrapper around EnvisionGraphAPI to provide Grep/Search capabilities.
    """
    def __init__(self, config: Dict[str, Any] = None):
        try:
            self.api = EnvisionGraphAPI()
        except Exception as e:
            logging.error(f"Failed to initialize GrepTools: {e}")
            self.api = None

    def search(self, pattern: str, node_type: str = None, node_ids: List[str] = None) -> Union[List[Dict], str]:
        """
        Search for a regex pattern within the CONTENT of nodes.
        Filters:
          - node_type: Only search nodes of this type (e.g., 'script', 'function', 'var').
          - node_ids: Only search within this specific list of node IDs.
        """
        if not self.api:
            return "Error: Graph API not initialized."
            
        try:
            matches = self.api.grep_nodes(pattern, node_type=node_type, node_ids=node_ids)
            
            # Format output for Agent
            if not matches:
                return "No matches found."
            
            # If error returned
            if isinstance(matches, list) and len(matches) > 0 and "error" in matches[0]:
                return matches[0]["error"]
                
            # Limit results
            limit = 50
            results = matches[:limit]
            
            summary = []
            for m in results:
                summary.append(f"[{m.get('type')}] {m.get('id')} ({m.get('path')})")
                
            output = f"Found {len(matches)} matches (showing first {len(summary)}):\n" + "\n".join(summary)
            if len(matches) > limit:
                output += f"\n...and {len(matches) - limit} more."
                
            return output

        except Exception as e:
            return f"Error executing grep_search: {e}"
