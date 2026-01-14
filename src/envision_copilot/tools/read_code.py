import json
from pathlib import Path
from ..utils import ConfigLoader

class CodeReader:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        net_path = Path(self.config.get("paths", {}).get("network_file", "data/network/network.json"))
        
        with open(net_path, 'r') as f:
            self.data = json.load(f)
            self.nodes = self.data.get("nodes", {})

    def read_code(self, script_id: str) -> str:
        """Returns the content of a script node."""
        # Exact match
        node = self.nodes.get(script_id)
        if node and node.get("content"):
             return node["content"]
             
        # Soft match
        for nid, n in self.nodes.items():
            if script_id in nid and n.get("content"):
                return f"# Content of {nid}\n{n['content']}"
                
        return f"Starting code for '{script_id}' not found or empty."
