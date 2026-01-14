import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..utils import ConfigLoader

class StructuralTools:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        net_path = Path(self.config.get("paths", {}).get("network_file", "data/network/network.json"))
        
        if not net_path.exists():
            raise FileNotFoundError(f"Network file not found at {net_path}")
            
        with open(net_path, 'r') as f:
            self.data = json.load(f)
            self.nodes = self.data.get("nodes", {})
            self.edges = self.data.get("edges", [])

    def scan_network_context(self, file_path_or_id: str) -> str:
        """Finds what a script reads/writes/imports."""
        # Normalize ID
        target_id = None
        for nid in self.nodes:
            if file_path_or_id in nid:
                target_id = nid
                break
        
        if not target_id:
            return f"Node '{file_path_or_id}' not found."

        # Find connections
        reads = []
        writes = []
        imports = []
        
        for edge in self.edges:
            if edge["source"] == target_id:
                if edge["type"] == "reads": reads.append(edge["target"])
                if edge["type"] == "writes": writes.append(edge["target"])
                if edge["type"] == "imports": imports.append(edge["target"])
                
        return json.dumps({
            "id": target_id,
            "reads": reads,
            "writes": writes,
            "imports": imports
        }, indent=2)

    def find_producers(self, output_path: str) -> str:
        """Finds which script writes to a given path (e.g. table or file)."""
        producers = []
        for edge in self.edges:
            if edge["target"] == output_path and edge["type"] in ["writes", "defines"]:
                producers.append(edge["source"])
                
        if not producers:
             # Try soft match
             for nid in self.nodes:
                 if output_path in nid:
                     return self.find_producers(nid)
                     
        return f"Producers for {output_path}: {producers}"
