from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
import networkx as nx
import json

class NodeType(str, Enum):
    SCRIPT = "script"
    TABLE = "table"   # Envision Table (e.g. "Orders")
    VARIABLE = "variable" # Column or Var (e.g. "Orders.Amount")
    FILE = "file"     # Physical file (e.g. "/Clean/Items.ion")

class EdgeType(str, Enum):
    READS = "reads"       # Script -> File
    WRITES = "writes"     # Script -> File
    DEFINES = "defines"   # Script -> Table/Variable
    DEPENDS_ON = "depends_on" # Variable -> Variable (Lineage)
    IMPORTS = "imports"   # Script -> Module

@dataclass
class Node:
    id: str  # Unique ID
    type: NodeType
    path: Optional[str] = None # File path if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type.value,
            "path": self.path,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            path=data.get("path"),
            metadata=data.get("metadata", {})
        )

@dataclass
class Edge:
    source: str
    target: str
    type: EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data):
        return cls(
            source=data["source"],
            target=data["target"],
            type=EdgeType(data["type"]),
            metadata=data.get("metadata", {})
        )

class DependencyGraph:
    """
    Wrapper around NetworkX DiGraph to store Envision dependencies.
    """
    def __init__(self):
        self._graph = nx.DiGraph()

    def add_node(self, node: Node):
        self._graph.add_node(node.id, **node.to_dict())

    def add_edge(self, edge: Edge):
        self._graph.add_edge(edge.source, edge.target, type=edge.type.value, **edge.metadata)

    def get_readers(self, file_id: str) -> List[str]:
        """Return list of script IDs that READ the given file."""
        readers = []
        if not self._graph.has_node(file_id):
            return []
            
        # In DiGraph: Script --reads--> File
        # So we look for Predecessors of File with edge type 'reads'
        for pred in self._graph.predecessors(file_id):
            edge_data = self._graph.get_edge_data(pred, file_id)
            if edge_data and edge_data.get("type") == EdgeType.READS.value:
                readers.append(pred)
        return readers

    def get_writers(self, file_id: str) -> List[str]:
        """Return list of script IDs that WRITE the given file."""
        writers = []
        if not self._graph.has_node(file_id):
            return []
            
        # In DiGraph: Script --writes--> File
        for pred in self._graph.predecessors(file_id):
            edge_data = self._graph.get_edge_data(pred, file_id)
            if edge_data and edge_data.get("type") == EdgeType.WRITES.value:
                writers.append(pred)
        return writers

    def save(self, path: str):
        """Save to JSON node-link data format"""
        data = nx.node_link_data(self._graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load from JSON"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._graph = nx.node_link_graph(data)

    def stats(self):
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges()
        }
