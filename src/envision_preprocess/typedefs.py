from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

class NodeType(str, Enum):
    SCRIPT = "script"
    FILE = "file"
    TABLE = "table"
    VAR = "var"
    FUNCTION = "function"

class EdgeType(str, Enum):
    READS = "reads"
    WRITES = "writes"
    IMPORTS = "imports"
    DEFINES = "defines"
    USES = "uses"
    EXPORT = "export"

@dataclass
class Node:
    id: str           # Unique Identifier (Logical Path or Name)
    type: NodeType
    path: Optional[str] = None # Physical path
    content: Optional[str] = None # Resolved content
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    source: str
    target: str
    type: EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Network:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def to_dict(self):
        return {
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
            "edges": [e.__dict__ for e in self.edges]
        }
