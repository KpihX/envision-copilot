from typing import List, Dict, Optional, Any
import json
from enum import Enum
from dataclasses import dataclass, field
from envision_copilot.utils.utils import smart_truncate
from rich.panel import Panel
from rich.tree import Tree
from rich import box

class NodeStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Node:
    goal: str
    id: str = "0"
    status: NodeStatus = NodeStatus.PENDING
    
    # Execution details
    tool_name: str = ""
    tool_args: Dict = field(default_factory=dict)
    reasoning: str = ""
    result: str = ""
    depth: int = 0

    def to_dict(self):
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status,
            "tool": self.tool_name,
            "args": self.tool_args,
            "reasoning": self.reasoning
        }

class Planner:
    """
    Linearized BFS Planner.
    Manages execution in 'Layers' (Depth).
    History is a list of layers (List[List[Node]]).
    """
    def __init__(self, root_goal: str, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_depth = self.config.get("agent", {}).get("constraints", {}).get("max_depth", 5)
        self.max_branches = self.config.get("agent", {}).get("constraints", {}).get("max_branches", 2)
        
        # History of layers. layers[0] is depth 0.
        self.layers: List[List[Node]] = []
        self.current_depth = 0
        
        # Simple Integer ID Counter
        self._node_counter = 0
        self.root: Node = Node(
            id=str(self._node_counter),
            goal="ROOT GOAL",
            tool_name="ROOT",
            tool_args={},
            result="INIT",
            depth=0
        )
        self._node_counter += 1
        
        # Linear tracking of the plan order
        self.nodes_sequence: List[Node] = [self.root]
        self.layers.append([self.root])

    def propose_next_layer(self, layer_proposals: List[Dict]) -> List[Node]:
        """
        Creates the next layer of nodes based on LLM suggestions.
        Input: List of dicts {goal, tool, args}
        Respects max_branches.
        """
        if self.current_depth >= self.max_depth:
            return []

        # Enforce max branches limit
        effective_proposals = layer_proposals[:self.max_branches]
        
        new_layer = []
        for prop in effective_proposals:
            node = Node(
                id=str(self._node_counter),
                goal=prop.get("goal", "Unknown Goal"),
                tool_name=prop.get("tool", ""),
                tool_args=prop.get("args", {})
            )
            self._node_counter += 1
            new_layer.append(node)
            
        if new_layer:
            self.layers.append(new_layer)
            self.current_depth += 1
            
        return new_layer

    def get_current_layer(self) -> List[Node]:
        """Returns the nodes in the current deepest layer."""
        if not self.layers:
            return []
        return self.layers[-1]

    def get_next_pending_node(self) -> Optional[Node]:
        """
        Returns the first PENDING node in the current layer.
        """
        current_layer = self.get_current_layer()
        for node in current_layer:
            if node.status == NodeStatus.PENDING:
                node.status = NodeStatus.ACTIVE
                return node
        return None

    def has_pending_nodes(self) -> bool:
        """Checks if there are any PENDING nodes in the current layer without modifying them."""
        current_layer = self.get_current_layer()
        for node in current_layer:
            if node.status == NodeStatus.PENDING:
                return True
        return False

    def is_layer_complete(self) -> bool:
        """Checks if all nodes in the current layer are terminal (DONE/FAILED/CANCELLED)."""
        current_layer = self.get_current_layer()
        if not current_layer:
            return True
        return all(node.status in [NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.CANCELLED] for node in current_layer)

    def mark_done(self, node: Node, reasoning: str = ""):
        node.status = NodeStatus.DONE
        node.reasoning = reasoning

    def mark_failed(self, node: Node, reasoning: str = ""):
        node.status = NodeStatus.FAILED
        node.reasoning = reasoning
        
    def mark_cancelled(self, node: Node, reasoning: str = ""):
        node.status = NodeStatus.CANCELLED
        node.reasoning = reasoning

    def __str__(self) -> str:
        """
        Returns a text representation for LLM context.
        Includes depth tracking: i/max_depth
        """
        limit = self.config.get("presentation", {}).get("max_lines", 100)
        
        buffer = [f"\n### 🗺️ EXPLORATION HISTORY (Depth: {self.current_depth}/{self.max_depth}):"]
        
        for i, layer in enumerate(self.layers):
            if i == 0: continue # Skip init layer usually
            
            buffer.append(f"\n**Layer {i}:**")
            for node in layer:
                icon = "⏳"
                if node.status == NodeStatus.DONE: icon = "✅"
                elif node.status == NodeStatus.FAILED: icon = "❌"
                elif node.status == NodeStatus.CANCELLED: icon = "🚫"
                elif node.status == NodeStatus.ACTIVE: icon = "👉"
                
                buffer.append(f"  {icon} [{node.id}] {node.goal}")
                if node.tool_name:
                    buffer.append(f"     [Tool: {node.tool_name}]")
                if node.reasoning:
                    # Use smart_truncate for reasoning
                    limit = self.config.get("presentation", {}).get("max_lines", 100)
                    list_limit = self.config.get("presentation", {}).get("max_items", 20)
                    truncated_reasoning = smart_truncate(node.reasoning, max_lines=limit, max_items=list_limit)
                    if isinstance(truncated_reasoning, str):
                        buffer.append(f"     (Reason: {truncated_reasoning})")
        
        return "\n".join(buffer)

    def print(self) -> Panel:
        """Returns a Rich Panel containing the visual tree of the plan for UI."""
        root = Tree(f"[bold gold1]🧭 Exploration Plan (Depth {self.current_depth}/{self.max_depth})[/bold gold1]")
        
        for i, layer in enumerate(self.layers):
            if i == 0: continue # Skip root prompt layer
            
            layer_branch = root.add(f"[bold]Layer {i}[/bold]")
            for node in layer:
                status_icon = "⏳"
                style = "dim"
                if node.status == NodeStatus.DONE: 
                    status_icon = "✅"
                    style = "green"
                elif node.status == NodeStatus.FAILED:
                    status_icon = "❌"
                    style = "red"
                elif node.status == NodeStatus.ACTIVE:
                    status_icon = "👉"
                    style = "bold cyan"
                
                label = f"{status_icon} {node.goal}"
                if node.tool_name:
                    label += f" [dim]({node.tool_name})[/dim]"
                
                layer_branch.add(f"[{style}]{label}[/{style}]")
                
        return Panel(root, border_style="gold1", title="Plan Status")
