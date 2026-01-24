from typing import List, Dict, Optional, Literal
import uuid
import json

NodeType = Literal["root", "plan", "action"]
NodeStatus = Literal["pending", "active", "done", "failed"]

class Node:
    def __init__(self, goal: str, parent: Optional['Node'] = None, node_type: NodeType = "plan"):
        self.id = str(uuid.uuid4())[:8]
        self.goal = goal
        self.parent = parent
        self.children: List['Node'] = []
        self.status: NodeStatus = "pending"
        self.type = node_type
        self.reasoning = ""
        self.result = ""
        self.tool_call: Optional[Dict] = None

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "goal": self.goal,
            "status": self.status,
            "reasoning": self.reasoning,
            "tool_call": self.tool_call,
            "children": [c.to_dict() for c in self.children]
        }

class TreePlanner:
    """
    Manages the 'Tree of Thoughts'.
    """
    def __init__(self, root_goal: str):
        self.root = Node("Init", node_type="root")
        self.root.status = "active"
        self.current_node = self.root

    def add_subtask(self, parent_node: Node, goal: str) -> Node:
        child = Node(goal, parent=parent_node, node_type="plan")
        child.status = "active"  # Immediately focus the new subtask
        parent_node.children.append(child)
        return child

    def get_active_leaf(self) -> Node:
        """
        DFS to find the deepest 'active' node or activate the next 'pending' node.
        """
        current = self.root
        
        while True:
            # 1. Check for an already active child to descend
            active_children = [c for c in current.children if c.status == "active"]
            if active_children:
                current = active_children[0]
                continue
            
            # 2. If no active child, check for a pending child to start
            pending_children = [c for c in current.children if c.status == "pending"]
            if pending_children:
                child = pending_children[0]
                child.status = "active"
                return child
            
            # 3. No active or pending children -> This node is the leaf
            return current

    def get_plan_text(self) -> str:
        """
        Returns an indented text representation of the tree for the LLM.
        """
        buffer = ["\n### CURRENT PLAN (Tree):"]
        
        def _recurse(node: Node, depth: int):
            indent = "  " * depth
            icon = "⏳"
            if node.status == "done": icon = "✅"
            elif node.status == "failed": icon = "❌"
            elif node.status == "active": icon = "👉"
            
            buffer.append(f"{indent}{icon} {node.type.upper()}: {node.goal}")
            if node.reasoning:
                buffer.append(f"{indent}  (Reason: {node.reasoning[:100]}...)")
                
            for child in node.children:
                _recurse(child, depth + 1)

        _recurse(self.root, 0)
        return "\n".join(buffer)
