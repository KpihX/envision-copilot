from typing import List, Dict, Any
import json
import uuid
from dataclasses import dataclass, field
from envision_copilot.utils.utils import smart_truncate

from rich.table import Table
from rich import box
from rich.markdown import Markdown
from rich.panel import Panel

@dataclass
class MemoryItem:
    """
    Represents a single unit of memory (observation/fact).
    """
    id: str
    tool_name: str
    tool_args: Dict[str, Any]
    compact_view: str  # Optimized for LLM Context
    full_content: Any  # Full raw data for Appendix
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "id": self.id,
            "tool": self.tool_name,
            "args": self.tool_args,
            "compact": self.compact_view,
            "content": self.full_content
        }

class Memory:
    """
    Interactive Memory System.
    The LLM explicitly decides what to KEEP and what to DISCARD at each step.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.items: List[MemoryItem] = []
        self._history_archive: List[MemoryItem] = [] # To keep track even if discarded from working memory

    def add_observation(self, tool_name: str, tool_args: Dict, result: Any, compact_view: str = None) -> MemoryItem:
        """
        Adds a new observation to the memory.
        """
        # Auto-generate compact view if not provided
        if compact_view is None:
            limit = self.config.get("presentation", {}).get("max_output_lines", 100)
            compact_data = {
                "tool": tool_name,
                "args": tool_args,
                "result": smart_truncate(result, max_lines=limit)
            }
            compact_view = json.dumps(compact_data, ensure_ascii=False)

        item = MemoryItem(
            id=str(uuid.uuid4()), # Short ID for LLM usage
            tool_name=tool_name,
            tool_args=tool_args,
            compact_view=compact_view,
            full_content=result
        )
        self.items.append(item)
        self._history_archive.append(item)
        return item

    def remove_by_indices(self, indices: List[int]):
        """
        Remove items at specific indices (0-based) from active memory.
        """
        # Sort indices descending to avoid shifting issues
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.items):
                self.items.pop(idx)

    def update_memory(self, keep_ids: List[str]):
        """
        Updates the working memory by keeping ONLY the specified IDs.
        (Legacy method, kept for compatibility if needed)
        """
        cleaned_ids = [k.strip() for k in keep_ids]
        self.items = [item for item in self.items if item.id in cleaned_ids]

    def __str__(self) -> str:
        """
        Returns the formatted string for the LLM Context with INDICES.
        """
        if not self.items:
            return "No active memory."

        buffer = ["\n### 🧠 ACTIVE MEMORY (Facts you decided to keep):"]
        for i, item in enumerate(self.items):
            buffer.append(f"[{i}] (ID: {item.id}) {item.compact_view}")
        
        return "\n".join(buffer)

    def print(self, title: str = None) -> Panel:
        """
        Returns a Rich Panel containing a formatted JSON view of the memory.
        Used for both standard UI and Appendix.
        """
        display_title = title or "💾 Active Memory"
        
        if not self.items:
            return Panel("No items in memory.", title=display_title, border_style="bold cyan")

        # Build structure with FULL CONTENT (not compact)
        struct = []
        for item in self.items:
            struct.append({
                "id": item.id,
                "tool": item.tool_name,
                "args": item.tool_args,
                "content": item.full_content 
            })
            
        # Serialize to JSON
        json_str = json.dumps(struct, indent=2, ensure_ascii=False)
        
        # Apply Smart Truncation on the JSON string directly
        limit = self.config.get("presentation", {}).get("max_output_lines", 500)
        truncated_json = smart_truncate(json_str, max_lines=limit)
        
        return Panel(Markdown(f"```json\n{truncated_json}\n```"), title=display_title, border_style="blue")
