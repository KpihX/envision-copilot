from pathlib import Path
from typing import Dict, Any, Union
import json
from envision_copilot.utils.utils import smart_truncate

from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box
from rich.syntax import Syntax

from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box
from rich.syntax import Syntax

class CodeReader:
    def __init__(self, api=None, config: Dict[str, Any] = None, result: Any = None):
        self.api = api
        self.config = config or {}
        self.result = result

    def read_section(self, script_id: str, start_line: Union[int, str] = 1, end_line: Union[int, str] = 100) -> 'CodeReader':
        """
        Reads a specific range of lines from a script using its Graph ID.
        Requires EnvisionGraphAPI to resolve ID to Content/Path.
        """
        if not self.api:
            return CodeReader(self.api, self.config, result={"error": "Graph API not initialized. Cannot parse script_id."})

        if not script_id:
            return CodeReader(self.api, self.config, result={"error": "Missing 'script_id'."})

        # 1. Fetch Node
        node = self.api.get_node(script_id)
        if not node:
             # Fallback: Try searching loosely (e.g. if user passed a path instead of ID)
             search_result = self.api.search_nodes(script_id)
             matches_list = search_result.get("matches", [])
             
             if matches_list:
                 # Ambiguous match: Return list of candidates
                 candidates = [f"- {m.get('id')} ({m.get('path')})" for m in matches_list]
                 candidates_str = "\n".join(candidates)
                 return CodeReader(self.api, self.config, result={
                     "error": f"Node '{script_id}' not found exactly. Did you mean one of these?\n{candidates_str}"
                 })
             else:
                 return CodeReader(self.api, self.config, result={"error": f"Script ID '{script_id}' not found in Graph (and no approximate match)."})

        content_str = None
        source_desc = f"Node [{script_id}]"
        script_metadata = {
             "id": node.get("id"),
             "path": node.get("path"),
             "type": node.get("type"),
             "defined_symbols": node.get("defined_symbols", []),
             "imports": node.get("imports", []),
             "keywords": node.get("keywords", [])
        }

        # 2. Get Content (Graph Cache or Disk)
        if node.get("content"):
            content_str = node["content"]
            source_desc += " (from Graph)"
        elif node.get("path"):
            # Fallback to disk using path from GRAPH (trusted)
            fpath = Path(node["path"])
            if fpath.exists():
                try:
                     with open(fpath, 'r', encoding='utf-8') as f:
                         content_str = f.read()
                     source_desc += f" (from Disk: {fpath.name})"
                except Exception as e:
                     return CodeReader(self.api, self.config, result={"error": f"Failed to read file for node '{script_id}': {e}"})
            else:
                 return CodeReader(self.api, self.config, result={"error": f"File path for node '{script_id}' does not exist: {fpath}"})
        else:
             return CodeReader(self.api, self.config, result={"error": f"Node '{script_id}' has no content and no valid path."})

        # 3. Extract Range
        lines = content_str.splitlines()
        total_lines = len(lines)
        
        # Resolve Start
        start_val = start_line
        if isinstance(start_val, str):
            s = str(start_val).lower().strip()
            if s in ["start", "begin", "first"]: start_val = 1
            else:
                try: start_val = int(s)
                except: start_val = 1
        
        # Resolve End
        end_val = end_line
        if isinstance(end_val, str):
            s = str(end_val).lower().strip()
            if s in ["end", "last", "finish"]: end_val = total_lines
            else:
                try: end_val = int(s)
                except: end_val = total_lines

        start = max(1, int(start_val))
        end = min(total_lines, int(end_val))
        
        if start > end:
            return CodeReader(self.api, self.config, result={"error": f"Invalid range: {start}-{end} (Total: {total_lines})"})
            
        snippet_lines = lines[start-1:end]
        content = "\n".join(snippet_lines)
        
        result_data = {
            "file": source_desc,
            "range": f"{start}-{end}",
            "content": content,
            "extracted_lines": end - start + 1,
            "file_total_lines": total_lines,
            "metadata": script_metadata
        }
        return CodeReader(self.api, self.config, result=result_data)

    def __str__(self) -> str:
        """Format result for LLM context (NO TRUNCATION)."""
        if self.result is None:
            return "CodeReader Tool (Ready)"

        if "error" in self.result:
            return self.result["error"]
        
        buffer = [f"\n### 📄 Code Section:"]
        buffer.append(f"File: {self.result.get('file', 'N/A')}")
        buffer.append(f"Range: {self.result.get('range', 'N/A')} ({self.result.get('extracted_lines', 0)} lines)")
        
        # Add metadata if available
        metadata = self.result.get("metadata", {})
        if metadata:
            # Get configured max output
            max_items = self.config.get("presentation", {}).get("max_items", 30)
            
            buffer.append(f"\n**Script Metadata:**")
            if metadata.get("defined_symbols"):
                buffer.append(f"  Defined Symbols: {', '.join(metadata['defined_symbols'][:max_items])}")
            if metadata.get("keywords"):
                buffer.append(f"  Keywords: {', '.join(metadata['keywords'][:max_items])}")
            if metadata.get("imports"):
                buffer.append(f"  Imports: {', '.join(metadata['imports'][:max_items])}")
        
        buffer.append(f"\n```envision")
        buffer.append(self.result.get('content', ''))  # NO TRUNCATION
        buffer.append(f"```")
        
        return "\n".join(buffer)

    def print(self) -> Panel:
        """Format result for Rich UI (Smart Truncate)."""
        if self.result is None:
             from rich.text import Text
             return Panel(Text("CodeReader Tool Ready", style="green"), title="📄 CodeReader", border_style="green")
             
        if "error" in self.result:
            from rich.text import Text
            return Panel(Text(self.result["error"], style="red"), title="📄 Code Reader Error", border_style="red")
        
        limit = self.config.get("presentation", {}).get("max_lines", 100)
        list_limit = self.config.get("presentation", {}).get("max_items", 20)
        content_preview = smart_truncate(self.result.get("content", ""), max_lines=limit, max_items=list_limit)
        
        # Syntax Highlighting
        syntax = Syntax(content_preview, "envision", theme="monokai", line_numbers=True, start_line=int(self.result.get("range", "1-1").split("-")[0]))
        
        source = f"{self.result.get('file')} [{self.result.get('range')}]"
        return Panel(syntax, title=f"📄 {source}", border_style="blue")

    def to_dict(self):
        return self.result
