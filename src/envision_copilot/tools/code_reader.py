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

    def read_section(self, file_path: str, start_line: Union[int, str] = 1, end_line: Union[int, str] = 100) -> 'CodeReader':
        """
        Reads a specific range of lines from a file or graph node.
        NO TRUNCATION: Returns the full requested range.
        Returns a NEW instance of CodeReader containing the result.
        """
        path = Path(file_path)
        content_str = None
        source_desc = str(path)
        script_metadata = {}

        # 1. Try resolving via Network API (Priority for DSL paths)
        if self.api:
             matches = self.api.search_nodes(file_path)
             if matches:
                 best = matches[0]
                 if "content" in best and best["content"]:
                     content_str = best["content"]
                     source_desc = f"Graph Node: {best.get('id')} ({best.get('path')})"
                     
                     # Extract metadata (keywords, defined symbols)
                     script_metadata = {
                         "id": best.get("id"),
                         "path": best.get("path"),
                         "type": best.get("type"),
                         "defined_symbols": best.get("defined_symbols", []),
                         "imports": best.get("imports", []),
                         "keywords": best.get("keywords", [])
                     }
                 elif "path" in best and Path(best["path"]).exists():
                     path = Path(best["path"])

        # 2. Try Physical File (Absolute or Relative)
        if not content_str:
            if not path.exists():
                path = Path.cwd() / file_path
                
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content_str = f.read()
                    source_desc = str(path)
                except Exception as e:
                    return CodeReader(self.api, self.config, result={"error": f"Failed to read file: {e}"})

        if not content_str:
             return CodeReader(self.api, self.config, result={"error": f"File/Node not found: {file_path}"})
             
        # 3. Extract range (NO TRUNCATION)
        lines = content_str.splitlines()
        total_lines = len(lines)
        
        # Resolve Start
        start_val = start_line
        if isinstance(start_val, str):
            s = str(start_val).lower().strip()
            if s in ["start", "begin", "first"]:
                start_val = 1
            elif s in ["end", "last"]:
                start_val = total_lines
            else:
                try: start_val = int(s)
                except: start_val = 1
        
        # Resolve End
        end_val = end_line
        if isinstance(end_val, str):
            s = str(end_val).lower().strip()
            if s in ["end", "last", "finish"]:
                end_val = total_lines
            elif s in ["start", "begin"]:
                end_val = 1
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
            "metadata": script_metadata  # Include script metadata
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
            max_output = self.config.get("presentation", {}).get("max_output", 30)
            
            buffer.append(f"\n**Script Metadata:**")
            if metadata.get("defined_symbols"):
                buffer.append(f"  Defined Symbols: {', '.join(metadata['defined_symbols'][:max_output])}")
            if metadata.get("keywords"):
                buffer.append(f"  Keywords: {', '.join(metadata['keywords'][:max_output])}")
            if metadata.get("imports"):
                buffer.append(f"  Imports: {', '.join(metadata['imports'][:max_output])}")
        
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
        
        limit = self.config.get("presentation", {}).get("max_output_lines", 100)
        content_preview = smart_truncate(self.result.get("content", ""), max_lines=limit)
        
        # Syntax Highlighting
        syntax = Syntax(content_preview, "envision", theme="monokai", line_numbers=True, start_line=int(self.result.get("range", "1-1").split("-")[0]))
        
        source = f"{self.result.get('file')} [{self.result.get('range')}]"
        return Panel(syntax, title=f"📄 {source}", border_style="blue")

    def to_dict(self):
        return self.result
