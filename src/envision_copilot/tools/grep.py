from typing import Dict, Any, Union, List
import logging
import json
from envision_preprocess.api import EnvisionGraphAPI
from envision_copilot.utils.utils import smart_truncate

from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box

from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box

class Grep:
    """
    Wrapper around EnvisionGraphAPI to provide Grep/Search capabilities.
    Acts as both Runner and Result Container.
    """
    def __init__(self, config: Dict[str, Any] = None, result: Any = None):
        self.config = config or {}
        self.result = result
        try:
             # Lazy init or shared usage?
             # If result is None, we are a runner
             if result is None:
                 self.api = EnvisionGraphAPI()
             else:
                 self.api = None
        except Exception as e:
            logging.error(f"Failed to initialize Grep: {e}")
            self.api = None

    def search(self, pattern: str, **kwargs) -> 'Grep':
        """
        Scan all nodes for occurrences of specific pattern.
        Returns a NEW instance of Grep containing the result.
        """
        if not self.api:
             return Grep(self.config, result="Error: Graph API not initialized.")

        try:
             # Use grep_search from API
             result_data = self.api.grep_search([pattern])
             return Grep(self.config, result=result_data)
        except Exception as e:
             return Grep(self.config, result=f"Error executing grep_search: {e}")

    def __str__(self) -> str:
        """Format result for LLM context."""
        if self.result is None:
             return "Grep Tool (Ready)"

        if isinstance(self.result, str):
            return self.result
        
        limit = self.config.get("presentation", {}).get("max_output_lines", 100)
        buffer = ["\n### 🔎 Grep Search Results:"]
        
        patterns = self.result.get("patterns", {})
        for pattern, data in patterns.items():
            total = data.get("total", 0)
            files = data.get("files", [])
            buffer.append(f"\nPattern: '{pattern}' → {total} occurrences in {len(files)} files")
            
            for file_info in files[:10]:  # Top 10 files
                buffer.append(f"  {file_info.get('file', 'N/A')}: {file_info.get('count', 0)} matches")
        
        return "\n".join(buffer)

    def print(self) -> Panel:
        """Format result for Rich UI display."""
        if self.result is None:
            from rich.text import Text
            return Panel(Text("Grep Tool Ready", style="green"), title="🔎 Grep", border_style="green")
            
        if isinstance(self.result, str):
            from rich.text import Text
            return Panel(Text(self.result, style="red"), title="🔎 Grep Error", border_style="red")
        
        patterns = self.result.get("patterns", {})
        root = Tree("🔎 [bold]Grep Search[/bold]")
        
        # Get configured max output
        max_output = self.config.get("presentation", {}).get("max_output", 30)

        for pattern, data in patterns.items():
            total = data.get("total", 0)
            branch = root.add(f"'{pattern}' ({total} matches)")
            
            files = data.get("files", [])
            for file_info in files[:max_output]: # Limit UI listing
                branch.add(f"{file_info.get('file')} [dim]({file_info.get('count')})[/dim]")
            
            if len(files) > max_output:
                branch.add(f"[italic dim]... +{len(files)-max_output} more files[/italic dim]")
                
        return Panel(root, title="🔎 Grep Results", border_style="blue")
    
    def to_dict(self):
        return self.result if isinstance(self.result, dict) else {"error": self.result}
