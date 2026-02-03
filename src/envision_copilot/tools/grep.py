from typing import Dict, Any, Union, List
import logging
import json
from envision_preprocess.api import EnvisionGraphAPI
from envision_copilot.utils.utils import smart_truncate

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
        self.max_lines = self.config.get("presentation", {}).get("max_lines", 50)
        self.max_items = self.config.get("presentation", {}).get("max_items", 20)
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
        
        # Generic Smart Truncation
        truncated = smart_truncate(self.result, max_lines=self.max_lines, max_items=self.max_items)
        return json.dumps(truncated, indent=2, ensure_ascii=False) if isinstance(truncated, (dict, list)) else str(truncated)

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
        
        # We manually build the tree here which is nice, but let's respect truncation limits
        # on the number of files shown.
        
        for pattern, data in patterns.items():
            total = data.get("total", 0)
            branch = root.add(f"'{pattern}' ({total} matches)")
            
            files = data.get("files", [])
            # Truncate file list if needed
            visible_files = files[:self.max_items]
            
            for file_info in visible_files: 
                f_name = file_info.get('file')
                count = file_info.get('count')
                branch.add(f"{f_name} [dim]({count})[/dim]")
            
            if len(files) > self.max_items:
                branch.add(f"[italic dim]... +{len(files)-self.max_items} more files[/italic dim]")
                
        return Panel(root, title="🔎 Grep Results", border_style="blue")
    
    def to_dict(self):
        return self.result if isinstance(self.result, dict) else {"error": self.result}
