from pathlib import Path
from typing import Dict, Any, Union

class CodeReader:
    def __init__(self, api=None):
        self.api = api

    def read_section(self, file_path: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """
        Reads a specific range of lines from a file or graph node.
        """
        path = Path(file_path)
        content_str = None
        source_desc = str(path)

        # 1. Try resolving via Network API (Priority for DSL paths)
        if self.api:
             # Search by exact ID or Path
             matches = self.api.search_nodes(file_path)
             if matches:
                 best = matches[0]
                 if "content" in best and best["content"]:
                     content_str = best["content"]
                     source_desc = f"Graph Node: {best.get('id')} ({best.get('path')})"
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
                    return {"error": f"Failed to read file: {e}"}

        if not content_str:
             return {"error": f"File/Node not found: {file_path}"}
             
        # 3. Extract range
        lines = content_str.splitlines(keepends=True) # Keep newlines for accurate reconstruction? Or no?
        # Standard: splitlines() drops \n, so we re-join with \n. 
        # Actually splitlines() is better.
        lines = content_str.splitlines()
        
        total_lines = len(lines)
        start = max(1, start_line)
        end = min(total_lines, end_line)
        
        if start > end:
            return {"error": f"Invalid range: {start}-{end} (Total: {total_lines})"}
            
        snippet_lines = lines[start-1:end]
        content = "\n".join(snippet_lines)
        
        return {
            "file": source_desc,
            "range": f"{start}-{end}",
            "content": content,
            "extracted_lines": end - start + 1,
            "file_total_lines": total_lines
        }
