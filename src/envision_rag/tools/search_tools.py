import re
from pathlib import Path
from typing import List, Dict, Any
import logging

class SearchTools:
    """
    Tools for exact pattern searching in the codebase (Grep-like).
    Handles mapping from internal IDs (68001.nvn) to Logical Paths (/1. utilities/...).
    """
    def __init__(self, root_dir: str = "scripts", mapping_path: str = "mapping.txt"):
        self.root_dir = Path(root_dir)
        self.mapping = {}
        self._load_mapping(mapping_path)

    def _load_mapping(self, mapping_path: str):
        """Loads ID -> Path mapping."""
        p = Path(mapping_path)
        if not p.exists():
            return
            
        try:
            with open(p, 'r') as f:
                for line in f:
                    if ',' in line:
                        parts = line.strip().split(',', 1)
                        if len(parts) == 2:
                            # 12345 -> /path/to/file
                            internal_id = parts[0].strip() + ".nvn"
                            logical_path = parts[1].strip()
                            self.mapping[internal_id] = logical_path
        except Exception as e:
            logging.warning(f"Failed to load mapping: {e}")

    def read_code(self, file_path_or_id: str, start_line: int = 1, end_line: int = -1) -> Dict[str, Any]:
        """
        Reads a specific range of lines from a file.
        Args:
            file_path_or_id: Logical Path (e.g. '/1. utilities/foo') or Internal ID (e.g. '12345.nvn').
            start_line: 1-based start line (inclusive).
            end_line: 1-based end line (inclusive). If -1, reads up to 50 lines from start.
        """
        # Resolve path
        # HARDENING: Handle cases where LLM passes "path, start, end" as a single string
        if "," in file_path_or_id:
            parts = [p.strip().strip("'").strip('"') for p in file_path_or_id.split(",")]
            file_path_or_id = parts[0]
            if len(parts) > 1 and str(parts[1]).isdigit():
                start_line = int(parts[1])
            if len(parts) > 2 and str(parts[2]).isdigit():
                end_line = int(parts[2])
        
        # Clean quotes
        file_path_or_id = file_path_or_id.strip("'").strip('"')
        
        target_path = None
        if file_path_or_id.endswith(".nvn"):
            target_path = self.root_dir / file_path_or_id
        else:
            # Reverse mapping search (Logical -> ID)
            # This is O(N) but N is small (250 files).
            for internal_id, logical in self.mapping.items():
                if logical == file_path_or_id:
                    target_path = self.root_dir / internal_id
                    break
            
        # Fallback: Try simple filename match if path fails (Handles concatenated paths)
        if not target_path or not target_path.exists():
            possible = self.root_dir / Path(file_path_or_id).name
            if possible.exists(): target_path = possible

        if not target_path or not target_path.exists():
            return {"error": f"File '{file_path_or_id}' not found."}

        try:
            with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            total_lines = len(lines)
            if start_line < 1: start_line = 1
            if end_line == -1: end_line = start_line + 50
            if end_line > total_lines: end_line = total_lines
            
            # Limit massive reads
            if end_line - start_line > 100:
                return {"error": "Range too large. Max 100 lines."}

            content = "".join(lines[start_line-1:end_line])
            
            return {
                "file": str(target_path),
                "lines": f"{start_line}-{end_line}",
                "content": content
            }
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}

    def grep_code(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs a regex search across all .nvn files.
        Args:
            query: The regex pattern to search for (e.g. "ReDispatchCycle =" or "table .*Locations").
        """
        results = []
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as e:
            return [{"error": f"Invalid Regex pattern: {e}"}]

        # Limit results to avoid flooding context
        MAX_RESULTS = 20
        matches_found = 0

        # Iterate all .nvn files
        if not self.root_dir.exists():
             return [{"error": f"Root directory '{self.root_dir}' not found."}]

        for file_path in self.root_dir.glob("*.nvn"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines):
                    if pattern.search(line):
                        # Capture context (1 line before/after)
                        start_idx = max(0, i - 1)
                        end_idx = min(len(lines), i + 2)
                        context = "".join(lines[start_idx:end_idx]).strip()
                        
                        # Translate ID to Logical Path
                        filename = file_path.name
                        logical_path = self.mapping.get(filename, filename)
                        
                        results.append({
                            "file": logical_path, # Return readable path
                            "internal_id": filename,
                            "line": i + 1,
                            "content": line.strip(),
                            "context": context
                        })
                        matches_found += 1
                        
                        if matches_found >= MAX_RESULTS:
                            break
            except Exception as e:
                pass # Skip unreadable files
                
            if matches_found >= MAX_RESULTS:
                        break
        
        if not results:
            return [{"message": f"No matches found for pattern '{query}'"}]

        return {
            "summary": f"Found {len(results)} matches for pattern '{query}'",
            "count": len(results),
            "results": results,
            "query": query
        }
