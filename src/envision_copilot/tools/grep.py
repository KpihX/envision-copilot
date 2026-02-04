"""
Grep Tool - Pattern-Based Text Search
=====================================

Provides regex-based text search across script contents.
Uses EnvisionGraphAPI.grep() for actual searching.

Unlike RAG (semantic), grep finds exact pattern matches.
Best for finding specific identifiers, keywords, or code patterns.
"""

from typing import Optional, List

from .base import BaseTool, ToolResult


class GrepTool(BaseTool):
    """
    Tool for regex-based text search in Envision scripts.
    
    Searches for pattern matches across all scripts.
    Returns matching files with occurrence counts and previews.
    
    Best for:
    - Finding exact identifiers (table names, variables)
    - Locating specific keywords or syntax
    - Counting pattern occurrences
    
    For semantic/conceptual search, use the rag tool instead.
    """
    
    name = "grep"
    description = "Search for regex patterns in Envision script contents"
    
    def execute(
        self,
        pattern: str,
        node_types: Optional[List[str]] = None,
        **_
    ) -> ToolResult:
        """
        Search for pattern matches.
        
        Args:
            pattern: Regex pattern to search
            node_types: Node types to search in
            
        Returns:
            ToolResult with matching files and counts
        """
        if not self.api:
            return self._error("Graph API not initialized")
        
        if not pattern:
            return self._error("pattern is required")
        
        try:
            result = self.api.grep(pattern, node_types=node_types)
            
            return self._success(result)
            
        except Exception as e:
            return self._error(f"Grep failed: {str(e)}")
