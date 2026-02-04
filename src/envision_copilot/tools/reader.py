"""
Reader Tool - Content Access
==============================

Provides access to script and function contents with line range selection.
Delegates to EnvisionGraphAPI.read() for actual content retrieval.

Key features:
- Read full file or specific line ranges
- Read functions directly by function ID
- Returns execution_order for workflow navigation
- Includes line count and range metadata
"""

from typing import Optional

from .base import BaseTool, ToolResult


class ReaderTool(BaseTool):
    """
    Tool for reading Envision script or function contents.
    
    Retrieves content by ID with optional line range.
    Works with:
    - Scripts: ID like '68006' or path
    - Functions: ID like '67992::func::StockEvol'
    
    The response includes `execution_order` indicating where
    this script sits in the execution workflow.
    """
    
    name = "reader"
    description = "Read Envision script or function content by ID with optional line range"
    
    def execute(
        self,
        node_id: str = None,
        script_id: str = None,  # Legacy alias
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **_
    ) -> ToolResult:
        """
        Read script or function content.
        
        Args:
            node_id: ID of the node to read (script or function)
            script_id: Legacy alias for node_id
            start_line: Starting line (1-indexed)
            end_line: Ending line (inclusive)
            
        Returns:
            ToolResult with content and metadata
        """
        if not self.api:
            return self._error("Graph API not initialized")
        
        # Support both node_id and legacy script_id
        target_id = node_id or script_id
        if not target_id:
            return self._error("node_id is required")
        
        try:
            result = self.api.read(
                node_id=target_id,
                start_line=start_line,
                end_line=end_line
            )
            
            # Check for errors in API response (both full and lite modes)
            if result.get("stats", {}).get("error") or result.get("error"):
                return self._error(result.get("error", "Read failed"))
            
            # In lite mode, result has {id, name, content} directly
            # In full mode, result has {stats, node: {...}}
            # Both are valid successful responses
            if "node" not in result and "id" not in result:
                return self._error(f"Node not found: {target_id}")
            
            return self._success(result)
            
        except Exception as e:
            return self._error(f"Read failed: {str(e)}")
