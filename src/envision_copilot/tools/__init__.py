"""
Envision Copilot Tools
======================

Unified tool interface for LLM agents.

Tools:
- GraphTool: Navigate dependency graph (tree, node, neighbors, edges, search)
- ReaderTool: Read script content with line ranges
- RagTool: Semantic search using embeddings
- GrepTool: Pattern-based text search

All tools inherit from BaseTool and return ToolResult.
"""

from .base import BaseTool, ToolResult
from .definitions import TOOLS, get_tool_definitions, get_tool_by_name, get_tools_summary
from .graph import GraphTool
from .reader import ReaderTool
from .rag import RagTool
from .grep import GrepTool

__all__ = [
    # Base classes
    "BaseTool",
    "ToolResult",
    # Tool classes
    "GraphTool",
    "ReaderTool", 
    "RagTool",
    "GrepTool",
    # Definitions
    "TOOLS",
    "get_tool_definitions",
    "get_tool_by_name",
    "get_tools_summary",
]