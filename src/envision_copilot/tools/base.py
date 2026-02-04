"""
Base classes for Copilot Tools (MCP-Ready)
==========================================

Provides the foundational contracts for all tools:
- ToolResult: Uniform result container with LLM/Rich/Dict outputs
- BaseTool: Abstract base class defining the tool interface

All tools inherit from BaseTool and return ToolResult instances.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
import json

from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Console

from envision_copilot.utils.utils import smart_truncate
from envision_copilot.utils.config_loader import ConfigLoader


# Load truncation settings from config.yaml
_config = ConfigLoader.load_config()
_presentation = _config.get("presentation", {})
DEFAULT_MAX_LINES = _presentation.get("max_lines", 200)
DEFAULT_MAX_ITEMS = _presentation.get("max_items", 30)


@dataclass
class ToolResult:
    """
    Uniform container for all tool execution results.
    
    Provides consistent output formatting for:
    - LLM consumption (to_llm_string)
    - Human display (to_rich_panel)
    - Serialization (to_dict)
    
    Attributes:
        success: Whether the tool executed successfully
        data: The result data (dict, list, or primitive)
        error: Error message if success=False
        tool_name: Name of the tool that produced this result
        action: Specific action/method called (for multi-action tools)
        metadata: Additional context (timing, counts, etc.)
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    tool_name: str = ""
    action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Truncation config (can be set per-result or use defaults)
    max_lines: int = DEFAULT_MAX_LINES
    max_items: int = DEFAULT_MAX_ITEMS
    
    def __str__(self) -> str:
        """
        Format result for LLM consumption with smart truncation.
        
        Returns a clean, parseable string representation
        suitable for inclusion in LLM context.
        """
        if not self.success:
            return f"Error [{self.tool_name}]: {self.error}"
        
        header = f"[{self.tool_name}"
        if self.action:
            header += f".{self.action}"
        header += "]"
        
        # Apply smart truncation before serializing
        truncated_data = smart_truncate(self.data, max_lines=self.max_lines, max_items=self.max_items)
        
        # Format data as readable JSON
        if isinstance(truncated_data, (dict, list)):
            data_str = json.dumps(truncated_data, indent=2, ensure_ascii=False)
        else:
            data_str = str(truncated_data)
        
        return f"{header}\n{data_str}"
    
    def print(self) -> None:
        """
        Print result to console using Rich formatting with smart truncation.
        
        Displays a Panel with syntax-highlighted JSON content.
        """
        title = self.tool_name
        if self.action:
            title += f".{self.action}"
        
        if not self.success:
            panel = Panel(
                f"[red]Error:[/red] {self.error}",
                title=f"❌ {title}",
                border_style="red"
            )
        else:
            # Apply smart truncation before display
            truncated_data = smart_truncate(self.data, max_lines=self.max_lines, max_items=self.max_items)
            
            # Format data with syntax highlighting
            if isinstance(truncated_data, (dict, list)):
                content = Syntax(
                    json.dumps(truncated_data, indent=2, ensure_ascii=False),
                    "json",
                    theme="monokai",
                    word_wrap=True
                )
            else:
                content = str(truncated_data)
            
            panel = Panel(
                content,
                title=f"✅ {title}",
                border_style="green"
            )
        
        console = Console()
        console.print(panel)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize result to dictionary.
        
        Useful for JSON serialization or logging.
        """
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "action": self.action,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }


class BaseTool(ABC):
    """
    Abstract base class for all Copilot tools.
    
    Defines the contract that all tools must implement:
    - name: Unique identifier for the tool
    - description: Short description for LLM
    - execute(): Main entry point returning ToolResult
    
    Tool definitions (parameters schema, documentation) are centralized
    in definitions.py to avoid duplication.
    
    Tools receive shared dependencies (API, config) via __init__.
    """
    
    # Class attributes to be overridden by subclasses
    name: str = "base_tool"
    description: str = "Base tool description"
    
    def __init__(self, api=None, config: Optional[Dict] = None, **kwargs):
        """
        Initialize tool with shared dependencies.
        
        Args:
            api: EnvisionGraphAPI instance (shared across tools)
            config: Tool-specific configuration
            **kwargs: Additional dependencies (retriever, etc.)
        """
        self.api = api
        self.config = config or {}
        # Store additional dependencies
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        This is the main entry point called by the agent.
        Must be implemented by all subclasses.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with success/failure and data
        """
        pass
    
    def _success(self, data: Any, action: str = "", **metadata) -> ToolResult:
        """Helper to create successful ToolResult."""
        return ToolResult(
            success=True,
            data=data,
            tool_name=self.name,
            action=action,
            metadata=metadata
        )
    
    def _error(self, message: str, action: str = "") -> ToolResult:
        """Helper to create error ToolResult."""
        return ToolResult(
            success=False,
            error=message,
            tool_name=self.name,
            action=action
        )
