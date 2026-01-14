#!/usr/bin/env python3
"""
Test Vector Index CLI.
Query the vector index and display top semantic matches.

Usage:
    uv run test-vector -q "stock calculation" -n 5
    uv run test-vector -q "forecast autodiff" -n 10
"""
import argparse
import yaml
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from envision_rag.index.vector_tools import VectorTools


console = Console()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        prog="test-vector",
        description="Test the vector index with semantic queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run test-vector -q "stock calculation" -n 5
  uv run test-vector -q "forecast function" -n 10
  uv run test-vector -q "dispatch cycle parameter" --full

Query is a natural language description of what you're looking for.
Results are ranked by semantic similarity.
        """
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        metavar="QUERY",
        help="Semantic search query (natural language)"
    )
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=5,
        metavar="N",
        help="Number of results to show (default: 5)"
    )
    parser.add_argument(
        "-f", "--full",
        action="store_true",
        help="Show full chunk content (not truncated)"
    )
    
    args = parser.parse_args()
    
    # Load vector index
    console.print(Panel.fit(
        f"[bold purple]🧠 Vector Query[/bold purple]\n"
        f"[dim]Query: {args.query}[/dim]",
        border_style="purple"
    ))
    
    try:
        vector_tools = VectorTools()
    except FileNotFoundError:
        console.print("[red]❌ Vector index not found. Run: uv run build -v[/red]")
        return
    
    # Execute query
    results = vector_tools.search_code(args.query, k=args.num)
    
    console.print(f"\n📊 Found [bold]{len(results)}[/bold] matching chunks\n")
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    # Display results
    for i, result in enumerate(results, 1):
        # Parse the result string
        lines = result.strip().split("\n")
        file_line = lines[0] if lines else "Unknown"
        type_line = lines[1] if len(lines) > 1 else ""
        
        # Extract content
        content_start = result.find("Code:\n")
        if content_start != -1:
            code = result[content_start + 6:].strip()
        else:
            code = result
        
        if not args.full and len(code) > 1000:
            code = code[:1000] + "\n... (use --full for complete)"
        
        # Display panel
        title = f"[bold]#{i}[/bold] {file_line}"
        console.print(Panel(
            Syntax(code, "python", theme="monokai", word_wrap=True),
            title=title,
            subtitle=type_line,
            border_style="purple"
        ))


if __name__ == "__main__":
    main()
