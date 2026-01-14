#!/usr/bin/env python3
"""
Test Graph Index CLI.
Query the graph index and display relationships.

Usage:
    uv run test-graph -q "read /Clean/Items.ion" -n 10
    uv run test-graph -q "write FcItems" -n 5
    uv run test-graph -q "import" -n 20
"""
import argparse
import yaml
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from envision_rag.graph.graph_types import DependencyGraph
from envision_rag.tools.graph_tools import GraphTools


console = Console()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        prog="test-graph",
        description="Test the graph index with sample queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run test-graph -q "read /Clean/Items.ion" -n 10
  uv run test-graph -q "write FcItems" -n 5
  uv run test-graph -q "import" --all
  uv run test-graph -q "any Items" -n 20

Query Format:
  "<keyword> <pattern>" where:
  - keyword: read, write, import, any
  - pattern: substring to match in file paths
        """
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        metavar="QUERY",
        help="Query string: 'keyword pattern' (e.g. 'read Items.ion')"
    )
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=10,
        metavar="N",
        help="Number of results to show (default: 10)"
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Show all results (no limit)"
    )
    
    args = parser.parse_args()
    
    # Load graph
    config = load_config()
    data_dir = config.get("system", {}).get("data_dir", "./data")
    graph_path = Path(data_dir) / "dependency_graph.json"
    
    if not graph_path.exists():
        console.print("[red]❌ Graph not found. Run: uv run build -g[/red]")
        return
    
    graph = DependencyGraph()
    graph.load(str(graph_path))
    tools = GraphTools(graph)
    
    # Execute query
    console.print(Panel.fit(
        f"[bold cyan]🔍 Graph Query[/bold cyan]\n"
        f"[dim]Query: {args.query}[/dim]",
        border_style="cyan"
    ))
    
    result = tools.scan_references(args.query)
    
    # Display results
    count = result['count']
    results = result['results']
    limit = len(results) if args.all else min(args.num, len(results))
    
    console.print(f"\n📊 Found [bold]{count}[/bold] references")
    console.print(f"   Unique targets: [dim]{result['unique_targets_count']}[/dim]")
    
    # Results table
    if results:
        table = Table(
            title=f"Top {limit} Results",
            show_header=True,
            header_style="bold"
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Relationship", style="cyan", width=10)
        table.add_column("Source Script", style="green")
        table.add_column("Target File", style="yellow")
        
        for i, r in enumerate(results[:limit], 1):
            table.add_row(
                str(i),
                r['relationship'],
                r['source_script'],
                r['target_file']
            )
        
        console.print(table)
        
        if len(results) > limit:
            console.print(f"\n[dim]... and {len(results) - limit} more. Use --all to see all.[/dim]")
    else:
        console.print("[yellow]No results found.[/yellow]")


if __name__ == "__main__":
    main()
