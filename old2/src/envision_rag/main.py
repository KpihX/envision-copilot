"""
Envision RAG System - Main CLI Entry Point.
Query the RAG system with graph-enhanced retrieval.

Usage:
    uv run envision-rag -q "Combien de scripts lisent /Clean/Items.ion?"
    uv run envision-rag -i                   # Interactive mode
    uv run envision-rag -v -q "..."          # Verbose trace
"""
import argparse
import yaml
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from envision_rag.graph.builder import GraphBuilder
from envision_rag.tools.graph_tools import GraphTools
from envision_rag.workflow.agent import AgentWorkflow
from envision_rag.logging.session_logger import SessionLogger


console = Console()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def extract_answer(response: str) -> str:
    """Extract the final answer from agent response."""
    if "Final Answer:" in response:
        return response.split("Final Answer:")[-1].strip()
    return response


def main():
    parser = argparse.ArgumentParser(
        prog="envision-rag",
        description="Query the Envision RAG system with natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run envision-rag -q "Combien de scripts lisent /Clean/Items.ion?"
  uv run envision-rag -v -q "Où est définie la variable ReDispatchCycle?"
  uv run envision-rag -i                 # Interactive mode
  uv run envision-rag -i -v              # Interactive with full trace

Before first use:
  uv run envision-build                  # Build graph and vector indexes
        """
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        metavar="QUESTION",
        help="Question to ask the RAG system"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive conversation mode (agent can ask clarifications)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show full agent reasoning trace"
    )
    
    args = parser.parse_args()
    
    # Validate args
    if not args.query and not args.interactive:
        parser.print_help()
        return

    # 1. Load config
    config = load_config()
    
    # Auto-build indexes if missing
    from envision_rag.cli.utils import ensure_indexes
    ensure_indexes(config)
    
    system_config = config.get("system", {})
    data_dir = system_config.get("data_dir", "./data")
    graph_path = Path(data_dir) / "dependency_graph.json"
    
    # Initialize logger
    logging_config = config.get("logging", {})
    logger = SessionLogger(
        log_type="main",
        log_dir=logging_config.get("log_dir", "data/logs"),
        enabled=logging_config.get("enabled", True)
    )

    # 2. Load Graph
    if not graph_path.exists():
        console.print("[red]❌ Graph index not found. Run:[/red] uv run envision-build -g")
        return
    
    from envision_rag.graph.graph_types import DependencyGraph
    graph = DependencyGraph()
    graph.load(str(graph_path))
    
    if not args.verbose:
        console.print(Panel.fit(
            "[bold cyan]Envision RAG[/bold cyan]",
            subtitle=f"Graph: {graph.stats()['nodes']} nodes"
        ))

    # 3. Build Agent
    # Pass interactive flag to agent so it can ask clarifications
    tools = GraphTools(graph)
    workflow = AgentWorkflow(
        config, tools, 
        verbose=args.verbose, 
        logger=logger,
        interactive=args.interactive
    )
    app = workflow.build_graph()

    # 4. Handle Query or Interactive Mode
    if args.query:
        _handle_single_query(app, args.query, args.verbose, logger, console)
    
    elif args.interactive:
        _handle_interactive(app, args.verbose, logger, console)


def _handle_single_query(app, query: str, verbose: bool, logger, console: Console):
    """Process a single query."""
    logger.start_session({"query": query, "mode": "single", "verbose": verbose})
    
    if not verbose:
        console.print(f"\n[bold]❓ Question:[/bold] {query}")
    
    result = app.invoke({
        "question": query, 
        "scratchpad": "", 
        "messages": [], 
        "facts": []
    })
    
    messages = result.get('messages', [])
    response = messages[-1] if messages else "No response."
    answer = extract_answer(response)
    
    if not verbose:
        console.print(Panel(
            Markdown(answer),
            title="[green]✅ Answer[/green]",
            border_style="green"
        ))
    
    # End and save session
    logger.end_session({"final_answer": answer[:500], "success": True})
    log_path = logger.save()
    if log_path and verbose:
        console.print(f"\n📝 Session saved: {log_path}")


def _handle_interactive(app, verbose: bool, logger, console: Console):
    """Interactive conversation mode."""
    console.print(Panel.fit(
        "[bold]💬 Interactive Mode[/bold]\n"
        "[dim]Type your questions. Press Ctrl+C to exit.[/dim]\n"
        "[dim]The agent may ask for clarifications if needed.[/dim]",
        border_style="cyan"
    ))
    
    session_count = 0
    
    while True:
        try:
            # Rich prompt
            query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if not query.strip():
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("\n[dim]👋 Goodbye![/dim]")
                break
            
            # Start session
            session_count += 1
            logger.start_session({
                "query": query, 
                "mode": "interactive", 
                "session_num": session_count
            })
            
            result = app.invoke({
                "question": query,
                "scratchpad": "",
                "messages": [],
                "facts": []
            })
            
            messages = result.get('messages', [])
            response = messages[-1] if messages else "No response."
            answer = extract_answer(response)
            
            if not verbose:
                console.print(Panel(
                    Markdown(answer),
                    title="[green]🤖 Agent[/green]",
                    border_style="green"
                ))
            
            # Save session
            logger.end_session({
                "final_answer": answer[:500], 
                "success": True
            })
            logger.save()
            
        except KeyboardInterrupt:
            console.print("\n[dim]👋 Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")


if __name__ == "__main__":
    main()
