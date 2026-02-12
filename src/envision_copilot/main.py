import sys
import logging
import typer
from typing import Optional
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from envision_copilot.core.main import EnvisionCopilot

# Initialize Typer App
app = typer.Typer(
    name="envision-copilot",
    help="Envision Copilot - The Agentic Brain",
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def start(
    query: Annotated[
        Optional[str], 
        typer.Option("--query", "-q", help="Question to ask the RAG system", metavar="QUESTION")
    ] = None,
    interactive: Annotated[
        bool, 
        typer.Option("--interactive", "-i", "--live", help="Interactive conversation mode (agent can ask clarifications)")
    ] = False,
    verbose: Annotated[
        bool, 
        typer.Option("--verbose", "-v", help="Show full agent reasoning trace (Tree of Thoughts)")
    ] = False,
):
    """
    Launch the Envision Copilot Agent.
    """
    
    if not query and not interactive:
        console.print("[yellow]Please provide a query (-q) or enable interactive mode (-i).[/yellow]")
        console.print("Run [bold]uv run copilot --help[/bold] for usage.")
        raise typer.Exit(code=1)
        
    console.print(Panel("[bold cyan]Envision Copilot[/bold cyan] initialized.", border_style="cyan"))
    
    # Initialize Agent
    copilot = EnvisionCopilot(verbose=verbose, interactive=interactive)
    
    # Run
    try:
        if interactive:
            # Interactive Mode (Self-Managed UI)
            copilot.run()
        
        elif query:
            # Execute (The agent handles its own standardized UI display)
            result = copilot.run(query)
                 
            # Display Appendix (via Encapsulated UI) if present
            if result.get("appendix"):
                console.print(result["appendix"])

    except KeyboardInterrupt:
        console.print("\n[red]Cancelled by User[/red]")
        raise typer.Exit(code=0)
    except Exception as e:
        console.print(f"\n[red]Fatal Error:[/red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)

def main():
    """Entry point for project.scripts"""
    app()

if __name__ == "__main__":
    app()
