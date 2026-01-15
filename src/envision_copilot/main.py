import argparse
import sys
import logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Add current directory to path if needed (though uv handles this usually)
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))

from envision_copilot.core.agent import EnvisionCopilot

def main():
    parser = argparse.ArgumentParser(
        description="Envision Copilot - The Agentic Brain",
        formatter_class=argparse.RawTextHelpFormatter
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
        help="Show full agent reasoning trace (Tree of Thoughts)"
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    if not args.query and not args.interactive:
        parser.print_help()
        sys.exit(1)
        
    console.print(Panel("[bold cyan]Envision Copilot[/bold cyan] initialized.", border_style="cyan"))
    
    # Initialize Agent
    copilot = EnvisionCopilot(verbose=args.verbose, interactive=args.interactive)
    
    # Run
    try:
        if args.query:
             
             # Execute
             result = copilot.run(args.query)
             
             # Display Final Answer (only in non-verbose mode, verbose shows it during run)
             if not args.verbose:
                 console.print("\n")
                 console.print(Panel(
                     Markdown(result["answer"]),
                     title="✅ Final Answer",
                     border_style="green",
                     subtitle="Envision Copilot"
                 ))
                 
                 # Display Appendix
                 if result.get("appendix"):
                     import json
                     appendix_json = json.dumps(result["appendix"], indent=2, ensure_ascii=False)
                     console.print("\n")
                     console.print(Panel(
                         appendix_json,
                         title="📎 Appendix (References)",
                         border_style="blue"
                     ))

    except KeyboardInterrupt:
        console.print("\n[red]Cancelled by User[/red]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Fatal Error:[/red] {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
