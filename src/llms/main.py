import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from llms import get_llm

console = Console()

def main():
    parser = argparse.ArgumentParser(description="LLM CLI - Query models directly.")
    parser.add_argument("-q", "--query", type=str, help="The prompt to send to the LLM.")
    parser.add_argument("-m", "--model", type=str, help="Provider to use (mistral, gemini, groq). Default: from config.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive chat mode.")
    
    args = parser.parse_args()

    try:
        llm = get_llm(args.model)
        console.print(f"[dim]Loaded provider: {llm.model_name}[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error loading LLM:[/bold red] {e}")
        sys.exit(1)

    if args.interactive:
        console.print(Panel("💬 Interactive Mode (Ctrl+C to exit)", title="LLM CLI", border_style="green"))
        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ")
                if not user_input.strip():
                    continue
                
                with console.status("Generating...", spinner="dots"):
                    response = llm.generate(user_input)
                
                console.print(Panel(Markdown(response), title="🤖 Assistant", border_style="blue"))
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting...[/yellow]")
                break
            except Exception as e:
                 console.print(f"[red]Error:[/red] {e}")

    elif args.query:
        with console.status("Generating...", spinner="dots"):
            response = llm.generate(args.query)
        console.print(Panel(Markdown(response), title="🤖 Assistant", border_style="blue"))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
