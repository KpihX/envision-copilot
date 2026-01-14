import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from envision_copilot.agent import EnvisionAgent

console = Console()

def extract_answer(response: str) -> str:
    if "Final Answer:" in response:
        return response.split("Final Answer:")[-1].strip()
    return response

def main():
    parser = argparse.ArgumentParser(description="Envision Copilot CLI")
    parser.add_argument("-q", "--query", type=str, help="Question about the codebase.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full agent reasoning trace.")

    args = parser.parse_args()
    
    try:
        agent = EnvisionAgent(verbose=args.verbose, interactive=args.interactive)
    except Exception as e:
         console.print(f"[bold red]Failed to init Agent:[/bold red] {e}")
         sys.exit(1)

    if args.interactive:
        console.print(Panel("Envision Copilot (Ctrl+C to exit)", title="Interactive", border_style="purple"))
        while True:
            try:
                user_input = console.input("[bold purple]User:[/bold purple] ")
                if not user_input.strip(): continue
                
                if args.verbose:
                    console.print("[dim]Thinking...[/dim]")
                
                response = agent.run(user_input)
                answer = extract_answer(response)
                
                if not args.verbose: # If verbose, we already printed thoughts/answers in real time
                     console.print(Panel(Markdown(answer), title="Copilot", border_style="blue"))
                     
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")

    elif args.query:
        if args.verbose:
             console.print("[dim]Reasoning...[/dim]")
        
        response = agent.run(args.query)
        answer = extract_answer(response)
        
        if not args.verbose:
            console.print(Panel(Markdown(answer), title="Copilot Result", border_style="blue"))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
