#!/usr/bin/env python3
"""
Log Viewer CLI.
Replays saved session logs with original Rich styling.

Usage:
    uv run envision-logs --type main --nth 1       # View last main.py run
    uv run envision-logs --type benchmark --nth 2  # View 2nd to last benchmark
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table

# Fix imports
sys.path.append(str(Path(__file__).parent.parent))

from envision_rag.logging.session_logger import SessionLogger, SessionLog


def display_session(session: SessionLog, console: Console):
    """Display a session log with Rich styling."""
    
    # Header
    console.print("\n" + "=" * 60)
    console.print(f"[bold cyan]📜 Session Log: {session.session_id}[/bold cyan]")
    console.print(f"   Type: {session.log_type}")
    console.print(f"   Started: {session.started_at}")
    console.print(f"   Ended: {session.ended_at or 'N/A'}")
    
    # Metadata
    if session.metadata:
        console.print(f"\n[bold]Metadata:[/bold]")
        for k, v in session.metadata.items():
            console.print(f"   {k}: {v}")
    
    console.print("\n" + "=" * 60)
    
    # Events
    for event in session.events:
        style = event.style
        title = event.title
        content = event.content
        
        if style == "thought":
            console.print(Panel(Markdown(content), title=f"🧠 {title}", border_style="purple"))
        elif style == "action":
            console.print(Panel(Markdown(content), title=f"🛠️ {title}", border_style="blue"))
        elif style == "observation":
            # Smart truncation for display
            lines = content.splitlines()
            N = 15
            if len(lines) > 2 * N:
                head = lines[:N]
                tail = lines[-N:]
                truncated = "\n".join(head) + f"\n\n... [Masked {len(lines) - 2*N} lines] ...\n\n" + "\n".join(tail)
            else:
                truncated = content
            console.print(Panel(Syntax(truncated, "python", word_wrap=True, theme="monokai"), 
                               title=f"👀 {title}", border_style="green"))
        elif style == "answer":
            console.print(Panel(Markdown(content), title=f"✅ {title}", border_style="cyan"))
        elif style == "question":
            console.print(Panel(Markdown(content), title=f"❓ {title}", border_style="yellow"))
        elif style == "expected":
            console.print(Panel(content, title=f"📋 {title}", border_style="blue"))
        elif style == "verdict":
            color = "green" if "✅" in title else "red"
            console.print(Panel(Markdown(content), title=title, border_style=color))
        elif style == "appendix":
            console.print(Panel(Markdown(content), title=f"📎 {title}", border_style="dim"))
        elif style == "error":
            console.print(Panel(Markdown(content), title=f"❌ {title}", border_style="red"))
        else:
            console.print(f"[bold]{title}[/bold]: {content}")
    
    # Summary
    if session.summary:
        console.print("\n" + "=" * 60)
        console.print("[bold]📊 Summary:[/bold]")
        summary_table = Table(show_header=False, box=None)
        for k, v in session.summary.items():
            summary_table.add_row(str(k), str(v))
        console.print(summary_table)
    
    console.print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="View saved session logs with Rich styling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run envision-logs -t main -n 1       # Last main.py run
    uv run envision-logs -t benchmark -n 2  # 2nd to last benchmark
    uv run envision-logs --list main        # List all main logs
        """
    )
    parser.add_argument(
        "-t", "--type",
        type=str,
        choices=["main", "benchmark"],
        required=True,
        help="Log type to view"
    )
    parser.add_argument(
        "-n", "--nth",
        type=int,
        default=1,
        help="Which log to view: 1=most recent, 2=second most recent, etc."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available logs instead of viewing one"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="data/logs",
        help="Base directory for logs"
    )
    
    args = parser.parse_args()
    console = Console()
    
    if args.list:
        # List mode
        logs = SessionLogger.list_logs(args.type, args.log_dir)
        if not logs:
            console.print(f"[yellow]No logs found for type '{args.type}'[/yellow]")
            sys.exit(1)
        
        console.print(f"\n[bold]📂 Available {args.type} logs:[/bold]")
        for i, log in enumerate(logs, 1):
            console.print(f"   {i}. {log.name}")
        console.print(f"\nUse: envision-logs -t {args.type} -n <number>")
        
    else:
        # View mode
        session = SessionLogger.load_nth(args.type, args.nth, args.log_dir)
        if not session:
            console.print(f"[red]❌ Log not found: type='{args.type}', nth={args.nth}[/red]")
            logs = SessionLogger.list_logs(args.type, args.log_dir)
            if logs:
                console.print(f"[yellow]Available logs: {len(logs)}[/yellow]")
            sys.exit(1)
        
        display_session(session, console)


if __name__ == "__main__":
    main()
