#!/usr/bin/env python3
"""
Log Viewer CLI.
Replays saved session logs with original Rich styling.

Usage:
    uv run logs -t main -n 1           # View last main.py run
    uv run logs -t benchmark -n 2      # View 2nd to last benchmark
    uv run logs -t main --clean 7      # Delete logs older than 7 days
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table

from envision_rag.logging.session_logger import SessionLogger, SessionLog


console = Console()


def display_session(session: SessionLog):
    """Display a session log with Rich styling."""
    
    # Header
    console.print(Panel.fit(
        f"[bold cyan]📜 Session: {session.session_id}[/bold cyan]\n"
        f"[dim]Type: {session.log_type} | {session.started_at}[/dim]",
        border_style="cyan"
    ))
    
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
    
    # No Summary section - Appendix is the last thing displayed


def clean_old_logs(log_type: str, days: int, log_dir: str) -> int:
    """Delete logs older than N days."""
    logs_path = Path(log_dir) / log_type
    if not logs_path.exists():
        return 0
    
    cutoff = datetime.now() - timedelta(days=days)
    deleted = 0
    
    for log_file in logs_path.glob("*.json"):
        # Parse date from filename: YYYY-MM-DD_HH-MM-SS_xxx.json
        try:
            parts = log_file.stem.split("_")
            date_str = parts[0]  # YYYY-MM-DD
            log_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            if log_date < cutoff:
                log_file.unlink()
                deleted += 1
        except (ValueError, IndexError):
            continue
    
    return deleted


def main():
    parser = argparse.ArgumentParser(
        prog="logs",
        description="View and manage saved session logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run logs -t main -n 1           View last main query
  uv run logs -t benchmark -n 2      View 2nd to last benchmark
  uv run logs -t main --list         List all main logs
  uv run logs -t benchmark --clean 7 Delete logs older than 7 days
        """
    )
    parser.add_argument(
        "-t", "--type",
        type=str,
        choices=["main", "benchmark"],
        required=True,
        help="Log type to view/manage"
    )
    parser.add_argument(
        "-n", "--nth",
        type=int,
        default=1,
        help="Which log to view: 1=most recent, 2=second most recent"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available logs"
    )
    parser.add_argument(
        "-c", "--clean",
        type=int,
        metavar="DAYS",
        help="Delete logs older than N days"
    )
    parser.add_argument(
        "-d", "--log-dir",
        type=str,
        default="data/logs",
        help="Base directory for logs"
    )
    
    args = parser.parse_args()
    
    if args.clean:
        # Cleanup mode
        deleted = clean_old_logs(args.type, args.clean, args.log_dir)
        console.print(f"🗑️ Deleted [yellow]{deleted}[/yellow] {args.type} logs older than {args.clean} days")
        
    elif args.list:
        # List mode
        logs = SessionLogger.list_logs(args.type, args.log_dir)
        if not logs:
            console.print(f"[yellow]No logs found for type '{args.type}'[/yellow]")
            sys.exit(1)
        
        console.print(Panel.fit(
            f"[bold]📂 {args.type.title()} Logs ({len(logs)} total)[/bold]",
            border_style="blue"
        ))
        for i, log in enumerate(logs[:10], 1):  # Show last 10
            console.print(f"  {i}. {log.name}")
        if len(logs) > 10:
            console.print(f"  [dim]... and {len(logs) - 10} more[/dim]")
        console.print(f"\n[dim]Use: uv run logs -t {args.type} -n <number>[/dim]")
        
    else:
        # View mode
        session = SessionLogger.load_nth(args.type, args.nth, args.log_dir)
        if not session:
            console.print(f"[red]❌ Log not found: type='{args.type}', nth={args.nth}[/red]")
            logs = SessionLogger.list_logs(args.type, args.log_dir)
            if logs:
                console.print(f"[yellow]Available logs: {len(logs)}[/yellow]")
            sys.exit(1)
        
        display_session(session)


if __name__ == "__main__":
    main()
