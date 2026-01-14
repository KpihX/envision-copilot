"""
Benchmark Runner CLI.
Runs LLM-as-Judge benchmark on selected questions.

Usage:
    uv run envision-benchmark                     # Questions 1-5 (default)
    uv run envision-benchmark -f 1 -t 10          # Questions 1-10
    uv run envision-benchmark -i 1 3 5 8          # Specific questions by ID
"""
import argparse
import sys
import yaml
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from envision_rag.benchmark.runner import BenchmarkRunner


console = Console()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_question_selection(args) -> List[int]:
    """Parse question selection from arguments."""
    selected_ids = set()
    
    start = args.start if args.start else 1
    end = args.end if args.end else 5
    
    if args.ids is None or args.start is not None or args.end is not None:
        for i in range(start, end + 1):
            selected_ids.add(i)
    
    if args.ids:
        for qid in args.ids:
            selected_ids.add(qid)
    
    return sorted(list(selected_ids))


def main():
    parser = argparse.ArgumentParser(
        prog="envision-benchmark",
        description="Run LLM-as-Judge benchmark on the Envision RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run envision-benchmark                   Default: questions 1-5
  uv run envision-benchmark -f 1 -t 10        Questions 1 to 10
  uv run envision-benchmark -i 1 3 5 8        Specific questions by ID
  uv run envision-benchmark -f 1 -t 3 -i 10   Combine range + IDs
  uv run envision-benchmark -q                Quiet mode (results only)

Benchmark Configuration:
  Edit config.yaml -> benchmark section to change:
  - judge_model: Model for semantic evaluation
  - questions_file: Path to questions JSON
        """
    )
    parser.add_argument(
        "-f", "--from",
        dest="start",
        type=int,
        metavar="N",
        help="Start of question range (default: 1)"
    )
    parser.add_argument(
        "-t", "--to",
        dest="end",
        type=int,
        metavar="N",
        help="End of question range, inclusive (default: 5)"
    )
    parser.add_argument(
        "-i", "--ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="Specific question IDs to run"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Show detailed agent trace (default: on)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output - only show final results"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    question_ids = parse_question_selection(args)
    
    if verbose:
        console.print(Panel.fit(
            "[bold]🔬 Envision RAG Benchmark[/bold]",
            border_style="blue"
        ))
        console.print(f"📋 Selected Questions: [cyan]{question_ids}[/cyan]")
    
    config = load_config()
    
    # Auto-build indexes if missing
    from envision_rag.cli.utils import ensure_indexes
    ensure_indexes(config)
    
    runner = BenchmarkRunner(config, verbose=verbose)
    report = runner.run(question_ids=question_ids)
    
    # Summary (always shown)
    if not verbose:
        # Compact summary for quiet mode
        console.print()
        table = Table(title="Benchmark Results", show_header=False, box=None)
        table.add_row("Questions", str(report.total))
        color = "green" if report.accuracy >= 0.8 else "red"
        table.add_row("Accuracy", f"[{color}]{report.accuracy:.0%}[/{color}]")
        console.print(table)
    
    sys.exit(0 if report.accuracy >= 0.8 else 1)


if __name__ == "__main__":
    main()
