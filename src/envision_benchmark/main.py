import argparse
import sys
from rich.console import Console
from rich.table import Table

from envision_benchmark.runner import BenchmarkRunner

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Benchmark CLI")
    parser.add_argument("--run", action="store_true", help="Run full benchmark.")
    parser.add_argument("--ids", type=str, help="Comma-separated IDs or ranges (e.g., '1,3,5-10').")
    
    args = parser.parse_args()
    
    if args.run or args.ids:
        runner = BenchmarkRunner()
        
        # Parse IDs if provided
        target_ids = None
        if args.ids:
            target_ids = []
            parts = args.ids.split(',')
            for part in parts:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    target_ids.extend(range(start, end + 1))
                else:
                    target_ids.append(int(part))
        
        results = runner.run(question_ids=target_ids)
        
        table = Table(title="Benchmark Results")
        table.add_column("Question")
        table.add_column("Score")
        table.add_column("Reasoning", no_wrap=False) # overflow wrap
        
        avg_score = 0
        for r in results:
            table.add_row(
                r["question"][:50]+"...",
                str(r["score"]),
                r["reasoning"][:100]+"..."
            )
            avg_score += r["score"]
            
        if results:
            avg_score /= len(results)
            
        console.print(table)
        console.print(f"[bold green]Average Score: {avg_score:.2f}/10[/bold green]")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
