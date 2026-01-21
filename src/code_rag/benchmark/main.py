"""
RAG Benchmark - Test if patterns are found in retrieved chunks.

For each question, retrieves top-k chunks and shows which patterns
are found in each chunk, with their ranking position.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box

from code_rag.retriever import GraphRetriever
from code_rag.utils import ConfigLoader

console = Console()


class RAGBenchmark:
    """Benchmark to test pattern retrieval in RAG chunks."""

    def __init__(
        self, 
        questions_path: str = None, 
        top_k: int = None, 
        recall_k: int = None, 
        use_reranking: bool = None
    ):
        # Load config
        self.config = ConfigLoader.load_config()
        bench_config = self.config.get("benchmark", {})
        ret_config = self.config.get("retrieval", {})
        
        # Use config defaults if not specified
        self.top_k = top_k if top_k is not None else bench_config.get("top_k", 5)
        self.recall_k = recall_k if recall_k is not None else bench_config.get("recall_k", 50)
        use_rerank = use_reranking if use_reranking is not None else ret_config.get("use_reranker", True)
        
        # Initialize retriever
        self.retriever = GraphRetriever()
        self.retriever.set_reranking(use_rerank)
        self.use_reranking = use_rerank

        # Resolve questions path
        if questions_path is None:
            questions_file = bench_config.get("questions_file", "src/code_rag/benchmark/questions.json")
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            questions_path = project_root / questions_file
            if not questions_path.exists():
                # Fallback to relative to this file
                questions_path = Path(__file__).parent / "questions.json"
        else:
            questions_path = Path(questions_path)

        with open(questions_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.pattern_questions = data.get("pattern_based", [])
        
        # Log file path
        log_file = bench_config.get("log_file", "src/code_rag/data/log/benchmark.json")
        project_root = Path(__file__).parent.parent.parent.parent
        self.log_path = project_root / log_file
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Header
        console.print(Panel.fit(
            f"[bold cyan]📊 RAG Benchmark[/bold cyan]\n"
            f"Questions: {len(self.pattern_questions)} | Top-K: {self.top_k} | Recall: {self.recall_k} | "
            f"Reranking: {'[green]ON[/green]' if use_rerank else '[red]OFF[/red]'}",
            border_style="cyan"
        ))

    def run(self, question_ids: List[int] = None) -> Dict[str, Any]:
        """
        Run benchmark on pattern-based questions.
        
        Args:
            question_ids: Optional list of specific question IDs to test
        """
        questions = self.pattern_questions
        if question_ids:
            questions = [q for q in questions if q["id"] in question_ids]

        total_found = 0
        total_patterns = 0
        results_summary = []
        detailed_results = []

        for q in track(questions, description="Evaluating...", transient=True):
            query = q["question"]
            patterns = q["patterns"]
            total_patterns += len(patterns)

            # Execute RAG search
            response = self.retriever.query(query, top_k=self.top_k, recall_k=self.recall_k)
            chunks = response.get("results", [])

            # Find patterns in each chunk
            chunk_patterns = []
            all_found_patterns: Set[str] = set()
            
            for i, chunk in enumerate(chunks):
                # Search in full text (includes header with Path, Imports, Reads + content)
                chunk_full_text = chunk.get("text", "")
                chunk_full_text += " " + chunk.get("source", "")
                chunk_full_text = chunk_full_text.lower()
                
                patterns_in_chunk = []
                for p in patterns:
                    if p.lower() in chunk_full_text:
                        patterns_in_chunk.append(p)
                        all_found_patterns.add(p)
                
                if patterns_in_chunk:
                    chunk_patterns.append({
                        "rank": i + 1,
                        "source": chunk.get("source", "?"),
                        "score": chunk.get("score", 0),
                        "patterns": patterns_in_chunk
                    })

            found_count = len(all_found_patterns)
            total_found += found_count
            
            result = {
                "id": q["id"],
                "question": query,
                "patterns": patterns,
                "found": list(all_found_patterns),
                "missing": [p for p in patterns if p not in all_found_patterns],
                "chunks_with_patterns": chunk_patterns,
                "success": found_count == len(patterns)
            }
            detailed_results.append(result)
            results_summary.append({
                "id": q["id"],
                "found": found_count,
                "total": len(patterns),
                "success": found_count == len(patterns)
            })

            # Display result
            self._print_question_result(q["id"], query, patterns, chunk_patterns, all_found_patterns)

        # Final summary
        summary = self._print_summary(results_summary, total_found, total_patterns)
        
        # Save log
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "top_k": self.top_k,
                "recall_k": self.recall_k,
                "reranking": self.use_reranking
            },
            "summary": summary,
            "results": detailed_results
        }
        self._save_log(report)
        
        return report

    def _print_question_result(
        self, 
        qid: int, 
        question: str,
        patterns: List[str], 
        chunk_patterns: List[Dict],
        found_patterns: Set[str]
    ):
        """Display result showing which chunks contain which patterns."""
        
        if len(found_patterns) == len(patterns):
            status = "[green]✓ ALL FOUND[/green]"
        elif len(found_patterns) > 0:
            status = f"[yellow]◐ {len(found_patterns)}/{len(patterns)}[/yellow]"
        else:
            status = "[red]✗ NONE[/red]"

        console.print(f"\n[bold]Q{qid}[/bold] {status}")
        console.print(f"[dim]{question}[/dim]\n")

        console.print("[bold]Patterns to find:[/bold]")
        for p in patterns:
            if p in found_patterns:
                console.print(f"  [green]✓[/green] {p}")
            else:
                console.print(f"  [red]✗[/red] {p}")

        if chunk_patterns:
            console.print(f"\n[bold]Chunks containing patterns:[/bold]")
            
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            table.add_column("Rank", justify="center", style="cyan", width=5)
            table.add_column("Source", style="dim")
            table.add_column("Patterns Found", style="green")
            
            for cp in chunk_patterns:
                patterns_str = ", ".join(cp["patterns"][:3])
                if len(cp["patterns"]) > 3:
                    patterns_str += f" (+{len(cp['patterns'])-3})"
                table.add_row(
                    f"#{cp['rank']}",
                    cp["source"][:50] + ("..." if len(cp["source"]) > 50 else ""),
                    patterns_str
                )
            
            console.print(table)
        else:
            console.print("\n[dim]No chunks contain the expected patterns.[/dim]")

        console.print("─" * 60)

    def _print_summary(self, results: List[Dict], total_found: int, total_patterns: int) -> Dict[str, Any]:
        """Display final summary and return summary dict."""
        success_count = sum(1 for r in results if r["success"])
        partial_count = sum(1 for r in results if 0 < r["found"] < r["total"])
        fail_count = sum(1 for r in results if r["found"] == 0)
        
        pct = (total_found / total_patterns * 100) if total_patterns > 0 else 0
        color = "green" if pct >= 70 else "yellow" if pct >= 40 else "red"
        
        console.print(Panel.fit(
            f"[bold]Results[/bold]\n\n"
            f"  [green]✓ All patterns:[/green] {success_count} questions\n"
            f"  [yellow]◐ Partial:[/yellow] {partial_count} questions\n"
            f"  [red]✗ None:[/red] {fail_count} questions\n\n"
            f"  [{color}]Patterns: {total_found}/{total_patterns} ({pct:.1f}%)[/{color}]",
            border_style=color,
            title="📊 Summary"
        ))
        
        return {
            "total_questions": len(results),
            "all_found": success_count,
            "partial": partial_count,
            "none_found": fail_count,
            "patterns_found": total_found,
            "patterns_total": total_patterns,
            "success_rate": pct
        }

    def _save_log(self, report: Dict[str, Any]):
        """Save benchmark report to log file."""
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        console.print(f"\n[dim]📄 Log saved to {self.log_path}[/dim]")


def main():
    # Load config for defaults
    config = ConfigLoader.load_config()
    bench_config = config.get("benchmark", {})
    
    parser = argparse.ArgumentParser(description="RAG Benchmark - Test pattern retrieval")
    parser.add_argument(
        "--top-k", "-t", "-k",
        type=int,
        default=None,
        help=f"Number of final chunks to evaluate (default: {bench_config.get('top_k', 5)})"
    )
    parser.add_argument(
        "--recall", "-r",
        type=int,
        default=None,
        help=f"Number of candidates from FAISS (default: {bench_config.get('recall_k', 50)})"
    )
    parser.add_argument(
        "--ids", "-i",
        type=int,
        nargs="+",
        help="Specific question IDs to test"
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        default=None,
        help="Path to custom questions.json"
    )
    parser.add_argument(
        "--no-rerank", "-n",
        action="store_true",
        help="Disable reranking"
    )

    args = parser.parse_args()

    try:
        benchmark = RAGBenchmark(
            questions_path=args.questions, 
            top_k=args.top_k,
            recall_k=args.recall,
            use_reranking=False if args.no_rerank else None
        )
        benchmark.run(question_ids=args.ids)
    except FileNotFoundError as e:
        console.print(f"[red]❌ Error:[/red] {e}")
        console.print("[yellow]💡 Run 'uv run index --build' first.[/yellow]")


if __name__ == "__main__":
    main()
