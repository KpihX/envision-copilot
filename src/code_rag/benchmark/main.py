"""
RAG Benchmark - Test if patterns are found in retrieved chunks.

For each question, retrieves top-k chunks and shows which patterns
are found in each chunk, with their ranking position.

Metrics:
- M1: Recall Score - Pattern found or not (1/n per pattern found)
- M2: Position Score - Penalizes patterns found at low ranks (α × 1/n)
- M3: Rerank Delta - Improvement from reranking (only in rerank mode)
"""

import json
import argparse
import uuid
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set, Callable

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box

from code_rag.vector_engines import get_retriever
from code_rag.utils import ConfigLoader

console = Console()


from .common import (
    alpha_log, 
    alpha_linear, 
    compute_position_score, 
    load_questions,
    ALPHA_FUNCTIONS,
    DEFAULT_ALPHA
)


# ============================================================================
# SCORING FUNCTIONS - Reusable by optimize_weights.py
# ============================================================================




class RAGBenchmark:
    """Benchmark to test pattern retrieval in RAG chunks."""

    def __init__(
        self, 
        questions_path: str = None, 
        top_k: int = None, 
        recall_k: int = None, 
        use_reranking: bool = None,
        alpha_fn: str = DEFAULT_ALPHA,
        pattern_as_keywords: bool = False
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
        self.retriever = get_retriever("src/code_rag/config.yaml", verbose=True)
        self.retriever.set_reranking(use_rerank)
        self.use_reranking = use_rerank
        
        # Alpha function for position scoring
        self.alpha_fn = ALPHA_FUNCTIONS.get(alpha_fn, alpha_log)
        self.alpha_name = alpha_fn
        self.pattern_as_keywords = pattern_as_keywords

        if questions_path is None:
             # Let common loader handle defaults
             self.pattern_questions = load_questions()
        else:
             self.pattern_questions = load_questions(str(questions_path))
             
        # Log file path
        
        # Log file path
        # Log directory path
        log_dir = bench_config.get("log_dir", "datas/code_rag/benchmark")
        project_root = Path(__file__).parent.parent.parent.parent
        
        # Determine unique filename: YYYY-MM-DD_HH-MM-SS_uuid.json
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{timestamp}_{unique_id}.json"
        
        self.log_path = project_root / log_dir / filename
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get model names for display
        idx_config = self.config.get("indexing", {})
        ret_config = self.config.get("retrieval", {})
        embedding_model = idx_config.get("model_name", "unknown").split("/")[-1]
        reranker_type = ret_config.get("reranker_type", "cross-encoder")
        
        # Header
        header_lines = [
            "[bold cyan]📊 RAG Benchmark[/bold cyan]",
            f"[dim]Embedding:[/dim] {embedding_model}"
        ]
        if use_rerank:
            header_lines.append(f"[dim]Reranker:[/dim] {reranker_type}")
        header_lines.append(
            f"Questions: {len(self.pattern_questions)} | Top-K: {self.top_k} | Recall: {self.recall_k} | "
            f"Reranking: {'[green]ON[/green]' if use_rerank else '[red]OFF[/red]'}"
        )
        
        console.print(Panel.fit(
            "\n".join(header_lines),
            border_style="cyan"
        ))

    def _compute_position_score(
        self, 
        patterns: List[str], 
        pattern_first_ranks: Dict[str, int]
    ) -> float:
        """Call global helper with self.alpha_fn"""
        return compute_position_score(patterns, pattern_first_ranks, self.alpha_fn)

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
        total_recall_score = 0.0
        total_position_score = 0.0
        max_possible_score = len(questions)  # Max is 1.0 per question
        
        results_summary = []
        detailed_results = []

        # If reranking enabled, we need to run twice to compare
        if self.use_reranking:
            # First pass: without reranking (for comparison)
            self.retriever.set_reranking(False)
            no_rerank_results = self._run_single_pass(questions, silent=True)
            
            # Second pass: with reranking
            self.retriever.set_reranking(True)
            rerank_results = self._run_single_pass(questions, silent=False, 
                                                    compare_ranks=no_rerank_results,
                                                    compare_scores=no_rerank_results)
            
            # Merge results
            for q_result in rerank_results["detailed"]:
                qid = q_result["id"]
                no_rerank_q = next(r for r in no_rerank_results["detailed"] if r["id"] == qid)
                q_result["position_score_no_rerank"] = no_rerank_q["position_score"]
                q_result["rerank_delta"] = q_result["position_score"] - no_rerank_q["position_score"]
            
            detailed_results = rerank_results["detailed"]
            results_summary = rerank_results["summary"]
            total_found = rerank_results["total_found"]
            total_patterns = rerank_results["total_patterns"]
            total_recall_score = rerank_results["total_recall_score"]
            total_position_score = rerank_results["total_position_score"]
            no_rerank_position_score = no_rerank_results["total_position_score"]
        else:
            # Single pass without reranking
            pass_results = self._run_single_pass(questions, silent=False)
            detailed_results = pass_results["detailed"]
            results_summary = pass_results["summary"]
            total_found = pass_results["total_found"]
            total_patterns = pass_results["total_patterns"]
            total_recall_score = pass_results["total_recall_score"]
            total_position_score = pass_results["total_position_score"]
            no_rerank_position_score = None

        # Final summary
        summary = self._print_summary(
            results_summary, 
            total_found, 
            total_patterns,
            total_recall_score,
            total_position_score,
            max_possible_score,
            no_rerank_position_score
        )
        
        # Save log
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "top_k": self.top_k,
                "recall_k": self.recall_k,
                "reranking": self.use_reranking,
                "alpha_function": self.alpha_name
            },
            "summary": summary,
            "results": detailed_results
        }
        self._save_log(report)
        
        return report

    def _run_single_pass(
        self, 
        questions: List[Dict], 
        silent: bool = False,
        compare_ranks: Dict = None,
        compare_scores: Dict = None
    ) -> Dict[str, Any]:
        """Run a single pass of evaluation."""
        
        total_found = 0
        total_patterns = 0
        total_recall_score = 0.0
        total_position_score = 0.0
        results_summary = []
        detailed_results = []

        iterator = questions if silent else track(questions, description="Evaluating...", transient=True)
        
        for q in iterator:
            query = q["question"]
            patterns = q["patterns"]
            total_patterns += len(patterns)

            # Execute RAG search
            keywords = patterns if self.pattern_as_keywords else None
            response = self.retriever.query(
                query, 
                top_k=self.top_k, 
                recall_k=self.recall_k,
                keywords=keywords
            )
            chunks = response.get("results", [])

            # Find patterns in each chunk and track first occurrence rank
            chunk_patterns = []
            all_found_patterns: Set[str] = set()
            pattern_first_ranks: Dict[str, int] = {}  # pattern -> first rank found
            
            for i, chunk in enumerate(chunks):
                rank = i + 1
                # Search in full text (includes header with Path, Imports, Reads + content)
                chunk_full_text = chunk.get("text", "")
                chunk_full_text += " " + chunk.get("source", "")
                chunk_full_text = chunk_full_text.lower()
                
                patterns_in_chunk = []
                for p in patterns:
                    if p.lower() in chunk_full_text:
                        patterns_in_chunk.append(p)
                        if p not in all_found_patterns:
                            all_found_patterns.add(p)
                            pattern_first_ranks[p] = rank
                
                if patterns_in_chunk:
                    chunk_info = {
                        "rank": rank,
                        "source": chunk.get("source", "?"),
                        "score": chunk.get("score", 0),
                        "patterns": patterns_in_chunk
                    }
                    
                    # Add rank delta if comparing
                    if compare_ranks:
                        old_q = next((r for r in compare_ranks["detailed"] if r["id"] == q["id"]), None)
                        if old_q:
                            # Find old rank for this source
                            old_chunks = old_q.get("chunks_with_patterns", [])
                            old_chunk = next((c for c in old_chunks if c["source"] == chunk.get("source")), None)
                            if old_chunk:
                                old_rank = old_chunk["rank"]
                                chunk_info["rank_delta"] = old_rank - rank  # positive = moved up
                    
                    chunk_patterns.append(chunk_info)

            found_count = len(all_found_patterns)
            total_found += found_count
            
            # Compute scores
            recall_score = found_count / len(patterns) if patterns else 0.0
            position_score = self._compute_position_score(patterns, pattern_first_ranks)
            
            total_recall_score += recall_score
            total_position_score += position_score
            
            result = {
                "id": q["id"],
                "question": query,
                "patterns": patterns,
                "found": list(all_found_patterns),
                "missing": [p for p in patterns if p not in all_found_patterns],
                "pattern_first_ranks": pattern_first_ranks,
                "chunks_with_patterns": chunk_patterns,
                "recall_score": recall_score,
                "position_score": position_score,
                "success": found_count == len(patterns)
            }
            detailed_results.append(result)
            results_summary.append({
                "id": q["id"],
                "found": found_count,
                "total": len(patterns),
                "recall_score": recall_score,
                "position_score": position_score,
                "success": found_count == len(patterns)
            })

            # Display result
            if not silent:
                # Get position score delta if comparing
                pos_delta = None
                if compare_scores:
                    old_q = next((r for r in compare_scores["detailed"] if r["id"] == q["id"]), None)
                    if old_q:
                        pos_delta = position_score - old_q["position_score"]
                
                self._print_question_result(q["id"], query, patterns, chunk_patterns, 
                                           all_found_patterns, position_score, pos_delta)

        return {
            "detailed": detailed_results,
            "summary": results_summary,
            "total_found": total_found,
            "total_patterns": total_patterns,
            "total_recall_score": total_recall_score,
            "total_position_score": total_position_score
        }

    def _print_question_result(
        self, 
        qid: int, 
        question: str,
        patterns: List[str], 
        chunk_patterns: List[Dict],
        found_patterns: Set[str],
        position_score: float,
        position_delta: float = None
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
            table.add_column("Rank", justify="center", style="cyan", width=12)
            table.add_column("Source", style="dim")
            table.add_column("Patterns Found", style="green")
            
            for cp in chunk_patterns:
                patterns_str = ", ".join(cp["patterns"][:3])
                if len(cp["patterns"]) > 3:
                    patterns_str += f" (+{len(cp['patterns'])-3})"
                
                # Format rank with delta if available
                rank_str = f"#{cp['rank']}"
                if "rank_delta" in cp:
                    delta = cp["rank_delta"]
                    if delta > 0:
                        rank_str = f"#{cp['rank']} [green]↑(+{delta})[/green]"
                    elif delta < 0:
                        rank_str = f"#{cp['rank']} [red]↓({delta})[/red]"
                
                table.add_row(
                    rank_str,
                    cp["source"][:50] + ("..." if len(cp["source"]) > 50 else ""),
                    patterns_str
                )
            
            console.print(table)
        else:
            console.print("\n[dim]No chunks contain the expected patterns.[/dim]")

        # Show position score for this question with delta if available
        pos_str = f"Position Score: {position_score:.3f}/1.000"
        if position_delta is not None:
            if position_delta > 0.001:
                pos_str += f" [green]↑(+{position_delta:.3f})[/green]"
            elif position_delta < -0.001:
                pos_str += f" [red]↓({position_delta:.3f})[/red]"
            else:
                pos_str += " [dim](=)[/dim]"
        console.print(f"[dim]{pos_str}[/dim]")
        console.print("─" * 60)

    def _print_summary(
        self, 
        results: List[Dict], 
        total_found: int, 
        total_patterns: int,
        total_recall_score: float,
        total_position_score: float,
        max_possible_score: float,
        no_rerank_position_score: float = None
    ) -> Dict[str, Any]:
        """Display final summary and return summary dict."""
        success_count = sum(1 for r in results if r["success"])
        partial_count = sum(1 for r in results if 0 < r["found"] < r["total"])
        fail_count = sum(1 for r in results if r["found"] == 0)
        
        recall_pct = (total_found / total_patterns * 100) if total_patterns > 0 else 0
        position_pct = (total_position_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        color = "green" if recall_pct >= 70 else "yellow" if recall_pct >= 40 else "red"
        
        # Build summary text
        summary_lines = [
            f"[bold]Results[/bold]\n",
            f"  [green]✓ All patterns:[/green] {success_count} questions",
            f"  [yellow]◐ Partial:[/yellow] {partial_count} questions",
            f"  [red]✗ None:[/red] {fail_count} questions\n",
            f"  [{color}]Recall: {total_found}/{total_patterns} ({recall_pct:.1f}%)[/{color}]",
            f"  Position: {total_position_score:.2f}/{max_possible_score:.0f} ({position_pct:.1f}%)"
        ]
        
        # Add rerank comparison if available
        if no_rerank_position_score is not None:
            no_rerank_pct = (no_rerank_position_score / max_possible_score * 100)
            delta = total_position_score - no_rerank_position_score
            delta_pct = (delta / max_possible_score * 100)
            
            if delta > 0:
                delta_str = f"[green]↑ +{delta:.2f} (+{delta_pct:.1f}%)[/green]"
            elif delta < 0:
                delta_str = f"[red]↓ {delta:.2f} ({delta_pct:.1f}%)[/red]"
            else:
                delta_str = "[dim]= 0[/dim]"
            
            summary_lines.append(f"\n  [dim]Position (no-rerank): {no_rerank_position_score:.2f} ({no_rerank_pct:.1f}%)[/dim]")
            summary_lines.append(f"  Rerank Gain: {delta_str}")
        
        console.print(Panel.fit(
            "\n".join(summary_lines),
            border_style=color,
            title="📊 Summary"
        ))
        
        summary_dict = {
            "total_questions": len(results),
            "all_found": success_count,
            "partial": partial_count,
            "none_found": fail_count,
            "patterns_found": total_found,
            "patterns_total": total_patterns,
            "recall_rate": recall_pct,
            "recall_score": total_recall_score,
            "position_score": total_position_score,
            "position_rate": position_pct,
            "max_score": max_possible_score
        }
        
        if no_rerank_position_score is not None:
            summary_dict["position_score_no_rerank"] = no_rerank_position_score
            summary_dict["rerank_gain"] = total_position_score - no_rerank_position_score
        
        return summary_dict

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
    parser.add_argument(
        "--alpha",
        type=str,
        choices=list(ALPHA_FUNCTIONS.keys()),
        default=DEFAULT_ALPHA,
        help=f"Alpha decay function for position scoring (default: {DEFAULT_ALPHA})"
    )

    args = parser.parse_args()

    try:
        benchmark = RAGBenchmark(
            questions_path=args.questions, 
            top_k=args.top_k,
            recall_k=args.recall,
            use_reranking=False if args.no_rerank else None,
            alpha_fn=args.alpha
        )
        benchmark.run(question_ids=args.ids)
    except FileNotFoundError as e:
        console.print(f"[red]❌ Error:[/red] {e}")
        console.print("[yellow]💡 Run 'uv run index --build' first.[/yellow]")


if __name__ == "__main__":
    main()
