"""
Benchmark Runner for Envision RAG.
Orchestrates benchmark execution and collects results.
"""
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from envision_rag.graph.builder import GraphBuilder
from envision_rag.tools.graph_tools import GraphTools
from envision_rag.workflow.agent import AgentWorkflow
from envision_rag.benchmark.judge import BenchmarkJudge, JudgeResult
from envision_rag.logging.session_logger import SessionLogger


@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: int
    question: str
    agent_answer: str
    judge_result: JudgeResult
    
    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "agent_answer": self.agent_answer,
            "judge_result": asdict(self.judge_result)
        }


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    total: int
    passed: int
    failed: int
    accuracy: float
    average_score: float
    results: List[QuestionResult]
    
    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "accuracy": self.accuracy,
            "average_score": self.average_score,
            "results": [r.to_dict() for r in self.results]
        }


class BenchmarkRunner:
    """
    Runs benchmarks on the RAG system.
    
    Usage:
        runner = BenchmarkRunner(config, verbose=True)
        report = runner.run(n=5)
    """
    
    def __init__(self, config: Dict[str, Any], verbose: bool = True):
        """
        Initialize benchmark runner.
        
        Args:
            config: Full config dict
            verbose: Show detailed output (default True)
        """
        self.config = config
        self.verbose = verbose
        self.console = Console()
        
        # Load config values
        benchmark_config = config.get("benchmark", {})
        self.judge_model = benchmark_config.get("judge_model", "mistral")
        self.questions_file = benchmark_config.get("questions_file", "questions.json")
        
        logging_config = config.get("logging", {})
        self.log_dir = logging_config.get("log_dir", "data/logs")
        self.logging_enabled = logging_config.get("enabled", True)
        
        # Initialize logger
        self.logger = SessionLogger(
            log_type="benchmark",
            log_dir=self.log_dir,
            enabled=self.logging_enabled
        )
        
        # Initialize components
        self.judge = BenchmarkJudge(judge_model=self.judge_model)
        self.agent_app = None
        self.questions = []
    
    def _load_questions(self) -> List[dict]:
        """Load questions from JSON file."""
        with open(self.questions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Only use answered questions for benchmark
        return data.get("answered", [])
    
    def _init_agent(self):
        """Initialize the RAG agent."""
        data_dir = Path("data")
        graph_path = data_dir / "dependency_graph.json"
        
        # Load graph
        from envision_rag.graph.graph_types import DependencyGraph
        graph = DependencyGraph()
        graph.load(str(graph_path))
        
        # Build agent
        tools = GraphTools(graph)
        workflow = AgentWorkflow(self.config, tools, verbose=self.verbose)
        self.agent_app = workflow.build_graph()
    
    def _log(self, title: str, content: str, style: str = "info"):
        """Log to console and logger."""
        self.logger.log_event(style, title, content, style)
        
        if not self.verbose:
            return
            
        if style == "question":
            self.console.print(Panel(Markdown(content), title=f"❓ {title}", border_style="yellow"))
        elif style == "answer":
            self.console.print(Panel(Markdown(content), title=f"🤖 {title}", border_style="cyan"))
        elif style == "expected":
            self.console.print(Panel(content, title=f"📋 {title}", border_style="blue"))
        elif style == "verdict":
            color = "green" if "✅" in title else "red"
            self.console.print(Panel(Markdown(content), title=title, border_style=color))
        else:
            self.console.print(f"[bold]{title}[/bold]: {content}")
    
    def run(self, question_ids: List[int] = None, n: int = None) -> BenchmarkReport:
        """
        Run benchmark on selected questions.
        
        Args:
            question_ids: List of question IDs to test (preferred)
            n: Number of questions (legacy, uses first n)
            
        Returns:
            BenchmarkReport with results
        """
        # Load all questions
        all_questions = self._load_questions()
        
        # Build question index by ID
        question_by_id = {q.get("id", i+1): q for i, q in enumerate(all_questions)}
        
        # Determine which questions to run
        if question_ids:
            # Use explicit IDs
            selected_questions = []
            for qid in question_ids:
                if qid in question_by_id:
                    selected_questions.append(question_by_id[qid])
                else:
                    self.console.print(f"[yellow]⚠️ Question ID {qid} not found, skipping[/yellow]")
        elif n:
            # Legacy: use first n questions
            selected_questions = all_questions[:n]
        else:
            # Default: first 5
            selected_questions = all_questions[:5]
        
        total = len(selected_questions)
        
        # Start logging session
        self.logger.start_session({
            "question_ids": question_ids or list(range(1, total + 1)),
            "total": total,
            "judge_model": self.judge_model,
            "questions_file": self.questions_file
        })
        
        self.console.print(f"\n🎯 [bold]Running Benchmark[/bold] ({total} questions)")
        self.console.print("=" * 60)
        
        # Initialize components
        self._init_agent()
        self.judge.initialize()
        
        results = []
        passed = 0
        total_score = 0.0
        
        for i, q in enumerate(selected_questions):
            q_id = q.get("id", i + 1)
            q_text = q.get("question", "")
            
            self.console.print(f"\n[bold cyan]Question {i+1}/{total} (ID: {q_id})[/bold cyan]")
            self._log(f"Question {q_id}", q_text, "question")
            
            # Run agent
            try:
                result = self.agent_app.invoke({
                    "question": q_text,
                    "scratchpad": "",
                    "messages": [],
                    "facts": []
                })
                
                messages = result.get("messages", [])
                agent_answer = messages[-1] if messages else "No response"
                
            except Exception as e:
                agent_answer = f"Error: {str(e)}"
            
            # Clean answer for display
            display_answer = agent_answer
            if len(display_answer) > 500:
                display_answer = display_answer[:500] + "..."
            
            self._log("Agent Answer", display_answer, "answer")
            
            # Show expected
            expected = q.get("answers", [])
            expected_str = "\n".join([f"• {a}" for a in expected])
            self._log("Expected Answer(s)", expected_str, "expected")
            
            # Judge
            judge_result = self.judge.evaluate(q, agent_answer)
            
            # Show verdict
            if judge_result.correct:
                passed += 1
                self._log("✅ CORRECT", f"Score: {judge_result.score:.0%}\n\n{judge_result.reasoning}", "verdict")
            else:
                self._log("❌ INCORRECT", f"Score: {judge_result.score:.0%}\n\n{judge_result.reasoning}", "verdict")
            
            total_score += judge_result.score
            
            results.append(QuestionResult(
                question_id=q_id,
                question=q_text,
                agent_answer=agent_answer,
                judge_result=judge_result
            ))
        
        # Summary
        accuracy = passed / total if total > 0 else 0
        avg_score = total_score / total if total > 0 else 0
        
        self.console.print("\n" + "=" * 60)
        self.console.print(f"[bold]📊 BENCHMARK RESULTS[/bold]")
        
        summary_table = Table(show_header=False, box=None)
        summary_table.add_row("Total Questions", str(total))
        summary_table.add_row("Passed", f"[green]{passed}[/green]")
        summary_table.add_row("Failed", f"[red]{total - passed}[/red]")
        summary_table.add_row("Accuracy", f"[bold]{accuracy:.1%}[/bold]")
        summary_table.add_row("Average Score", f"{avg_score:.2f}")
        self.console.print(summary_table)
        
        report = BenchmarkReport(
            total=total,
            passed=passed,
            failed=total - passed,
            accuracy=accuracy,
            average_score=avg_score,
            results=results
        )
        
        # End and save session
        self.logger.end_session({
            "total": total,
            "passed": passed,
            "accuracy": accuracy,
            "average_score": avg_score
        })
        
        log_path = self.logger.save()
        if log_path:
            self.console.print(f"\n📝 Log saved: {log_path}")
        
        return report

