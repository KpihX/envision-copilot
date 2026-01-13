#!/usr/bin/env python3
"""
Benchmark Runner CLI.
Runs LLM-as-Judge benchmark on selected questions.

Usage:
    uv run envision-benchmark                     # Questions 1-5 (default)
    uv run envision-benchmark --from 1 --to 10   # Questions 1-10
    uv run envision-benchmark --ids 1 3 5 8      # Specific questions by ID
    uv run envision-benchmark --from 1 --to 5 --ids 8 12  # Range + extras
"""
import argparse
import sys
import yaml
from pathlib import Path
from typing import List, Optional

from envision_rag.benchmark.runner import BenchmarkRunner


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_question_selection(args) -> List[int]:
    """
    Parse question selection from arguments.
    Supports:
      - Range: --from X --to Y (inclusive)
      - Explicit IDs: --ids 1 3 5 8
      - Both combined
    """
    selected_ids = set()
    
    # 1. Add range if specified
    start = args.start if args.start else 1
    end = args.end if args.end else 5
    
    # Only add range if no explicit --ids OR if --from/--to were explicitly provided
    if args.ids is None or args.start is not None or args.end is not None:
        for i in range(start, end + 1):
            selected_ids.add(i)
    
    # 2. Add explicit IDs if provided
    if args.ids:
        for qid in args.ids:
            selected_ids.add(qid)
    
    return sorted(list(selected_ids))


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark on Envision RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run envision-benchmark                      # Default: questions 1-5
    uv run envision-benchmark --from 1 --to 10     # Questions 1-10
    uv run envision-benchmark --ids 1 3 5 8        # Specific questions
    uv run envision-benchmark --from 1 --to 3 --ids 10 12  # Combine range + IDs
        """
    )
    parser.add_argument(
        "--from", "-f",
        dest="start",
        type=int,
        default=None,
        help="Start of question range (default: 1)"
    )
    parser.add_argument(
        "--to", "-t",
        dest="end",
        type=int,
        default=None,
        help="End of question range, inclusive (default: 5)"
    )
    parser.add_argument(
        "--ids", "-i",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of question IDs to run (can be combined with range)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Show detailed output (default: True)"
    )
    
    args = parser.parse_args()
    
    # Determine which questions to run
    question_ids = parse_question_selection(args)
    
    print("🔬 Envision RAG Benchmark")
    print("=" * 40)
    print(f"📋 Selected Questions: {question_ids}")
    
    # Load config
    config = load_config()
    
    # Run benchmark
    runner = BenchmarkRunner(config, verbose=args.verbose)
    report = runner.run(question_ids=question_ids)
    
    # Exit with appropriate code
    if report.accuracy >= 0.8:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
