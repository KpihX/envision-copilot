#!/usr/bin/env python3
"""
Benchmark Runner CLI.
Runs LLM-as-Judge benchmark on first N questions.

Usage:
    uv run envision-benchmark -n 5
    uv run envision-benchmark --num-questions 10 --verbose
"""
import argparse
import sys
import yaml
from pathlib import Path

# Fix imports
sys.path.append(str(Path(__file__).parent.parent))

from envision_rag.benchmark.runner import BenchmarkRunner


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark on Envision RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run envision-benchmark -n 5         # Test first 5 questions
    uv run envision-benchmark -n 10        # Test first 10 questions
        """
    )
    parser.add_argument(
        "-n", "--num-questions",
        type=int,
        default=5,
        help="Number of questions to test (default: 5)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Show detailed output (default: True)"
    )
    
    args = parser.parse_args()
    
    print("🔬 Envision RAG Benchmark")
    print("=" * 40)
    
    # Load config
    config = load_config()
    
    # Run benchmark
    runner = BenchmarkRunner(config, verbose=args.verbose)
    report = runner.run(n=args.num_questions)
    
    # Exit with appropriate code
    if report.accuracy >= 0.8:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
