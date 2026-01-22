#!/usr/bin/env python3
"""
Oriented RAG Benchmark (Agent Simulation)

Runs the standard benchmark but simulates an Agentic workflow:
1. Extracts "Patterns" from the known ground truth (questions.json).
2. Injects them as "Keywords" into the Retriever query.
3. Reranker uses these keywords to boost specific chunks.

This validates the effectiveness of the OrientedReranker when receiving proper hints.
"""

from datetime import datetime
import uuid
import pathlib
from pathlib import Path
from code_rag.benchmark.main import RAGBenchmark

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Oriented RAG Benchmark (Agent Simulation)")
    parser.add_argument("-t", "--top-k", type=int, help="Number of results to evaluate (k) - Default 5")
    parser.add_argument("--recall", "-r", type=int, default=50, help="Initial retrieval depth")
    parser.add_argument("--ids", type=str, help="Comma-separated list of Question IDs to run (e.g. 16,18)")
    args = parser.parse_args()
    
    q_ids = [int(i.strip()) for i in args.ids.split(",")] if args.ids else None

    print(f"🚀 Running Oriented RAG Benchmark (Agent Simulation) [Top-K: {args.top_k or 'Config'}]...")
    print("ℹ️  Injecting question patterns as Reranker Keywords.")
    print("    (Simulating an Agent that successfully identified terms)")
    
    # Initialize Benchmark with keyword injection enabled
    # We assume 'oriented' reranker is active in config.yaml
    bench = RAGBenchmark(
        top_k=args.top_k,
        recall_k=args.recall,
        use_reranking=True,
        pattern_as_keywords=True
    )
    
    # Override Log Dir (as requested by user)
    # Use oriented_log_dir from config if available
    bench_conf = bench.config.get("benchmark", {})
    or_log_dir = bench_conf.get("oriented_log_dir", "datas/code_rag/benchmark/oriented")
    
    if "benchmark" not in bench.config: bench.config["benchmark"] = {}
    bench.config["benchmark"]["log_dir"] = or_log_dir
    
    # Recalculate Logic path immediately
    project_root = Path(__file__).parent.parent.parent.parent
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{timestamp}_{unique_id}.json"
    bench.log_path = project_root / or_log_dir / filename
    
    # Ensure dir exists
    bench.log_path.parent.mkdir(parents=True, exist_ok=True)

    # Force Oriented Reranker (even if config says heuristic)
    print("ℹ️  Forcing Reranker Type: 'oriented'")
    bench.retriever.ret_config["reranker_type"] = "oriented"
    bench.retriever._init_reranker()
    
    # Run
    bench.run(question_ids=q_ids)

if __name__ == "__main__":
    main()
