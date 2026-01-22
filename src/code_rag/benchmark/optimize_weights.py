#!/usr/bin/env python3
"""
Heuristic Reranker Weight Optimizer

Grid search over weight combinations to find the best rerank gain.
Imports scoring functions from benchmark.main for consistency.
"""

import json
import itertools
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from code_rag.retriever import GraphRetriever
from code_rag.rerankers import HeuristicReranker
from code_rag.utils import ConfigLoader

# Import scoring functions from main.py (single source of truth)
from .main import alpha_linear, compute_position_score


def run_benchmark_pass(retriever: GraphRetriever, questions: List[Dict], top_k: int = 50) -> float:
    """
    Run a single benchmark pass and return total position score.
    Uses the same scoring as benchmark.main.
    """
    total_score = 0.0
    
    for q in questions:
        query = q["question"]
        patterns = q["patterns"]
        
        response = retriever.query(query, top_k=top_k, recall_k=50)
        chunks = response.get("results", [])
        
        # Find first rank for each pattern (1-indexed)
        pattern_first_ranks = {}
        for pattern in patterns:
            pattern_lower = pattern.lower()
            for i, chunk in enumerate(chunks):
                chunk_text = (chunk.get("text", "") + " " + chunk.get("source", "")).lower()
                if pattern_lower in chunk_text:
                    if pattern not in pattern_first_ranks:
                        pattern_first_ranks[pattern] = i + 1
                    break
        
        # Use shared scoring function
        score = compute_position_score(patterns, pattern_first_ranks, alpha_linear)
        total_score += score
    
    return total_score


def evaluate_weights(heuristic_config: Dict, questions: List[Dict], top_k: int = 50):
    """Evaluate a weight configuration. Returns (score_rerank, score_no_rerank, gain)."""
    retriever = GraphRetriever()
    retriever._use_reranker = True
    retriever.reranker = HeuristicReranker(config=heuristic_config)
    
    score_rerank = run_benchmark_pass(retriever, questions, top_k)
    
    retriever.set_reranking(False)
    score_no_rerank = run_benchmark_pass(retriever, questions, top_k)
    
    return score_rerank, score_no_rerank, score_rerank - score_no_rerank


def grid_search(questions: List[Dict], top_k: int = 50, num_iterations: int = 15000) -> Dict:
    """Perform grid search over weight combinations."""
    import random
    
    # Weight ranges: 0.0 to 1.0 by 0.1 step
    weight_range = [round(x * 0.1, 1) for x in range(11)]
    
    weight_options = {
        "technical_term_boost": weight_range,
        "pattern_density": weight_range,
        "definition_boost": weight_range,
        "diversity_penalty": weight_range,
    }
    
    keys = list(weight_options.keys())
    all_combinations = list(itertools.product(*[weight_options[k] for k in keys]))
    
    print(f"[OPT] {len(all_combinations)} combos, testing {min(num_iterations, len(all_combinations))}")
    
    results = []
    best_gain = float('-inf')
    best_weights = None
    
    combos = random.sample(all_combinations, min(num_iterations, len(all_combinations)))
    
    for i, combo in enumerate(combos):
        weights = dict(zip(keys, combo))
        heuristic_config = {"weights": weights}
        
        print(f"[{i+1}/{len(combos)}] TTB={weights['technical_term_boost']:.1f} "
              f"PDS={weights['pattern_density']:.1f} DB={weights['definition_boost']:.1f} "
              f"SDP={weights['diversity_penalty']:.1f}", end="")
        
        try:
            score_rerank, score_no_rerank, gain = evaluate_weights(heuristic_config, questions, top_k)
            max_score = len(questions)
            pct_rerank = (score_rerank / max_score) * 100
            pct_gain = ((score_rerank - score_no_rerank) / max_score) * 100
            
            results.append({
                "weights": weights,
                "score_rerank": score_rerank,
                "score_no_rerank": score_no_rerank,
                "pct_rerank": pct_rerank,
                "pct_gain": pct_gain,
                "gain": gain
            })
            
            print(f" → {pct_rerank:.1f}% | gain: {pct_gain:+.1f}%")
            
            if gain > best_gain:
                best_gain = gain
                best_weights = weights.copy()
                print(f"   ⭐ NEW BEST!")
                
        except Exception as e:
            print(f" → ERROR: {e}")
    
    return {
        "best_weights": best_weights,
        "best_gain": best_gain,
        "max_score": len(questions),
        "all_results": sorted(results, key=lambda x: x["gain"], reverse=True)
    }


def main():
    print("=" * 60)
    print("Heuristic Reranker Weight Optimizer")
    print("=" * 60)
    
    config = ConfigLoader.load_config()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    questions_file = config.get("benchmark", {}).get("questions_file", "src/code_rag/benchmark/questions.json")
    questions_path = project_root / questions_file
    
    if not questions_path.exists():
        questions_path = script_dir / "questions.json"
    
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data.get("pattern_based", [])
    top_k = 50  # Match benchmark -t 50
    
    print(f"Loaded {len(questions)} questions, top_k={top_k}\n")
    
    results = grid_search(questions, top_k=top_k, num_iterations=15000)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60 + "\n")
    
    best = results["all_results"][0]
    max_score = results["max_score"]
    
    print("🏆 BEST WEIGHTS:")
    for key, value in results["best_weights"].items():
        print(f"   {key}: {value}")
    
    print(f"\n   Position (rerank): {best['score_rerank']:.2f}/{max_score} ({best['pct_rerank']:.1f}%)")
    print(f"   Position (no-rerank): {best['score_no_rerank']:.2f}/{max_score}")
    print(f"   Rerank Gain: {best['gain']:+.2f} ({best['pct_gain']:+.1f}%)")
    
    print("\n📊 Top 5 Configurations:")
    for i, r in enumerate(results["all_results"][:5]):
        print(f"   {i+1}. Gain={r['pct_gain']:+.1f}% | "
              f"TTB={r['weights']['technical_term_boost']:.1f}, "
              f"PDS={r['weights']['pattern_density']:.1f}, "
              f"DB={r['weights']['definition_boost']:.1f}, "
              f"SDP={r['weights']['diversity_penalty']:.1f}")
    
    output_file = project_root / "data/logs/weight_optimization.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "top_k": top_k,
            "num_questions": len(questions),
            "best_weights": results["best_weights"],
            "best_gain": best["gain"],
            "best_pct_gain": best["pct_gain"],
            "top_10": results["all_results"][:10]
        }, f, indent=2)
    
    print(f"\n📄 Results saved to {output_file}")
    return results


if __name__ == "__main__":
    main()
