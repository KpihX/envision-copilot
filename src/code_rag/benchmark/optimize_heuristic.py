#!/usr/bin/env python3
"""
Heuristic Reranker Parameter Optimizer

Intelligent staged grid search over ALL heuristic parameters.
Reuses benchmarking logic from main.py to ensuring consistency.
"""

import json
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import uuid
from dataclasses import dataclass

# Import core benchmark logic
from code_rag.benchmark.main import RAGBenchmark
from code_rag.benchmark.common import (
    alpha_linear, 
    alpha_log,
    compute_position_score
)
from code_rag.vector_engines import get_retriever, BaseRetriever
from code_rag.rerankers import HeuristicReranker
from code_rag.utils import ConfigLoader

# Suppress verbose output
logging.getLogger("code_rag").setLevel(logging.WARNING)


# ============================================================================
# PARAMETER SPACE DEFINITION
# ============================================================================

@dataclass
class ParamSpec:
    name: str
    values: List[float]

PARAM_SPECS = {
    # Stage 1: Main Weights (~2400 combos)
    # Step 0.1 instead of 0.2 for finer granularitry
    "weights": [
        ParamSpec("technical_term_boost", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),  # Likely high
        ParamSpec("pattern_density", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),           # Likely medium
        ParamSpec("definition_boost", [0.0, 0.1, 0.2]),                         # Likely low/zero
        ParamSpec("diversity_penalty", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), # Full range
    ],
    
    # Stage 2: TTB Scores (~125 combos)
    # Finer tuning around 0.2-0.6 range
    "ttb_scores": [
        ParamSpec("path_match", [0.3, 0.4, 0.5, 0.6, 0.7]),
        ParamSpec("formula_match", [0.2, 0.3, 0.4, 0.5, 0.6]),
        ParamSpec("identifier_match", [0.1, 0.2, 0.3, 0.4, 0.5]),
    ],
    
    # Stage 3: PDS Params (~20 combos)
    "pds_params": [
        ParamSpec("density_bonus_threshold", [0.4, 0.5, 0.6, 0.7, 0.8]),
        ParamSpec("density_bonus", [0.1, 0.15, 0.2, 0.25]),
    ],
    
    # Stage 4: SDP Penalties (~64 combos)
    "sdp_penalties": [
        ParamSpec("first_repeat", [0.05, 0.1, 0.15, 0.2]),
        ParamSpec("second_repeat", [0.15, 0.2, 0.25, 0.3]),
        ParamSpec("third_plus", [0.3, 0.35, 0.4, 0.45]),
    ],
}

# Global retriever instance to avoid reloading FAISS index repeatedly
GLOBAL_RETRIEVER = None

def get_retriever():
    global GLOBAL_RETRIEVER
    if GLOBAL_RETRIEVER is None:
        GLOBAL_RETRIEVER = get_retriever(verbose=False)
        # Force load index now
        GLOBAL_RETRIEVER._ensure_loaded()
    return GLOBAL_RETRIEVER

def run_benchmark_pass(retriever: BaseRetriever, questions: List[Dict], top_k: int) -> float:
    """Run a silent benchmark pass and return total position score."""
    total_score = 0.0
    recall_k = 50 
    
    for q in questions:
        query = q["question"]
        patterns = q["patterns"]
        
        # Retrieval
        response = retriever.query(query, top_k=top_k, recall_k=recall_k)
        chunks = response.get("results", [])
        
        # Calculate score using shared logic
        pattern_first_ranks = {}
        for pattern in patterns:
            pattern_lower = pattern.lower()
            for i, chunk in enumerate(chunks):
                chunk_text = (chunk.get("text", "") + " " + chunk.get("source", "")).lower()
                if pattern_lower in chunk_text:
                    if pattern not in pattern_first_ranks:
                        pattern_first_ranks[pattern] = i + 1
                    break
        
        # Reuse robust scoring function from main.py
        score = compute_position_score(patterns, pattern_first_ranks, alpha_linear)
        total_score += score
        
    return total_score

def run_evaluation(config: Dict, questions: List[Dict], top_k: int) -> Tuple[float, float, float]:
    """
    Run evaluation using shared retriever.
    Returns (score_rerank, score_no_rerank, gain).
    """
    retriever = get_retriever()
    
    # 1. Evaluate with Reranker
    retriever.set_reranking(True)
    retriever.reranker = HeuristicReranker(config=config)
    score_rerank = run_benchmark_pass(retriever, questions, top_k)
    
    # 2. Evaluate without Reranker
    retriever.set_reranking(False)
    score_no_rerank = run_benchmark_pass(retriever, questions, top_k)
    
    return score_rerank, score_no_rerank, score_rerank - score_no_rerank

def optimize_section(section_name: str, base_config: Dict, questions: List[Dict], top_k: int) -> Tuple[Dict, float]:
    specs = PARAM_SPECS[section_name]
    keys = [s.name for s in specs]
    value_lists = [s.values for s in specs]
    all_combos = list(itertools.product(*value_lists))
    
    best_gain = float('-inf')
    best_params = None
    
    for i, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))
        test_config = {**base_config}
        test_config[section_name] = params
        
        # Progress indicator every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"   Testing [{i+1}/{len(all_combos)}]...", end="\r", flush=True)

        try:
            _, _, gain = run_evaluation(test_config, questions, top_k)
            
            if gain > best_gain:
                best_gain = gain
                best_params = params.copy()
                print(f"   ⭐ New best: {params} → gain={gain:.3f}") # flush implicitly
                
        except Exception:
            pass 
            
    return best_params, best_gain

def staged_optimization(questions: List[Dict], top_k: int = 50) -> Dict:
    print("=" * 70)
    print("STAGED HEURISTIC PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Questions: {len(questions)}, Top-K: {top_k}\n")
    
    loader = ConfigLoader.load_config()
    baseline_heuristic = loader.get("heuristic_reranking", {})
    
    base_config = {
        "weights": baseline_heuristic.get("weights", {"technical_term_boost": 0.7, "pattern_density": 0.4, "definition_boost": 0.0, "diversity_penalty": 0.9}),
        "ttb_scores": baseline_heuristic.get("ttb_scores", {"path_match": 0.5, "formula_match": 0.4, "identifier_match": 0.3}),
        "pds_params": baseline_heuristic.get("pds_params", {"density_bonus_threshold": 0.5, "density_bonus": 0.2}),
        "db_scores": baseline_heuristic.get("db_scores", {"definition_match": 0.5, "module_location": 0.2}),
        "sdp_penalties": baseline_heuristic.get("sdp_penalties", {"first_repeat": 0.15, "second_repeat": 0.30, "third_plus": 0.45}),
    }

    print("📊 Baseline evaluation...")
    _, _, baseline_gain = run_evaluation(base_config, questions, top_k)
    print(f"   Baseline gain: {baseline_gain:.3f}\n")
    
    for stage, section in enumerate(["weights", "ttb_scores", "pds_params", "sdp_penalties"], 1):
        print(f"STAGE {stage}: Optimizing {section}")
        best, _ = optimize_section(section, base_config, questions, top_k)
        if best:
            base_config[section] = best
        print()
    
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    score_rerank, score_no_rerank, final_gain = run_evaluation(base_config, questions, top_k)
    max_score = len(questions)
    
    print(f"\n🏆 OPTIMIZED CONFIGURATION:\n")
    print(json.dumps(base_config, indent=2))
    
    print(f"\n   Position (rerank): {score_rerank:.2f}/{max_score}")
    print(f"   Position (no-rerank): {score_no_rerank:.2f}/{max_score}")
    print(f"   Rerank Gain: {final_gain:+.2f}")
    
    return {
        "config": base_config,
        "score_rerank": score_rerank,
        "score_no_rerank": score_no_rerank,
        "gain": final_gain
    }

def main():
    try:
        # Prefer loading via standard Benchmark class
        bench = RAGBenchmark()
        questions = bench.pattern_questions
    except Exception:
        # Fallback: Load from config
        config = ConfigLoader.load_config()
        project_root = Path(__file__).parent.parent.parent.parent
        q_file = config.get("benchmark", {}).get("questions_file", "src/code_rag/benchmark/questions.json")
        q_path = project_root / q_file
        
        if not q_path.exists():
             # Last resort relative path
             q_path = Path(__file__).parent / "questions.json"
        
        with open(q_path) as f:
            questions = json.load(f).get("pattern_based", [])

    print(f"Loaded {len(questions)} questions\n")
    results = staged_optimization(questions, top_k=50)

    # Save results to standardized log
    # data/logs/optimize_heuristic/YYYY-MM-DD_HH-MM-SS_uuid.json
    output_dir = project_root / "datas/code_rag/optimize_heuristic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = uuid.uuid4().hex[:8]
    output_file = output_dir / f"{timestamp}_{unique_id}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": results["config"],
            "metrics": {
                "score_rerank": results["score_rerank"],
                "score_no_rerank": results["score_no_rerank"],
                "gain": results["gain"]
            }
        }, f, indent=2)
        
    print(f"\n📄 Results saved to {output_file}")

if __name__ == "__main__":
    main()
