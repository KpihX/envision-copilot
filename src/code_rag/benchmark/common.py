"""
Common Benchmark Utilities - Shared metrics and IO for ranking evaluations.
"""

import math
import json
from typing import List, Dict, Callable, Any
from pathlib import Path

# ============================================================================
# ALPHA FUNCTIONS - Position decay functions for ranking score
# ============================================================================

def alpha_log(rank: int) -> float:
    """
    Logarithmic decay: α = 1 / (1 + log(rank))
    
    Smooth decay that doesn't penalize too harshly:
    - Rank 1: 1.00
    - Rank 5: 0.59
    - Rank 10: 0.43
    - Rank 50: 0.25
    """
    return 1.0 / (1.0 + math.log(rank))


def alpha_linear(rank: int, max_rank: int = 100) -> float:
    """
    Linear decay: α = max(0, 1 - rank/max_rank)
    
    Simple linear decay:
    - Rank 1: 0.99
    - Rank 50: 0.50
    - Rank 100: 0.00
    """
    return max(0.0, 1.0 - rank / max_rank)


# Default alpha function registry
ALPHA_FUNCTIONS = {
    "log": alpha_log,
    "linear": alpha_linear
}
# Default strategy (can be overridden by config, but defined here for fallback)
DEFAULT_ALPHA = "linear"


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def compute_position_score(
    patterns: List[str], 
    pattern_first_ranks: Dict[str, int],
    alpha_fn: Callable[[int], float] = None
) -> float:
    """
    Compute position score for a question.
    
    Score = Σ (α(rank) × 1/n) for each pattern found
    
    Where:
    - n = total number of patterns for the question
    - α(rank) = decay function based on first occurrence rank
    - If pattern not found, contributes 0
    
    Args:
        patterns: List of patterns to find
        pattern_first_ranks: Dict mapping pattern -> first rank where found
        alpha_fn: Position decay function (defaults to alpha_linear)
    
    Returns:
        Position score in [0, 1]
    """
    if alpha_fn is None:
        alpha_fn = alpha_linear
    
    n = len(patterns)
    if n == 0:
        return 0.0
        
    score_sum = 0.0
    for pattern in patterns:
        if pattern in pattern_first_ranks:
            rank = pattern_first_ranks[pattern]
            decay = alpha_fn(rank)
            score_sum += decay * (1.0 / n)
            
    return score_sum

    return score_sum

def load_questions(filepath: str = None) -> List[Dict[str, Any]]:
    """
    Load benchmark questions from JSON file.
    If filepath is None, tries to find questions.json in standard locations.
    """
    if filepath is None:
        # Try standard locations relative to project root or this file
        # Assuming this file is in src/code_rag/benchmark/common.py
        # root is ../../../..
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        
        candidates = [
            current_dir / "questions.json",
            project_root / "src/code_rag/benchmark/questions.json",
            project_root / "datas/code_rag/benchmark/questions.json"
        ]
        
        for cand in candidates:
            if cand.exists():
                path = cand
                break
        else:
            raise FileNotFoundError("Could not locate questions.json in standard paths.")
    else:
        path = Path(filepath)
        if not path.exists():
            # Try relative to project root if absolute failed
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent
            path_rel = project_root / filepath
            if path_rel.exists():
                path = path_rel
            else:
                raise FileNotFoundError(f"Questions file not found: {filepath}")
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Flatten if structured key exists (e.g. "pattern_based")
    if "pattern_based" in data:
        return data["pattern_based"]
    elif isinstance(data, list):
        return data
    else:
        # Fallback empty or error
        return []
