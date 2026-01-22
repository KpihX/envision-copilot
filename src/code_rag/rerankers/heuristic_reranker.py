"""
Heuristic Reranker - Domain-specific reranking for Envision code.

Strategies implemented:
1. TTB (Technical Term Boost): Boost chunks containing exact technical terms from query
2. SDP (Source Diversity Penalty): Penalize consecutive chunks from same source
4. PDS (Pattern Density Score): Favor chunks with multiple query terms
5. DB (Definition Boost): Boost chunks that define/declare variables

All parameters are configurable via config.yaml under heuristic_reranking section.
Note: Strategy 3 (QSCM - Query-Source Category Match) is deferred for later.
"""

import re
from typing import List, Dict, Any, Tuple, Set
from .base import BaseReranker


# Regex patterns for technical term detection
CAMEL_CASE_RE = re.compile(r'[a-z]+[A-Z][a-zA-Z]*')  # camelCase
PASCAL_CASE_RE = re.compile(r'[A-Z][a-z]+[A-Z][a-zA-Z]*')  # PascalCase
SNAKE_CASE_RE = re.compile(r'[a-z]+_[a-z_]+')  # snake_case
PATH_PATTERN_RE = re.compile(r'/[0-9]*\.?\s*[A-Za-z][^/\s]*(?:/[^/\s]+)*')  # Script paths
FORMULA_RE = re.compile(r'[\w.]+\s*=\s*[^=]')  # Variable assignments


class HeuristicReranker(BaseReranker):
    """
    Heuristic reranker optimized for Envision code retrieval.
    
    Uses domain-specific signals instead of ML models:
    - Technical term matching (camelCase, PascalCase, paths)
    - Source diversity to avoid redundant results
    - Pattern density for multi-term queries
    - Definition detection for code declarations
    
    All parameters are loaded from config.yaml.
    """
    
    # Default values (used as fallback if config is missing)
    DEFAULT_CONFIG = {
        "weights": {
            "technical_term_boost": 0.7,
            "pattern_density": 0.4,
            "definition_boost": 0.0,
            "diversity_penalty": 0.9,
        },
        "ttb_scores": {
            "path_match": 0.5,
            "formula_match": 0.4,
            "identifier_match": 0.3,
        },
        "pds_params": {
            "density_bonus_threshold": 0.5,
            "density_bonus": 0.2,
        },
        "db_scores": {
            "definition_match": 0.5,
            "module_location": 0.2,
        },
        "sdp_penalties": {
            "first_repeat": 0.15,
            "second_repeat": 0.30,
            "third_plus": 0.45,
        },
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize heuristic reranker.
        
        Args:
            config: Full heuristic_reranking config from config.yaml
        """
        super().__init__("heuristic")
        
        # Merge config with defaults
        cfg = config or {}
        
        # Load all parameters from config with defaults
        self.weights = {**self.DEFAULT_CONFIG["weights"], **cfg.get("weights", {})}
        self.ttb_scores = {**self.DEFAULT_CONFIG["ttb_scores"], **cfg.get("ttb_scores", {})}
        self.pds_params = {**self.DEFAULT_CONFIG["pds_params"], **cfg.get("pds_params", {})}
        self.db_scores = {**self.DEFAULT_CONFIG["db_scores"], **cfg.get("db_scores", {})}
        self.sdp_penalties = {**self.DEFAULT_CONFIG["sdp_penalties"], **cfg.get("sdp_penalties", {})}
        
        self._loaded = True  # No model to load
    
    def load(self) -> None:
        """No model to load for heuristic reranker."""
        pass
    
    def _predict(self, pairs: List[List[str]]) -> List[float]:
        """Not used - we override rank() directly."""
        raise NotImplementedError("HeuristicReranker uses rank() directly")
    
    def rank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 5,
        use_contextual: bool = True  # Ignored for heuristic
    ) -> List[Tuple[int, float]]:
        """
        Rerank candidates using domain-specific heuristics.
        
        Args:
            query: The search query
            candidates: List of chunk dicts with 'text', 'source', 'content'
            top_k: Number of top results to return
            use_contextual: Ignored (always uses full chunk context)
            
        Returns:
            List of (original_index, score) sorted by score desc
        """
        if not candidates:
            return []
        
        # Extract technical terms from query
        query_terms = self._extract_technical_terms(query)
        query_words = self._extract_significant_words(query)
        
        # Score each candidate
        scored = []
        for idx, cand in enumerate(candidates):
            # Get chunk text (handle both dict and string)
            if isinstance(cand, str):
                text = cand
                source = ""
            else:
                text = cand.get("text", "") or cand.get("content", "")
                source = cand.get("source", "")
            
            # Base score (normalized position - earlier = higher baseline)
            base_score = 1.0 - (idx / len(candidates)) * 0.3
            
            # Strategy 1: Technical Term Boost (TTB)
            ttb_score = self._compute_ttb(query_terms, text, source)
            
            # Strategy 4: Pattern Density Score (PDS)
            pds_score = self._compute_pds(query_words, text)
            
            # Strategy 5: Definition Boost (DB)
            db_score = self._compute_db(query_terms, text)
            
            # Combine scores using configurable weights
            final_score = (
                base_score +
                self.weights["technical_term_boost"] * ttb_score +
                self.weights["pattern_density"] * pds_score +
                self.weights["definition_boost"] * db_score
            )
            
            scored.append((idx, final_score, source))
        
        # Sort by score desc
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Strategy 2: Source Diversity Penalty (SDP)
        diversified = self._apply_diversity_penalty(scored)
        
        # Return top-k with just (idx, score)
        return [(idx, score) for idx, score, _ in diversified[:top_k]]
    
    def _extract_technical_terms(self, text: str) -> Set[str]:
        """
        Extract technical terms from text.
        
        Detects:
        - camelCase identifiers (stockEvol, carryingCost)
        - PascalCase identifiers (IsTopItem, ReDispatchCycle)
        - snake_case identifiers
        - Script paths (but NOT comment markers //)
        - Formulas with = sign
        """
        terms = set()
        
        # camelCase
        for match in CAMEL_CASE_RE.finditer(text):
            terms.add(match.group())
        
        # PascalCase
        for match in PASCAL_CASE_RE.finditer(text):
            terms.add(match.group())
        
        # snake_case
        for match in SNAKE_CASE_RE.finditer(text):
            terms.add(match.group())
        
        # Script paths - must start with / and contain real path components
        for match in PATH_PATTERN_RE.finditer(text):
            path = match.group()
            if not path.startswith("//") and "://" not in path:
                terms.add(path)
        
        # Formula patterns (e.g., "carryingCost = 0.3")
        for match in FORMULA_RE.finditer(text):
            var_part = match.group().split("=")[0].strip()
            if var_part and not var_part.isdigit():
                terms.add(var_part)
        
        return terms
    
    def _extract_significant_words(self, text: str) -> Set[str]:
        """Extract significant words from text (excluding stopwords)."""
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "what", "which", "who", "whom", "this",
            "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
            "he", "him", "his", "she", "her", "it", "its", "they", "them", "their"
        }
        
        words = set()
        for word in re.findall(r'\b[a-zA-Z]+\b', text.lower()):
            if word not in stopwords and len(word) > 2:
                words.add(word)
        
        return words
    
    def _compute_ttb(self, query_terms: Set[str], chunk_text: str, chunk_source: str) -> float:
        """
        Strategy 1: Technical Term Boost.
        
        Boost chunks containing exact technical terms from the query.
        Uses configurable scores from ttb_scores.
        """
        if not query_terms:
            return 0.0
        
        score = 0.0
        chunk_lower = chunk_text.lower()
        source_lower = chunk_source.lower()
        combined = chunk_lower + " " + source_lower
        
        for term in query_terms:
            term_lower = term.lower()
            
            if term_lower in combined:
                # Use configurable scores
                if term.startswith("/"):
                    score += self.ttb_scores["path_match"]
                elif "=" in term or "." in term:
                    score += self.ttb_scores["formula_match"]
                else:
                    score += self.ttb_scores["identifier_match"]
        
        # Normalize by number of terms (max 1.0)
        return min(score / max(len(query_terms), 1), 1.0)
    
    def _compute_pds(self, query_words: Set[str], chunk_text: str) -> float:
        """
        Strategy 4: Pattern Density Score.
        
        Favor chunks containing multiple significant words from the query.
        Uses configurable thresholds from pds_params.
        """
        if not query_words:
            return 0.0
        
        chunk_lower = chunk_text.lower()
        matches = sum(1 for word in query_words if word in chunk_lower)
        
        density = matches / len(query_words)
        
        # Extra bonus if density exceeds threshold (configurable)
        if density > self.pds_params["density_bonus_threshold"]:
            density += self.pds_params["density_bonus"]
        
        return min(density, 1.0)
    
    def _compute_db(self, query_terms: Set[str], chunk_text: str) -> float:
        """
        Strategy 5: Definition Boost.
        
        Boost chunks that appear to define/declare variables or functions.
        Uses configurable scores from db_scores.
        """
        score = 0.0
        
        # Definition patterns in Envision
        definition_patterns = [
            r'export\s+const\s+(\w+)',
            r'export\s+table\s+(\w+)',
            r'const\s+(\w+)\s*=',
            r'def\s+(\w+)',
            r'table\s+(\w+)\s*=',
            r'^(\w+)\s*=\s*\{',
        ]
        
        for pattern in definition_patterns:
            for match in re.finditer(pattern, chunk_text, re.MULTILINE):
                defined_name = match.group(1)
                for term in query_terms:
                    if term.lower() in defined_name.lower() or defined_name.lower() in term.lower():
                        score += self.db_scores["definition_match"]
                        break
        
        # Boost for Modules/Functions paths (configurable)
        if "modules" in chunk_text.lower() or "functions" in chunk_text.lower():
            score += self.db_scores["module_location"]
        
        return min(score, 1.0)
    
    def _apply_diversity_penalty(
        self, 
        scored: List[Tuple[int, float, str]]
    ) -> List[Tuple[int, float, str]]:
        """
        Strategy 2: Source Diversity Penalty.
        
        Penalize chunks from the same source progressively.
        Uses configurable penalties from sdp_penalties.
        """
        if not scored:
            return scored
        
        penalty_factor = self.weights["diversity_penalty"]
        result = []
        source_counts = {}
        
        for idx, score, source in scored:
            source_key = source.strip().lower() if source else f"unknown_{idx}"
            count = source_counts.get(source_key, 0)
            source_counts[source_key] = count + 1
            
            # Apply progressive penalty (configurable)
            if count == 0:
                penalty = 0.0
            elif count == 1:
                penalty = self.sdp_penalties["first_repeat"] * penalty_factor
            elif count == 2:
                penalty = self.sdp_penalties["second_repeat"] * penalty_factor
            else:
                penalty = self.sdp_penalties["third_plus"] * penalty_factor
            
            adjusted_score = max(0, score - penalty)
            result.append((idx, adjusted_score, source))
        
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def __repr__(self) -> str:
        return f"HeuristicReranker(weights={self.weights})"
