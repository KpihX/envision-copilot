"""
LLM-as-Judge for benchmark evaluation.
Evaluates agent answers against ground truth using semantic understanding.
"""
import json
import re
from dataclasses import dataclass
from typing import List, Any, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from envision_rag.agents.prepare_agent import prepare_agent
from envision_rag.config_manager import get_config


@dataclass
class JudgeResult:
    """Result of LLM-as-Judge evaluation."""
    question_id: int
    question: str
    expected: List[Any]
    got: str
    score: float          # 0.0 - 1.0
    correct: bool         # Binary pass/fail
    reasoning: str        # LLM explanation
    deterministic: bool   # Was this an exact match question?


class BenchmarkJudge:
    """
    LLM-as-Judge for evaluating agent answers.
    
    For deterministic questions: exact match check
    For non-deterministic: semantic evaluation via LLM
    """
    
    def __init__(self, judge_model: str = "mistral"):
        """
        Initialize the judge.
        
        Args:
            judge_model: Model to use for semantic evaluation
        """
        self.judge_model = judge_model
        self.llm = None
    
    def initialize(self):
        """Initialize the judge LLM."""
        self.llm = prepare_agent(self.judge_model)
    
    def evaluate(self, question: dict, agent_answer: str) -> JudgeResult:
        """
        Evaluate agent answer against ground truth.
        
        Args:
            question: Question dict from questions.json
            agent_answer: The agent's full response
            
        Returns:
            JudgeResult with score, correctness, and reasoning
        """
        q_id = question.get("id", 0)
        q_text = question.get("question", "")
        expected = question.get("answers", [])
        deterministic = question.get("deterministic", True)
        appendix = question.get("appendix", [])
        
        # Extract just the answer part (after "Final Answer:" if present)
        clean_answer = agent_answer
        if "Final Answer:" in agent_answer:
            clean_answer = agent_answer.split("Final Answer:")[-1].strip()
        
        if deterministic:
            return self._evaluate_deterministic(q_id, q_text, expected, clean_answer, appendix)
        else:
            return self._evaluate_semantic(q_id, q_text, expected, clean_answer, appendix)
    
    def _evaluate_deterministic(self, q_id: int, q_text: str, expected: List[Any], 
                                 got: str, appendix: List[str]) -> JudgeResult:
        """
        Evaluate deterministic questions with exact/partial match.
        Handles numbers, lists, and yes/no answers.
        """
        got_lower = got.lower()
        
        # Case 1: Numeric answer
        if len(expected) == 1 and isinstance(expected[0], (int, float)):
            expected_num = expected[0]
            # Try to find the number in the answer
            numbers = re.findall(r'\b(\d+)\b', got)
            if numbers:
                # Check if any extracted number matches
                for num in numbers:
                    if int(num) == expected_num:
                        return JudgeResult(
                            question_id=q_id, question=q_text, expected=expected, got=got,
                            score=1.0, correct=True,
                            reasoning=f"Correct! Found expected number {expected_num}.",
                            deterministic=True
                        )
            
            # Fuzzy: check if close (within 10%)
            for num in numbers:
                if abs(int(num) - expected_num) / max(expected_num, 1) < 0.1:
                    return JudgeResult(
                        question_id=q_id, question=q_text, expected=expected, got=got,
                        score=0.8, correct=False,
                        reasoning=f"Close! Got {num}, expected {expected_num}.",
                        deterministic=True
                    )
            
            return JudgeResult(
                question_id=q_id, question=q_text, expected=expected, got=got,
                score=0.0, correct=False,
                reasoning=f"Incorrect. Expected {expected_num}, answer did not contain it.",
                deterministic=True
            )
        
        # Case 2: Yes/No/Oui/Non answer
        if len(expected) == 1 and str(expected[0]).lower() in ["oui", "non", "yes", "no"]:
            expected_bool = str(expected[0]).lower()
            positive = ["oui", "yes", "existe", "il y a", "effectivement"]
            negative = ["non", "no", "n'existe pas", "pas de"]
            
            is_positive = any(p in got_lower for p in positive)
            is_negative = any(n in got_lower for n in negative)
            
            expected_positive = expected_bool in ["oui", "yes"]
            
            if (expected_positive and is_positive and not is_negative) or \
               (not expected_positive and is_negative and not is_positive):
                return JudgeResult(
                    question_id=q_id, question=q_text, expected=expected, got=got,
                    score=1.0, correct=True,
                    reasoning=f"Correct boolean answer.",
                    deterministic=True
                )
            else:
                return JudgeResult(
                    question_id=q_id, question=q_text, expected=expected, got=got,
                    score=0.0, correct=False,
                    reasoning=f"Expected {'positive' if expected_positive else 'negative'} response.",
                    deterministic=True
                )
        
        # Case 3: List of paths/strings - check if any expected is mentioned
        matches = 0
        for exp in expected:
            exp_str = str(exp).lower()
            # Check variants: full path, basename, key terms
            if exp_str in got_lower:
                matches += 1
            elif "/" in exp_str:
                # Try basename
                basename = exp_str.split("/")[-1]
                if basename and basename in got_lower:
                    matches += 1
        
        total = len(expected)
        if total == 0:
            return JudgeResult(
                question_id=q_id, question=q_text, expected=expected, got=got,
                score=0.5, correct=False,
                reasoning="No expected answer provided.",
                deterministic=True
            )
        
        score = matches / total
        correct = score >= 0.8  # Allow 80% match for lists
        
        return JudgeResult(
            question_id=q_id, question=q_text, expected=expected, got=got,
            score=score, correct=correct,
            reasoning=f"Matched {matches}/{total} expected items ({score:.0%}).",
            deterministic=True
        )
    
    def _evaluate_semantic(self, q_id: int, q_text: str, expected: List[Any], 
                           got: str, appendix: List[str]) -> JudgeResult:
        """
        Evaluate non-deterministic questions using LLM-as-Judge.
        """
        if not self.llm:
            self.initialize()
        
        # Build judge prompt
        expected_str = "\n".join([f"- {e}" for e in expected])
        appendix_str = "\n".join([f"- {a}" for a in appendix]) if appendix else "None"
        
        prompt = f"""You are a benchmark evaluator for a RAG system analyzing Envision DSL code.

TASK: Evaluate if the Agent's Answer is semantically correct.

QUESTION: {q_text}

EXPECTED ANSWER(S):
{expected_str}

ADDITIONAL CONTEXT (Appendix):
{appendix_str}

AGENT'S ANSWER:
{got}

---

EVALUATION CRITERIA:
1. Does the answer address the question correctly?
2. Is the key information from the expected answer present (even if worded differently)?
3. Are there any critical errors or hallucinations?

OUTPUT FORMAT (JSON only):
{{"score": <0.0-1.0>, "correct": <true/false>, "reasoning": "<brief explanation>"}}

Respond with ONLY the JSON object, no other text.
"""
        
        try:
            response = self.llm.generate_response(prompt)
            
            # Parse JSON from response
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return JudgeResult(
                    question_id=q_id, question=q_text, expected=expected, got=got,
                    score=float(result.get("score", 0)),
                    correct=bool(result.get("correct", False)),
                    reasoning=str(result.get("reasoning", "No reasoning provided.")),
                    deterministic=False
                )
        except Exception as e:
            pass  # Fallback below
        
        # Fallback: simple keyword matching
        expected_words = set(" ".join(str(e) for e in expected).lower().split())
        got_words = set(got.lower().split())
        overlap = len(expected_words & got_words) / max(len(expected_words), 1)
        
        return JudgeResult(
            question_id=q_id, question=q_text, expected=expected, got=got,
            score=overlap, correct=overlap > 0.5,
            reasoning=f"Fallback: {overlap:.0%} keyword overlap.",
            deterministic=False
        )
