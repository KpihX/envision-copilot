import json
import logging
from pathlib import Path
from rich.progress import track

from envision_copilot.agent import EnvisionAgent
from llms import get_llm
from .utils import ConfigLoader

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.agent = EnvisionAgent()
        
        # Load Judge LLM
        judge_model = self.config.get("judge", {}).get("model", "mistral")
        self.judge = get_llm(judge_model)

    def load_questions(self):
        q_path = Path(self.config.get("input", {}).get("questions_file", "questions.json"))
        # Fallback to older questions.json in root if it exists (which we moved to old2, so check root)
        # Actually user said assume centralized stuff, so we should look at root questions.json if we restored it?
        # Or look in package default
        if not q_path.exists():
             # Try root
             q_path = Path("questions.json")
             
        if not q_path.exists():
            # Create dummy if missing for demo
            return [{"id": 1, "question": "What is Envision?", "expected_topics": ["DSL", "Supply Chain"]}]

        with open(q_path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
             # Prefer 'answered' list if available, else combine or pick one
             return data.get("answered", []) + data.get("unanswered", [])
        return []

    def run(self, question_ids=None):
        questions = self.load_questions()
        
        # Filter by IDs if provided
        if question_ids:
            questions = [q for q in questions if q.get("id") in question_ids]
            desc = f"Running Benchmark (IDs={question_ids})..."
        else:
            limit = self.config.get("benchmark", {}).get("default_n", 5)
            questions = questions[:limit]
            desc = f"Running Benchmark (N={limit})..."
        
        results = []
        
        for q in track(questions, description=desc):
            question_text = q.get("question")
            
            # 1. Run Agent
            try:
                response = self.agent.run(question_text)
            except Exception as e:
                response = f"Error: {e}"
                
            # 2. Judge (LLM-as-Judge)
            score, reasoning = self._evaluate(question_text, response, q.get("expected_topics", []))
            
            results.append({
                "question": question_text,
                "response": response,
                "score": score,
                "reasoning": reasoning
            })
            
        self._save_report(results)
        return results

    def _evaluate(self, question, response, expected_topics):
        prompt = f"""
        You are an impartial judge evaluating an AI assistant's answer.
        
        Question: {question}
        Assistant Answer: {response}
        Expected Topics to Cover: {", ".join(expected_topics)}
        
        Task:
        1. Rate usefulness on 1-10 scale.
        2. Provide reasoning.
        
        Format output strictly as:
        Score: X
        Reasoning: ...
        """
        
        eval_res = self.judge.generate(prompt)
        
        # Parse score
        try:
            score_line = [l for l in eval_res.splitlines() if "Score:" in l][0]
            score = int(score_line.split(":")[1].strip())
        except:
            score = 0
            
        return score, eval_res

    def _save_report(self, results):
        out_path = Path(self.config.get("output", {}).get("report_file", "data/logs/benchmark_report.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
