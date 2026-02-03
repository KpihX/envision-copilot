from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from .base import BaseAgent
from ..state import CopilotState

class SynthesizerAgent(BaseAgent):
    """
    Agent responsible for generating the final answer in the user's language.
    """
    def __init__(self, config: Dict[str, Any], llm: Any, console: Console, prompt_loader: Any, verbose: bool = False, debug: bool = False):
        super().__init__(config, llm, console, prompt_loader, verbose=verbose, debug=debug)
        self.prompt_loader = prompt_loader

    def run(self, state: CopilotState, appendix: str, max_depth: int, original_question: str = None, reformulated_question: str = None, plan_thought: str = None) -> Dict:
        """
        Executes the Synthesis.
        """
        stop_reason = state.get("stop_reason", "unknown")
        user_language = state.get("user_language", "English")
        
        self.console.print(Panel(f"Synthesizing answer in {user_language}...", title="🎯 Synthesize", border_style="cyan"))
        
        # Build prompt using loader
        prompt = self.prompt_loader.get_synthesizer_prompt(
            appendix=appendix,
            max_depth=max_depth,
            user_language=user_language,
            stop_reason=stop_reason,
            original_question=original_question or "Unknown",
            reformulated_question=reformulated_question or "Unknown",
            plan_thought=plan_thought or "None"
        )
        
        # Query LLM (Synthesis usually doesn't output JSON, so direct generate)
        # We reuse the centralized logging method from BaseAgent
        final_answer = self._generate_with_logs(prompt, title_suffix="(Synthesizer)")
        
        return {"final_answer": final_answer}
