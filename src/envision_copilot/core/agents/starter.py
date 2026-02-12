from typing import Dict, Any, Optional
from rich.console import Console
from .base import BaseAgent
from ..state import CopilotState

class StarterAgent(BaseAgent):
    """
    Agent responsible for initial triage and translation.
    Decides if the workflow should proceed to Thinker or stop.
    """
    def __init__(self, config: Dict[str, Any], llm: Any, console: Console, prompt_loader: Any, verbose: bool = False, debug: bool = False):
        super().__init__(config, llm, console, prompt_loader, verbose=verbose, debug=debug)
        self.prompt_loader = prompt_loader

    def run(self, state: CopilotState, history: str = "", interactive_mode: bool = False, exploration_history: str = "") -> Dict:
        """
        Executes the starter logic.
        Updates state with detected language and translated question.
        """
        user_input = state["original_question"]
        
        # Build prompt using loader
        prompt = self.prompt_loader.get_starter_prompt(
            user_input=user_input, 
            history=history,
            interactive_mode=interactive_mode,
            exploration_history=exploration_history
        )
        
        # Query LLM with specific validation
        result = self.query_llm_robust(prompt, schema_validation=self._validate_schema)
        
        if not result:
            # Fallback for critical failure
            return {
                "stop_reason": "llm_error",
                "should_stop": True,
                "final_answer": "Sorry, a technical error prevents me from analyzing your request."
            }
        
        # Update State
        updates = {
            "user_language": result.get("user_language", "English"),
            "question": result.get("english_question") or user_input,
            "should_stop": not result.get("needs_exploration", False),
            "stop_reason": "direct_answer" if not result.get("needs_exploration") else "",
            "final_answer": result.get("direct_response", "")
        }
        
        return updates

    def _validate_schema(self, data: Dict) -> bool:
        """Validation specific to Starter Agent output."""
        required = ["needs_exploration", "user_language"]
        if not all(k in data for k in required):
            return False
            
        if not isinstance(data["needs_exploration"], bool): return False
        if not isinstance(data["user_language"], str): return False
        
        return True
