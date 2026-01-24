from typing import Dict, Any, Optional
from rich.console import Console
from .base import BaseAgent
from ..state import CopilotState

class StarterAgent(BaseAgent):
    """
    Agent responsible for initial triage and translation.
    Decides if the workflow should proceed to Thinker or stop.
    """
    def __init__(self, config: Dict[str, Any], llm: Any, console: Console, prompt_loader: Any, verbose: bool = False):
        super().__init__(config, llm, console, prompt_loader, verbose=verbose)
        self.prompt_loader = prompt_loader

    def run(self, state: CopilotState) -> Dict:
        """
        Executes the starter logic.
        Updates state with detected language and translated question.
        """
        user_input = state["original_question"]
        
        # Build prompt using loader
        prompt = self.prompt_loader.get_starter_prompt(user_input=user_input)
        
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
            "should_stop": not result.get("is_relevant", False),
            "stop_reason": "irrelevant" if not result.get("is_relevant") else "",
            "final_answer": result.get("direct_response", "")
        }
        
        return updates

    def _validate_schema(self, data: Dict) -> bool:
        """Validation specific to Starter Agent output."""
        required = ["is_relevant", "user_language"]
        if not all(k in data for k in required):
            return False
            
        if not isinstance(data["is_relevant"], bool): return False
        if not isinstance(data["user_language"], str): return False
        
        return True
