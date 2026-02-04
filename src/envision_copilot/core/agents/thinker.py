from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from .base import BaseAgent
from ..state import CopilotState

class ThinkerAgent(BaseAgent):
    """
    Agent responsible for reasoning and planning (The actual Copilot Brain).
    """
    def __init__(self, config: Dict[str, Any], llm: Any, console: Console, 
                 prompt_loader: Any, verbose: bool = False, debug: bool = False):
        super().__init__(config, llm, console, prompt_loader, verbose=verbose, debug=debug)
        self.prompt_loader = prompt_loader

    def run(self, state: CopilotState, planner: Any, memory: Any, result_formatter: Any) -> Dict:
        """
        Executes the Think Phase.
        """
        question = state["question"]
        last_results = state.get("last_layer_results", [])
        
        # UI Header
        if self.verbose or self.debug: 
            depth_info = f"Depth: {planner.current_depth}/{planner.max_depth}"
            self.console.print(Panel(f"[bold]{question}[/bold]", title=f"🧠 Think Phase ({depth_info})", border_style="purple"))

        # Context
        memory_text = str(memory)
        plan_text = str(planner)
        results_text = result_formatter(last_results)
        last_thought = state.get("last_thought_process", "")
        
        # Construct Prompt
        sys_prompt = self.prompt_loader.get_think_prompt(
            question=question,
            memory=memory_text,
            history=plan_text, 
            last_results=results_text,
            current_depth=planner.current_depth,
            last_thought_process=last_thought
        )
        
        # Query LLM
        # Query LLM
        response_json = self.query_llm_robust(
            sys_prompt, 
            schema_validation=self._validate_schema
        )
        
        if not response_json:
            return {"should_stop": True, "stop_reason": "llm_error"}

        # UI Thought Process
        if "thought_process" in response_json: 
             # Show thought process if Verbose OR Debug (Informational)
             if self.verbose or self.debug:
                 self.console.print(Panel(Markdown(response_json["thought_process"]), title="💭 Thought Process", border_style="dim"))

        # Return the parsed decision for the Orchestrator to handle (State updates)
        return {
            "think_response": response_json
        }

    def _validate_schema(self, data: Dict) -> bool:
        """
        Validation specific to Thinker Agent.
        Ensures all critical decision fields are present and correctly typed.
        """
        required_fields = ["thought_process", "should_stop", "memory_remove_indices", "add_result_indices"]
        
        # 1. Check presence of required keys
        if not all(k in data for k in required_fields):
            return False
            
        # 2. Check types
        if not isinstance(data["thought_process"], str): return False
        if not isinstance(data["should_stop"], bool): return False
        if not isinstance(data["memory_remove_indices"], list): return False
        
        # 3. Check Conditional Logic (Next Steps vs Stop)
        # If NOT stopping, we usually expect next_steps, but empty next_steps is allowed if just memory update (though rare).
        # But if next_steps is present, it must be a list of dicts with 'tool' and 'args'.
        add = data.get("add_result_indices")
        if add is not None:
             if not isinstance(add, (list, dict)):
                 return False #, f"'add_result_indices' must be a List or Dict, got {type(add)}"
             if isinstance(add, list) and not all(isinstance(x, int) for x in add):
                 return False #, "All items in 'add_result_indices' list must be integers"
             # Dict validation is looser: keys are result indices (str/int), values are list of ints
        
        # The original code had this check here, which is now handled by the 'add' block above.
        # if not isinstance(data["add_result_indices"], list): return False
        
        # 3. Check Conditional Logic (Next Steps vs Stop)
        # If NOT stopping, we usually expect next_steps, but empty next_steps is allowed if just memory update (though rare).
        # But if next_steps is present, it must be a list of dicts with 'tool' and 'args'.
        
        if "next_steps" in data:
            if not isinstance(data["next_steps"], list): return False
            for step in data["next_steps"]:
                if not isinstance(step, dict): return False
                if "tool" not in step or "args" not in step: return False
                
        return True
