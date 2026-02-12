from typing import List, Dict, Any
import json
from langgraph.graph import StateGraph, END, START
from rich.panel import Panel
from rich.markdown import Markdown

from envision_copilot.core.base import BaseCopilot

from envision_copilot.core.memory import Memory
from envision_copilot.core.planner import Planner
from envision_copilot.core.state import CopilotState

from envision_copilot.core.agents.starter import StarterAgent
from envision_copilot.core.agents.thinker import ThinkerAgent
from envision_copilot.core.agents.synthesizer import SynthesizerAgent


class BatchCopilot(BaseCopilot):
    """
    Standard Copilot implementation for One-Shot Execution (Batch Mode).
    Uses LangGraph: Starter -> Thinker -> Act -> Synthesizer.
    """
    def __init__(self, config_path: str = "config.yaml", verbose: bool = False, debug: bool = False):
        super().__init__(config_path, verbose, debug)

    def run(self, question: str):
        initial_state: CopilotState = {
            "original_question": question,
            "user_language": "English", # Default
            "question": question,
            "current_node_id": None,
            "last_layer_results": [],
            "should_stop": False,
            "stop_reason": "",
            "final_answer": "",
            "interactive_mode": False
        }
        
        # RUN UNIFIED BRAIN (Starter -> [Thinker -> Act]* -> Synthesizer)
        recursion_limit = self.config.get("agent", {}).get("constraints", {}).get("recursion_limit", 50)
        final_state = self.workflow.invoke(initial_state, config={"recursion_limit": recursion_limit})
        
        # Output
        answer = final_state.get("final_answer") or "Task Ended Without Final Answer."
        
        return {
            "answer": answer, 
            "appendix": self.memory.print(title="📎 Appendix") if self.memory else None
        }
