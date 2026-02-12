from typing import Dict, Any, List
import sys
import json
from rich.panel import Panel
from rich.console import Console
from rich.markdown import Markdown

from envision_copilot.core.base import BaseCopilot

from envision_copilot.core.memory import Memory
from envision_copilot.core.planner import Planner
from envision_copilot.core.state import CopilotState

class InteractiveSessionState(CopilotState):
    """Extended state for interactive sessions."""
    # Keeps track of the session history
    interaction_history: List[Dict[str, str]] # [{"role": "user", "content": "..."}, ...]

class LiveCopilot(BaseCopilot):
    """
    Interactive Copilot implementation (Live Mode).
    Features:
    - Persistent REPL Loop.
    - Persistent Memory & Planner across turns.
    - Conversation History.
    """
    def __init__(self, config_path: str = "config.yaml", verbose: bool = False, debug: bool = False):
        super().__init__(config_path, verbose, debug)
        
        # Persistent Components
        self.memory = Memory(self.config)
        # Planner is initialized with a dummy goal, updated per interaction
        self.planner = Planner("Interactive Session Start", self.config) 
        
        self.interaction_history: List[Dict[str, str]] = [] # [{"user": "...", "agent": "..."}, ...]
        
    def run(self):
        """
        Starts the Interactive REPL Loop.
        """
        self.console.print(Panel(
            "[bold green]Envision Copilot - LIVE MODE[/bold green]\n"
            "Type [bold]/exit[/bold] or [bold]/quit[/bold] to stop.\n"
            "Type [bold]/clear[/bold] to reset screen.\n"
            "Type [bold]/reset[/bold] to clear memory/plan.",
            title="🎮 Interactive Session",
            border_style="green"
        ))

        while True:
            try:
                # 1. READ
                user_input = self.console.input("\n[bold green]User > [/bold green]").strip()
                
                if not user_input:
                    continue

                if user_input.lower() in ("/exit", "/quit"):
                    self.console.print("[dim]Goodbye![/dim]")
                    break
                
                if user_input.lower() == "/clear":
                    self.console.clear()
                    continue
                    
                if user_input.lower() == "/reset":
                    self.memory = Memory(self.config)
                    self.planner = Planner("Interactive Session Reset", self.config)
                    self.interaction_history = []
                    self.console.print("[yellow]Session Reset.[/yellow]")
                    continue

                # 2. EVAL (Unified Agent Graphe)
                self._process_turn(user_input)

            except KeyboardInterrupt:
                self.console.print("[dim]Goodbye![/dim]")
                break
            except Exception as e:
                self.console.print(f"[red]Session Error: {e}[/red]")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def _process_turn(self, user_input: str):
        """
        Executes a reasoning loop for a single user interaction using the unified workflow.
        """
        # Live Mode Constraints from Config
        max_depth_live = self.config.get("agent", {}).get("constraints", {}).get("max_depth_live", 2)
        
        # Initial State for this turn
        state: CopilotState = {
            "original_question": user_input,
            "user_language": "English", # Starter will detect
            "question": user_input, 
            "last_layer_results": [],
            "should_stop": False,
            "stop_reason": "",
            "last_thought_process": "", 
            "interactive_mode": True,
            "turn_start_depth": self.planner.current_depth,
            "max_depth": max_depth_live
        }

        # RUN UNIFIED BRAIN
        # Since we use self.workflow, it calls self.starter_node, which will use self.interaction_history
        recursion_limit = self.config.get("agent", {}).get("constraints", {}).get("recursion_limit", 50)
        final_state = self.workflow.invoke(state, config={"recursion_limit": recursion_limit})
        
        # 3. UPDATE HISTORY
        final_answer = final_state.get("final_answer", "No answer produced.")
        self.interaction_history.append({
            "user": user_input,
            "agent": final_answer
        })
