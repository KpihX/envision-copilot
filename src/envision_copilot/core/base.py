from typing import Dict, Any, Optional, List
import json
from abc import ABC, abstractmethod
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from llms import get_llm
from envision_copilot.utils.config_loader import ConfigLoader
from envision_copilot.utils.prompt_loader import PromptLoader
from envision_copilot.core.memory import Memory
from envision_copilot.core.planner import Planner
from envision_copilot.core.state import CopilotState

from envision_preprocess.api import EnvisionGraphAPI
from envision_copilot.tools import GraphTool, ReaderTool, RagTool, GrepTool
from envision_copilot.core.agents.thinker import ThinkerAgent
from envision_copilot.core.agents.starter import StarterAgent
from envision_copilot.core.agents.synthesizer import SynthesizerAgent
from envision_copilot.tools.definitions import TOOLS

from langgraph.graph import StateGraph, END, START

# --- Standardized Error Container ---
class ErrorResult:
    """Standardized error container compatible with Tool Result Interface."""
    def __init__(self, msg, title="❌ Error"): 
        self.msg = msg
        self.title = title
    def __str__(self): return f"Error: {self.msg}"
    def to_dict(self): return {"error": self.msg, "title": self.title}
    def print(self): return Panel(self.msg, title=self.title, border_style="red")

class BaseCopilot(ABC):
    """
    Abstract Base Class for Envision Copilot.
    Handles unified initialization of Config, LLM, API, and Tools.
    Also provides shared execution nodes (Think, Act).
    """
    def __init__(self, config_path: str = "config.yaml", verbose: bool = False, debug: bool = False):
        self.config = ConfigLoader.load_config(config_path)
        self.verbose = verbose or self.config.get("presentation", {}).get("verbose", False)
        self.debug = debug or self.config.get("presentation", {}).get("debug", False)
        self.console = Console()
        
        self.prompts = PromptLoader(self.config)
        
        # Tools (using unified BaseTool pattern)
        # Init shared API once
        self.api = EnvisionGraphAPI()
                
        # Initialize all tools - they encapsulate their own dependencies
        self.tools_mapping = {
            "graph": GraphTool(api=self.api, config=self.config),
            "reader": ReaderTool(api=self.api, config=self.config),
            "grep": GrepTool(api=self.api, config=self.config),
            "rag": RagTool(api=self.api, config=self.config),
        }

        # Load tool hints
        self.tools_hint = {}
        for too_name in self.tools_mapping.keys():
            self.tools_hint[too_name] = TOOLS[too_name]['usage_hint']
        
        # LLM
        self.llm = get_llm(config=self.config)
        
        # Agents
        self.starter_agent = StarterAgent(self.config, self.llm, self.console, self.prompts, verbose=self.verbose, debug=self.debug)
        self.thinker_agent = ThinkerAgent(self.config, self.llm, self.console, self.prompts, verbose=self.verbose, debug=self.debug)
        self.synthesizer_agent = SynthesizerAgent(self.config, self.llm, self.console, self.prompts, verbose=self.verbose, debug=self.debug)

        # Shared State Placeholders
        self.memory: Optional[Memory] = None
        self.planner: Optional[Planner] = None
        self.interaction_history: List[Dict] = [] # User-Synthesizer history

        # Workflow graph
        self.workflow = self._build_graph()

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Main execution entry point."""
        pass

    # --- Graph Definition ---

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(CopilotState)
        
        workflow.add_node("starter", self.starter_node)
        workflow.add_node("thinker", self.thinker_node)
        workflow.add_node("act", self.act_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        
        workflow.add_edge(START, "starter")
        
        # Decisions for Starter
        workflow.add_conditional_edges(
            "starter",
            self.decide_after_starter,
            {
                "thinker": "thinker",
                "synthesizer": "synthesizer",
                END: END
            }
        )
        
        # Decisions for Thinker
        workflow.add_conditional_edges(
            "thinker",
            self.decide_after_thinker,
            {
                "act": "act",
                "synthesizer": "synthesizer"
            }
        )
        
        # Decisions for Act
        workflow.add_conditional_edges(
            "act",
            self.decide_after_act,
            {
                "continue_layer": "act",
                "thinker": "thinker"
            }
        )
        
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()

    # --- Shared Nodes ---

    def starter_node(self, state: CopilotState) -> Dict:
        """Starter Phase: Gatekeeper & Triage."""
        if self.verbose:
             self.console.print(Panel("Analyzing User Request...", title="🚦 Starter Agent", border_style="yellow"))

        # Convert interaction history to plain text for LLM
        history_str = self._format_interaction_history()

        # Exploration summary (the "technical digest" of what was found so far)
        exploration_history = self.planner.get_visible_history(plan_history_depth=-1) if self.planner else "(No exploration performed yet)"

        updates = self.starter_agent.run(
            state, 
            history=history_str, 
            interactive_mode=state.get("interactive_mode", False),
            exploration_history=exploration_history
        )
        
        # If technical exploration is required, initialize resources
        if not updates.get("should_stop"):
            if self.memory is None: self.memory = Memory(self.config)
            
            # Use English Question for Planner
            plan_goal = updates.get("question", state["original_question"])
            
            if self.planner is None: 
                self.planner = Planner(plan_goal, self.config)
            
            # Critical for Interactive Mode: Mark turnover
            if state.get("interactive_mode"):
                self.planner.mark_interaction_boundary(state["original_question"])
                # Update planner goal if it changed significantly
                self.planner.goal = plan_goal
            
            if self.verbose or self.debug:
                self.console.print(self.planner.print())
        
        # If IRRELEVANT: Display directly using the unified method
        if updates.get("should_stop"):
            final_ans = updates.get("final_answer") or updates.get("direct_response") or ""
            if final_ans:
                self._display_final_answer(final_ans)
                updates["final_answer"] = final_ans # Ensure it's in the state for history
            
        return updates

    def thinker_node(self, state: CopilotState) -> Dict:
        """Think Phase: Reasoning & Planning."""
        
        if not self.thinker_agent:
            return {"should_stop": True, "stop_reason": "system_error_no_thinker"}

        # Execute Thinker Agent (Injecting Session Dependencies)
        result = self.thinker_agent.run(
            state, 
            planner=self.planner, 
            memory=self.memory, 
            result_formatter=self._format_last_results
        )
        
        response_json = result.get("think_response")
        
        if not response_json:
            return {"should_stop": True, "stop_reason": "llm_error"}

        # Store results_digest in the nodes of the PREVIOUS layer (the one just executed)
        results_digest = response_json.get("results_digest", {})
        
        if results_digest and hasattr(self.planner, 'layers') and len(self.planner.layers) > 1:
            previous_layer = self.planner.layers[-1]
            for idx_str, digest_lines in results_digest.items():
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(previous_layer):
                        if isinstance(digest_lines, list):
                            previous_layer[idx].summary = digest_lines
                except (ValueError, IndexError):
                    continue

        # Handle Memory Updates Logic
        # 1. Remove items requested by LLM
        remove_indices = response_json.get("memory_remove_indices", [])
        if remove_indices:
            self.memory.remove_by_indices(remove_indices)

        # 2. Add new items requested by LLM (by index in last_results)
        add_result_indices = response_json.get("add_result_indices", [])
        last_results = state.get("last_layer_results", [])
        
        # Normalize to Map: { int_idx: [chunk_list] }
        selection_map = {}
        if isinstance(add_result_indices, list):
            for idx in add_result_indices:
                selection_map[int(idx)] = [0] # Default: Keep All
        elif isinstance(add_result_indices, dict):
            for k, v in add_result_indices.items():
                try:
                    selection_map[int(k)] = v 
                except ValueError:
                    continue

        for idx, chunk_indices in selection_map.items():
            if 0 <= idx < len(last_results):
                result_entry = last_results[idx]
                tool_name = result_entry.get("tool", "unknown")
                raw_result = result_entry.get("result", "")
                
                # Use standardized str() on the result object for compact view
                # raw_result is now the Result Object (or simple types)
                final_stored_result = raw_result
                compact_view = str(raw_result)
                
                # Granular Logic for RAG results
                if tool_name == "rag" and hasattr(raw_result, 'data'):
                    rag_data = raw_result.data
                    if isinstance(rag_data, dict) and "results" in rag_data:
                        results_list = rag_data["results"]
                        filtered_chunks = []
                        
                        for chunk_idx in chunk_indices:
                            if 0 <= chunk_idx < len(results_list):
                                filtered_chunks.append(results_list[chunk_idx])
                                
                        if filtered_chunks:
                            final_stored_result = {"results": filtered_chunks, "stats": rag_data.get("stats", {})}
                            compact_view = json.dumps(final_stored_result, indent=2, ensure_ascii=False)
                        else:
                            continue 

                self.memory.add_observation(
                    tool_name=tool_name,
                    tool_args=result_entry.get("args", {}),
                    result=final_stored_result.to_dict() if hasattr(final_stored_result, 'to_dict') else final_stored_result,
                    compact_view=compact_view
                )
        
        # RICH UI: Display updated memory (Optional debug)
        # if self.debug:
            # self.console.print(self.memory.print())

        # Check stop conditions
        current_depth, max_depth = self.planner.get_effective_depths(state)

        at_max_depth = current_depth >= max_depth
        llm_wants_stop = response_json.get("should_stop", False)
        
        if at_max_depth or llm_wants_stop:
            stop_reason = "max_depth" if at_max_depth else "llm_decision"
            if self.verbose or self.debug:
                self.console.print(f"[yellow]🛑 Stopping: {stop_reason}[/yellow]")
            return {
                "should_stop": True, 
                "stop_reason": stop_reason, 
                "last_layer_results": [],
                "plan_thought": response_json.get("thought_process", ""), 
                "last_thought_process": response_json.get("thought_process", "")
            }

        # Propose new layer
        new_steps = response_json.get("next_steps", [])
        if new_steps:
            self.planner.propose_next_layer(new_steps)
            if self.verbose or self.debug:
                self.console.print(self.planner.print())
        
        return {
            "last_layer_results": [], 
            "plan_thought": response_json.get("thought_process", ""), 
            "last_thought_process": response_json.get("thought_process", "")
        }

    def synthesizer_node(self, state: CopilotState) -> Dict:
        """Synthesize Phase: Produce final answer."""
        appendix = str(self.memory) if self.memory else ""
        exploration_history = self.planner.get_visible_history(plan_history_depth=-1) if self.planner else "(No exploration performed)"
        
        # Convert interaction history to plain text for LLM
        history_str = self._format_interaction_history()

        updates = self.synthesizer_agent.run(
            state, 
            appendix=appendix, 
            max_depth=self.planner.max_depth if self.planner else 0,
            original_question=state.get("original_question"),
            reformulated_question=state.get("question"), 
            plan_thought=state.get("plan_thought"), 
            exploration_history=exploration_history,
            history=history_str,
            interactive_mode=state.get("interactive_mode", False)
        )
        
        # Display final response using the unified method
        self._display_final_answer(updates.get("final_answer", "No answer produced."))
        
        return updates

    def _format_interaction_history(self) -> str:
        """Helper to format the chat history for LLM prompts."""
        constraints = self.config.get("agent", {}).get("constraints", {})
        history_depth = constraints.get("interaction_history_depth", 5)
        
        relevant_turns = self.interaction_history
        if history_depth != -1 and len(self.interaction_history) > history_depth:
            relevant_turns = self.interaction_history[-history_depth:]

        history_str = ""
        for i, turn in enumerate(relevant_turns):
            history_str += f"Turn {i+1}:\nUser: {turn['user']}\nAgent: {turn['agent']}\n\n"
        
        return history_str

    def _display_final_answer(self, answer: str):
        """Standardized UI for the Copilot's final response."""
        title = self.config.get("presentation", {}).get("title", "Envision Copilot")
        self.console.print("\n")
        self.console.print(Panel(Markdown(answer), title=f"🤖 {title}", border_style="green"))

    def act_node(self, state: CopilotState) -> Dict:
        """Act Phase: Execute Tools."""
        node = self.planner.get_next_pending_node()
        if not node:
            return {}

        state["current_node_id"] = node.id
        
        if self.verbose or self.debug:
            # Show plan evolution
            self.console.print(self.planner.print())
            
            args_fmt = json.dumps(node.tool_args, ensure_ascii=False, indent=2)
            md_content = f"**Tool**: `{node.tool_name}`\n**Args**: \n```json\n{args_fmt}\n```"
            self.console.print(Panel(
                Markdown(md_content), 
                title=f"🚀 Executing [{node.id}]: {node.goal}", 
                border_style="magenta"
            ))
            
        result_raw = self._execute_tool(node.tool_name, node.tool_args)
        
        # Store result in state for next Think
        result_entry = {
            "id": node.id,
            "tool": node.tool_name,
            "args": node.tool_args,
            "result": result_raw
        }
        
        state.setdefault("last_layer_results", []).append(result_entry)
        
        self.planner.mark_done(node, reasoning="Executed")
        
        if self.verbose or self.debug:
            self._display_tool_result(node.tool_name, result_raw)

        return {"last_layer_results": state["last_layer_results"]}

    # --- Decisions (Edges) ---
    
    def decide_after_starter(self, state: CopilotState) -> str:
        if state.get("should_stop"):
            return END
        return "thinker"
    
    def decide_after_thinker(self, state: CopilotState) -> str:
        if state.get("should_stop"):
            return "synthesizer"
        
        if self.planner.has_pending_nodes():
            return "act"
            
        return "synthesizer" # Fallback

    def decide_after_act(self, state: CopilotState) -> str:
        if self.planner.is_layer_complete():
            return "thinker"
        return "continue_layer"

    # --- Helpers ---
    
    def _execute_tool(self, name: str, args: Dict) -> Any:
        try:
            tool = self.tools_mapping.get(name)
            
            if tool is None:
                available = list(self.tools_mapping.keys())
                
                hints_list = [f"- [{tool_name}]: {hint}" for tool_name, hint in self.tools_hint.items()]
                hint = "\n".join(hints_list)
                
                return ErrorResult(f"Unknown tool: '{name}'. Available: {available}.\n\nUsage Hints:\n{hint}", title="❌ System Error")
            
            return tool.execute(**args)
            
        except Exception as e:
            return ErrorResult(f"Tool Execution Error: {e}", title="❌ Execution Exception")

    def _format_last_results(self, results: List[Dict]) -> str:
        """Format last layer results for LLM context."""
        if not results:
            return "No results from last layer."
        
        buffer = []
        for i, r in enumerate(results):
            tool_name = r.get("tool", "unknown tool")
            result_obj = r.get("result") 
            formatted = str(result_obj)
            buffer.append(f"\n--- Result [{i}] ({tool_name}) ---\n{formatted}")
        
        return "\n".join(buffer)

    def _display_tool_result(self, tool_name: str, result: Any):
        """Display tool execution result using its own print() method."""
        if hasattr(result, 'print'):
            self.console.print(result.print())
        else:
            self.console.print(f"[bold red]Result object for {tool_name} has no print() method![/bold red]")
            self.console.print(str(result))
