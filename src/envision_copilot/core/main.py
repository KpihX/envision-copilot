from typing import List, Dict, Any
import json
from langgraph.graph import StateGraph, END, START
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from llms import get_llm
from envision_copilot.utils.config_loader import ConfigLoader
from envision_copilot.utils.prompt_loader import PromptLoader
from envision_copilot.core.memory import Memory
from envision_copilot.core.planner import Planner, NodeStatus
from envision_copilot.core.state import CopilotState

from envision_preprocess.api import EnvisionGraphAPI

from envision_copilot.tools import GraphTool, ReaderTool, RagTool, GrepTool, BaseTool

from envision_copilot.core.agents.starter import StarterAgent
from envision_copilot.core.agents.thinker import ThinkerAgent
from envision_copilot.core.agents.synthesizer import SynthesizerAgent

# --- Helpers Classes ---
class ErrorResult:
    """Standardized error container compatible with Tool Result Interface."""
    def __init__(self, msg, title="❌ Error"): 
        self.msg = msg
        self.title = title
    def __str__(self): return f"Error: {self.msg}"
    def to_dict(self): return {"error": self.msg, "title": self.title}
    def print(self): return Panel(self.msg, title=self.title, border_style="red")

# --- The Brain ---
class EnvisionCopilot:
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
        
        # LLM
        self.llm = get_llm(config=self.config)
        
        # Agents Intialization (Service Pattern)
        self.starter_agent = StarterAgent(self.config, self.llm, self.console, self.prompts, verbose=self.verbose, debug=self.debug)
        self.thinker_agent = ThinkerAgent(self.config, self.llm, self.console, self.prompts, verbose=self.verbose, debug=self.debug)
        self.synthesizer_agent = SynthesizerAgent(self.config, self.llm, self.console, self.prompts, verbose=self.verbose, debug=self.debug)
        
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(CopilotState)
        
        # Note: Starter is run manually before the graph to allow lazy init of heavy objects
        workflow.add_node("thinker", self.thinker_node)
        workflow.add_node("act", self.act_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        
        # Entry Point -> Thinker (Starter is strictly pre-process now)
        workflow.add_edge(START, "thinker")
        
        # Think Logic
        workflow.add_conditional_edges(
            "thinker",
            self.decide_after_thinker,
            {
                "act": "act",
                "synthesizer": "synthesizer"
            }
        )
        
        # Act Logic
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

    def run(self, question: str):
        # 1. PRE-FLIGHT CHECK (Starter Agent)
        # We run this BEFORE initializing Memory/Planner/Workflow to save resources
        # and avoid "empty" appendixes for greetings.
        
        if self.verbose or self.debug:
             self.console.print(Panel("Analyzing User Request...", title="🚦 Starter Agent", border_style="yellow"))

        initial_state: CopilotState = {
            "original_question": question,
            "user_language": "English", # Default
            "question": question,
            "current_node_id": None,
            "last_layer_results": [],
            "should_stop": False,
            "stop_reason": "",
            "final_answer": ""
        }
        
        starter_updates = self.starter_agent.run(initial_state)
        
        # Merge updates into state
        initial_state.update(starter_updates)
        
        # 2. DECISION: Stop or Go?
        if initial_state.get("should_stop"):
            # Irrelevant query (Greeting, etc.)
            # We return output IMMEDIATELY. No memory, no planner, no appendix.
            answer = initial_state.get("final_answer") or starter_updates.get("direct_response") or "Conversation ended."
            return {
                "answer": answer,
                "appendix": None # Explicitly None as requested
            }
            
        # 3. INITIALIZE WORKFLOW RESOURCES (Only if relevant)
        self.memory = Memory(self.config)
        self.planner = Planner(initial_state["question"], self.config) # Use translated question
        
        # 4. RUN BRAIN (Thinker -> Act -> Synthesizer)
        recursion_limit = self.config.get("agent", {}).get("constraints", {}).get("recursion_limit", 50)
        final_state = self.workflow.invoke(initial_state, config={"recursion_limit": recursion_limit})
        
        # Output
        answer = final_state.get("final_answer") or "Task Ended Without Final Answer."
        
        return {
            "answer": answer, 
            "appendix": self.memory.print(title="📎 Appendix")
        }

    # --- Nodes ---



    def thinker_node(self, state: CopilotState) -> Dict:
        """Think Phase: Reasoning & Planning."""
        
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

        # Handle Memory Updates Logic
        # 1. Remove items requested by LLM
        remove_indices = response_json.get("memory_remove_indices", [])
        if remove_indices:
            self.memory.remove_by_indices(remove_indices)

        # 2. Add new items requested by LLM (by index in last_results)
        add_result_indices = response_json.get("add_result_indices", [])
        last_results = state.get("last_layer_results", [])
        
        # Normalize to Map: { int_idx: [chunk_list] }
        # Legacy support: List[int] -> { i: [0] } (Keep All)
        selection_map = {}
        if isinstance(add_result_indices, list):
            for idx in add_result_indices:
                selection_map[int(idx)] = [0] # Default: Keep All (Chunk 0 = Full Result logic if not chunkable)
        elif isinstance(add_result_indices, dict):
            for k, v in add_result_indices.items():
                try:
                    selection_map[int(k)] = v # v is List[int] of chunks
                except ValueError:
                    continue # Skip invalid keys

        for idx, chunk_indices in selection_map.items():
            if 0 <= idx < len(last_results):
                result_entry = last_results[idx]
                tool_name = result_entry.get("tool", "unknown")
                raw_result = result_entry.get("result", "")
                
                # ToolResult has .data attribute with the actual result data
                # For RAG tool, data contains {"results": [...], "stats": {...}}
                
                final_stored_result = raw_result # Default
                compact_view = str(raw_result)
                
                # Granular Logic for RAG results (filter specific result indices)
                if tool_name == "rag" and hasattr(raw_result, 'data'):
                    rag_data = raw_result.data
                    if isinstance(rag_data, dict) and "results" in rag_data:
                        results_list = rag_data["results"]
                        filtered_chunks = []
                        
                        for chunk_idx in chunk_indices:
                            if 0 <= chunk_idx < len(results_list):
                                filtered_chunks.append(results_list[chunk_idx])
                                
                        if filtered_chunks:
                            # Create filtered version
                            final_stored_result = {"results": filtered_chunks, "stats": rag_data.get("stats", {})}
                            compact_view = json.dumps(final_stored_result, indent=2, ensure_ascii=False)
                        else:
                            continue 
                
                # For other tools, ToolResult wraps data - chunk indices [0] means "Add this result"

                self.memory.add_observation(
                    tool_name=tool_name,
                    tool_args=result_entry.get("args", {}),
                    result=final_stored_result.to_dict() if hasattr(final_stored_result, 'to_dict') else final_stored_result, # Store raw data
                    compact_view=compact_view
                )
        
        # RICH UI: Display updated memory
        # if self.verbose or self.debug:
        #     self.console.print(self.memory.print())

        # Check stop conditions
        at_max_depth = self.planner.current_depth >= self.planner.max_depth
        llm_wants_stop = response_json.get("should_stop", False)
        
        if at_max_depth or llm_wants_stop:
            stop_reason = "max_depth" if at_max_depth else "llm_decision"
            if self.verbose or self.debug:
                self.console.print(f"[yellow]🛑 Stopping: {stop_reason}[/yellow]")
            return {
                "should_stop": True, 
                "stop_reason": stop_reason, 
                "last_layer_results": [],
                "plan_thought": response_json.get("thought_process", ""), # Capture final thought
                "last_thought_process": response_json.get("thought_process", "") # For next iteration
            }

        # Propose new layer
        new_steps = response_json.get("next_steps", [])
        if new_steps:
            self.planner.propose_next_layer(new_steps)
            if self.verbose or self.debug:
                # RICH UI: Display updated plan
                self.console.print(self.planner.print())
        
        return {
            "last_layer_results": [], # Reset for new layer
            "plan_thought": response_json.get("thought_process", ""), # Capture thought for this step
            "last_thought_process": response_json.get("thought_process", "") # For next iteration
        }

    def act_node(self, state: CopilotState) -> Dict:
        """Act Phase: Execute Tools."""
        node = self.planner.get_next_pending_node()
        if not node:
            return {}

        state["current_node_id"] = node.id
        
        if self.verbose or self.debug:
            # Formatter Tool Execution Log using Markdown
            args_fmt = json.dumps(node.tool_args, ensure_ascii=False, indent=2)
            md_content = f"""
**Tool**: `{node.tool_name}`
**Args**: 
```json
{args_fmt}
```
"""
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
        
        # Mark Node Done
        self.planner.mark_done(node, reasoning="Executed")
        
        # Display result
        if self.verbose or self.debug:
            self._display_tool_result(node.tool_name, result_raw)

        return {"last_layer_results": state["last_layer_results"]}

    def synthesizer_node(self, state: CopilotState) -> Dict:
        """Synthesize Phase."""
        # Use Synthesizer Agent
        # Pass compact memory text as appendix for Synthesis (Trusted Facts)
        appendix = str(self.memory)
        
        updates = self.synthesizer_agent.run(
            state, 
            appendix=appendix, 
            max_depth=self.planner.max_depth,
            original_question=state.get("original_question"),
            reformulated_question=state.get("question"), # This is the reformulated one
            plan_thought=state.get("plan_thought") # Inject Thinker's logic
        )
        return updates
    
    # --- Decisions (Edges) ---
    
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
        """Execute a tool by name with given arguments.
        
        All tools use unified execute(**kwargs) -> ToolResult interface.
        """
        try:
            tool = self.tools_mapping.get(name)
            
            if tool is None:
                # Provide helpful hints for common mistakes
                available = list(self.tools_mapping.keys())
                hint = ""
                if name in ("tree", "node", "neighbors", "edges", "search"):
                    hint = f" HINT: '{name}' is an ACTION of the 'graph' tool. Use: {{\"tool\": \"graph\", \"args\": {{\"action\": \"{name}\", ...}}}}"
                return ErrorResult(f"Unknown tool: {name}. Available: {available}.{hint}", title="❌ System Error")
            
            # Execute with unified interface
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
            result_obj = r.get("result") # This is now a Result Object
            
            # Use standardized str() on the result object
            formatted = str(result_obj)
            
            buffer.append(f"\n--- Result [{i}] ({tool_name}) ---\n{formatted}")
        
        return "\n".join(buffer)

    def _display_tool_result(self, tool_name: str, result: Any):
        """Display tool execution result using its own print() method."""
        # result is assumed to be a standardized Result Object with .print()
        if hasattr(result, 'print'):
            self.console.print(result.print())
        else:
            # Fallback
            self.console.print(f"[bold red]Result object for {tool_name} has no print() method![/bold red]")
            self.console.print(str(result))
