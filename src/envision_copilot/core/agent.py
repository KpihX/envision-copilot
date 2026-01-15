from typing import TypedDict, List, Dict, Any, Literal
import json
import logging
import re
from langgraph.graph import StateGraph, END, START
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.tree import Tree
from rich.table import Table

from llms import LLM, MistralLLM, GeminiLLM

from envision_copilot.utils.config_loader import ConfigLoader
from envision_copilot.utils.prompt_loader import PromptLoader
from envision_copilot.core.memory import FactStore
from envision_copilot.core.planner import TreePlanner, Node

from envision_copilot.tools.structural import StructuralTools
from envision_copilot.tools.semantic import SemanticTools
from envision_copilot.tools.code_reader import CodeReader
from envision_copilot.tools.grep import GrepTools
from envision_copilot.tools.definitions import TOOLS_SCHEMA

def clean_json_string(s: str) -> str:
    """
    Cleans up common LLM JSON syntax errors.
    """
    # Remove markdown code blocks
    s = re.sub(r'```json\s*', '', s)
    s = re.sub(r'```', '', s)
    return s.strip()

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    messages: List[str]      # Chat History
    scratchpad: str          # ReAct trace (Thought/Action/Observation history)
    iterations: int
    latest_response: Dict    # Tool call and thought transfer
    last_observation: Dict   # Last tool result (for selection)
    appendix: List[Dict]     # Selected references with full data + reason
    final_answer: str        # Extracted final answer text

# --- The Brain ---
class EnvisionCopilot:
    def __init__(self, config_path: str = "config.yaml", verbose: bool = False, interactive: bool = False):
        self.config = ConfigLoader.load_config(config_path)
        self.verbose = verbose
        self.interactive = interactive
        self.console = Console()
        
        self.prompts = PromptLoader(self.config)
        
        # Memory & Planner are Session-specific (reset in run)
        self.memory = None 
        self.planner = None
        
        # Tools
        self.structural = StructuralTools(self.config)
        self.semantic = SemanticTools(self.config)
        self.reader = CodeReader(self.config)
        self.grep = GrepTools(self.config)
        
        # LLM
        model_name = self.config.get("agent", {}).get("main_model", "mistral")
        if "mistral" in model_name.lower():
             self.llm = MistralLLM(self.config)
        elif "gemini" in model_name.lower():
             self.llm = GeminiLLM(self.config)
        else:
             self.llm = MistralLLM(self.config)
             
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("think", self.think_node)
        workflow.add_node("act", self.act_node)
        
        workflow.add_edge(START, "think")
        
        workflow.add_conditional_edges(
            "think",
            self.decide_next,
            {
                "act": "act",
                "end": END
            }
        )
        
        # Determine if we loop back to think or end after acting
        workflow.add_conditional_edges(
            "act", 
            self.check_post_act,
            {
                "think": "think", 
                "end": END
            }
        )
        
        return workflow.compile()

    def _log_header(self, question: str):
        """Standardized One-Time Header."""
        t = Table(show_header=False, box=None)
        t.add_row("❓ ", f"[bold]{question}[/bold]")
        self.console.print(Panel(t, border_style="red", padding=(1, 2)))

    def _log_status(self, root: Node, current_focus: Node = None, title: str = "Plan Status"):
        """Visualizes Tree with Color Encoding."""
        tree = Tree(f"[bold]{title}[/bold]")
        
        def add_node(parent_tree, node: Node):
            # Style: Green=Done, Orange=Pending/Active, Red=Failed
            style = "dim"
            icon = "⚪"
            if node.status == "done":
                style = "green"
                icon = "✅"
            elif node.status == "failed":
                style = "red"
                icon = "❌"
            elif node == current_focus:
                style = "bold orange3"
                icon = "👉"
            
            # Determine Label
            # Meta-tools (manage_plan, final_answer) should NOT appear as "Tool:" - show Goal instead
            META_TOOLS = {"manage_plan", "final_answer", "ask_user"}
            if node.tool_call and node.tool_call.get('name') not in META_TOOLS:
                 label_text = f"Tool: {node.tool_call.get('name')}"
            else:
                 # Default to Goal (Question or Subtask)
                 label_text = node.goal[:60] + "..." if len(node.goal) > 60 else node.goal
                 
            label = f"[{style}]{icon} {label_text}[/{style}]"
            
            branch = parent_tree.add(label)
            for child in node.children:
                add_node(branch, child)
        
        add_node(tree, root)
        self.console.print(tree)

    def run(self, question: str):
        self.memory = FactStore(self.config)
        self.planner = TreePlanner(question)
        
        final_state = self.workflow.invoke({
            "question": question,
            "messages": [],
            "scratchpad": "",
            "iterations": 0,
            "last_observation": {},
            "appendix": [],
            "final_answer": ""
        }, config={"recursion_limit": 100})
        
        # Display Final Answer
        answer = final_state.get("final_answer") or self.planner.root.reasoning or "Task End"
        if self.verbose:
            self.console.print(Panel(
                Markdown(answer),
                title="✅ Final Answer",
                border_style="green",
                subtitle="Envision Copilot"
            ))
        
        # Display Appendix (Selected References Only) with Markdown
        appendix = final_state.get("appendix", [])
        if self.verbose and appendix:
            appendix_json = json.dumps(appendix, indent=2, ensure_ascii=False)
            self.console.print(Panel(
                Markdown(f"```json\n{appendix_json}\n```"),
                title="📎 Appendix (Selected References)",
                border_style="blue"
            ))
        
        return {
            "answer": answer,
            "appendix": appendix
        }

    def think_node(self, state: AgentState) -> Dict:
        iteration = state["iterations"]
        question = state["question"]
        scratchpad = state.get("scratchpad", "")
        
        # Get active node early for prompting context
        active_node = self.planner.get_active_leaf()
        
        # 1. Header
        if iteration == 0 and self.verbose:
            self._log_header(question)

        # 2. Context & Limits
        facts_text = self.memory.get_facts_text()
        facts_count = len(self.memory.facts)
        
        # Load from Config
        prompts_cfg = self.config.get("prompts", {})
        sys_template = prompts_cfg.get("system", "")
        philosophy = prompts_cfg.get("philosophy", "")
        context_setup = prompts_cfg.get("context_setup", "")
        instructions = prompts_cfg.get("instructions", "")
        tools_usage = prompts_cfg.get("tools_usage", "")
        
        max_depth = self.config.get("agent", {}).get("constraints", {}).get("max_depth", 5)
        max_iter = self.config.get("agent", {}).get("constraints", {}).get("max_iterations", 10)
        
        # Calculate current depth
        current_depth = 0
        temp = active_node
        while temp.parent:
            current_depth += 1
            temp = temp.parent
        
        # Quota info for LLM
        remaining_iter = max(0, max_iter - iteration)
        remaining_depth = max(0, max_depth - current_depth)
        quota_info = f"\n[QUOTA: Iter {iteration+1}/{max_iter} | Depth {current_depth}/{max_depth} | Remaining: {remaining_iter} iters, {remaining_depth} depth levels]"
        
        # Format Dynamic Values in system template
        system_prompt = sys_template.format(
            facts_count=facts_count,
            iteration=iteration,
            max_iterations=max_iter,
            max_depth=max_depth
        )
        
        # Combine ALL prompt sections
        full_system = f"{system_prompt}\n\n{philosophy}\n\n{context_setup}\n\n{instructions}\n\n{tools_usage}"
        
        # Dynamic Depth Hint
        depth_hint = ""
        if iteration > 0 and active_node.parent is None:
             depth_hint = "\n\nTIP: You are currently at Root Level (Depth 0). If you have found relevant information (e.g., from search), you MUST create a SUBTASK (`manage_plan`) to investigate it distinctively. Do not just keep calling tools at the root level."

        full_prompt = f"{full_system}\n\nOBJECTIVE: {question}\n\nPLAN & HISTORY:\n{scratchpad}\n\nFACTS:\n{facts_text}{depth_hint}{quota_info}\n\nThought:"
        
        # ... [Rest of loop logic] ...
        
        # 3. LLM Call (With Internal Retry)
        import sys
        max_retries = 2
        
        tool_call = None
        thought = ""
        response_text = ""
        
        rejection_reason = ""
        
        for attempt in range(max_retries + 1):
             if attempt > 0:
                 error_msg = rejection_reason if rejection_reason else "You failed to output the Action JSON in the previous attempt (or syntax was wrong)."
                 # CRITICAL: Keep History & Facts in Retry Prompt to prevent Amnesia/Loops
                 current_prompt = f"{system_prompt}\n\nOBJECTIVE: {question}\n\nPLAN & HISTORY:\n{scratchpad}\n\nFACTS:\n{facts_text}\n\nERROR: {error_msg}\nINSTRUCTION: Output the Action JSON block again. You may add a brief 'Thought' before it to correct yourself.\nAction:"
             else:
                 current_prompt = full_prompt

             response_text = self.llm.generate(current_prompt)
             
             # ... (Parsing logic remains the same) ...
             if "Action:" in response_text:
                 parts = response_text.split("Action:", 1)
                 thought = parts[0].replace("Thought:", "").strip()
                 json_part = parts[1].strip()
             else:
                 # Implicit
                 json_part = response_text
                 json_start = json_part.find("```json")
                 if json_start != -1:
                     thought = json_part[:json_start].replace("Thought:", "").strip()
             
             # Extract JSON
             try:
                 # 1. Regex for ```json blocks
                 match = re.search(r'```json(.*?)```', json_part, re.DOTALL)
                 if match:
                     json_str = match.group(1).strip()
                     tool_call = json.loads(json_str)
                 else:
                     # 2. Direct Load Aggressive
                     clean_json = json_part.replace("```json", "").replace("```", "").strip()
                     idx = clean_json.find("{")
                     if idx != -1:
                         clean_json = clean_json[idx:]
                     tool_call = json.loads(clean_json)
             except Exception:
                 tool_call = None
             
             # 2b. Fallback: Parse Python-style function calls like `tool_name({'arg': 'val'})`
             if not tool_call:
                 # Match: tool_name({...}) or tool_name({...}, ...)
                 func_match = re.search(r'(\w+)\s*\(\s*(\{.*\})\s*\)', json_part, re.DOTALL)
                 if func_match:
                     tool_name_parsed = func_match.group(1)
                     args_str = func_match.group(2)
                     # Convert single quotes to double quotes for JSON parsing
                     args_str = args_str.replace("'", '"')
                     try:
                         args_parsed = json.loads(args_str)
                         tool_call = {"name": tool_name_parsed, "arguments": args_parsed}
                     except Exception:
                         pass
             
             # 2c. Fallback: Detect "Final Answer:" as implicit final_answer tool
             if not tool_call and "Final Answer:" in response_text:
                 # Extract the answer after "Final Answer:"
                 fa_idx = response_text.find("Final Answer:")
                 answer_text = response_text[fa_idx + len("Final Answer:"):].strip()
                 tool_call = {"name": "final_answer", "arguments": {"answer": answer_text}}
                 # Extract thought before Final Answer
                 thought = response_text[:fa_idx].replace("Thought:", "").strip()
             
             # LOOP DETECTION
             if tool_call:
                 previous_tool = state.get("latest_response", {}).get("tool_call")
                 if previous_tool and tool_call == previous_tool:
                      # Detected exact repetition
                      if attempt < max_retries:
                           rejection_reason = f"You proposed the exact same action as the previous step ({json.dumps(tool_call)}). This is a loop. You MUST propose a DIFFERENT action or arguments."
                           tool_call = None # Invalidate
                           thought += " [System: Loop Detected - Retrying]"
                           continue # Go to next attempt
                      else:
                           # Last attempt failed
                           pass
                 else:
                     break 
                 
             if attempt < max_retries:
                 pass
             else:
                 thought = "Failed to generate valid Action JSON after retries."
                 if self.verbose:
                     self.console.print(Panel(f"Raw Output:\n{response_text}", title="🐞 Debug: Model Failure", border_style="red"))
        
        # 4. Visualization - Always show Thought (Final Answer is shown separately by run())
        if self.verbose:
            # Fallback for empty thought
            display_thought = thought if thought.strip() else f"Thinking... (Model did not output explicit 'Thought')"

            # 4. Visualization - Thought with Iter x/y
            self.console.print(Panel(Markdown(display_thought), title=f"🧠 Thought (Iter {iteration+1}/{max_iter})", border_style="purple"))

            # 4b. Visualization - Memory (Facts)
            if facts_count > 0:
                 # Truncate for display if needed
                 display_facts = facts_text
                 if len(display_facts) > 300:
                      display_facts = display_facts[:300] + "\n... [Truncated]"
                 
                 self.console.print(Panel(
                      Markdown(f"**Facts ({facts_count})**:\n{display_facts}"),
                      title="🗃️ Memory Evolution",
                      border_style="blue"
                 ))

        return {
            "latest_response": {"thought": thought, "tool_call": tool_call},
            "iterations": iteration + 1,
            "scratchpad": scratchpad + f"\n\nThought: {thought}\n"
        }

    def act_node(self, state: AgentState) -> Dict:
        latest = state.get("latest_response", {})
        tool_call = latest.get("tool_call")
        
        if not tool_call:
            return {"messages": ["Error: No tool call generated."]}
            
        tool_name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        
        # Meta-tools don't create subtasks
        META_TOOLS = {"manage_plan", "final_answer", "ask_user"}
        
        # 1. Create Subtask Node for this Action (Tree Evolution) - BEFORE showing status
        new_node = None
        if tool_name not in META_TOOLS:
            active = self.planner.get_active_leaf()
            new_node = self.planner.add_subtask(active, f"{tool_name}")
            new_node.tool_call = tool_call
        
        # 2. Show Plan Status BEFORE Action (Plan first, execute second)
        if self.verbose:
            iteration = state.get("iterations", 0)
            max_depth = self.config.get("agent", {}).get("constraints", {}).get("max_depth", 5)
            max_iter = self.config.get("agent", {}).get("constraints", {}).get("max_iterations", 10)
            
            # Calculate Depth
            current_focus = self.planner.get_active_leaf()
            current_depth = 0
            temp = current_focus
            while temp.parent:
                current_depth += 1
                temp = temp.parent
            
            status_title = f"Plan Status (Iter {iteration}/{max_iter} | Depth {current_depth}/{max_depth})"
            self._log_status(self.planner.root, current_focus, title=status_title)
        
        # 3. Action Panel (Cyan Markdown)
        if self.verbose:
            self.console.print(Panel(
                Markdown(f"**Tool**: `{tool_name}`\n**Args**: \n```json\n{json.dumps(args, indent=2)}\n```"),
                title="🚀 Action",
                border_style="cyan"
            ))
        
        # 3. Execute
        try:
            if tool_name == "semantic_search":
                result = self.semantic.search(**args)
            elif tool_name == "structural_explorer":
                result = self.structural.explore(**args)
            elif tool_name == "read_file":
                file_path = args.get("path") or args.get("file_path")
                result = self.reader.read_section(
                    file_path=file_path,
                    start_line=args.get("start_line", 1),
                    end_line=args.get("end_line", 100)
                )
            elif tool_name == "grep_search":
                result = self.grep.search(**args)
            elif tool_name == "manage_plan":
                result = self._handle_plan_update(action=args.get("action"), description=args.get("description"))
            elif tool_name == "ask_user":
                if self.interactive:
                    q = args.get("question")
                    self.console.print(Panel(q, title="❓ Copilot Request", border_style="yellow"))
                    ans = input(">> ")
                    result = f"User Answer: {ans}"
                    self.memory.add_fact(f"User: {ans}")
                else:
                    result = "Error: Interactive mode disabled (use -i)."
            elif tool_name == "add_to_appendix":
                # Add last observation to appendix with reason
                reason = args.get("reason", "Relevant to query")
                last_obs = state.get("last_observation", {})
                if last_obs:
                    entry = {**last_obs, "reason": reason}
                    appendix = state.get("appendix", [])
                    appendix.append(entry)
                    result = f"Added to appendix: {last_obs.get('tool')} with reason: {reason}"
                    return {
                        "appendix": appendix,
                        "scratchpad": state["scratchpad"] + f"Action: add_to_appendix({args})\nObservation: {result}\n"
                    }
                else:
                    result = "Error: No observation to add to appendix"
            else:
                result = f"Error: Unknown tool '{tool_name}'"
                
        except Exception as e:
            result = f"Error executing tool: {e}"

        # 4. Store observation for potential appendix selection
        observation_entry = {
            "tool": tool_name,
            "args": args,
            "result": result  # Full result, not truncated
        }
        
        # 5. Visual Log (Observation)
        res_str = str(result)
        if self.verbose:
            # Create a nice summary for the panel (truncated for display only)
            display_content = self._format_observation(result)
            
            self.console.print(Panel(
                Markdown(display_content),
                title=f"👀 Observation ({tool_name})",
                border_style="green"
            ))

        # 6. Update Scratchpad + Last Observation
        return {
            "scratchpad": state["scratchpad"] + f"Action: {tool_name}({args})\nObservation: {res_str}\n",
            "last_observation": observation_entry
        }
    
    def _format_observation(self, result: Any) -> str:
        """Formats the tool result for nice markdown display with line-based truncation."""
        n = self.config.get("presentation", {}).get("max_output_lines", 20)
        
        if isinstance(result, (dict, list)):
            try:
                # Format full JSON with indentation
                formatted = json.dumps(result, indent=2, ensure_ascii=False)
                
                # Truncate by LINES only (N first + N last)
                lines = formatted.splitlines()
                max_lines = n * 2
                if len(lines) > max_lines:
                    head = lines[:n]
                    tail = lines[-n:]
                    formatted = "\n".join(head) + f"\n\n... [Masked {len(lines)-max_lines} lines] ...\n\n" + "\n".join(tail)
                
                return f"```json\n{formatted}\n```"
            except Exception:
                return str(result)
                
        elif isinstance(result, list):
            try:
                formatted = json.dumps(result, indent=2, ensure_ascii=False)
                lines = formatted.splitlines()
                max_lines = n * 2
                if len(lines) > max_lines:
                    head = lines[:n]
                    tail = lines[-n:]
                    formatted = "\n".join(head) + f"\n\n... [Masked {len(lines)-max_lines} lines] ...\n\n" + "\n".join(tail)
                return f"```json\n{formatted}\n```"
            except Exception:
                return str(result)
        
        return str(result)

    def decide_next(self, state: AgentState) -> str:
        latest = state.get("latest_response", {})
        tool_call = latest.get("tool_call")

        # 1. Critical Failure Check
        if not tool_call:
             # Model failed to generate a valid action (and retries exhausted)
             if self.verbose:
                 self.console.print(Panel("❌ Agent Failed to determine next step. Stopping.", title="System Error", border_style="red"))
             return "end"

        # 2. Handle final_answer: Auto-add last observation to appendix, run synthesis, then end
        if tool_call.get("name") == "final_answer":
            answer = tool_call.get("arguments", {}).get("answer", "")
            self.planner.root.reasoning = answer
            state["final_answer"] = answer
            
            # AUTO-ADD last observation to appendix if not empty
            last_obs = state.get("last_observation", {})
            current_appendix = state.get("appendix", [])
            if last_obs and last_obs.get("tool"):
                # Add with default reason from config
                default_reason = self.config.get("prompts", {}).get("default_appendix_reason", "Referenced in final answer")
                entry = {**last_obs, "reason": default_reason}
                current_appendix.append(entry)
            
            # SYNTHESIS PHASE: Review appendix before ending
            if current_appendix:
                curated_appendix = self._synthesize_appendix(answer, current_appendix)
                state["appendix"] = curated_appendix
            
            return "end"

        # 3. Constraints - Use config max_iterations
        max_iter = self.config.get("agent", {}).get("constraints", {}).get("max_iterations", 10)
            
        if state.get("iterations", 0) > max_iter:
            # FORCE BEST EFFORT ANSWER
            if self.verbose:
                 self.console.print(Panel("⏳ Max iterations reached. Synthesizing Best Effort Answer...", style="bold red"))
            
            # 1. Synthesize Answer using LLM
            synthesis_prompt = f"""
            You have reached the maximum number of iterations ({max_iter}).
            You must provide a BEST EFFORT answer based on what you have found so far.
            
            USER QUESTION: {self.memory.facts[0] if self.memory.facts else 'Unknown'}
            Facts Found:
            {self.memory.get_facts_text()}
            
            PLAN STATUS:
            {self.planner.get_plan_status()}
            
            GENERATE A FINAL ANSWER (French). Summarize findings, state what is missing, and provide leads.
            """
            try:
                best_effort_answer = self.llm.generate(synthesis_prompt)
            except Exception:
                best_effort_answer = "Max iterations reached. Partial findings: " + self.memory.get_facts_text()
            
            state["final_answer"] = best_effort_answer
            self.planner.root.reasoning = best_effort_answer

            # 2. Appendix Synthesis
            current_appendix = state.get("appendix", [])
            if current_appendix:
                 curated_appendix = self._synthesize_appendix(best_effort_answer, current_appendix)
                 state["appendix"] = curated_appendix
            
            return "end"
        
        # 4. Plan Status
        if self.planner and self.planner.root.status == "done":
             pass

        return "act"

    def check_post_act(self, state: AgentState) -> str:
        """Determines if we should stop after an action."""
        latest = state.get("latest_response", {})
        tool_call = latest.get("tool_call")
        
        if tool_call and tool_call.get("name") == "final_answer":
            return "end"
            
        return "think"

    # --- Helpers ---
    def _handle_plan_update(self, action: str, description: str):
        active_node = self.planner.get_active_leaf()
        if action == "add_subtask":
            self.planner.add_subtask(active_node, description)
            return f"Plan Updated: Added subtask '{description}'"
        elif action == "mark_done":
            active_node.status = "done"
            active_node.reasoning = description
            return f"Plan Updated: Marked focus as DONE. Reasoning: {description}"
        elif action == "mark_failed":
             active_node.status = "failed"
             active_node.reasoning = description
             return f"Plan Updated: Marked focus as FAILED."
        return "Invalid plan action"

    def _synthesize_appendix(self, final_answer: str, current_appendix: List[Dict]) -> List[Dict]:
        """
        Synthesis phase: LLM reviews appendix entries and keeps only relevant ones.
        Updates reasons if needed.
        """
        if not current_appendix:
            return []
        
        # Prepare appendix summary for LLM (without full content to save tokens)
        appendix_summary = []
        for i, entry in enumerate(current_appendix):
            summary = {
                "index": i,
                "tool": entry.get("tool"),
                "reason": entry.get("reason"),
                "source": entry.get("args", {}).get("query") or entry.get("args", {}).get("path") or str(entry.get("args"))[:50]
            }
            appendix_summary.append(summary)
        
        prompt_template = self.config.get("prompts", {}).get("synthesis_prompt", "")
        if not prompt_template:
            return current_appendix  # No synthesis if no prompt
        
        prompt = prompt_template.format(
            final_answer=final_answer[:500] + "...",
            appendix_summary=json.dumps(appendix_summary, indent=2)
        )

        try:
            response = self.llm.generate(prompt)
            
            # Parse JSON from response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                decisions = json.loads(match.group())
                
                curated = []
                for decision in decisions:
                    idx = decision.get("index")
                    if decision.get("keep", False) and 0 <= idx < len(current_appendix):
                        entry = current_appendix[idx].copy()
                        # Update reason if provided
                        if "reason" in decision:
                            entry["reason"] = decision["reason"]
                        curated.append(entry)
                
                if self.verbose:
                    self.console.print(f"[dim]📎 Appendix: {len(current_appendix)} → {len(curated)} entries after synthesis[/dim]")
                
                return curated
        except Exception as e:
            if self.verbose:
                self.console.print(f"[dim]Synthesis warning: {e}[/dim]")
        
        # Fallback: return original appendix
        return current_appendix
