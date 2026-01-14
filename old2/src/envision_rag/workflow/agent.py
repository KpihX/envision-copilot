"""Agent Workflow - ReAct-style agent with LangGraph."""
from typing import TypedDict, List, Dict, Any
import re
import pprint
from langgraph.graph import StateGraph, END, START
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from envision_rag.agents.prepare_agent import prepare_agent
from envision_rag.tools.graph_tools import GraphTools
from envision_rag.tools.search_tools import SearchTools
from envision_rag.index.vector_tools import VectorTools

import warnings
# Suppress Google Generative AI deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
warnings.filterwarnings("ignore", category=FutureWarning, module="envision_rag.agents.gemini_agent")

class AgentState(TypedDict):
    question: str
    messages: List[str]  # Chat/Interaction history
    scratchpad: str      # Working memory (ReAct trace)
    final_answer: str
    step_count: int
    facts: List[Any]     # Structured data from tools for Appendix
    plan: List[str]      # Explicit plan of action (checklist)

class AgentWorkflow:
    def __init__(self, config: Dict[str, Any], graph_tools: GraphTools, 
                 verbose: bool = False, logger = None, interactive: bool = False):
        self.config = config
        self.tools = graph_tools
        self.search_tools = SearchTools()
        self.verbose = verbose
        self.interactive = interactive  # If True, agent can ask clarifications
        self.console = Console()
        self.logger = logger
        # Lazy load vector tools (might require index to exist)
        try:
             index_config = config.get('index', {})
             store_path = index_config.get('store_path', 'data/vector_store')
             model_name = index_config.get('embedding_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
             self.vector_tools = VectorTools(index_dir=store_path, model_name=model_name)
        except Exception as e:
             print(f"⚠️ Vector Index not found or error loading: {e}. Semantic search disabled.")
             self.vector_tools = None

        self.llm = prepare_agent(config.get('agent', {}).get('main_model', 'mistral'))
        
        # Tool Registry
        self.tool_map = {
            "scan_references": self.tools.scan_references,
            "describe_impact": self.tools.describe_impact,
            "grep_code": self.search_tools.grep_code,
            "read_code": self.search_tools.read_code
        }
        if self.vector_tools:
            self.tool_map["search_code"] = self.vector_tools.search_code

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("reason", self.reason_node)
        workflow.add_node("act", self.execution_node)
        # Note: A "reflect" node was considered for explicit self-critique after observations,
        # but currently the reflection logic is merged into reason_node for simplicity.

        workflow.add_edge(START, "reason")
        
        # Conditional edge from Reason:
        # If "Action: ..." -> Go to Act
        # If "Final Answer: ..." -> END
        workflow.add_conditional_edges(
            "reason",
            self.decide_next_step,
            {
                "act": "act",
                "end": END
            }
        )
        
        workflow.add_edge("act", "reason")

        return workflow.compile()



    def _log(self, title: str, content: str, style: str = "info"):
        """
        Unified Live Logging using Rich + Session Logger.
        """
        # Always log to session logger if available (for replay)
        if self.logger:
            self.logger.log_event(style, title, content, style)
        
        if not self.verbose: return
        
        if style == "thought":
            # Thought often contains markdown (lists, bold), so render it
            self.console.print(Panel(Markdown(content), title=f"🧠 {title}", border_style="purple"))
        elif style == "action":
            self.console.print(Panel(Markdown(content), title=f"🛠️ {title}", border_style="blue"))
        elif style == "observation":
             # Smart Line-based Truncation
             lines = content.splitlines()
             N = 15 # Number of lines to keep at head and tail
             
             if len(lines) > 2 * N:
                 head = lines[:N]
                 tail = lines[-N:]
                 truncated_content = "\n".join(head) + f"\n\n... [Masked {len(lines) - 2*N} lines] ...\n\n" + "\n".join(tail)
             else:
                 truncated_content = content
                 
             # Render as Python syntax for coloring dictionary structures
             self.console.print(Panel(Syntax(truncated_content, "python", word_wrap=True, theme="monokai"), title=f"👀 {title}", border_style="green"))
        elif style == "answer":
             # Markdown renders beautifully and avoids raw text truncation issues
             self.console.print(Panel(Markdown(content), title=f"✅ {title}", border_style="cyan"))
        elif style == "error":
             self.console.print(Panel(Markdown(content), title=f"❌ {title}", border_style="red"))
        elif style == "appendix":
             self.console.print(Panel(Markdown(content), title=f"📎 {title}", border_style="dim"))
        else:
            self.console.print(f"[bold]{title}[/bold]: {content}")



    def reason_node(self, state: AgentState) -> Dict:
        """
        The 'Brain'. Generates Thought + Action OR Final Answer.
        """
        question = state["question"]
        scratchpad = state.get("scratchpad", "")
        facts = state.get("facts", [])
        
        # Format Textual Facts for Prompt Context
        facts_text = ""
        if facts:
            facts_text = "\n\nOBSERVED FACTS from Tools:\n"
            seen = set()
            for f in facts:
                # Simple string representation for the LLM to understand
                s = str(f)
                if s not in seen:
                    facts_text += f"- {s}\n"
                    seen.add(s)
        
        # Load prompts from config
        prompts = self.config.get('prompts', {})
        system_prompt = prompts.get('agent_system', "You are an expert software architect...")
        tools_desc = prompts.get('tools_description', "")
        instructions = prompts.get('agent_instructions', "")
        interactive_instr = prompts.get('interactive_instruction', "") if self.interactive else ""
        
        prompt = f"""
{system_prompt}

TOOLS AVAILABLE:
{tools_desc}

QUESTION: {question}
PLAN: {state.get('plan', [])}
{facts_text}
PREVIOUS REASONING:
{scratchpad}

{instructions}
{interactive_instr}

RESPONSE FORMAT:
Thought: <reasoning>
Action: <tool_call>
(System will append Observation here)

OR

Thought: <reasoning>
Final Answer: <Narrative, convincing summary of findings. Point to Appendix for details.>
"""
        response = self.llm.generate_response(prompt)
        
        # Parse Response for logging
        if "Thought:" in response:
            thought = response.split("Thought:")[-1].split("Action:")[0].split("Final Answer:")[0].strip()
            self._log("Thought", thought, "thought")
        
        if "Action:" in response:
            action = response.split("Action:")[-1].strip()
            self._log("Action Proposed", action, "action")

        # Post-process for Appendix if Final Answer
        if "Final Answer:" in response:
             final_parts = response.split("Final Answer:")
             final_ans = final_parts[-1].strip()
             
             appendix = ""
             if facts:
                appendix = "\n\n## Appendix\n"
                
                # 1. Flatten and Deduplicate
                unique_refs = {} # key -> formatted string
                unique_texts = set()
                
                for fact in facts:
                    if isinstance(fact, list):
                        items = fact
                    else:
                        items = [fact]
                        
                    for item in items:
                        if isinstance(item, dict) and 'source_script' in item:
                            # It's a reference object
                            key = f"{item['source_script']}|{item.get('relationship','')}|{item.get('target_file','')}"
                            if key not in unique_refs:
                                rel = item.get('relationship', 'ref')
                                fmt = f"- **{rel}**: `{item['source_script']}` -> `{item['target_file']}`"
                                unique_refs[key] = fmt
                        elif isinstance(item, dict) and 'file' in item and 'line' in item:
                             # It's a Grep result (new)
                             key = f"{item['file']}|{item['line']}"
                             if key not in unique_refs:
                                 fmt = f"- **Match** `SearchTools` in `{item['file']}` (Line {item['line']}): `{item['content'].strip()}`"
                                 unique_refs[key] = fmt
                        else:
                            # It's text or other
                            s = str(item).strip()
                            if s:
                                unique_texts.add(s)
                
                # 2. Build Appendix String
                if unique_refs:
                    appendix += "### References\n"
                    # Sort by Source Script for readability
                    appendix += "\n".join(sorted(unique_refs.values())) + "\n"
                
                if unique_texts:
                    if unique_refs: appendix += "\n"
                    # Filter out purely summary strings if they are redundant with detailed refs
                    clean_notes = [t for t in unique_texts if not t.startswith("Found")]
                    if clean_notes:
                        appendix += "### Notes\n"
                        appendix += "\n".join([f"- {t}" for t in sorted(clean_notes)]) + "\n"
             
             # Log the Final Answer cleanly
             self._log("Final Answer", final_ans, "answer")
             
             if appendix:
                 # Explicitly log the appendix if it exists
                 self._log("Appendix", appendix, "appendix")
             
             # Append appendix to response so it is recorded in history,
             # but we rely on _log for the UI display.
             response += appendix

        # Parse Plan updates if any
        new_plan = state.get("plan", [])
        if "Plan:" in response:
             try:
                 # Regex to find list-like structure
                 plan_match = re.search(r'Plan:\s*(\[.*?\])', response, re.DOTALL)
                 if plan_match:
                     import ast
                     new_plan = ast.literal_eval(plan_match.group(1))
             except:
                 pass # Keep old plan if parsing fails

        return {
            "messages": [response],
            "scratchpad": scratchpad + f"\nAgent: {response}\n",
            "step_count": state.get("step_count", 0) + 1,
            "plan": new_plan
        }

    def execution_node(self, state: AgentState) -> Dict:
        """
        Parses the last message for 'Action: ...' and runs it.
        """
        last_message = state["messages"][-1]
        scratchpad = state.get("scratchpad", "")
        
        # Parse Action
        match = re.search(r'Action:\s*(\w+)\s*\((.*?)\)', last_message, re.DOTALL)
        if not match:
            return {
                "scratchpad": scratchpad + "\nSystem: Error: No valid Action found. Please use 'Action: tool(\"arg\")'.\n"
            }
            
        tool_name = match.group(1)
        tool_arg = match.group(2).strip('"\' ')
        
        if tool_name not in self.tool_map:
             return {
                "scratchpad": scratchpad + f"\nSystem: Error: Tool '{tool_name}' not found.\n"
            }
            
        try:
            # print(f"🔧 Executing {tool_name}('{tool_arg}')...") # Handled by _log
            result = self.tool_map[tool_name](tool_arg)
            
            # Format observation summary for log
            # Use pprint for nice structure of lists/dicts
            obs_str = pprint.pformat(result, indent=2, width=100)
            
            self._log("Observation", obs_str, "observation")

            observation = f"\nObservation: {result}\n"
            
            # Store result in facts if it's substantial (list or dict)
            facts = state.get("facts", [])
            # Handle Structured Output (Dict with 'results')
            if isinstance(result, dict) and "results" in result:
                 facts.append(result["results"]) # Store the list for Appendix
                 # Also store the summary string as a fact for the LLM context
                 if "summary" in result:
                     facts.append(result["summary"])
                 # Store unique targets summary if available (for deterministic counting)
                 if "unique_targets_count" in result and result["unique_targets_count"] > 0:
                     unique_summary = f"Distinct Targets ({result['unique_targets_count']}): {', '.join(result['unique_targets'])}"
                     facts.append(unique_summary)
            elif isinstance(result, (list, dict)) and result:
                 facts.append(result)
            elif isinstance(result, str) and len(result) > 50:
                 facts.append(result)

        except Exception as e:
            observation = f"\nObservation: Error executing tool: {e}\n"
            result = None
            
        return {
            "scratchpad": scratchpad + observation,
            "facts": facts if 'facts' in locals() else []
        }

    def decide_next_step(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if "Final Answer:" in last_message:
            return "end"
        if "Action:" in last_message:
            return "act"
        return "end" # Fallback
