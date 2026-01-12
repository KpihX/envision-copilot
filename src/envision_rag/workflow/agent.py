from typing import TypedDict, List, Annotated, Dict, Any, Union
import re
import operator
import pprint
from langgraph.graph import StateGraph, END, START
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

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
    def __init__(self, config: Dict[str, Any], graph_tools: GraphTools, verbose: bool = False):
        self.config = config
        self.tools = graph_tools
        self.search_tools = SearchTools() # Initialize SearchTools
        self.verbose = verbose
        self.console = Console()
        # Lazy load vector tools (might require index to exist)
        try:
            self.vector_tools = VectorTools()
        except:
             print("⚠️ Vector Index not found. Semantic search disabled.")
             self.vector_tools = None

        self.llm = prepare_agent(config.get('agent', {}).get('main_model', 'mistral'))
        
        # Tool Registry
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
        # workflow.add_node("reflect", self.reflection_node) # Merged into 'reason' for simplicity? 
        # Actually user wants explicit Reflect. Let's keep it simple first: Reason -> Act -> Reason...

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
        Unified Live Logging using Rich.
        """
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
        
        prompt = f"""
You are an expert software architect analyzing a Supply Chain Optimization Codebase written in Envision DSL (Lokad).
You have access to a Structural Graph (Imports/Reads/Writes) and a Semantic Code Index (Vector Search).

CONTEXT:
- The codebase is organized directly in folders (e.g. `/3. Inspectors/`).
- Scripts are identified by "Logical Paths" (e.g. `/1. utilities/foo`).
- Filenames (e.g. `12345.nvn`) are internal IDs. Prefer Logical Paths when using Graph Tools.
- Supply Chain Concepts: Stock, Forecast, Ordering, Dispatch.
- **PRIORITY RULE**: When asked "Where to configure X?", prefer the location where the **Business Logic** is defined (e.g. `match` statements, formulas) over the location where the data is loaded (e.g. `Manual Inputs`, `read`). The *Why* and *How* (Logic) is more important than the *What* (Data).

TOOLS AVAILABLE:
- scan_references(query): Finds relationships. Input format: "keyword path" 
    * keyword: "read", "write", "import" (or "any").
    * path: (Optional) Segment of the file path. Default "*".
    * Example: `scan_references("import")` lists all modules used.
    * Example: `scan_references("read Items.ion")` finds who reads items.
- grep_code(pattern): EXACT REGEX search. Essential for finding variable definitions/assignments.
    * Example: `grep_code("ReDispatchCycle =")` finds where the variable is assigned.
- read_code(file_path, start_line, end_line): Read specific lines of a file.
    * Use to inspect Logic Blocks found by grep/search.
    * Example: `read_code("/1. utilities/foo", 10, 50)`
- describe_impact(script_path): What files does this script generate and who reads them?
- search_code(query): Semantic search for logic/snippets. (Input: Simple string query).

QUESTION: {question}
PLAN: {state.get('plan', [])}
{facts_text}
PREVIOUS REASONING:
{scratchpad}

INSTRUCTIONS:
1. Analyze the Question and any OBSERVED FACTS.
2. If you have the answer, output 'Final Answer: ...'. 
   - **CRITICAL**: Do NOT just give a dry one-line answer. 
   - **SYNTHESIZE**: Explain *how* you arrived at the conclusion.
   - **JUSTIFY**: Use the facts to convince the user.
   - **REFERENCE**: Explicitly mention "Please refer to the Appendix for the full list of..." if facts were moved there.
   - The Appendix is automatically generated, so you don't need to repeat the raw list, but you MUST contextually point to it.
3. If you need more facts, use a Tool. Format: 'Action: tool_name("arg")'
4. ALWAYS explain your reasoning with 'Thought: ...' before acting.
5. **REFORMULATION STRATEGY**:
   - If `search_code` returns references (graphs, usage) but NO DEFINITION:
     - **DO NOT give up**.
     - **DO NOT hallucinate**.
     - **REFORMULATE** your query (e.g., try English keywords like "function", "def", "logic", or synonyms).
     - Example: If "calculer stock" fails, try "stock calculation logic" or "stockEvol".
     - **USE `read_code`**: If `grep_code` finds a `def` or function signature, ALWAYS use `read_code` to verify the logic inside before answering.
6. **PLANNING**:
   - At the beginning, propose a Plan: `Plan: ["Step 1", "Step 2"]`
   - If a step fails or is insufficient, UPDATE the plan.
   - Mention the plan status in your thought process.
7. **STOPPING CRITERIA**:
   - DO NOT simulate the "Observation" part. 
   - After outputting "Action: ...", **STOP generating**. The system will provide the observation.

8. **VERIFICATION & EXHAUSTIVENESS** (Phase 6):
   - **BREADTH-FIRST**: When `search_code` returns multiple relevant candidates (e.g. A, B, C), you MUST acknowledge ALL of them in your Plan.
     - BAD: "I found A, so I will check A."
     - GOOD: "I found A (Rank 1) and B (Rank 3). Both seem relevant. I will check A *and* B."
   - **DEPTH-FIRST**: You are FORBIDDEN from generating a Final Answer until you have verified (via `read_code`) at least the top 2-3 most promising candidates.
     - **CRITICAL**: `search_code` results contain `File: ...` and `Context: ...`. **USE THESE PATHS DIRECTLY**.
     - Do NOT use `grep_code` to find what you already found. 
     - If `search_code` says "File: /1. util/foo", JUST RUN `read_code("/1. util/foo", 1, 50)`.
     - Compare candidates: "Why is A better than B?"
   - **REPORTING**: Your Final Answer MUST explicity mention the alternatives you explored.
     - "I found `StockEndWeek` and `StockEvol`. I selected `StockEndWeek` because X, but `StockEvol` is also valid for Y."

9. **SELF-CORRECTION**:
   - If `grep_code` finds a definition, asking "Where is it defined?" is stupid. You just found it. Read it!
   - If `search_code` gives you a file path, **READ IT**. Do not ask "Where is it?".
   - If you found a function name (e.g. `StockEndWeek`), ask yourself: "Is there a more specific function mentioned in the context or query?"
   - Use `grep_code` to search for `def` definitions of key concepts.

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
