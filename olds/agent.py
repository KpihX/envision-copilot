import operator
import re
import pprint
from typing import Annotated, TypedDict, Union, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from llms import get_llm
from .utils import ConfigLoader
from .tools.structural import StructuralTools
from .tools.semantic import SemanticTools
from .tools.read_code import CodeReader

# Define State (Restored from old2/agent.py)
class AgentState(TypedDict):
    question: str
    messages: List[str]  # Chat/Interaction history
    scratchpad: str      # Working memory (ReAct trace)
    final_answer: str
    step_count: int
    facts: List[Any]     # Structured data from tools for Appendix
    plan: List[str]      # Explicit plan of action (checklist)

class EnvisionAgent:
    def __init__(self, config_path: str = "config.yaml", verbose: bool = False, interactive: bool = False):
        self.config = ConfigLoader.load_config(config_path)
        self.verbose = verbose
        self.interactive = interactive
        self.console = Console()
        
        # Tools
        self.structural = StructuralTools(config_path)
        self.semantic = SemanticTools(config_path)
        self.reader = CodeReader(config_path)
        
        # Tool Map (Manual Dispatch)
        self.tool_map = {
            "scan_network_context": self.structural.scan_network_context,
            "find_producers": self.structural.find_producers,
            "search_code": self.semantic.search_code,
            "read_code": self.reader.read_code
        }
        
        # LLM
        # Note: We use the raw text generation capability for ReAct prompting
        self.llm = get_llm(self.config.get("defaults", {}).get("model", "mistral"))
        
        # Graph
        self.app = self.build_graph()

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("reason", self.reason_node)
        workflow.add_node("act", self.execution_node)
        
        workflow.add_edge(START, "reason")
        workflow.add_conditional_edges("reason", self.decide_next_step, {"act": "act", "end": END})
        workflow.add_edge("act", "reason")
        
        return workflow.compile()

    def _log(self, title: str, content: str, style: str = "info"):
        """Rich logging for Verbose Mode."""
        if not self.verbose: return
        
        if style == "thought":
            self.console.print(Panel(Markdown(content), title=f"🧠 {title}", border_style="purple"))
        elif style == "action":
            self.console.print(Panel(Markdown(content), title=f"🛠️ {title}", border_style="blue"))
        elif style == "observation":
            lines = content.splitlines()
            if len(lines) > 20:
                content = "\n".join(lines[:10]) + f"\n\n... [Masked {len(lines)-20} lines] ...\n\n" + "\n".join(lines[-10:])
            self.console.print(Panel(Syntax(content, "python", theme="monokai", word_wrap=True), title=f"👀 {title}", border_style="green"))
        elif style == "answer":
            self.console.print(Panel(Markdown(content), title=f"✅ {title}", border_style="cyan"))
        elif style == "appendix":
            self.console.print(Panel(Markdown(content), title=f"📎 {title}", border_style="dim"))

    def reason_node(self, state: AgentState) -> Dict:
        """The Brain: Generates Thought + Action OR Final Answer."""
        question = state["question"]
        scratchpad = state.get("scratchpad", "")
        facts = state.get("facts", [])
        
        # Context from facts
        facts_text = ""
        if facts:
             facts_text = "\n\nOBSERVED FACTS:\n" + "\n".join([f"- {str(f)[:200]}" for f in facts[-5:]]) # Limit context

        prompts = self.config.get("prompts", {})
        prompt = f"""
{prompts.get("system", "")}

TOOLS AVAILABLE:
{prompts.get("tools_description", "")}

QUESTION: {question}
{facts_text}

PREVIOUS REASONING:
{scratchpad}

{prompts.get("agent_instructions", "")}
{prompts.get("interactive_instruction", "") if self.interactive else ""}
"""
        response = self.llm.generate(prompt)
        
        # Logging
        if "Thought:" in response:
             thought = response.split("Thought:")[-1].split("Action:")[0].split("Final Answer:")[0].strip()
             self._log("Thought", thought, "thought")
        if "Action:" in response:
             action = response.split("Action:")[-1].strip()
             self._log("Action Proposed", action, "action")
        if "Final Answer:" in response:
             ans = response.split("Final Answer:")[-1].strip()
             self._log("Final Answer", ans, "answer")
             
             # Appendix Logic (Simple version)
             if facts:
                 appendix = "\n\n## Appendix\n" + "\n".join([f"- {str(f)[:100]}..." for f in facts])
                 self._log("Appendix", appendix, "appendix")
                 response += appendix

        return {
            "messages": [response],
            "scratchpad": scratchpad + f"\nAgent: {response}\n",
            "step_count": state.get("step_count", 0) + 1
        }

    def execution_node(self, state: AgentState) -> Dict:
        """The Tool Executor."""
        last_message = state["messages"][-1]
        scratchpad = state.get("scratchpad", "")
        
        # Regex to capture Action: tool("arg")
        # Support both 'tool("arg")' and "tool('arg')"
        match = re.search(r'Action:\s*(\w+)\s*\((.*?)\)', last_message, re.DOTALL)
        if not match:
             return {"scratchpad": scratchpad + "\nSystem: Error: No valid Action format found.\n"}
             
        tool_name = match.group(1)
        tool_arg = match.group(2).strip('"\' ')
        
        if tool_name not in self.tool_map:
             return {"scratchpad": scratchpad + f"\nSystem: Error: Tool '{tool_name}' not found.\n"}
             
        try:
            result = self.tool_map[tool_name](tool_arg)
            self._log("Observation", str(result), "observation")
            
            facts = state.get("facts", [])
            facts.append(result) # Add to facts
            
            return {
                "scratchpad": scratchpad + f"\nObservation: {result}\n",
                "facts": facts
            }
        except Exception as e:
            return {"scratchpad": scratchpad + f"\nSystem: Error executing tool: {e}\n"}

    def decide_next_step(self, state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if "Final Answer:" in last_msg:
            return "end"
        if "Action:" in last_msg:
            return "act"
        return "end" # Fallback

    def run(self, query: str):
        init_state = {
            "question": query,
            "scratchpad": "",
            "messages": [],
            "facts": [],
            "step_count": 0,
            "plan": [],
            "final_answer": ""
        }
        res = self.app.invoke(init_state)
        return res["messages"][-1]
