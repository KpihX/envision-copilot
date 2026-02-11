from typing import Dict, Any, List
import json
from envision_copilot.tools.definitions import TOOLS

class PromptLoader:
    """
    Loads and assembles prompts from the tiered config structure (Generic + Agents).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompts_root = config.get("prompts", {})
        self.generic = self.prompts_root.get("generic", {})
        self.agents = self.prompts_root.get("agents", {})

    def get_guardrails_retry_template(self) -> str:
        """Get the generic guardrails retry prompt."""
        return self.generic.get("guardrails_retry", "")

    def get_guardrails_tool_format_error_template(self) -> str:
        """Get the guardrails template for tool-call format errors."""
        return self.generic.get("guardrails_tool_format_error", "")

    def get_starter_prompt(self, user_input: str) -> str:
        """Assembles: Generic Identity + Generic Envision Doc + Starter Specific."""
        template = self.agents.get("starter", "")
        return template.format(
            identity=self.generic.get("identity", ""),
            envision_doc=self.generic.get("envision_doc", ""),
            user_input=user_input,
            guidelines=self.generic.get("guidelines", "")
        )

    def get_think_prompt(self, question, memory, history, last_results, current_depth: int = 0, last_thought_process: str = "") -> str:
        """Assembles: Identity + Envision Doc + Tools + Thinker (Objective/Workflow/Instr)."""
        # 1. Load config values
        constraints = self.config.get("agent", {}).get("constraints", {})
        presentation = self.config.get("presentation", {})
        
        max_depth = constraints.get("max_depth", 7)
        max_branches = constraints.get("max_branches", 2)
        max_lines = presentation.get("max_lines", 200)
        plan_history_depth = constraints.get("plan_history_depth", 1)
        
        # 2. Get Thinker Template (Specialized)
        thinker_template = self.agents.get("thinker", "")
        
        # 3. Generate Dynamic Tools Doc
        tools_doc = self._generate_tools_doc()
        
        # 4. Format Thinker specific parts using dictionary unpacking for clarity
        return thinker_template.format(
            identity=self.generic.get("identity", ""),
            envision_doc=self.generic.get("envision_doc", ""),
            tools_doc=tools_doc,
            question=question,
            current_depth=current_depth,
            max_depth=max_depth,
            max_branches=max_branches,
            max_lines=max_lines,
            plan_history_depth=plan_history_depth,
            history=history,
            memory=memory,
            last_results=last_results,
            last_thought_process=last_thought_process if last_thought_process else "(No previous reasoning yet)",
            guidelines=self.generic.get("guidelines", "")
        )

    def get_synthesizer_prompt(self, appendix: str, max_depth: int, user_language: str, stop_reason: str, original_question: str, reformulated_question: str, plan_thought: str, exploration_history: str = "") -> str:
        """Assembles: Identity + Envision Doc + Synthesizer (Unified)."""
        synthesizer_template = self.agents.get("synthesizer", "")
        
        return synthesizer_template.format(
            identity=self.generic.get("identity", ""),
            envision_doc=self.generic.get("envision_doc", ""),
            appendix=appendix,
            max_depth=max_depth,
            user_language=user_language,
            stop_reason=stop_reason,
            original_question=original_question,
            reformulated_question=reformulated_question,
            plan_thought=plan_thought,
            exploration_history=exploration_history or "(No exploration history)",
            guidelines=self.generic.get("guidelines", "")
        )

    def _generate_tools_doc(self) -> str:
        """Generates markdown documentation from TOOLS definitions."""
        buffer = ["### AVAILABLE TOOLS & DOCUMENTATION"]
        
        for tool in TOOLS:
            name = tool['name']
            doc = tool.get("documentation", "").strip()
            
            buffer.append(f"\n#### Tool: `{name}`")
            buffer.append(doc)
            
            if tool.get("examples"):
                buffer.append("\n**Examples:**")
                for ex in tool["examples"]:
                    buffer.append(f"- `{json.dumps(ex)}`")
        
        return "\n".join(buffer)
