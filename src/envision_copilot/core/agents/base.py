import json
import re
from typing import Dict, Any, Optional, Callable
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

class BaseAgent:
    """
    Base Agent providing robust LLM interaction capabilities.
    Implements the "Guardrails" middleware for JSON validation and Retries.
    """
    def __init__(self, config: Dict[str, Any], llm: Any, console: Console, prompt_loader: Any = None, verbose: bool = False, debug: bool = False):
        self.config = config
        self.llm = llm
        self.console = console
        self.prompt_loader = prompt_loader
        self.verbose = verbose
        self.debug = debug
        self.retry_limit = config.get("agent", {}).get("constraints", {}).get("retry_limit", 2)
        
        # Load Retry Prompt Template via Loader or Fallback
        if self.prompt_loader:
            self.retry_template = self.prompt_loader.get_guardrails_retry_template()
        else:
            prompts = config.get("prompts", {})
            generic = prompts.get("generic", {})
            self.retry_template = generic.get("guardrails_retry", "")

    def query_llm_robust(self, prompt: str, schema_validation: Optional[Callable[[Dict], bool]] = None) -> Optional[Dict]:
        """
        Queries the LLM with built-in retries and JSON validation.
        
        Args:
            prompt: The system prompt.
            schema_validation: Optional function to validate the JSON structure.
                               Should return True if valid, False otherwise.
        
        Returns:
            Dict: Validated JSON response.
            None: If retries are exhausted.
        """
        current_prompt = prompt
        
        for attempt in range(self.retry_limit + 1):
            
            # 1. Generate with Logs
            try:
                raw_response = self._generate_with_logs(current_prompt)
            except Exception as e:
                self._log_error(f"LLM Generation Error: {e}")
                return None

            # 2. Extract JSON
            json_data = self._extract_json(raw_response)
            
            # 3. Validate
            error_message = None
            if json_data is None:
                error_message = "Could not parse JSON from response."
            elif schema_validation and not schema_validation(json_data):
                error_message = "JSON failed schema validation."
            
            # 4. Success Path
            if not error_message:
                return json_data
            
            # 5. Failure Path (Guardrails Triggered)
            self._log_error(f"Guardrail Alert (Attempt {attempt+1}/{self.retry_limit + 1}): {error_message}", raw_response)
            
            if attempt < self.retry_limit:
                # Prepare Vigilance Prompt for Next Try
                vigilance_msg = self.retry_template.format(
                    error_message=error_message,
                    previous_output=raw_response
                )
                # Append to prompt (Simulation of conversation turn)
                current_prompt = f"{prompt}\n\n{vigilance_msg}"
            else:
                self._log_error("❌ Max Retries Exhausted. Stopping.")
                return None
                
        return None

    def _sanitize_json_string(self, text: str) -> str:
        """
        Attempts to fix common JSON format errors, specifically unescaped newlines within strings.
        """
        # Regex to find content inside double quotes and replace literal newlines with \n
        # This is a heuristic and might not cover all edge cases but handles standard LLM multi-line drift.
        def replace_newlines(match):
            return match.group(0).replace('\n', '\\n')
        
        # Apply replacement only inside string values
        return re.sub(r'(?<=: ")(.*?)(?=")', replace_newlines, text, flags=re.DOTALL)

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Robust JSON extraction (Strip Think Tags -> Regex -> Raw -> Flexible -> Sanitized)."""
        
        def try_parse(content):
            try: return json.loads(content)
            except: return None

        # 0. Strip <think>...</think> tags (Qwen3/QwQ reasoning blocks)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

        # 1. Regex Markdown
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        candidate = match.group(1) if match else text

        # 2. Direct Parse
        res = try_parse(candidate)
        if res: return res

        # 3. Soft Repair: Sanitize Newlines and Retry
        sanitized = self._sanitize_json_string(candidate)
        res = try_parse(sanitized)
        if res: return res
        
        # 4. Flexible (First { to last })
        try:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1:
                flexible_candidate = candidate[start:end+1]
                res = try_parse(flexible_candidate)
                if res: return res
                # Try sanitizing flexible candidate too
                res = try_parse(self._sanitize_json_string(flexible_candidate))
                if res: return res
        except:
            pass
             
        return None

    def _generate_with_logs(self, prompt: str, title_suffix: str = "") -> str:
        """
        Wraps LLM generation with Debug Logging (Raw Prompt/Response).
        Centralizes the logic so subclasses (like Synthesizer) can reuse it.
        """
        # Debug: Show Raw Prompt
        # if self.debug:
        #     self.console.print(Panel(
        #         Markdown(f"```text\n{prompt}\n```"), 
        #         title=f"🐞 Debug: Raw Prompt {title_suffix}", 
        #         border_style="dim yellow"
        #     ))

        # Generate
        response = self.llm.generate(prompt)

        # Debug: Show Raw Response
        if self.debug:
            self.console.print(Panel(
                Markdown(f"```text\n{response}\n```"), 
                title=f"🐞 Debug: Raw Response {title_suffix}", 
                border_style="dim yellow"
            ))
            
        return response

    def _log_error(self, title: str, response: str = None):
        """Standardized Error Logging for Agents."""
        content = title
        if response:
            content += f"\n\n**Response Context**:\n```text\n{response}\n```"
            
        self.console.print(Panel(
            Markdown(content),
            title="⚠️ Guardrail Alert",
            border_style="red"
        ))
