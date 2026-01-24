import json
import re
from typing import Dict, Any, Optional, Callable
from rich.console import Console
from rich.panel import Panel

class BaseAgent:
    """
    Base Agent providing robust LLM interaction capabilities.
    Implements the "Guardrails" middleware for JSON validation and Retries.
    """
    def __init__(self, config: Dict[str, Any], llm: Any, console: Console, prompt_loader: Any = None, verbose: bool = False):
        self.config = config
        self.llm = llm
        self.console = console
        self.prompt_loader = prompt_loader
        self.verbose = verbose
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
            # 1. Generate
            try:
                raw_response = self.llm.generate(current_prompt)
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

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Robus JSON extraction (Regex -> Raw -> Flexible)."""
        # 1. Regex Markdown
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
            
        # 2. Raw
        try: return json.loads(text)
        except: pass
            
        # 3. Flexible (First { to last })
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except:
            pass
             
        return None

    def _log_error(self, title: str, content: str = ""):
        """Display error in Rich UI."""
        if content:
            self.console.print(Panel(content, title=f"[bold red]{title}[/bold red]", border_style="red"))
        else:
            self.console.print(f"[bold red]{title}[/bold red]")
