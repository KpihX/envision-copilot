from typing import Dict, Any

class PromptLoader:
    def __init__(self, config: Dict[str, Any]):
        self.prompts = config.get("prompts", {})

    def get_system_prompt(self) -> str:
        """Combines system, philosophy, and context setup."""
        parts = [
            self.prompts.get("system", ""),
            self.prompts.get("philosophy", ""),
            self.prompts.get("context_setup", ""),
            self.prompts.get("instructions", ""),
            self.prompts.get("tools_usage", "")
        ]
        return "\n\n".join([p.strip() for p in parts if p])

    def get_instructions(self) -> str:
        return self.prompts.get("instructions", "")
