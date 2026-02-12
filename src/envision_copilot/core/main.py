from typing import Dict, Any, Optional
from envision_copilot.core.batch import BatchCopilot
from envision_copilot.core.live import LiveCopilot

class EnvisionCopilot:
    """
    Facade class that routes to the appropriate implementation:
    - BatchCopilot (One-shot, default)
    - LiveCopilot (Interactive, future)
    """
    def __init__(self, config_path: str = "config.yaml", verbose: bool = False, debug: bool = False, interactive: bool = False):
        self.interactive = interactive
        
        if self.interactive:
            self.copilot = LiveCopilot(config_path, verbose, debug)
        else:
            self.copilot = BatchCopilot(config_path, verbose, debug)

    def run(self, input_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Delegates execution to the underlying implementation.
        """
        if self.interactive:
            return self.copilot.run()
        else:
            if input_data is None:
                raise ValueError("Batch mode requires input_data (question).")
            return self.copilot.run(input_data)
