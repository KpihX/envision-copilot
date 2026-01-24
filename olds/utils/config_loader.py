import logging
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
        """
        Loads configuration, handling both relative and project-root paths.
        """
        path = Path(config_path)
        if not path.exists():
             # Try relative to this file's parent (src/envision_copilot/utils/../config.yaml)
             path = Path(__file__).parent.parent / "config.yaml"
        
        if not path.exists():
            # Try project root heuristic
             path = Path(__file__).parent.parent.parent.parent / "src/envision_copilot/config.yaml"

        if not path.exists():
            # Fallback to local if running from src/envision_copilot
            path = Path("src/envision_copilot/config.yaml")

        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path} or resolved paths.")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {}
