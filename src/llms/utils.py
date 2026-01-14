import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
        """Loads YAML config relative to the package root or absolute path."""
        # Priority 1: Package default config
        package_root = Path(__file__).parent
        path = package_root / "config.yaml"
        
        # Priority 2: Override from user provided path (if exists)
        if Path(config_path).exists() and config_path != "config.yaml":
             path = Path(config_path)

        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
