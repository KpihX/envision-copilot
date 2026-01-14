import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
        packages_root = Path(__file__).parent
        path = packages_root / "config.yaml"
        
        if Path(config_path).exists() and config_path != "config.yaml":
            path = Path(config_path)

        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
