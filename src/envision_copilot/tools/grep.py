from typing import Dict, Any, Union, List
import logging
from envision_preprocess.api import EnvisionGraphAPI

class GrepTools:
    """
    Wrapper around EnvisionGraphAPI to provide Grep/Search capabilities.
    """
    def __init__(self, config: Dict[str, Any] = None):
        try:
            self.api = EnvisionGraphAPI()
        except Exception as e:
            logging.error(f"Failed to initialize GrepTools: {e}")
            self.api = None

    def grep_search(self, patterns: List[str]) -> Union[Dict[str, Any], str]:
        """
        Scan all nodes for occurrences of specific patterns (keywords, terms, paths).
        Returns rich statistics to help the Agent decide between RAG or Direct Read.
        """
        if not self.api:
             return "Error: Graph API not initialized."

        try:
             # Use the new multi-pattern grep_search from API
             # API returns {"patterns": { "p": { "total": N, "files": [...] } }}
             result = self.api.grep_search(patterns)
             return result

        except Exception as e:
             return f"Error executing grep_search: {e}"

        except Exception as e:
            return f"Error executing grep_search: {e}"
