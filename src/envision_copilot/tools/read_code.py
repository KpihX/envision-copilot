from envision_preprocess.api import EnvisionGraphAPI

class CodeReader:
    def __init__(self, config_path: str = "config.yaml"):
        # We rely on API which loads config internally
        try:
             self.api = EnvisionGraphAPI(config_path)
             # Preload to be fast
             self.api.get_stats() 
        except Exception as e:
             self.api = None
             print(f"Error init CodeReader: {e}")

    def read_code(self, script_id: str) -> str:
        """
        Returns the content of a script node, prepended with Context Header.
        """
        if not self.api:
            return "Error: Graph API not initialized."
            
        # 1. Resolve ID (if path provided)
        real_id = self.api.resolve_node_id(script_id) or script_id
        
        # 2. Get Node Content
        node = self.api.get_node(real_id)
        if not node:
             # Try soft search if exact fail?
             # For now, stick to exact or path resolution.
             return f"Script/Node '{script_id}' not found."
             
        content = node.get("content", "")
        if not content:
             return f"Node found but no content available (Type: {node.get('type')}). Use get_node() for structure inspection."
             
        # 3. Get Context Header (Docs, Symbols)
        header = self.api.get_file_context(real_id)
        
        return f"{header}\n\nCONTENT:\n{content}"
