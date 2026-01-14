from code_rag.retriever import Retriever
from ..utils import ConfigLoader

class SemanticTools:
    def __init__(self, config_path: str = "config.yaml"):
        # We need to initialize the retriever with ITS OWN config
        # But Copilot config points to it
        self.config = ConfigLoader.load_config(config_path)
        rag_config_path = self.config.get("paths", {}).get("vector_config", "src/code_rag/config.yaml")
        
        self.retriever = Retriever(config_path=rag_config_path)

    def search_code(self, query: str) -> str:
        """Semantically searches the codebase."""
        results = self.retriever.retrieve(query, k=10, rerank_top_k=3)
        
        output = []
        for res in results:
            output.append(f"--- File: {res['source_id']} ---\n{res['text']}\n")
            
        return "\n".join(output) if output else "No results found."
