from typing import Dict, Any, Union, List
import logging
from code_rag.retriever import GraphRetriever

class SemanticTools:
    """
    Wrapper around Code RAG Retriever.
    """
    def __init__(self, config: Dict[str, Any] = None):
        try:
             self.retriever = GraphRetriever() # Load config internally
        except Exception as e:
            logging.error(f"Failed to initialize SemanticTools: {e}")
            self.retriever = None

    def search(self, query: str, top_k: int = 5, **kwargs) -> Union[List[Dict], str]:
        """
        Executes semantic search with Graph-Aware chunks.
        """
        if not self.retriever:
            return "Error: RAG Retriever not initialized (Run 'uv run index --build' first)."

        try:
            # Query the RAG
            response = self.retriever.query(query, top_k=top_k)
            
            # Extract only relevant info for the agent to save tokens
            # We want: Source, Content Wrapper, Score
            
            summary = []
            results = response.get("results", [])
            
            for res in results:
                summary.append({
                    "source_id": res.get("source_id"),
                    "source": res.get("source"), # Add Full Path
                    "lines": res.get("lines"),
                    "score": res.get("score"),
                    # Truncate content slightly if massive? 
                    # Chunks are usually 512 tokens (~2000 chars).
                    # Let's keep full content but maybe structure it.
                    "content": res.get("content"),
                    "context": res.get("context") # Graph Header
                })
                
            # Enrich stats
            stats = response.get("stats", {})
            stats["displayed_count"] = len(summary)

            return {
                "query": query,
                "stats": stats,
                "results": summary
            }

        except Exception as e:
            return f"Error during semantic search: {e}"
