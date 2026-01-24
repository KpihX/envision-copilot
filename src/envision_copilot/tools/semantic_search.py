from typing import Dict, Any, Union, List
import logging
import json
from code_rag.vector_engines import get_retriever
from envision_copilot.utils.utils import smart_truncate

from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box

from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box

class SemanticSearch:
    """
    Wrapper around Code RAG Retriever.
    Acts as both Runner (stateless) and Result Container (stateful).
    """
    def __init__(self, config: Dict[str, Any] = None, result: Any = None):
        self.config = config or {}
        self.result = result # Holds the result data if this instance is a result container
        try:
             # Only init retriever if we are a runner (no result yet) or if we want to reuse?
             # Actually, we can just init it always or lazy load.
             # Ideally, result containers don't need the heavyweight retriever.
             if result is None:
                 self.retriever = get_retriever()
             else:
                 self.retriever = None 
        except Exception as e:
            logging.error(f"Failed to initialize SemanticSearch: {e}")
            self.retriever = None

    def search(self, query: str, top_k: int = 5, keywords: List[str] = None, terms: List[str] = None, horizon: bool = False, **kwargs) -> 'SemanticSearch':
        """
        Executes semantic search with Graph-Aware chunks, Oriented Reranking, and Horizon scan.
        Returns a NEW instance of SemanticSearch containing the result.
        """
        if not self.retriever:
            return SemanticSearch(self.config, result="Error: RAG Retriever not initialized (Run 'uv run index --build' first).")

        try:
            # Query the RAG
            response = self.retriever.query(
                query, 
                top_k=top_k, 
                targets=terms, 
                keywords=keywords,
                horizon=horizon
            )
            
            # Extract only relevant info for the agent to save tokens
            summary = []
            results = response.get("results", [])
            
            for res in results:
                summary.append({
                    "source_id": res.get("source_id"),
                    "source": res.get("source"),
                    "lines": res.get("lines"),
                    "score": res.get("score"),
                    "content": res.get("content"),
                    "context": res.get("context")
                })
                
            # Enrich stats
            stats = response.get("stats", {})
            stats["displayed_count"] = len(summary)

            data = {
                "query": query,
                "stats": stats,
                "results": summary,
                "horizon": response.get("horizon", [])
            }
            return SemanticSearch(self.config, result=data)

        except Exception as e:
            return SemanticSearch(self.config, result=f"Error during semantic search: {e}")

    def __str__(self) -> str:
        """Format result for LLM context."""
        if self.result is None:
            return "SemanticSearch Tool (Ready)"
            
        if isinstance(self.result, str):
            return self.result
        
        limit = self.config.get("presentation", {}).get("max_output_lines", 100)
        
        buffer = [f"\n### 🔍 Semantic Search Results:"]
        buffer.append(f"Query: {self.result.get('query', 'N/A')}")
        buffer.append(f"Found: {self.result.get('stats', {}).get('displayed_count', 0)} results")
        
        for i, res in enumerate(self.result.get("results", []), 1):  
            buffer.append(f"\n**Result {i}** (score: {res.get('score', 0):.2f})")
            buffer.append(f"  Source: {res.get('source', 'N/A')} (lines {res.get('lines', 'N/A')})")
            content = smart_truncate(res.get('content', ''), max_lines=limit)
            buffer.append(f"  Content: {content}")
        
        return "\n".join(buffer)

    def print(self) -> Panel:
        """Format result for Rich UI."""
        if self.result is None:
             from rich.text import Text
             return Panel(Text("SemanticSearch Tool Ready", style="green"), title="🔍 Semantic Search", border_style="green")
             
        if isinstance(self.result, str):
            from rich.text import Text
            return Panel(Text(self.result, style="red"), title="🔍 Semantic Error", border_style="red")
            
        limit = self.config.get("presentation", {}).get("max_output_lines", 100)
        
        stats = self.result.get("stats", {})
        query = self.result.get("query", "N/A")
        
        # Main Table
        table = Table(title=f"Query: {query} ({stats.get('displayed_count')} results)", show_header=True, box=box.ROUNDED)
        table.add_column("Score", style="magenta", width=6)
        table.add_column("Location", style="cyan")
        table.add_column("Match Preview", style="white")
        
        for res in self.result.get("results", []):
            score = f"{res.get('score', 0):.2f}"
            loc = f"{res.get('source')}:{res.get('lines')}"
            content = smart_truncate(res.get('content', '').replace("\n", " "), max_lines=200) # One line per row ideal
            table.add_row(score, loc, content)
            
        # Add Horizon if present
        if self.result.get("horizon"):
            horizon_text = ", ".join(self.result.get("horizon"))
            # if len(self.result.get("horizon")) > 5: horizon_text += "..." # Removed limit
            table.caption = f"Horizon: {horizon_text}"
            
        return Panel(table, title="🔍 Semantic Search", border_style="magenta")
    
    def to_dict(self):
         return self.result if isinstance(self.result, dict) else {"error": self.result}
