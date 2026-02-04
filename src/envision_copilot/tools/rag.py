"""
RAG Tool - Semantic Code Search
===============================

Provides semantic search over Envision codebase using RAG retrieval.
Uses embedding-based similarity to find relevant code chunks.

This tool is independent from the graph structure and uses a
separate vector index for semantic matching.
"""

from typing import Optional, List, Dict, Any

from .base import BaseTool, ToolResult


class RagTool(BaseTool):
    """
    Tool for semantic code search using RAG retrieval.
    
    Uses vector embeddings to find code chunks semantically
    similar to the query. Best for:
    - Finding code by description ("calculate demand forecast")
    - Understanding concepts across files
    - Discovering related code patterns
    
    For exact text matching, use the grep tool instead.
    """
    
    name = "rag"
    description = "Semantic search over Envision codebase using embeddings"
    
    def __init__(self, api=None, config=None, **kwargs):
        """
        Initialize RAG tool with lazy retriever loading.
        
        Args:
            api: Graph API (not used directly but kept for consistency)
            config: Tool configuration
        """
        super().__init__(api=api, config=config, **kwargs)
        self._retriever = None  # Lazy loaded
    
    @property
    def retriever(self):
        """Lazy load retriever on first access."""
        if self._retriever is None:
            try:
                from code_rag.vector_engines import get_retriever
                self._retriever = get_retriever()
            except Exception as e:
                # Will be handled in execute()
                pass
        return self._retriever
    
    def execute(
        self,
        query: str,
        top_k: int = 5,
        keywords: Optional[List[str]] = None,
        terms: Optional[List[str]] = None,
        horizon: bool = False,
        **_
    ) -> ToolResult:
        """
        Perform semantic search with optional reranker boosting.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            keywords: Envision DSL keywords to boost (def, each, read, etc.)
            terms: Domain identifiers to boost (Items, SafetyStock, etc.)
            horizon: If true, include nearby chunks from same file
            
        Returns:
            ToolResult with matching code chunks
        """
        if not self.retriever:
            return self._error("RAG retriever not initialized (run 'uv run index --build' first)")
        
        if not query:
            return self._error("query is required")
        
        try:
            # Query the RAG with optional boost parameters
            response = self.retriever.query(
                query, 
                top_k=top_k,
                keywords=keywords,
                targets=terms,  # API uses 'targets' for terms
                horizon=horizon
            )
            
            # Format results
            formatted = self._format_results(response, query)
            
            return self._success(
                formatted, 
                top_k=top_k, 
                query=query,
                keywords=keywords,
                terms=terms,
                horizon=horizon
            )
            
        except Exception as e:
            return self._error(f"Search failed: {str(e)}")
    
    def _format_results(self, response: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Format retriever response for output.
        
        Args:
            response: Raw retriever response (dict with results, stats, horizon)
            query: Original query
            
        Returns:
            Formatted result dictionary
        """
        results = response.get("results", [])
        chunks = []
        
        for res in results:
            chunk = {
                "source_id": res.get("source_id"),
                "source": res.get("source"),
                "lines": res.get("lines"),
                "score": res.get("score"),
                "content": res.get("content"),
                "context": res.get("context")
            }
            chunks.append(chunk)
        
        # Build output
        stats = response.get("stats", {})
        stats["displayed_count"] = len(chunks)
        
        output = {
            "query": query,
            "stats": stats,
            "results": chunks
        }
        
        # Add horizon if present
        if response.get("horizon"):
            output["horizon"] = response["horizon"]
        
        return output
