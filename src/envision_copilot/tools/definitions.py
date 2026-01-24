# =============================================================================
# Tool Definitions for LLM Function Calling and Dynamic Prompting
# Central Source of Truth for Tools
# =============================================================================

TOOLS = [
    {
        "name": "semantic_search",
        "description": "Searches the codebase for concepts, logic, or definitions using semantic search (RAG) with Graph-Aware chunking.",
        "documentation": """
    **Purpose**: Semantic code search with oriented reranking (RAG).
    
    **Index Details (Graph-Aware Chunking)**:
    Every chunk is enriched with its graph context. 
    Example: `[Script: SalesAnalysis] [Imports: GlobalParams] [Reads: Items.ion]`.
    Searching "Items.ion consumer" naturally finds scripts that read it.
    
    **Parameters**:
    - `query`: Natural language question (MUST BE IN ENGLISH). Reformulate if needed.
    - `keywords`: Envision keywords to boost (def, show, when, by...). Use when looking for a TYPE of construct.
    - `terms`: Specific technical terms (variable names, function names...) to heavily boost. **VERIFY with grep_search first!**
    - `top_k`: Number of results (default 5).
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query (English preferred)."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (default 5).",
                    "default": 5
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Envision DSL keywords to boost (def, show, when, table...)."
                },
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific technical terms to heavily boost."
                }
            },
            "required": ["query"]
        },
        "examples": [
            {"query": "stock calculation logic", "keywords": ["def", "when"], "terms": ["StockCalc"]}
        ]
    },
    {
        "name": "structural_explorer",
        "description": "Explores the dependency graph (imports, reads, writes, defines).",
        "documentation": """
    **Purpose**: Explore the Dependency Graph (Nodes/Edges).
    
    **Node Types**:
    - `script`: Envision code file (ID e.g. "68010")
    - `file`: Data file .ion/.csv (Path e.g. "/Clean/Items.ion")
    - `table`: Symbol "script::table::Name"
    - `function`: Symbol "script::func::Name"
    
    **Edge Types**: `imports`, `reads`, `writes`, `defines`, `export`.
    
    **Actions**:
    - `stats`: Global graph statistics.
    - `neighbors`: Get connections. Returns rich metadata:
      - Docs: structure (///), business (//'), user ('""')
      - Symbols: defined variables/functions/tables
    
    **Directions**: `incoming` (used by), `outgoing` (uses), `all`.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["stats", "nodes", "edges", "neighbors"],
                    "description": "Action to perform."
                },
                "node_id": {
                    "type": "string",
                    "description": "Node ID for 'neighbors' action (script ID or file path)."
                },
                "type": {
                    "type": "string",
                    "description": "Filter by node/edge type."
                },
                "relation_type": {
                    "type": "string",
                    "description": "Filter neighbors by edge type."
                },
                "direction": {
                    "type": "string",
                    "enum": ["incoming", "outgoing", "all"],
                    "description": "Direction for neighbors (default: all)."
                }
            },
            "required": ["action"]
        },
        "examples": [
            {"action": "neighbors", "node_id": "68010"},
            {"action": "neighbors", "node_id": "/Clean/Items.ion", "direction": "incoming"}
        ]
    },
    {
        "name": "read_file",
        "description": "Reads a specific section of a script file (.nvn).",
        "documentation": """
    **Purpose**: Read code section from a script.
    
    **IMPORTANT**:
    - **NO LINE LIMIT** - request any range.
    - **ONLY for .nvn scripts** - NOT for .ion data files.
    - **Expand window**: If RAG says lines 40-50, read 20-80 for context.
    - **Metadata included** at top (symbols, imports).
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the script (ID, logical path, or physical path)."
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line (1-indexed). Use 1 if unsure.",
                    "default": 1
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line.",
                    "default": 100
                }
            },
            "required": ["path"]
        },
        "examples": [
            {"path": "/3. Inspectors/2 - Sales Analysis", "start_line": 1, "end_line": 100}
        ]
    },
    {
        "name": "grep_search",
        "description": "Regex search across all scripts to verify terms.",
        "documentation": """
    **Purpose**: Verify if a term exists before using in RAG (semantic_search terms).
    **Returns**: Scripts containing pattern + occurrence count.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search (case-insensitive)."
                }
            },
            "required": ["pattern"]
        },
        "examples": [
            {"pattern": "StockEvol"}
        ]
    }
]
