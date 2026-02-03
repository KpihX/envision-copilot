# =============================================================================
# Tool Definitions for LLM Function Calling and Dynamic Prompting
# Central Source of Truth for Tools
# =============================================================================

TOOLS = [
    {
        "name": "semantic_search",
        "description": "Searches for concepts/logic (RAG). Use ONLY for ambiguous/conceptual questions. NOT for file paths, deterministic questions.",
        "documentation": """
    **Purpose**: Semantic code search for concepts (e.g., "How is safety stock calculated?").
    
    **Usage (JSON Argument)**:
    ```json
    {
      "query": "how is safety stock calculated?",
      "keywords": ["def", "when"],  // Optional: Envision keywords to boost
      "terms": ["Stock", "Safety"], // Optional: Technical terms to boost
      "top_k": 5
    }
    ```
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
        "description": "PRIMARY tool for structural queries (Dependencies, reads, writes, imports, defines, File Paths).",
        "documentation": """
    **Purpose**: Explore the Dependency Graph (Nodes/Edges).
    
    **Node Types**: `script` (Envision files), `file` (Data .ion), `table (script::table::Name)`, `function (script::func::Name)`.
    **Edge Types**: `imports` (Modules), `reads` (Data), `writes` (Data), `defines`, `export`.
    
    **WHEN TO USE**: 
    - **Global Analysis**: "Which modules are used?" -> `action="edges", type="imports"`
    - **Global Analysis**: "Which files are read?" -> `action="edges", type="reads"`
    - **Type Listing**: "List all scripts" -> `action="nodes", type="script"`
    - **Dependency**: "Who reads Items.ion?" -> `action="neighbors", node_id="...", relation_type="reads"`
    
    **CRITICAL**: 
    - IF searching for "Modules" or "Imports", use `action="edges", type="imports"`. 
    - **DO NOT** use `nodes` type="script" to find imports. The graph ALREADY has them as edges.
    - **DO NOT** list all scripts to find imports. It is slow and unnecessary.
    
    **Usage (JSON Argument)**:
    ```json
    {
      "action": "neighbors",   // OR: "stats", "nodes", "edges", "search_node", "get_node"
      "node_id": "68010",      // Required for neighbors/get_node
      "direction": "incoming", // Optional (neighbors) incoming/outgoing/all
      "type": "script",        // Required for nodes (filter) ; to use just to check if a node exists having its ID or name
      "relation_type": "reads"/"writes"/"imports"/"defines"/"export" // Required for edges (general filter) or neighbors ; you can use it to find all edges of a specific type for example
    }
    ```
    
    **Actions Details**:
    - `nodes`: List ALL nodes of a specific `type` (e.g. "script").
    - `edges`: List ALL global relations of a specific `relation_type` (e.g. "imports").
    - `neighbors`: Find dependencies relative to a `node_id` (e.g., "incoming" + "reads" = Who reads me?)
    - `search_node`: **FUZZY STRING MATCH** on Name/Path. 
      - Use ONLY if you know part of the name (e.g. "Items").
    - `get_node`: Get details of a known Node ID.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["stats", "nodes", "edges", "neighbors", "search_node", "get_node"],
                    "description": "Action to perform."
                },
                "node_id": {
                    "type": "string",
                    "description": "Node ID (for 'neighbors', 'get_node')."
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for 'search_node'). Matches Name or Path."
                },
                "type": {
                    "type": "string",
                    "description": "Filter by node type (e.g. 'script', 'file')."
                },
                "relation_type": {
                    "type": "string",
                    "enum": ["reads", "writes", "imports", "defines", "export"],
                    "description": "Filter by edge type (e.g. 'imports')."
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
            {"action": "edges", "relation_type": "imports"},
            {"action": "nodes", "type": "script"},
            {"action": "neighbors", "node_id": "/Clean/Items.ion", "relation_type": "reads", "direction": "incoming"}
        ]
    },
    {
        "name": "read_file",
        "description": "Reads a specific section of a script file (.nvn) using its ID.",
        "documentation": """
    **Purpose**: Read code section from a script using its Graph ID.
    
    **Usage (JSON Argument)**:
    ```json
    {
      "script_id": "68010",   // REQUIRED. Found via structural_explorer.
      "start_line": 1,
      "end_line": 100         // Use -1 for end of file
    }
    ```
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "script_id": {
                    "type": "string",
                    "description": "Script ID (Found in structural_explorer results). e.g. '68010'."
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line (1-indexed). Use 1 if unsure.",
                    "default": 1
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line (Use -1 or large number for end).",
                    "default": 100
                }
            },
            "required": ["script_id"]
        },
        "examples": [
            {"script_id": "68010", "start_line": 1, "end_line": 50}
        ]
    },
    {
        "name": "grep_search",
        "description": "FALLBACK: Regex search across all scripts.",
        "documentation": """
    **Purpose**: Text/Pattern Search in files.
    
    **WHEN TO USE**:
    - **FALLBACK ONLY**: Use this ONLY if `structural_explorer` yielded NO RESULTS or if the graph lookup failed.
    - Useful for searching literal strings inside code content when graph metadata is insufficient.
    
    **Usage (JSON Argument)**:
    ```json
    {
      "pattern": "StockEvol.*"  // Regex pattern (case-insensitive)
    }
    ```
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
