"""
Tool Definitions for LLM Function Calling (MCP-Ready)
=====================================================

Central registry of tool definitions for LLM prompts.
Each tool has: name, description, parameters (JSON Schema), documentation.

Structure follows MCP (Model Context Protocol) conventions.

DOMAINS:
- Navigation: Browse folder hierarchy (scripts/data trees)
- Content: Read file contents
- Search: Find code (semantic RAG or regex grep)

EXECUTION ORDER:
Scripts and folders have `execution_order` extracted from name prefix:
- "01 - Catalog" → execution_order: 1
- "3. Inspectors" → execution_order: 3
This indicates the workflow execution sequence.
To find the next script, look for execution_order + 1.
"""

from typing import List, Dict, Any


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # NAVIGATION DOMAIN: Graph exploration and folder hierarchy
    # -------------------------------------------------------------------------
    {
        "name": "graph",
        "description": "Navigate and explore the Envision dependency graph (folders, scripts, data, relationships)",
        "documentation": """
## Graph Tool

Explore the Envision codebase structure through the dependency graph.

### ⚠️ RECOMMENDED FIRST STEP

**Always start your exploration with a `tree` action on the `scripts` domain** to understand the project structure before using other tools. This provides essential context about the workflow organization.

```json
{"action": "tree", "domain": "scripts"}
```

### Actions

| Action | Purpose | Key Parameters |
|--------|---------|----------------|
| `tree` | Browse folder hierarchy | `path`, `domain`, `max_depth` |
| `node` | Get single node details | `node_id` |
| `neighbors` | Find connected nodes | `node_id`, `direction`, `relation_type` |
| `edges` | List edges by type | `relation_type` |
| `search` | Fuzzy search by name/path | `query`, `node_types` |

### Domains (for tree action)

- **scripts**: Envision script files (.nvn) organized in workflow folders
- **data**: Data files (.ion, .csv) organized by usage

### Execution Order

Scripts/folders have `execution_order` from name prefix (e.g., "01 - Catalog" → 1).
Use this to navigate workflow sequence: to find next script, look for neighbor (sibling) with execution_order + 1.

### Node Types

- `script`: Envision script (.nvn)
- `data_file`: Data file (.ion, .csv)  
- `table`: Table defined in script
- `function`: Function defined in script
- `folder`: Directory container

### Relation Types (for neighbors/edges)

- `imports`: Script imports module
- `reads`: Script reads data file
- `writes`: Script writes data file
- `defines`: Script defines table/function
- `contains`: Folder contains item
- `sibling`: Same-folder relationship

### Direction Logic for `neighbors` (CRITICAL)

**Think from the perspective of `node_id`:**
- `incoming` = edges pointing **TO** node_id = "Who/what targets ME?"
- `outgoing` = edges going **FROM** node_id = "Who/what do I target?"
- `siblings` = **NO DIRECTION needed** (bidirectional, same-folder peers)

**Common patterns:**
| I want to find... | node_id | direction | relation_type |
|-------------------|---------|-----------|---------------|
| Scripts that READ a file | the file | `incoming` | `reads` |
| Files that a script READS | the script | `outgoing` | `reads` |
| Scripts that WRITE a file | the file | `incoming` | `writes` |
| Files that a script WRITES | the script | `outgoing` | `writes` |
| Scripts that IMPORT a module | the module | `incoming` | `imports` |
| Modules that a script IMPORTS | the script | `outgoing` | `imports` |
| Scripts in same folder | the script | `siblings` | - |

### Examples

```json
// FIRST: Get project overview
{"action": "tree", "domain": "scripts", "max_depth": 2}

// Get script details by ID
{"action": "node", "node_id": "68006"}

// Find all scripts that READ Items.ion → edges come INTO the file
{"action": "neighbors", "node_id": "/Clean/Items.ion", "direction": "incoming", "relation_type": "reads"}

// Find what files a script READS → edges go OUT FROM the script
{"action": "neighbors", "node_id": "68006", "direction": "outgoing", "relation_type": "reads"}

// Find scripts that IMPORT the Functions module → edges come INTO the module
{"action": "neighbors", "node_id": "67992", "direction": "incoming", "relation_type": "imports"}

// List all reads relationships
{"action": "edges", "relation_type": "reads"}

// Search for scripts containing "Forecast" in name/path
{"action": "search", "query": "Forecast", "node_types": ["script"]}
```
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["tree", "node", "neighbors", "edges", "search"],
                    "description": "Action to perform"
                },
                "path": {
                    "type": "string",
                    "description": "[tree] Folder path (default: '/' for root)"
                },
                "domain": {
                    "type": "string",
                    "enum": ["scripts", "data"],
                    "description": "[tree] Folder tree: 'scripts' or 'data'"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "[tree] Max depth to traverse (default: 1, omit or null for unlimited)"
                },
                "node_id": {
                    "type": "string",
                    "description": "[node, neighbors] Node ID to query (script ID like '68006' or path like '/Clean/Items.ion')"
                },
                "direction": {
                    "type": "string",
                    "enum": ["incoming", "outgoing", "all", "siblings"],
                    "description": "[neighbors] FROM node_id's perspective: 'incoming' = who targets me?, 'outgoing' = what do I target?"
                },
                "relation_type": {
                    "type": "string",
                    "enum": ["reads", "writes", "imports", "defines", "contains", "sibling"],
                    "description": "[neighbors, edges] Filter by relation type"
                },
                "query": {
                    "type": "string",
                    "description": "[search] Search query (substring match on name/path)"
                },
                "node_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "[search] Filter by node types: script, data_file, table, function, folder"
                }
            },
            "required": ["action"]
        }
    },
    
    # -------------------------------------------------------------------------
    # CONTENT DOMAIN: Read file contents
    # -------------------------------------------------------------------------
    {
        "name": "reader",
        "description": "Read Envision script or function content by ID with optional line range",
        "documentation": """
## Reader Tool

Read script or function contents with line range selection.

**IMPORTANT**: This tool reads SCRIPTS, not folders. Use numeric IDs from `graph.tree` output.

### Parameters

- `node_id` (required): **Use the numeric ID** from graph.tree (e.g., "68006"), NOT folder paths
- `start_line`: First line to read (1-indexed, default: 1)
- `end_line`: Last line (inclusive, default: end of file)

### Node ID Formats (CRITICAL)

| Format | Example | Works? | Description |
|--------|---------|--------|-------------|
| Script ID | `68006` | ✅ YES | Use this! Get IDs from graph.tree |
| Function ID | `67992::func::StockEvol` | ✅ YES | Function within a script |
| Table ID | `68010::table::Orders` | ✅ YES | Table definition |
| Folder path | `/1. utilities/2. preprocess` | ❌ NO! | Folders have no content - use graph.tree instead |

### Workflow

1. First: `graph.tree` to explore folders → get script IDs like `68000`, `68001`
2. Then: `reader` with the numeric ID → `{"node_id": "68000"}`

### Response

Returns content with metadata:
- `stats`: Line counts and range info
- `node`: Node info including `execution_order`
- `content`: The actual code

### Examples

```json
// Read first 50 lines of a script
{"node_id": "68006", "start_line": 1, "end_line": 50}

// Read entire script
{"node_id": "68006"}

// Read a specific function
{"node_id": "67992::func::StockEvol"}

// Read specific section
{"node_id": "68006", "start_line": 100, "end_line": 150}
```
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "NUMERIC script ID from graph.tree (e.g., '68006'), or function ID ('67992::func::StockEvol'). NOT folder paths!"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line (1-indexed, default: 1)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line (inclusive, default: end of file)"
                }
            },
            "required": ["node_id"]
        }
    },
    
    # -------------------------------------------------------------------------
    # SEARCH DOMAIN: RAG semantic search
    # -------------------------------------------------------------------------
    {
        "name": "rag",
        "description": "Semantic search over Envision codebase using embeddings (for conceptual queries)",
        "documentation": """
## RAG Tool (Semantic Search)

Find code by meaning using vector embeddings with reranking boost.

### When to Use

- Conceptual questions: "How is safety stock calculated?"
- Finding related code: "demand forecasting logic"
- Understanding patterns: "error handling approach"

### When NOT to Use

- Exact identifiers: use `grep` instead
- File paths: use `graph.search` instead  
- Structural queries: use `graph` instead

### Parameters

- `query` (required): Natural language description of what you're looking for
- `top_k`: Number of results (default: 5)
- `keywords`: List of Envision DSL keywords to boost results containing them
  - Function definitions: `def`, `autodiff`
  - Loop constructs: `each`, `for`, `while`, `where`
  - Table operations: `read`, `write`, `keep`, `by`
  - Aggregations: `sum`, `max`, `min`, `avg`, `first`, `last`
- `terms`: List of domain-specific identifiers to boost in reranking
  - Table names: `Items`, `Orders`, `Catalog`, `Suppliers`
  - File patterns: `Items.ion`, `Orders.csv`
  - Concepts: `SafetyStock`, `LeadTime`, `Forecast`
- `horizon`: If true, also returns "horizon" - nearby related chunks in the same file

### Reranker Boosting

The reranker scores chunks higher if they contain your `keywords` (DSL syntax) 
or `terms` (domain identifiers). Use these to guide results:

```json
// Find function definitions about stock
{"query": "stock calculation", "keywords": ["def"], "terms": ["SafetyStock"]}

// Find table read operations for Items
{"query": "reading item data", "keywords": ["read"], "terms": ["Items.ion"]}
```

### Examples

```json
// Basic semantic search
{"query": "safety stock calculation logic", "top_k": 5}

// With keyword boost for function definitions
{"query": "demand forecast", "keywords": ["def", "autodiff"]}

// With term boost for specific tables
{"query": "purchase orders generation", "terms": ["Orders", "PO"]}

// Full search with horizon context
{"query": "how is lead time computed", "keywords": ["def"], "terms": ["LeadTime"], "horizon": true}
```
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query (English preferred)"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (default: 5)"
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Envision DSL keywords to boost: def, each, read, write, sum, etc."
                },
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Domain identifiers to boost: Items, Orders, SafetyStock, etc."
                },
                "horizon": {
                    "type": "boolean",
                    "description": "If true, include nearby chunks from same file (context window)"
                }
            },
            "required": ["query"]
        }
    },
    
    # -------------------------------------------------------------------------
    # SEARCH DOMAIN: Grep pattern search
    # -------------------------------------------------------------------------
    {
        "name": "grep",
        "description": "Regex pattern search in script contents (for exact text matching)",
        "documentation": """
## Grep Tool (Pattern Search)

Find exact text patterns using regex.

### When to Use

- Exact identifiers: table names, variable names
- Specific keywords: function calls, syntax
- Counting occurrences

### When NOT to Use

- Conceptual queries: use `rag` instead
- Structural queries: use `graph` instead

### Parameters

- `pattern` (required): Regex pattern
- `node_types`: Filter by node types (default: scripts only)

### Examples

```json
{"pattern": "Items\\.Stock"}
{"pattern": "def CalculateSafetyStock"}
{"pattern": "read.*\\.ion"}
```
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search"
                },
                "node_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by node types (default: ['script'])"
                }
            },
            "required": ["pattern"]
        }
    }
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return all tool definitions for LLM prompts."""
    return TOOLS


def get_tool_by_name(name: str) -> Dict[str, Any] | None:
    """Get a specific tool definition by name."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool
    return None


def get_tools_summary() -> str:
    """Generate a compact summary of available tools."""
    lines = ["Available Tools:"]
    for tool in TOOLS:
        lines.append(f"  - {tool['name']}: {tool['description']}")
    return "\n".join(lines)
