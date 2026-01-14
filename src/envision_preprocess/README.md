# Envision Preprocess

**The "Eyes" of the System.**

This package is the core parsing engine for the Envision DSL. It transforms a collection of raw `.nvn` scripts into a semantic **Dependency Network**, enabling advanced analysis, RAG, and visualization.

It goes beyond simple regex matching by implementing a complete **Resolution Engine** that understands how Envision scripts interact with data and each other.

---

## 🏗️ Architecture & Philosophy

The Envision ecosystem distinguishes between **Code Structure** (Control Flow) and **Data Lineage** (Data Flow).

### 1. The Two Flows
*   **Control Flow (Code)**: Scripts importing other scripts (Modules, Libraries).
    *   Represented by `IMPORTS` edges.
    *   Connects `SCRIPT` -> `SCRIPT`.
*   **Data Flow (I/O)**: Scripts reading/writing/exporting data files (`.ion`, `.csv`).
    *   Represented by `READS`, `WRITES`, `EXPORT` edges.
    *   Connects `SCRIPT` -> `FILE`.

---

## 🧠 Advanced Resolution Engine

Envision scripts are dynamic. A simple "read" statement often involves variables, paths, and patterns. The preprocessor resolves these to concrete dependencies.

### 1. Variable Cascades (Const Propagation)
Scripts often define paths using variables:
```python
const root = "/Clean/Data"
read "\{root}/Sales.ion"
```
The engine recursively resolves these variables.
*   **Result**: The edge is recorded as pointing to `/Clean/Data/Sales.ion`.
*   **Visibility**: Use `uv run network --cascades` to see the resolution log.

### 2. Glob Resolution
Envision supports reading multiple files via wildcards:
```python
read "/Input/*.csv"
```
*   **Logic**: The engine detects the `*` pattern and scans the graph for **existing concrete files** that match this pattern.
*   **Graph**: Instead of linking to a phantom `*.csv` node, it creates explicit edges to every matching file (e.g., `Input/A.csv`, `Input/B.csv`).
*   **Visibility**: Use `uv run network --globs` to see matched patterns.

#### Unresolved Globs (Virtual Nodes)
Sometimes a pattern like `/Clean/Category_*.ion` matches **no existing files** (e.g. outputs of a future process).
*   **Behavior**: The system preserves the **Pattern Node** (containing `*`) in the graph.
*   **Meaning**: "This script depends on files matching this pattern, but none currently exist on disk."
*   **Display**: In CLI queries, these appear as `Category_*.ion`, signifying a virtual or future dependency.

### 3. Path Interpolation (Dynamic Paths)
Scripts often use data-driven paths:
```python
read "/Clean/Category_\{Category}.ion"
```
This implies a set of files dependent on the `Category` variable.
*   **Logic**: The engine treats `{...}` segments as **Wildcards** (`*`).
*   **Resolution**: It converts the path to `/Clean/Category_*.ion` and attempts to resolve it against existing files using the Glob engine.
*   **Result**:
    *   If files exist (`Category_1.ion`): Explicit links are created.
    *   If no files found (yet): A representative **Pattern Node** (`Category_*.ion`) is created to signify the dependency.

---

## 📊 Data Model (Network Structure)

The graph uses a rich schema with semantic types to represent the Envision ecosystem transparently.

### Node Types
| Type | Description | Naming Convention |
|------|-------------|-------------------|
| `script` | Envision Script file (`.nvn`). Contains code and logic. | Derived from **Logical Path** (e.g., `PathSchemas` from `/1. Utilities/PathSchemas`). ID is numeric (e.g., `82278`). |
| `file` | External Data File, Table, or Schema (e.g., `.ion`, `.csv`). | **Logical Path** (e.g., `/Clean/Stock.ion`). |
| `table` | Table defined *within* a script code. | `script_id::table::TableName` |
| `function`| Function defined in a script (`process`, `def`). | `script_id::func::FunctionName` |
| `var` | Constant or Global Variable. Content often contains the resolved value. | `script_id::var::VarName` |

### Edge Types
| Type | Description | Typical Direction |
|------|-------------|-------------------|
| `reads` | Script reads data from a File/Table. | `SCRIPT` -> `FILE` |
| `writes` | Script writes data to a File. | `SCRIPT` -> `FILE` |
| `export` | Script exports a Schema or File definition. | `SCRIPT` -> `FILE` |
| `imports` | Script imports another Script (Module). | `SCRIPT` -> `SCRIPT` |
| `defines` | Script defines a Symbol (Func, Const, Table). | `SCRIPT` -> `SYMBOL` |
| `uses` | (Reserved) General usage relationship. | `SCRIPT` -> `ANY` |

### Metadata Fields
The parser extracts rich metadata to support RAG and Understanding:
*   **Nodes**:
    *   `name`: Human-readable identifier (e.g., `Stock.ion`, `PathSchemas`).
    *   `path`: Full Logical Path.
    *   `docs`: Structured documentation extracted from comments:
        - `structure` (`///`): Architectural/Hierarchy notes.
        - `business` (`//'`): Business logic explanations.
        - `user` (`"""..."""`): User-facing Markdown blocks.
        - `memos` (`////`): Internal developer notes/todos.
    *   `qualifiers` (Functions): Keywords like `pure`, `process`.
*   **Edges**:
    *   `count`: Number of times this relationship appears in the source code.
    *   `occurrences`: List of raw strings found in the code (e.g. `read "/path/to/file"`), useful for auditing resolution.

---

## 🚀 CLI Usage

### Build & Stats
```bash
# Build the graph (Scans all scripts)
uv run network --build

# General Statistics (Counts, Globs, Cascades)
uv run network --stats

# Node Type Stats (with examples)
uv run network -s -tn script -n 5  # -tn = Type Node

# Edge Type Stats (with examples)
uv run network -s -te imports -n 5 # -te = Type Edge
```

### Query & Inspection (`-q`)
Inspect specific nodes with rich formatting (Tree view, Metadata, Content Snippets).

```bash
# Query by ID
uv run network -q 82278

# Query by Logical Path (Fuzzy match supported)
uv run network -q "/Clean/Items.ion"

# Filter Relationships
uv run network -q 82278 -r exports   # Show only Export edges
uv run network -q 82278 -r imports   # Show only Import edges
```

### Search (`-f`)
Broad search for nodes matching a string.
```bash
uv run network -q "Stock" -f
```

---

## 📦 Python API Reference

The `EnvisionGraphAPI` is designed to be highly flexible, offering **feature parity** with the CLI. It returns rich JSON structures suitable for automated reasoning (LLMs).

```python
from envision_preprocess.api import EnvisionGraphAPI
api = EnvisionGraphAPI()
```

### 1. `get_node(node_id: str)`
Retrieves full details of a node.

*   **Input**: `node_id` (e.g., `"82278"` or `"/Clean/Items.ion"` if resolved).
*   **Output**:
    ```json
    {
      "id": "82278",
      "type": "script",
      "name": "PathSchemas",
      "path": "/1. Utilities/PathSchemas",
      "content": "...",
      "metadata": { "docs": { ... } }
    }
    ```

### 2. `get_neighbors(node_id, direction="all", relation_type=None)`
Retrieves connected nodes. **Includes statistical summary**.

*   **Input**:
    *   `direction`: `"incoming"`, `"outgoing"`, or `"all"`.
    *   `relation_type`: Optional filter (e.g., `"reads"`, `"imports"`).
*   **Output**:
    ```json
    {
      "stats": {
        "incoming": {
          "total": 29,
          "unique_nodes": 28,
          "by_type": { "reads": 28, "writes": 1 }
        },
        "outgoing": {
          "total": 5,
          "unique_nodes": 3,
          "by_type": { "imports": 5 }
        },
        "filter_applied": null
      },
      "incoming": [
        {
          "source_id": "68010",
          "source_label": "Sales Analysis",
          "edge_type": "reads",
          "count": 1,
          "metadata": { "occurrences": ["read \"/Clean/Items.ion\""] }
        },
        ...
      ],
      "outgoing": []
    }
    ```

### 3. `search_nodes(query: str)`
Performs a fuzzy search across IDs, Names, and Logical Paths.

*   **Input**: `query` (e.g., `"Items"`).
*   **Output**: List of Node objects (same format as `get_node`).

### 4. `get_stats()`
Returns global graph statistics (same as CLI `--stats`).

*   **Output**:
    ```json
    {
      "node_count": 450,
      "edge_count": 1200,
      "resolutions": {
        "globs": [...],
        "placeholders": [...]
      },
      "nodes_by_type": { "script": 100, "file": 350 }
    }
    ```

## ⚙️ Configuration
The system is highly configurable via `config.yaml`:
```yaml
output:
  snippet_lines: 10 # Number of lines to show in CLI content snippets (Head + Tail)
```
