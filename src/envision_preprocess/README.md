# Envision Preprocess

**The "Eyes" of the System.**

This package is the core parsing engine for the Envision DSL. It transforms a collection of raw `.nvn` scripts into a semantic **Dependency Network**, enabling advanced analysis, RAG, and visualization.

It goes beyond simple regex matching by implementing a complete **Resolution Engine** that understands how Envision scripts interact with data, defines symbols, and documents intent.

---

## 🏗️ Architecture & Philosophy

The system parses two parallel layers of information:
1.  **The Structural Layer (The Code)**: Dependencies, Variables, Functions.
2.  **The Semantic Layer (The Intent)**: Documentation, Business Rules, User Instructions.

### 1. Symbol Extraction ("The Mini-Map")
Code is not just text. It's a graph of symbols. We extract high-level symbols to give LLMs a "Mini-Map" of the codebase before they dive into the details.
*   **Functions**: `StockEvol`, `PurchaseQty`.
*   **Variables**: `Catalog.Sku`, `Items.Sold`.
*   **Constants**: `DcUnit`, `ColorB`.

### 2. Documentation Segregation
Envision scripts are hybrid: they contain Math, Business Logic, and UI code. To avoid noise, we segregate comments by **audience** using strict syntactic markers:

| Section       | Marker | Audience      | Purpose                                                              | Example                                        |
| :------------ | :----- | :------------ | :------------------------------------------------------------------- | :--------------------------------------------- |
| **Structure** | `///`  | **Developer** | Technical Table of Content. The definition of the script's skeleton. | `/// Stock evolution function`                 |
| **Business**  | `//'`  | **Scientist** | Business Logic & Economic Intent. "Why" we apply this rule.          | `//' Reduce stock by 20% for perishable items` |
| **User**      | `"""`  | **End-User**  | Dashboard UI Documentation (Markdown). What the client sees.         | `""" ## Sales Overview """`                    |
| **Memos**     | `////` | **Author**    | Private notes, TODOs, scratchpad. Ignored in production.             | `//// Fix this bug later`                      |

---

## 📊 Data Model (Node Structure)

Each node in the graph is a rich JSON object containing both the raw code and the extracted intelligence.

### Node Types
| Type       | Description                     | Naming ID                                |
| ---------- | ------------------------------- | ---------------------------------------- |
| `script`   | Envision Script file (`.nvn`).  | Script ID (e.g., `82278`).               |
| `file`     | External Data (`.ion`, `.csv`). | Logical Path (e.g., `/Clean/Stock.ion`). |
| `table`    | Internal Table definition.      | `script_id::table::TableName`            |
| `function` | Internal Function definition.   | `script_id::func::FunctionName`          |
| `var`      | Constant or Global Variable.    | `script_id::var::VarName`                |

### Node JSON Structure
```json
{
  "id": "67992",
  "type": "script",
  "content": "... raw source code ...",
  "metadata": {
    "docs": {
      "structure": ["Stock evolution function", ...],
      "business": ["We ignore returns..."],
      "user": ["## Documentation..."],
      "memos": []
    },
    "symbols": {
      "functions": {"StockEvol": 1, "GrowthMap": 1},
      "variables": {"Catalog.Sku": 29},
      "tables": {"Catalog": 1}
    }
  }
}
```

---

## 🧠 Resolution Engine

Envision scripts use dynamic paths and variables. The engine resolves them to build a concrete graph.

### 1. Variable Cascades
Resolves path variables like `read "\{root}/Sales.ion"` by tracking `const root = "..."`.
*   **Visibility**: `uv run network --cascades`

### 2. Glob Resolution
Resolves wildcards like `read "/Input/*.csv"` finding all matching files on disk and creating explicit edges.
*   **Visibility**: `uv run network --globs`

---

## 🚀 CLI Usage

### Build & Stats
```bash
# Build the graph (Scans all scripts & extracts symbols)
uv run network --build

# General Statistics
uv run network --stats
```

### Inspection (The "X-Ray")
Inspect a specific script to see its Metadata, Symbols, and Documentation structure.
```bash
# Inspect Script 67992
uv run network -q 67992

# Inspect a Table
uv run network -q "/Clean/Catalog.ion"
```

### Search
```bash
# Find scripts referencing "Stock"
uv run network -q "Stock" -f
```

---

## 📦 Python API
The `EnvisionGraphAPI` exposes all data programmatically.

```python
from envision_preprocess.api import EnvisionGraphAPI
api = EnvisionGraphAPI()
node = api.get_node("67992")
print(node["metadata"]["symbols"])
```
