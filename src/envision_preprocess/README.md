# Envision Preprocess

**The "Eyes" of the System.**

This package parses Envision DSL scripts (`.nvn`), resolves constants, and builds a comprehensive structural graph (Network).

## Key Features
- **Deep Parsing**: Extracts `read`, `write`, `export`, `import`, and `table` definitions.
- **Function Body Extraction**: Captures full code blocks for `process` and `def` nodes.
- **Comment Intelligence**: 
    - Strips logical comments (`//`, `/*`) to prevent false positives.
    - Extracts structured docs (`///` structure, `//'` business, `"""` user, `////` memos).
- **Constant Resolution**: Recursively resolves string constants to find real file paths.

## Usage

### 1. Build the Network
Scans specific directory (configured in `config.yaml`), builds the graph, and saves it to `data/network/`.

```bash
uv run network --build
# OR
uv run network -b
```

### 2. Inspect Statistics
View node types, counts, and examples.

```bash
uv run network --stats
# OR
uv run network -s

# Filter by Type (e.g., function, table, script)
uv run network -s -t function -n 3
```

### 3. Query the Network (`-q`)
Inspect a specific node (by ID or Logical Path) to see its relationships, metadata, and content.
```bash
# Query by File ID (e.g. 82278)
uv run network -q 82278

# Query by Logical Path (fuzzy matching supported)
uv run network -q "/Clean/Items.ion"

# Filter relationships (e.g., only show Exports or Reads)
uv run network -q 82278 -r export
uv run network -q "/Clean/Items.ion" --relation reads
```

## Data Model (Network)
- **Nodes**: 
    - `SCRIPT`: ID is the File ID (e.g., `68000`). `metadata.logical_path` stores the mapped path.
    - `FILE`: External files read/written/exported.
    - `TABLE`, `VAR`, `FUNCTION`: Internal definitions.
- **Edges**: `READS`, `WRITES`, `EXPORT`, `IMPORTS`, `DEFINES`.

## Configuration
- **Mapping**: Uses `mapping.txt` to resolve `import "/Logical/Path"` to correct File IDs.
- **Regex**: Advanced patterns for `export schema`, `export ... as`, `write ... as`.

Everything is saved to:
- `data/network/network.json`: The graph.
- `data/network/metadata.json`: Summary stats.
