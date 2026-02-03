# Envision Preprocess

**The "Eyes" of the System.**

This package is the core parsing engine for the Envision DSL. It transforms a collection of raw `.nvn` scripts into a semantic **Hierarchical Dependency Graph**, enabling advanced analysis, RAG, and visualization.

The graph is hierarchical with **two separate folder domains**: 
- **SCRIPTS domain**: Contains script files organized by workflow
- **DATA domain**: Contains data files (inputs/outputs) organized by storage paths

---

## 📦 Package Structure

```
envision_preprocess/
├── api.py          # Public API (EnvisionGraphAPI)
├── builder.py      # Graph construction engine
├── typedefs.py     # Data types (NodeType, EdgeType, TreeDomain, Node, Edge, Network)
├── extractor.py    # Symbol extraction from code
├── utils.py        # Configuration loading utilities
├── network.py      # CLI interface (uv run network)
└── config.yaml     # Configuration file
```

---

## 🏗️ Architecture

### Two-Domain Tree Structure

The graph maintains **two completely separate folder hierarchies**:

```
SCRIPTS DOMAIN                    DATA DOMAIN
==============                    ===========
📁 /                              📁 /
├── 📁 /1. utilities              ├── 📁 /Input
│   ├── 📁 /1. populating         │   ├── 📁 /Catalog
│   │   └── 📜 67982              │   │   └── 📄 Catalog.csv
│   │   └── 📜 67983              │   └── 📁 /Orders
│   └── 📁 /Modules               ├── 📁 /Clean
│       └── 📜 68000              │   ├── 📄 Items.ion
├── 📁 /2. Data sanity            │   └── 📄 Orders.ion
└── 📁 /3. Inspectors             └── 📁 /Manual
```

### Cross-Domain Relationships

**Intra-domain edges** (within same domain):
- `contains`: folder → folder/file (hierarchy)
- `sibling`: file ↔ file (same parent folder)
- `imports`: script → script (SCRIPTS only)

**Cross-domain edges** (between domains):
- `reads`: script → data_file (SCRIPTS → DATA)
- `writes`: script → data_file (SCRIPTS → DATA)

**Internal edges**:
- `defines`: script → table/function

```
67982 (script) ──reads────> /Clean/Items.ion (data_file)
       ↑ sibling             (NO sibling - different domain)
67983 (script) ──writes───> /Clean/Output.ion (data_file)
```

### Data Flow

1. **Builder** scans `.nvn` scripts in `scripts/` directory
2. **Extractor** parses symbols, documentation, and dependencies
3. **Graph** is constructed with TWO folder hierarchies (scripts + data)
4. **Siblings** are established WITHIN each domain (never cross-domain)
5. **API** provides programmatic access with domain filtering
6. **CLI** offers exploration with `--domain` option

---

## 📊 Data Model

### Node Types

| Type        | Domain  | Description                      | ID Format                         | Example                          |
|-------------|---------|----------------------------------|------------------------------------|----------------------------------|
| `folder`    | Both    | Directory in hierarchy           | `folder::{domain}::/path`          | `folder::scripts::/1. utilities` |
| `script`    | Scripts | Envision script file (`.nvn`)    | Script ID (numeric)                | `67982`                          |
| `data_file` | Data    | External data file (`.ion`)      | Full path                          | `/Clean/Items.ion`               |
| `table`     | Scripts | Table definition in a script     | `{script}::table::{name}`          | `67982::table::Items`            |
| `function`  | Scripts | Function definition in a script  | `{script}::func::{name}`           | `67982::func::StockEvol`         |

### Edge Types

| Type       | Direction        | Domain Constraint              | Example                           |
|------------|------------------|--------------------------------|-----------------------------------|
| `contains` | folder → child   | Same domain                    | `folder::scripts::/` → `67982`    |
| `reads`    | script → file    | SCRIPTS → DATA (cross-domain)  | `67982` → `/Clean/Items.ion`      |
| `writes`   | script → file    | SCRIPTS → DATA (cross-domain)  | `67982` → `/Output/result.ion`    |
| `imports`  | script → script  | SCRIPTS only (intra-domain)    | `67983` → `67982`                 |
| `defines`  | script → symbol  | SCRIPTS only                   | `67982` → `67982::table::Items`   |
| `sibling`  | file ↔ file      | Same domain + same folder      | `67982` ↔ `67983`                 |

### Node JSON Structure

```json
{
  "id": "folder::scripts::/1. utilities",
  "type": "folder",
  "name": "1. utilities",
  "path": "/1. utilities",
  "metadata": {
    "domain": "scripts"
  }
}
```

```json
{
  "id": "67982",
  "type": "script",
  "name": "01 - Catalog Loader",
  "path": "/1. utilities/1. populating dataset/01 - Catalog Loader",
  "content": "/// Full source code...",
  "metadata": {
    "execution_order": 67982,
    "docs": {...},
    "symbols": {...}
  }
}
```

### Edge JSON Structure (sibling with domain)

```json
{
  "source": "67982",
  "target": "67983",
  "type": "sibling",
  "metadata": {
    "folder": "/1. utilities/1. populating dataset",
    "domain": "scripts"
  }
}
```

---

## 🔌 Python API

### Quick Start

```python
from envision_preprocess.api import EnvisionGraphAPI

api = EnvisionGraphAPI()

# Build the graph
result = api.build()
print(f"Built {result['stats']['node_count']} nodes")
```

### API Reference

The API is organized into 4 domains:

#### Navigation Domain

```python
# Get folder hierarchy for SCRIPTS domain (default)
tree = api.get_tree("/1. utilities", domain="scripts")

# Get folder hierarchy for DATA domain
data_tree = api.get_tree("/Clean", domain="data")

# Get BOTH trees at once
both = api.get_tree("/", domain="both")
# Returns: {"trees": {"scripts": {...}, "data": {...}}}

for child in tree["children"]:
    print(f"{child['name']} ({child['type']})")
```

#### Stats Domain

```python
# Get network statistics (includes per-domain stats)
stats = api.get_stats()
# Returns: {
#   "generated_at": "2024-01-15T10:30:00",
#   "node_count": 295, "edge_count": 773,
#   "nodes_by_type": {"script": 60, "folder": 49, ...},
#   "edges_by_type": {"reads": 270, "contains": 181, ...},
#   "domains": {
#     "scripts": {"folders": 15, "files": 60, "siblings": 59},
#     "data": {"folders": 34, "files": 73, "siblings": 32}
#   }
# }
```

#### Content Domain

```python
# Read node content
content = api.read("67982", start_line=1, end_line=50)
# Returns: {"stats": {...}, "node": {"id": ..., "content": "...", ...}}

# Search with grep
results = api.grep("table Items")
# Returns: {"stats": {...}, "results": [{"node_id": ..., "previews": [...]}]}
```

#### Graph Exploration Domain

```python
# Get node by ID
node = api.get_node("67982")
# Returns: {"stats": {"found": True}, "node": {...}}

# Search nodes
results = api.search("loader", node_types=["script"], top_k=10)
# Returns: {"stats": {...}, "matches": [...]}

# Explore neighbors
neighbors = api.get_neighbors("67982", direction="all")
# Returns: {
#   "stats": {"incoming": {...}, "outgoing": {...}, "siblings": {...}},
#   "incoming": [...], "outgoing": [...], "siblings": [...]
# }

# Filter by edge type
reads = api.get_neighbors("67982", direction="outgoing", relation_type="reads")

# Get only siblings
siblings = api.get_neighbors("67982", direction="siblings")
```

### Direction Options for `get_neighbors`

| Direction   | Returns                              | Notes                                    |
|-------------|--------------------------------------|------------------------------------------|
| `incoming`  | Nodes pointing TO this node          |                                          |
| `outgoing`  | Nodes this node points TO            |                                          |
| `all`       | Both + siblings                      | Default                                  |
| `siblings`  | Only sibling relationships           | Auto-sets `relation_type="sibling"`      |

---

## 🖥️ CLI Reference

The CLI tool `network` provides full access to all API functionality:

### Build & Stats

```bash
# Build the graph from scripts
uv run network --build

# Show general statistics
uv run network --stats

# Filter by node type
uv run network --stats --type script -n 5

# Filter by edge type
uv run network --stats --edge-type reads -n 3
```

### Navigation

```bash
# Show root folder hierarchy (SCRIPTS domain - default)
uv run network --tree

# Show DATA domain
uv run network --tree --domain data

# Show BOTH domains
uv run network --tree --domain both

# Navigate in SCRIPTS domain
uv run network --tree "/1. utilities"

# Navigate in DATA domain
uv run network --tree /Clean --domain data

# JSON output
uv run network --tree "/" --domain scripts --json
```

### Content

```bash
# Read full node content
uv run network --read 67982

# Read specific line range
uv run network --read 67982 -l 1-50

# Raw output (no formatting)
uv run network --read 67982 --raw

# Grep for patterns
uv run network --grep "table Items"

# Multiple patterns with context
uv run network --grep "table Items" "read.*Stock" -c

# Grep specific node types
uv run network --grep "def.*" --grep-types function
```

### Graph Exploration

```bash
# Get node metadata
uv run network --node 67982

# Search by name/path
uv run network --search loader
uv run network --search loader --type script

# Explore neighbors
uv run network --neighbors 67982

# Filter direction
uv run network --neighbors 67982 -d outgoing
uv run network --neighbors 67982 -d incoming
uv run network --neighbors 67982 -d siblings

# Filter by edge type
uv run network --neighbors 67982 -r reads
```

### Resolution Inspection

```bash
# Show resolved glob patterns
uv run network --globs

# Show resolved placeholder cascades
uv run network --cascades
```

### JSON Output

All commands support `--json` for machine-readable output:

```bash
uv run network --tree "/" --json
uv run network --node 67982 --json
uv run network --neighbors 67982 --json
```

---

## 🧠 Resolution Engine

### Path Normalization

Paths are normalized for consistency:
- Brackets: `[1]`, `[ 1]`, `[1 ]` → `[ X ]`
- Slashes: `\` → `/`
- Leading slash added for absolute paths

### Placeholder Cascades

Resolves variable interpolation in paths:

```envision
const root = "/project"
const dataPath = "{root}/data"
read "{dataPath}/catalog.ion"  // → /project/data/catalog.ion
```

View with: `uv run network --cascades`

### Glob Resolution

Resolves wildcards to concrete files:

```envision
read "/Input/*.ion"  // → Expanded to all matching files
```

View with: `uv run network --globs`

---

## 📝 Documentation Extraction

Envision scripts contain structured documentation:

| Marker | Audience   | Purpose                              |
|--------|------------|--------------------------------------|
| `///`  | Developer  | Structural documentation             |
| `//'`  | Scientist  | Business logic explanations          |
| `"""`  | End-User   | Dashboard UI documentation (Markdown)|
| `////` | Author     | Private memos and TODOs              |

All documentation is extracted and stored in `metadata.docs`.

---

## ⚙️ Configuration

Edit `config.yaml` to customize behavior:

```yaml
parsing:
  script_dir: "scripts"
  script_ext: "nvn"
  mapping_file: "mapping.txt"
  recursion_limit: 10

output:
  network_file: "datas/network/network.json"
  metadata_file: "datas/network/metadata.json"

data_extensions:
  - "ion"

normalize_brackets: true

search:
  top_k: 20

grep:
  default_node_types:
    - "script"
```

---

## 🔧 Extending

### Adding New Node Types

1. Add to `NodeType` enum in `typedefs.py`
2. Update extraction logic in `builder.py`
3. Update documentation

### Adding New Edge Types

1. Add to `EdgeType` enum in `typedefs.py`
2. Add extraction pattern in `builder.py`
3. Update `descriptions` dict in `network.py` CLI
