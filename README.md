# Envision RAG

**Hybrid RAG system for Envision DSL codebase analysis with LLM-as-Judge benchmarking.**

A ReAct-style agent combining **Graph Analysis** (NetworkX) and **Semantic Search** (FAISS) to answer questions about Envision DSL codebases.

---

## 📦 Installation

```bash
git clone <repo-url> && cd envision-rag
uv sync
```

---

## 🚀 Quick Start

### 1. Prepare Your Scripts

Place your Envision scripts in the `scripts/` folder:
```
scripts/
├── 12345.nvn
├── 67890.nvn
└── ...
```

> **📝 Configuration (config.yaml):**
> ```yaml
> system:
>   scripts_dir: "./scripts"      # Change folder name
>   file_extension: "nvn"         # Change extension (without dot)
>   mapping_file: "mapping.txt"   # ID -> Path mapping
> ```

### 2. Create Mapping File

Create `mapping.txt` to map IDs to logical paths:
```
12345, /1. utilities/Script A
67890, /3. Inspectors/Report B
```

### 3. Build Indexes

```bash
uv run build                 # Build both (graph + vector)
uv run build -g              # Graph only (fast)
uv run build -v              # Vector only (slower)
uv run build -s              # With detailed stats
```

### 4. Query the System

```bash
uv run main -q "Combien de scripts lisent /Clean/Items.ion?"
uv run main -v -q "..."      # Verbose trace
uv run main -i               # Interactive mode
```

---

## 📖 CLI Commands

| Command | Description |
|---------|-------------|
| `uv run main` | Query the RAG system |
| `uv run build` | Build graph/vector indexes |
| `uv run benchmark` | Run LLM-as-Judge evaluation |
| `uv run logs` | View/manage saved sessions |
| `uv run test-graph` | Test graph index queries |
| `uv run test-vector` | Test vector index queries |

### `main` - Query the RAG system

```
uv run main -q "..."         Single query
uv run main -i               Interactive mode (agent can ask clarifications)
uv run main -v -q "..."      Verbose agent trace
```

### `build` - Build indexes

```
uv run build                 Build both (default)
uv run build -g              Graph only (NetworkX)
uv run build -v              Vector only (FAISS)
uv run build -s              Show detailed statistics
uv run build -q              Quiet mode
```

### `benchmark` - LLM-as-Judge evaluation

```
uv run benchmark             Default: questions 1-5
uv run benchmark -f 1 -t 10  Questions 1 to 10
uv run benchmark -i 1 3 5    Specific IDs
uv run benchmark -q          Quiet mode
```

### `logs` - View/manage sessions

```
uv run logs -t main -n 1       View last main session
uv run logs -t benchmark -l    List benchmark logs
uv run logs -t main -c 7       Delete logs older than 7 days
```

### `test-graph` - Test graph queries

```
uv run test-graph -q "read /Clean/Items.ion" -n 10
uv run test-graph -q "write FcItems" -n 5
uv run test-graph -q "import" --all
```

### `test-vector` - Test semantic search

```
uv run test-vector -q "stock calculation" -n 5
uv run test-vector -q "forecast function" --full
```

---

## ⚙️ Configuration

All settings in `config.yaml`:

```yaml
system:
  scripts_dir: "./scripts"       # Source folder
  file_extension: "nvn"          # File extension
  mapping_file: "mapping.txt"    # ID -> Path mapping

agent:
  main_model: "mistral"          # Default LLM
  max_iterations: 10             # Max reasoning steps

benchmark:
  judge_model: "mistral"         # Evaluation model
  questions_file: "questions.json"
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                 User Query                       │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│            ReAct Agent (LLM)                     │
│      Thought → Action → Observation → ...        │
└─────────────────────────────────────────────────┘
            │                       │
            ▼                       ▼
┌───────────────────┐   ┌─────────────────────────┐
│   Graph Tools     │   │    Search Tools         │
│  (Deterministic)  │   │    (Semantic)           │
├───────────────────┤   ├─────────────────────────┤
│ • scan_references │   │ • search_code (FAISS)   │
│ • describe_impact │   │ • grep_code (Regex)     │
└───────────────────┘   │ • read_code             │
         │              └─────────────────────────┘
         ▼                          │
┌───────────────────┐   ┌─────────────────────────┐
│  NetworkX Graph   │   │  FAISS Vector Index     │
└───────────────────┘   └─────────────────────────┘
```

---

## 🛠️ Agent Tools

| Tool | Description |
|------|-------------|
| `scan_references(query)` | Find read/write/import relationships |
| `describe_impact(script)` | Trace downstream data flow |
| `grep_code(pattern)` | Regex search across scripts |
| `read_code(path, start, end)` | Read file lines |
| `search_code(query)` | Semantic search |

---

## 📁 Project Structure

```
envision-rag/
├── src/envision_rag/
│   ├── agents/           # LLM agents
│   ├── benchmark/        # LLM-as-Judge
│   ├── cli/              # CLI entry points
│   ├── graph/            # NetworkX builder
│   ├── index/            # FAISS indexing
│   ├── logging/          # Session logging
│   ├── tools/            # Agent tools
│   └── workflow/         # LangGraph agent
├── scripts/              # Source scripts (.nvn)
├── data/                 # Generated indexes
├── config.yaml
├── mapping.txt
└── questions.json
```

---

## 📄 License

MIT License
