# Gemini Memory Context: Envision DSL RAG (Architecture 2.0)

## 1. Project Overview
This system is an advanced **Hybrid RAG** (Retrieval-Augmented Generation) engine designed specifically for **Envision DSL** codebases. It combines structural analysis (Graph) and semantic understanding (Vector) to answer complex technical questions about script dependencies and logic.

### Key Logic (Architecture 2.0):
- **Deterministic Graph**: Parses `.nvn` files to identify `read`, `write`, `export`, and `import` links.
- **Constant Resolution**: The parser recursively resolves Envision constants (`const variable = "..."`) to accurately track dynamic file paths (e.g., `read "\{path}file.ion"`).
- **Semantic Vector Index**: Chunks scripts by logical blocks (`read`, `show`, `table`) while preserving preceding comments as context.
- **LangGraph Agent**: A ReAct-style agent orchestrates tools to find readers/writers or search code snippets.

## 2. Updated Directory Structure
- **`src/envision_rag/`**: Core package.
    - `graph/`: `builder.py` (Scanner 2.0 with constant resolution), `graph_types.py` (NetworkX wrapper).
    - `index/`: `chunker.py` (Semantic block chunking), `vector_tools.py` (FAISS retrieval).
    - `workflow/`: `agent.py` (LangGraph ReAct loop, **Appendix** generation).
    - `tools/`: `graph_tools.py` (Agent-ready tools).
    - `agents/`: LLM client wrappers (Mistral, Gemini).
- **`main.py`**: Main CLI supporting verbose mode and graph rebuilds.
- **`build_index_v2.py`**: Indexing script for the Vector database.
- **`data/`**: Stores `dependency_graph.json` and `vector_store/` (FAISS + metadata).
- **`env_scripts/`**: Source Envision files (Gitignored, use `uv run` to process).

## 3. Core Features
- **Verbose Mode (`-v`)**: Displays the full internal reasoning (Thought/Action/Observation) for total transparency.
- **Automatic Appendix**: Every answer includes an "Appendix" section containing the raw data used by the agent (e.g., the exact list of scripts found in the graph).
- **Robustness**: Handles 25/27 readers for core files (e.g., `/Clean/Items.ion`) by resolving variables.

## 4. Usage Commands

### Setup & First Build
```bash
# Install dependencies (requires uv)
uv add sentence-transformers faiss-cpu numpy networkx langgraph

# 1. Build the Structural Graph
uv run src/envision_rag/main.py --rebuild --query "INIT"

# 2. Build the Semantic Vector Index
uv run build_index_v2.py
```

### Querying
```bash
# Simple Query
uv run src/envision_rag/main.py --query "Combien de scripts lisent /Clean/Items.ion ?"

# Verbose Mode (Transparency + Trace)
uv run src/envision_rag/main.py -v --query "Comment est calculé le flux de stock ?"

# Interactive Mode
uv run src/envision_rag/main.py -i
```

## 5. Next Steps for Takeover
- **Vector Tuning**: Improve retrieval by fine-tuning chunk sizes or metadata weighting.
- **Graph Expansion**: Add more specific tools for `keep` and `where` block analysis.
- **Logic Mapping**: Enhance the `describe_impact` tool to follow multi-hop data lineage.
