# Envision Code RAG

**The "Memory" of the System.**

This module builds a **Graph-Aware Vector Index** of the Envision codebase.
Unlike standard RAG which blindly chunks text, this engine leverages the dependency graph to enrich each chunk with context (dependencies, imports, purposes), and uses a Two-Stage retrieval process for high precision.

> ⚠️ **IMPORTANT**: This RAG system is **optimized for English queries**. The embedding model performs best with English text. If your copilot receives French questions, translate them to English before calling the retriever.

---

## 🏗️ Architecture

### 1. Graph-Aware Chunking (`chunker.py`)
Code is not just text; it's a web of dependencies.
Before embedding a script, we prepend a **Context Header**:
```text
[Script: /Utilities/PathSchemas | Path: /1. utilities/PathSchemas]
[Imports: Global.ion]
[Reads: /Clean/Items.ion]

<Actual Code Content>
```
*   **Benefit**: The embedding model "sees" the connections. Searching for "Items.ion consumer" naturally retrieves this script.
*   **Source Tracking**: Each chunk stores the full `source` path in metadata, ensuring the agent knows exactly which file is being referenced.

### 2. Two-Stage Retrieval (`retriever.py`)
1.  **Recall (Dense Search)**: Fetch top `recall_k` candidates (default: 50) using `sentence-transformers/all-MiniLM-L6-v2` via FAISS.
2.  **Precision (Rerank)**: Re-score using `cross-encoder/ms-marco-MiniLM-L-6-v2`. High accuracy, understands logical query-document relationship.

---

## 🚀 Usage

### CLI (`index.py`)

#### Build Index
```bash
uv run index --build
```

#### Query
```bash
uv run index -q "Where are best sellers calculated?"
```

#### Statistics
```bash
uv run index --stats
```

---

## 📊 Benchmark (`benchmark/`)

Evaluate retrieval quality by testing if expected patterns are found in retrieved chunks.

### Run Benchmark
```bash
uv run rag-benchmark                    # All questions with defaults from config
uv run rag-benchmark -t 10              # Top-10 final chunks
uv run rag-benchmark -r 100             # Recall 100 candidates from FAISS
uv run rag-benchmark -n                 # Disable reranking
uv run rag-benchmark --ids 1 2 3        # Specific questions only
```

### Output
For each question, the benchmark shows:
- Which patterns are found (✓) or missing (✗)
- Which chunks contain patterns, with their rank position
- Summary statistics (patterns found / total)

### Questions File
Located at `benchmark/questions.json` with two categories:
- **`pattern_based`**: Questions with syntactic patterns to find (paths, formulas, variable names)
- **`non_pattern_based`**: Questions without verifiable patterns

> 📝 All benchmark questions are in **English** for optimal retrieval performance.

---

## ⚙️ Configuration (`config.yaml`)

```yaml
indexing:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512

retrieval:
  use_reranker: true
  top_k_recall: 50
  top_k_final: 5

benchmark:
  questions_file: "src/code_rag/benchmark/questions.json"
  log_file: "data/logs/benchmark.json"
  top_k: 5
  recall_k: 50
```

---

## 📁 Structure

```
code_rag/
├── chunker.py      # Graph-aware chunking
├── indexer.py      # FAISS index builder
├── retriever.py    # Two-stage retrieval (dense + rerank)
├── reranker.py     # Cross-encoder reranker
├── index.py        # CLI entry point
├── config.yaml     # All configuration
└── benchmark/
    ├── main.py         # Benchmark runner
    └── questions.json  # Test questions (English)
```
