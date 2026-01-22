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
[Script: PathSchemas | Node: /1. utilities/PathSchemas]
[Imports: Global.ion]
[Reads: /Clean/Items.ion]

<Actual Code Content>
```
*   **Benefit**: The embedding model "sees" the connections. Searching for "Items.ion consumer" naturally retrieves this script.
*   **Source Tracking**: Each chunk stores the full `source` path in metadata, ensuring the agent knows exactly which file is being referenced.

### 2. Two-Stage Retrieval (`retriever.py`)
1.  **Recall (Dense Search)**: Fetch top `recall_k` candidates (default: 50) using `sentence-transformers/all-MiniLM-L6-v2` via FAISS.
2.  **Precision (Rerank)**: Re-score using a cross-encoder reranker for semantic matching.

### 3. Modular Rerankers (`rerankers/`)
Extensible reranker package supporting multiple models:

| Type            | Model                                            | Notes                       |
| --------------- | ------------------------------------------------ | --------------------------- |
| `cross-encoder` | `cross-encoder/ms-marco-MiniLM-L-6-v2`           | Default, fast               |
| `bge`           | `BAAI/bge-reranker-base`                         | Best for technical terms    |
| `mxbai`         | `mixedbread-ai/mxbai-rerank-base-v1`             | Best speed/accuracy (gated) |
| `answerai`      | `tomaarsen/reranker-modernbert-base-msmarco-bce` | Optimized Q&A (gated)       |

**Contextual Reranking**: When enabled, chunks are enriched with metadata (FILE, SCRIPT, CONTEXT) before reranking for better relevance.

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
uv run index -q "stockEvol function" --no-rerank   # Skip reranking
uv run index -q "Items table" -r 100               # Recall 100 candidates
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
uv run rag-benchmark --alpha log        # Use logarithmic decay for position score
```

### Metrics
The benchmark computes three key metrics:

1. **Recall Score**: Percentage of patterns found in top-K chunks
2. **Position Score**: Quality-weighted score using α-decay (penalizes lower ranks)
3. **Rerank Gain**: Improvement from reranking vs dense-only retrieval

### Output
For each question, the benchmark shows:
- Which patterns are found (✓) or missing (✗)
- Which chunks contain patterns, with their rank position
- Position delta (↑/↓) when reranking is active
- Summary statistics with rerank gain

### Questions File
Located at `benchmark/questions.json` with pattern-based questions containing syntactic patterns to find (paths, formulas, variable names).

> 📝 All benchmark questions are in **English** for optimal retrieval performance.

---

## ⚙️ Configuration (`config.yaml`)

```yaml
indexing:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 350      # Characters per chunk
  overlap: 5           # Lines overlap between chunks
  max_deps: 5          # Max dependencies to include in context

retrieval:
  reranker_type: "bge"           # cross-encoder, bge, mxbai, answerai
  # reranker_name: "custom/model" # Override default model
  use_reranker: true
  contextual_reranking: true     # Enrich chunks with metadata before reranking
  top_k_recall: 50
  top_k_final: 5

benchmark:
  questions_file: "src/code_rag/benchmark/questions.json"
  log_file: "data/logs/rag_benchmark.json"
  top_k: 50
  recall_k: 50
```

---

## 📁 Structure

```
code_rag/
├── chunker.py        # Graph-aware chunking
├── indexer.py        # FAISS index builder
├── retriever.py      # Two-stage retrieval (dense + rerank)
├── index.py          # CLI entry point
├── config.yaml       # All configuration
├── utils.py          # Shared utilities
├── rerankers/        # Modular reranker package
│   ├── __init__.py       # Factory function & registry
│   ├── base.py           # Abstract base class
│   └── sentence_reranker.py  # Unified cross-encoder implementation
└── benchmark/
    ├── main.py           # Benchmark runner
    └── questions.json    # Test questions (English)
```
