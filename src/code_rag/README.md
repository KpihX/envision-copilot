# Envision Code RAG

**The "Memory" of the System.**

This module builds a **Graph-Aware Vector Index** of the Envision codebase.
Unlike standard RAG which blindly chunks text, this engine leverages the dependency graph to enrich each chunk with context (dependencies, imports, purposes), and uses a Two-Stage retrieval process for high precision.

---

## 🏗️ Architecture

### 1. Graph-Aware Chunking (`chunker.py`)
Code is not just text; it's a web of dependencies.
Before embedding a script, we prepend a **Context Header**:
```text
[Script: /Utilities/PathSchemas]
[Imports: Global.ion]
[Reads: /Clean/Items.ion]
[Docs: "Calculates global path schemas..."]

<Actual Code Content>
```
*   **Benefit**: The embedding model "sees" the connections. Searching for "Items.ion consumer" naturally retrieves this script.
*   **Source Tracking**: Each chunk now explicitly stores the `source` (Full Logical Path) in metadata, ensuring the agent constantly knows exactly which file is being referenced.

### 2. Two-Stage Retrieval
1.  **Recall (Dense Search)**: Fetch top 50 candidates using `sentence-transformers/all-MiniLM-L6-v2` via FAISS. High speed, broad net.
2.  **Precision (Rerank)**: Re-score top 50 using `cross-encoder/ms-marco-MiniLM-L-6-v2`. High accuracy, understands logical query-document relationship.

---

## 🚀 Usage

### CLI (`index.py`)

#### Build Index
Chunk -> Embed -> Index.
```bash
uv run index --build
```

#### Query
Search for concepts or specific code.
```bash
uv run index -q "Where is dispatch calculated?"
```
*   **Output**: Shows formatted code blocks with syntax highlighting and context headers.
*   **Visual**: Displays Reranking progress and scores.

#### Statistics
Inspect index health and metadata.
```bash
uv run index --stats
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
indexing:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
```

```yaml
retrieval:
  use_reranker: true
  top_k_recall: 50
  top_k_final: 5
```
