# Envision Code RAG

**The Semantic Memory of the System.**

This module builds a **Graph-Aware Vector Index** of the Envision codebase, enabling the Copilot to retrieve relevant code snippets not just by keyword, but by *meaning* and *structural relationship*.

It features a pilotable retrieval engine that allows the Agent to "focus" the search on specific symbols (functions, variables) identified via the Mini-Map.

---

## 💡 Why? The Evolution

### The Problem
Standard RAG pipelines treat code as flat text. They blindly chunk files, losing critical structural information.
*   *Query*: "Where is the `IsTopItem` logic?"
*   *Naive RAG*: Might find a comment mentioning `IsTopItem`, but miss the actual function definition if it's in a file named differently or hidden in a generic `utils.py`.
*   *Result*: The LLM hallucinates or says "I don't know".

### The Solution: Graph-Aware & Oriented
We shifted to a **Graph-Aware** architecture that leverages the dependency graph (Envision Network) to enrich chunks, combined with an **Oriented Reranker** driven by symbol extraction.
*   **Contextual**: Every chunk knows who imports it and what it consumes.
*   **Modular**: We separated the *Vector Engine* (Storage) from the *Reranking Strategy* (Precision).
*   **Pilotable**: The Agent can instruct the Reranker to find specific keys ("Find StockEvol").

---

## 🏗️ Architecture

### 1. Vector Engines (`vector_engines/`)
The foundational storage layer. Uses a **Factory Pattern**.
*   **Role**: Handles Embedding (Indexing) and Dense Retrieval (Recall).
*   **Implementation**: `sentence-transformers` engine (FAISS + SentenceBERT).

### 2. Graph-Aware Chunking (`chunker.py`)
Before embedding, code is **Enriched**.
```text
[Script: PathSchemas | Node: /1. utilities/PathSchemas]
[Imports: Global.ion]   <-- Graph Context
```
Searching for "Items.ion consumer" naturally finds this script.

### 3. Modular Rerankers (`rerankers/`)
Retrieval is a two-stage process: **Recall** (Broad) -> **Rerank** (Precise).
We support diverse strategies:

| Type            | Strategy              | Best For...                               |
| :-------------- | :-------------------- | :---------------------------------------- |
| **`oriented`**  | **Guided Heuristics** | **Copilot Usage** (Targets specific keys) |
| `heuristic`     | Domain-Specific Rules | Code Search (Passive)                     |
| `bge`           | Deep Learning (BAAI)  | Technical Documentation & Concepts        |
| `cross-encoder` | MS-MARCO              | General Purpose Q&A                       |

#### 🧠 Spotlight: The Oriented Reranker
Designed for Agentic workflows. It extends the `HeuristicReranker` to allow surgical retrieval.
1.  **Inherits Heuristic Intelligence**: Uses all TTB, PDS, and Diversity logic.
2.  **Targeted Boosting**: Heavily boosts chunks containing injected `keywords` (concepts) or `terms` (symbols).
    *   *Agent*: "I see 'StockEvol' in the query. Reranker, boost 'StockEvol'!"
    *   *Reranker*: *Boosts chunks containing the function definition.*

---

## 🚀 Usage

### 1. Indexing (Build the Memory)
The CLI uses the factory to load the configured engine and build the index.
```bash
uv run index --build
```
*   *Input*: `datas/network/network.json`
*   *Output*: `datas/code_rag/index/`

### 2. Querying (Test it)
```bash
uv run index -q "Where are best sellers calculated?"
uv run index -q "IsTopItem" --no-rerank      # Raw vector search
```

### 3. Benchmarking (`benchmark/`)
We don't guess; we measure.
```bash
uv run rag-benchmark            # Run full suite
uv run rag-benchmark -t 50      # Check top-50 recall
```

---

## ⚙️ Configuration (`config.yaml`)

Everything is configurable. No hardcoding.

```yaml
indexing:
  engine_type: "sentence-transformers"
  engine_name: "sentence-transformers/all-MiniLM-L6-v2"
  chunker_type: "graph"

retrieval:
  # The Precision Layer
  reranker_type: "oriented"  # Recommended for Copilot
  use_reranker: true
  
  # Base Heuristic Logic
  heuristic_reranking:
    weights:
      technical_term_boost: 0.7
      pattern_density: 0.4
      diversity_penalty: 0.9

  # Oriented Logic (The Pilot)
  oriented_reranking:
    weights:
      keyword_match: 1.5  # Boost per general keyword (def, show)
      term_match: 2.0     # Boost per technical term (StockEvol)
```

---

## 📁 Project Structure

```
code_rag/
├── vector_engines/       # 🏭 ENGINE FACTORY
│   ├── ...
├── rerankers/            # 🧠 RERANKING STRAT
│   ├── oriented_reranker.py  # <-- The Pilot
│   ├── heuristic_reranker.py # Custom Domain Logic
│   └── sentence_reranker.py  # ML Wrappers
├── benchmark/            # 📊 MEASUREMENT
├── chunkers/             # 🔪 PREPROCESSING
├── index.py              # CLI Entry Point
└── config.yaml           # Configuration
```
