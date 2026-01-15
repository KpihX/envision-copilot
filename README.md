# Envision Agentic RAG System (Architecture 3.0)

**An Advanced Agentic System for Parsing, Indexing, and Querying DSL Codebases.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Env](https://img.shields.io/badge/Dependency-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20Agentic-green.svg)](docs/architecture.md)

---

## 📖 Overview

This project implements a specialized **Information Extraction & Reasoning Engine** for the **Envision DSL** (a proprietary Supply Chain Language).
Unlike standard RAG systems that treat code as plain text, this engine builds a **Semantic Dependency Network** to understand the deep relationships between Scripts, Data Files, and Global Variables.

It is powered by a **Code Copilot (Agent)** that reasons over this graph to answer complex business questions like *"How is the Dispatch Reorder Point calculated?"* or *"Where is the stock variable impacted?"*.

---

## 🏗️ Modular Architecture (v3.0)

The system is decoupled into 5 standalone packages managed by a single `uv` workspace:

### 1. `src/envision_preprocess` (The Eyes)
*   **Role**: Parsing & Graph Construction.
*   **Tech**: Custom Regex Parser + Recursive Glob Resolution.
*   **Output**: A directed graph of dependencies (Reads, Writes, Imports).
*   **Key Feature**: Resolves dynamic paths (`read "/Data/\{Category}.ion"`).

### 2. `src/code_rag` (The Memory)
*   **Role**: Indexing & Retrieval.
*   **Tech**: FAISS (Dense) + Cross-Encoder (Reranking).
*   **Strategy**: "Graph-Aware Chunking". Every code chunk includes a metadata header describing its imports and I/O context.

### 3. `src/envision_copilot` (The Brain)
*   **Role**: Agent Orchestration.
*   **Tech**: LangGraph (ReAct Loop).
*   **Capabilities**:
    *   **Tree of Thoughts**: Break down complex queries.
    *   **Tools**: `structural_explorer`, `semantic_search`, `read_file`, `grep_search`.
    *   **Black Box Protocol**: Knows how to handle Data Files vs. Logic Scripts.

### 4. `src/llms` (The Interface)
*   **Role**: Unified LLM Abstraction Layer.
*   **Models**: Mistral (via Groq/Local), Gemini, OpenAI.

### 5. `src/envision_benchmark` (The Judge)
*   **Role**: Automated Evaluation.
*   **Tech**: LLM-as-a-Judge to score answer accuracy against a Golden Dataset.

---

## 🚀 Quick Start

### Prerequisites
*   **Linux/MacOS**
*   **[uv](https://github.com/astral-sh/uv)** (Fast Python package manager).

### Installation
```bash
# 1. Clone
git clone ...
cd llm-DSL-info-extraction0

# 2. Sync Dependencies (Reinstall workspace)
uv sync --reinstall
```

### Usage

#### 1. Build the Knowledge Base
Parse the codebase and build the vector index.
```bash
# Build Graph
uv run network --build

# Build RAG Index
uv run index --build
```

#### 2. Run the Copilot
Interact with the agent in your terminal.
```bash
# Interactive Chat Mode
uv run copilot -i

# One-Shot Query
uv run copilot -q "Where is the variable 'ReDispatchCycle' defined?"
```

#### 3. Inspect the Graph
Query the dependency network directly.
```bash
# Show file statistics
uv run network --stats

# List all scripts reading 'Items.ion'
uv run structural --action neighbors --node_id "Items.ion" --relation_type reads --direction incoming
```

---

## 📂 Directory Structure

```text
.
├── src/
│   ├── envision_preprocess/  # Parsing Engine
│   ├── code_rag/             # Vector Database
│   ├── envision_copilot/     # Agent Logic
│   ├── llms/                 # LLM Interfaces
│   └── envision_benchmark/   # Evaluation Framework
├── data/                     # (Ignored) Generated Indexes & Networks
│   ├── network/
│   └── vector_store/
├── pyproject.toml            # Workspace Definition
└── uv.lock
```

---

## 🛠️ Advanced Features

### Data File Protocol
The Agent is strictly instructed to treat `.ion` and `.csv` files as **Black Box Data**.
*   It checks **Scripts** that read these files to understand data structure.
*   It never attempts to read raw data files directly, preventing hallucination and token waste.

### Grep Search
When RAG is too broad, the Agent uses `grep_search` to perform regex scans on script content to find local variables or hardcoded strings.

---

## 📄 License
Internal / Proprietary.
