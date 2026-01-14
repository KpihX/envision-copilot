# Envision RAG System (Architecture 3.0)

A modular, containerized RAG system for Envision DSL codebases.

## Packages

### 1. `llms`
Abstract layer for LLM providers.
- **CLI**: `uv run llm`

### 2. `envision_preprocess`
DSL Parser and Network Builder.
- **CLI**: `uv run network`

### 3. `code_rag`
Code indexer and retrieval engine.
- **CLI**: `uv run index`

### 4. `envision_copilot`
Main Agent application.
- **CLI**: `uv run copilot`

### 5. `envision_benchmark`
Evaluation framework.
- **CLI**: `uv run benchmark`

## Setup
1. Define `.env` in `src/llms/.env` and `src/envision_copilot/.env`.
2. Run `uv sync`.
