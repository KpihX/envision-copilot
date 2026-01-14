# Gemini Memory Context: Envision RAG (Architecture 3.0)

## 1. Project Overview
This system is an **Agentic RAG Engine** refactored into a **Modular Architecture (3.0)**. It is designed to understand, index, and query **Envision DSL** codebases using a combination of structural analysis (Network) and semantic search (Vector).

### Core Philosophy
- **Modular & Standalone**: Everything is split into 5 independent packages (`llms`, `envision_preprocess`, `code_rag`, `envision_copilot`, `envision_benchmark`). Each can be run separately via CLI.
- **Data Centric**: All artifacts are stored in a centralized `data/` directory (`data/network`, `data/vector_store`), with dedicated lightweight `metadata.json` for fast introspection.
- **Strict `uv` Usage**: We use `uv` exclusively for dependency management and execution (`uv run ...`). No `pip`, no direct `python`, no `hatchlet` commands.
- **No Global Config**: Each package manages its own configuration in `src/<package>/config.yaml`.

## 2. Package Ecology
- **`src/llms`**: Abstract interface for generic LLM access (Mistral, Gemini, Groq).
  - CLI: `uv run llm -q "..." -m mistral`
- **`src/envision_preprocess`**: The "Eyes". Parses .nvn files, resolves constants deeply, and builds a rich Node Network (Script, File, Table, Var, Func) with full context.
  - CLI: `uv run network --build`, `uv run network --stats`
- **`src/code_rag`**: The "Memory". Indexes the Network into FAISS (Dense) + Hybrid Search.
  - CLI: `uv run index --build`, `uv run index --query "..."`
- **`src/envision_copilot`**: The "Brain". A LangGraph Agent that orchestrates the other packages to answer complex user questions.
  - CLI: `uv run copilot -q "..." (-i)`
- **`src/envision_benchmark`**: The "Judge". Automated evaluation framework.
  - CLI: `uv run benchmark`

## 3. Data Structure
- `data/network/network.json`: The source of truth for structural dependencies.
- `data/vector_store/faiss.index`: The semantic index.
- `data/**/*metadata.json`: Quick-access stats files.

## 4. Workflow Rules
- **Environment**: Use `.env` files located inside packages (`src/llms/.env`, `src/envision_copilot/.env`).
- **Development**: Always use `uv sync --reinstall` when adding/moving packages to ensure the editable install is up to date.
- **Mapping**: The mapping file is located at `src/envision_preprocess/mapping.txt`.
- **Testing**: **ALWAYS use Verbose Mode (`-v`)** during development and testing to detect issues. Never run "blind".
- **Documentation**: **ALWAYS update package README.md** after completing a feature or refactor plan.

## 5. Next Steps
- Implement robust benchmarking.
- Refine "Function" detection in preprocess.
- Explore Cross-Encoder tuning in `code_rag`.
