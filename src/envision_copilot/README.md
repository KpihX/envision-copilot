# Envision Copilot (`src/envision_copilot`)

**The "Brain" of the System.**

This module implements an **Agentic AI Assistant** capable of investigating, explaining, and debugging the Envision DSL codebase. It uses a **Tree-of-Thoughts** planner and a suite of specialized tools to solve complex queries.

---

## 🏗️ Architecture: The "ReAct" Agent

The Copilot follows a Reasoning-Action (ReAct) loop:
1.  **Plan**: Uses a Tree Planner to decompose variables and strategies.
2.  **Act**: Selects the best tool (`structural`, `semantic`, `grep`, `read`).
3.  **Observe**: Analyzes tool output.
4.  **Synthesize**: Updates its internal "Fact Store" and refines the plan.

### Core Strategies
*   **"Best Effort" Handling**: If the agent hits iteration limits (e.g. 10 steps) or cannot find an exact answer, it **synthesizes** all gathered facts into a "Best Effort" response rather than failing.
*   **Data File Protocol**: Explicitly treats `.ion` and `.csv` files as **Black Boxes**. It trusts script references (`reads /Clean/Items.ion`) and avoids trying to read data files directly, focusing instead on the logic that processes them.

---

## 🛠️ Tool Suite

### 1. `structural_explorer` (The Graph)
Query the Dependency Network directly.
*   **Actions**: `nodes`, `edges`, `neighbors`.
*   **Use Case**: "What scripts import X?", "Who writes to Items.ion?", "List all tables".
*   **Optimization**: Returns truncated list (max 50) for broad queries to save context.

### 2. `semantic_search` (The RAG)
Semantic Vector Search over the codebase.
*   **Context**: Returns snippets with full dependency context headers.
*   **Metadata**: Includes the **Full Source Path** for every result.
*   **Use Case**: "How is stock calculated?", "Explain the dispatch logic."

### 3. `read_file` (The Reader)
Reads raw source code from **Scripts** or **Functions**.
*   **Constraint**: CANNOT read data files (`.ion`, `.csv`). Only executable logic.

### 4. `grep_search` (The Scanner)🆕
Regex-based content search.
*   **Function**: Searches for text pattern inside the *content* of nodes.
*   **Use Case**: Finding local variables, specific hardcoded strings, or when RAG results are too ambiguous.
*   **Example**: `grep_search(pattern="myLocalVar", node_type="script")`

---

## 🚀 Usage

### CLI (`main.py`)

#### Interactive Mode (Recommended)
Full chat session with memory.
```bash
uv run copilot -i
```

#### One-Shot Query
Quick answer for CLI pipelines.
```bash
uv run copilot -q "Where is the ReDispatchCycle defined?"
```

#### Verbose Mode
See the "Thought Process" (Plan, Tools, Observations).
```bash
uv run copilot -i -v
```

---

## ⚙️ Configuration (`config.yaml`)
*   **Prompts**: System prompt, Tool definitions.
*   **Constraints**: Max iterations, Context window size.
*   **Models**: Default LLM provider (Mistral, Gemini).
