# Envision Copilot (`src/envision_copilot`)

**Role**: The main Agent Application.

## Features
- **Orchestrator**: LangGraph ReAct Agent.
- **Tools Integrator**: Uses `network` and `code_rag` as tools.
- **Context Aware**: Reads full code context.

## Usage
```bash
# Ask a question
uv run copilot -q "Où est défini le stock ?"

# Interactive
uv run copilot -i
```

## Dependencies
- `src/llms`
- `src/envision_preprocess`
- `src/code_rag`
