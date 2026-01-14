# LLMs Package (`src/llms`)

**Role**: Abstract generic layer for LLM providers.

## Features
- Unified `LLM` interface.
- Providers: Mistral, Gemini, Groq.
- Interactive CLI for direct prompting.

## Usage
```bash
# Query specific model
uv run llm -q "Explique le code" -m mistral

# Interactive mode
uv run llm -i
```

## Configuration
- `config.yaml`: Default models.
- `.env`: API Keys (MISTRAL_API_KEY, etc.)
