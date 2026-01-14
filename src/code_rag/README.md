# Code RAG (`src/code_rag`)

**Role**: Generic RAG Engine for Code.

## Features
- **Indexer**: Embeds code chunks using SentenceTransformers + FAISS.
- **Retriever**: Hybrid Search (Dense + Keyword).
- **Reranker**: Cross-Encoder refinement.

## Usage
```bash
# Build Index
uv run index --build

# Query
uv run index --query "stock calculation logic"
```

## Input
- `data/network/network.json` (from envision_preprocess)

## Output
- `data/vector_store/faiss.index`
- `data/vector_store/metadata.json`
