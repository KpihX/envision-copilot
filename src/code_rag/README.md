# Code RAG (`src/code_rag`)

**La mémoire sémantique du système.**

Ce module construit un **index vectoriel enrichi par le graphe** de la codebase Envision, permettant au Copilot de retrouver du code pertinent non pas seulement par mots-clés, mais par *sens* et *relations structurelles*.

---

## 🚀 Quick Start

```bash
# 1. Construire l'index (après avoir généré le network)
uv run index --build

# 2. Tester une requête
uv run index -q "Where are best sellers calculated?"

# 3. Requête sans reranking (debug)
uv run index -q "IsTopItem" --no-rerank
```

---

## 📋 CLI Reference

```
uv run index [OPTIONS]

Options:
  --build               Construire l'index vectoriel
  -q, --query TEXT      Rechercher dans l'index
  --no-rerank           Désactiver le reranking (résultats bruts)
  -k, --top-k INT       Nombre de résultats (défaut: 10)
  -v, --verbose         Afficher les détails des chunks
```

### Benchmark RAG

```bash
# Évaluer la qualité du retrieval
uv run rag-benchmark

# Vérifier le recall top-50
uv run rag-benchmark -t 50
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
indexing:
  chunker_type: "graph"           # Chunking enrichi par le graphe
  engine_type: "sentence-transformers"
  engine_name: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 350                 # Tokens max par chunk
  overlap: 5                      # Lignes de contexte entre chunks

retrieval:
  reranker_type: "oriented"       # Recommandé pour Copilot
  use_reranker: true
  
  heuristic_reranking:
    weights:
      technical_term_boost: 0.7
      pattern_density: 0.4
      diversity_penalty: 0.9

  oriented_reranking:
    weights:
      keyword_match: 1.5          # Boost mots-clés généraux
      term_match: 2.0             # Boost termes techniques
```

### Modèles d'Embedding Disponibles

| Modèle | Dim | MTEB | Vitesse |
|--------|-----|------|---------|
| `all-MiniLM-L6-v2` | 384 | ~55% | Très rapide |
| `BAAI/bge-base-en-v1.5` | 768 | ~63% | 2x plus lent |
| `intfloat/multilingual-e5-base` | 768 | ~61% | Multilingue |
| `qwen/text-embedding-v4` (API) | 1024 | ~68% | Cloud |

---

## 🏗️ Architecture

```
code_rag/
├── index.py              # CLI entry point
├── utils.py              # Helpers
├── config.yaml           # Configuration
├── chunkers/             # 🔪 Découpage du code
│   └── graph_chunker.py  # Enrichi par contexte graphe
├── vector_engines/       # 🏭 Moteurs d'embedding
│   ├── base.py           # Interface abstraite
│   └── sentence_transformers.py
├── rerankers/            # 🧠 Stratégies de reranking
│   ├── oriented_reranker.py   # Pour Copilot (pilotable)
│   ├── heuristic_reranker.py  # Règles domaine
│   └── sentence_reranker.py   # ML (cross-encoder)
└── benchmark/            # 📊 Évaluation
```

### Pipeline de Retrieval

```
Query → Embedding → Vector Search (Recall) → Reranker (Precision) → Top-K
```

---

## 💡 Concepts Clés

### Graph-Aware Chunking

Chaque chunk est **enrichi avec son contexte structurel** :

```
[Script: PathSchemas | Node: /1. utilities/PathSchemas]
[Imports: Global.ion]   ← Contexte du graphe
[Defines: table Items, function StockEvol]

const inputFolder = "/Clean/"
read "{inputFolder}Items.ion" as Items...
```

→ Rechercher "Items.ion consumer" trouve naturellement ce script.

### Rerankers

| Type | Stratégie | Usage |
|------|-----------|-------|
| **`oriented`** | Heuristiques pilotables | **Copilot** (boost mots-clés injectés) |
| `heuristic` | Règles métier | Recherche passive |
| `cross-encoder` | Deep Learning | Q&A général |

Le **Oriented Reranker** permet au Copilot d'injecter des mots-clés à booster :
- Agent : "Je vois 'StockEvol' dans la query. Reranker, boost 'StockEvol' !"
- Reranker : *Priorise les chunks contenant la définition de la fonction*

---

## 📁 Données

**Input** : `datas/network/network.json` (généré par `envision_preprocess`)

**Output** : `datas/code_rag/index/`
```
index/
├── faiss.index    # Index vectoriel FAISS
└── metadata.json  # Métadonnées des chunks
```

