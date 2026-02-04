# 🔬 Envision Agentic RAG System

**Moteur d'extraction d'information et de raisonnement sur des codebases DSL.**

Un système agentique qui combine **analyse structurelle** (graphe de dépendances) et **recherche sémantique** (RAG) pour répondre à des questions complexes sur du code Envision.

---

## 🚀 Quick Start

### Prérequis

- Python 3.10+
- **uv** (gestionnaire de dépendances) — [Installation](https://github.com/astral-sh/uv)

### Installation

```bash
# Cloner le projet
git clone https://github.com/ClementLokad/llm-DSL-info-extraction.git
cd llm-DSL-info-extraction

# Installer les dépendances (uv uniquement !)
uv sync

# Configurer les API keys
cp src/llms/.env.example src/llms/.env
# Éditer src/llms/.env avec vos clés API
```

### Première Utilisation

```bash
# 1. Placer vos scripts .nvn dans scripts/

# 2. Construire le graphe de dépendances
uv run network --build

# 3. Construire l'index sémantique
uv run index --build

# 4. Lancer le Copilot !
uv run copilot -i
```

---

## 📦 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENVISION AGENTIC RAG SYSTEM                          │
│                                                                          │
│   ┌───────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐  │
│   │  Scripts  │───▶│  Graph API   │───▶│ Vector Index│───▶│  Copilot │  │
│   │  (.nvn)   │    │  (Network)   │    │   (FAISS)   │    │  (Agent) │  │
│   └───────────┘    └──────────────┘    └─────────────┘    └──────────┘  │
│                                                                          │
│        "Eyes"            "Eyes"           "Memory"          "Brain"      │
│     envision_preprocess                   code_rag      envision_copilot │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ CLI Tools

### `copilot` — Assistant IA Principal

```bash
uv run copilot -i                    # Mode interactif (recommandé)
uv run copilot -q "Question"         # Query unique
uv run copilot -v -q "Question"      # Avec détails du raisonnement
```

📖 Détails : [src/envision_copilot/README.md](src/envision_copilot/README.md)

### `network` — Exploration du Graphe

```bash
uv run network --build               # Construire le graphe
uv run network --tree                # Afficher l'arborescence
uv run network --stats               # Statistiques
uv run network --neighbors 68006     # Explorer les voisins d'un nœud
```

📖 Détails : [src/envision_preprocess/README.md](src/envision_preprocess/README.md)

### `index` — Index Sémantique

```bash
uv run index --build                 # Construire l'index
uv run index -q "stock calculation"  # Recherche sémantique
```

📖 Détails : [src/code_rag/README.md](src/code_rag/README.md)

### `llm` — Interface LLM

```bash
uv run llm -q "Hello" -m mistral     # Query directe
uv run llm -i                        # Mode interactif
```

📖 Détails : [src/llms/README.md](src/llms/README.md)

### `benchmark` — Évaluation

```bash
uv run benchmark                     # Lancer tous les tests
uv run benchmark --id 15             # Tester une question spécifique
```

📖 Détails : [src/envision_benchmark/README.md](src/envision_benchmark/README.md)

---

## ⚙️ Configuration

Chaque package a son propre `config.yaml` :

| Package | Config | Clés Principales |
|---------|--------|------------------|
| `llms` | Providers LLM | `defaults.model`, API keys dans `.env` |
| `envision_preprocess` | Parsing & API | `api.mode` (lite/full) |
| `code_rag` | Indexation & Retrieval | `indexing.engine_name`, `reranker_type` |
| `envision_copilot` | Agent & Prompts | `agent.llm_type`, `constraints` |

---

## 📁 Structure des Données

```
scripts/               # Scripts Envision (.nvn) à analyser
datas/
├── network/           # Graphe de dépendances (généré)
│   ├── network.json
│   └── metadata.json
├── code_rag/          # Index vectoriel (généré)
│   └── index/
├── benchmark/         # Rapports d'évaluation
└── copilot/           # Logs des sessions
```

---

## 🔑 API Keys

Créer `src/llms/.env` :

```bash
MISTRAL_API_KEY=your_key      # Mistral AI
GOOGLE_API_KEY=your_key       # Google Gemini
GROQ_API_KEY=your_key         # Groq
DASHSCOPE_API_KEY=your_key    # Qwen (Alibaba)
```

---

## 💡 Philosophie

> **"Prompt engineering appliqué à l'architecture d'agents"**  
> **"Tout raisonneur peut performer si l'environnement est bien scaffoldé"**

Le système est conçu pour maximiser les performances avec des **LLMs modestes/gratuits** via :

- **Scaffolding cognitif** : Documentation riche des outils avec tables et exemples
- **Mode lite** : Réponses API optimisées (-80% tokens)
- **Anti-boucle** : Injection du raisonnement précédent
- **Stopping rule** : Checklist explicite avant abandon

---

## 📚 Documentation Approfondie

- **Architecture & Évolution** : [GEMINI.md](GEMINI.md) (guide développeur)
- **Copilot (Agent)** : [src/envision_copilot/README.md](src/envision_copilot/README.md)
- **Graphe** : [src/envision_preprocess/README.md](src/envision_preprocess/README.md)
- **RAG** : [src/code_rag/README.md](src/code_rag/README.md)
- **LLMs** : [src/llms/README.md](src/llms/README.md)
- **Benchmark** : [src/envision_benchmark/README.md](src/envision_benchmark/README.md)

---

## 📜 License

MIT
