# 🔬 Envision Agentic RAG System (Architecture 3.0)

**Un Système Agentique Avancé pour l'Extraction d'Information et le Raisonnement sur des Codebases DSL**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/Dependency-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20Agentic-green.svg)](docs/architecture.md)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph%20ReAct-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/Index-FAISS-red.svg)](https://github.com/facebookresearch/faiss)

---

## 📑 Table des Matières

1. [Présentation Générale](#-présentation-générale)
2. [Philosophie du Projet](#-philosophie-du-projet)
3. [Architecture Globale](#-architecture-globale)
4. [Pipeline Complète](#-pipeline-complète)
5. [Modules et Packages](#-modules-et-packages)
6. [Structures de Données](#-structures-de-données)
7. [Installation et Prise en Main](#-installation-et-prise-en-main)
8. [Utilisation Détaillée](#-utilisation-détaillée)
9. [Simulation de Fonctionnement](#-simulation-de-fonctionnement)
10. [Configuration Avancée](#-configuration-avancée)
11. [Protocoles et Conventions](#-protocoles-et-conventions)
12. [License](#-license)

---

## 🎯 Présentation Générale

### Le Problème

Les codebases **Envision DSL** (langage propriétaire pour Supply Chain chez Lokad) contiennent des centaines de scripts interconnectés avec des dépendances complexes : lectures de fichiers de données, imports entre scripts, définitions de variables et fonctions. Comprendre "Comment est calculé le stock ?" nécessite de naviguer dans un réseau de dépendances.

### La Solution

Ce projet implémente un **Moteur d'Extraction d'Information Sémantique** qui :

1. **Parse** les scripts pour construire un **Graphe de Dépendances**
2. **Indexe** le code en **chunks enrichis de contexte structurel** 
3. **Raisonne** via un **Agent LLM** utilisant une boucle **ReAct** (Reason + Act)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     ENVISION AGENTIC RAG SYSTEM                                  │
│                                                                                  │
│   ┌───────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐   │
│   │  Scripts  │───▶│  Graph API   │───▶│ Vector DB   │───▶│   Agent LLM    │   │
│   │  (.nvn)   │    │  (NetworkX)  │    │   (FAISS)   │    │  (LangGraph)   │   │
│   └───────────┘    └──────────────┘    └─────────────┘    └────────────────┘   │
│                                                                   │             │
│                           ┌───────────────────────────────────────┘             │
│                           ▼                                                      │
│                   ┌─────────────────┐                                           │
│                   │  User Question  │                                           │
│                   │  "Où est défini │                                           │
│                   │   ReorderPoint?"│                                           │
│                   └─────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Philosophie du Projet

### Pourquoi pas un RAG classique ?

Un RAG standard traite le code comme du texte brut. Ce système va plus loin :

| Approche RAG Classique | Notre Approche "Graph-Aware" |
|------------------------|------------------------------|
| Chunks de texte pur | Chunks enrichis de contexte structurel |
| Pas de notion de dépendances | Graphe de dépendances (imports, reads, writes) |
| Recherche sémantique seule | Recherche sémantique + structurelle + grep |
| LLM en one-shot | Agent en boucle ReAct avec outils |

### Concepts Clés

**1. Graph-Aware Chunking** : Chaque chunk de code inclut un header décrivant ses imports et I/O.

```
[Script: MyScript | Path: /Clean/Processing.nvn]
[Reads: /Data/Items.ion]
[Imports: /Lib/Utils.nvn]

// Contenu du chunk réel...
table Items = read "/Data/Items.ion" with ...
```

**2. Data File Protocol** : Les fichiers `.ion`/`.csv` sont des "boîtes noires". L'agent ne les lit jamais directement, il analyse les scripts qui les utilisent.

**3. Tree of Thoughts** : L'agent décompose les questions complexes en sous-tâches et planifie son exploration.

---

## 🏗 Architecture Globale

### Vue d'Ensemble

```
                          ┌─────────────────────────────────────┐
                          │            USER QUERY               │
                          │  "Comment est calculé le stock ?"    │
                          └───────────────┬─────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               ENVISION COPILOT                                   │
│                          (Agent LangGraph ReAct)                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────────┐ │   │
│  │ │   REASON   │───▶│    ACT     │───▶│  OBSERVE   │───▶│   REASON...    │ │   │
│  │ │  (Thought) │    │  (Tool)    │    │  (Result)  │    │  (Next Step)   │ │   │
│  │ └────────────┘    └────────────┘    └────────────┘    └────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              │ Tools Available:                                  │
│            ┌─────────────────┼─────────────────┬────────────────────┐           │
│            ▼                 ▼                 ▼                    ▼           │
│  ┌─────────────────┐ ┌───────────────┐ ┌────────────────┐ ┌──────────────┐     │
│  │structural_explo │ │ semantic_     │ │  read_file     │ │ grep_search  │     │
│  │     rer         │ │   search      │ │                │ │              │     │
│  │ (Graph Query)   │ │ (RAG Query)   │ │ (File Reader)  │ │ (Regex Scan) │     │
│  └────────┬────────┘ └───────┬───────┘ └───────────────┘ └──────────────┘     │
│           │                  │                                                   │
└───────────┼──────────────────┼───────────────────────────────────────────────────┘
            │                  │
            ▼                  ▼
┌───────────────────┐  ┌──────────────────────────────────────────────────────────┐
│  ENVISION GRAPH   │  │                    CODE RAG                              │
│     (Network)     │  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  ┌─────────────┐  │  │  │   Chunker    │  │    FAISS      │  │  Reranker    │  │
│  │   Nodes     │  │  │  │ (Graph-Aware)│  │   Index       │  │ (CrossEnc.)  │  │
│  │   Edges     │  │  │  └──────────────┘  └───────────────┘  └──────────────┘  │
│  └─────────────┘  │  └──────────────────────────────────────────────────────────┘
└───────────────────┘
         ▲
         │ Built from
         │
┌───────────────────────────────────────┐
│       ENVISION PREPROCESS             │
│  ┌─────────────┐  ┌─────────────────┐ │
│  │   Parser    │  │   Builder       │ │
│  │   (Regex)   │  │   (Network)     │ │
│  └─────────────┘  └─────────────────┘ │
│            │                          │
│            ▼                          │
│  ┌──────────────────────────────────┐ │
│  │  .nvn Scripts (Envision DSL)     │ │
│  │  scripts/*.nvn                   │ │
│  └──────────────────────────────────┘ │
└───────────────────────────────────────┘
```

### Arborescence du Projet

```
llm-DSL-info-extraction0/
├── 📁 src/                         # CODE SOURCE PRINCIPAL
│   ├── 📁 envision_preprocess/     # 👁️ Module de Parsing (The Eyes)
│   │   ├── typedefs.py            # Définitions: Node, Edge, Network
│   │   ├── builder.py             # NetworkBuilder (Regex Parser)
│   │   ├── network.py             # CLI `uv run network`
│   │   ├── api.py                 # EnvisionGraphAPI (Interface)
│   │   ├── utils.py               # ConfigLoader
│   │   └── config.yaml            # Config parsing
│   │
│   ├── 📁 code_rag/                # 🧠 Module RAG (The Memory)
│   │   ├── chunker.py             # GraphChunker (Graph-Aware)
│   │   ├── indexer.py             # GraphIndexer (FAISS Builder)
│   │   ├── retriever.py           # GraphRetriever (Query Engine)
│   │   ├── reranker.py            # Cross-Encoder Reranking
│   │   ├── index.py               # CLI `uv run index`
│   │   └── config.yaml            # Config indexing
│   │
│   ├── 📁 envision_copilot/        # 🤖 Agent Principal (The Brain)
│   │   ├── agent.py               # EnvisionAgent (LangGraph)
│   │   ├── main.py                # CLI `uv run copilot`
│   │   ├── config.yaml            # Prompts & Config Agent
│   │   ├── 📁 tools/              # Outils de l'Agent
│   │   │   ├── structural.py      # Graph exploration
│   │   │   ├── semantic.py        # RAG search
│   │   │   ├── grep.py            # Regex search
│   │   │   └── read_code.py       # File reader
│   │   └── 📁 utils/
│   │       └── config.py          # Config loader
│   │
│   ├── 📁 llms/                    # 🔌 Abstraction LLM (The Interface)
│   │   ├── __init__.py            # Factory get_llm()
│   │   ├── interface.py           # Abstract LLM class
│   │   ├── mistral.py             # Mistral via API
│   │   ├── gemini.py              # Google Gemini
│   │   ├── groq.py                # Groq (Mistral fast)
│   │   └── config.yaml            # API keys & defaults
│   │
│   └── 📁 envision_benchmark/      # 📊 Évaluation (The Judge)
│       ├── runner.py              # BenchmarkRunner
│       ├── main.py                # CLI `uv run benchmark`
│       ├── questions.json         # Golden Dataset (Q&A)
│       └── config.yaml            # Config évaluation
│
├── 📁 scripts/                     # Scripts Envision (.nvn)
│   ├── 68000.nvn                  # Ex: Technical Preprocessing
│   ├── 68001.nvn                  # Ex: Business Processing
│   └── ... (60+ scripts)
│
├── 📁 data/                        # DONNÉES GÉNÉRÉES (gitignored)
│   ├── 📁 network/                # Graphe de dépendances
│   │   ├── network.json           # Nodes + Edges
│   │   └── metadata.json          # Stats & Résolutions
│   └── 📁 vector_store/           # Index vectoriel
│       ├── faiss.index            # Index FAISS
│       └── metadata.json          # Chunks + Métadonnées
│
├── pyproject.toml                  # Config workspace uv
├── uv.lock                         # Lockfile dépendances
└── README.md                       # 📖 Ce fichier
```

---

## 🔄 Pipeline Complète

### Étape 1 : Parsing & Construction du Graphe

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PARSING (envision_preprocess)                                          │
│                                                                                  │
│  ┌─────────────────┐                                                            │
│  │  scripts/*.nvn  │ ─────────────────────────────────────────────────────┐     │
│  └─────────────────┘                                                      │     │
│           │                                                               │     │
│           ▼                                                               │     │
│  ┌───────────────────────────────────────────────────────────────────┐   │     │
│  │                      NetworkBuilder                                │   │     │
│  │                                                                    │   │     │
│  │  ┌──────────────────────────────────────────────────────────────┐ │   │     │
│  │  │  1. REGEX PATTERNS                                            │ │   │     │
│  │  │     • read_pattern:    read "path"                            │ │   │     │
│  │  │     • write_pattern:   write T as "path"                      │ │   │     │
│  │  │     • import_pattern:  import "path"                          │ │   │     │
│  │  │     • table_pattern:   table Name =                           │ │   │     │
│  │  │     • func_pattern:    def/process FuncName                   │ │   │     │
│  │  │     • const_pattern:   const Name = "value"                   │ │   │     │
│  │  └──────────────────────────────────────────────────────────────┘ │   │     │
│  │                              │                                     │   │     │
│  │                              ▼                                     │   │     │
│  │  ┌──────────────────────────────────────────────────────────────┐ │   │     │
│  │  │  2. PLACEHOLDER RESOLUTION                                    │ │   │     │
│  │  │     • "\{filestorage}Items.ion" → "/Input/Items.ion"          │ │   │     │
│  │  │     • Recursive resolution with max_depth=10                  │ │   │     │
│  │  └──────────────────────────────────────────────────────────────┘ │   │     │
│  │                              │                                     │   │     │
│  │                              ▼                                     │   │     │
│  │  ┌──────────────────────────────────────────────────────────────┐ │   │     │
│  │  │  3. GLOB PATTERN RESOLUTION                                   │ │   │     │
│  │  │     • "/Data/*.ion" → ["/Data/A.ion", "/Data/B.ion"]         │ │   │     │
│  │  │     • fnmatch matching against known file nodes              │ │   │     │
│  │  └──────────────────────────────────────────────────────────────┘ │   │     │
│  └───────────────────────────────────────────────────────────────────┘   │     │
│                              │                                            │     │
│                              ▼                                            │     │
│  ┌────────────────────────────────────────────────────────────────────────┘     │
│  │                                                                              │
│  ▼                                                                              │
│ ┌───────────────────────────────────────────────────────────────────────────┐  │
│ │                         NETWORK (Graphe)                                   │  │
│ │                                                                            │  │
│ │   NODES:                           EDGES:                                  │  │
│ │   ┌──────────────────────┐        ┌─────────────────────────────────────┐ │  │
│ │   │ script: 68000        │───────▶│ 68000 --[reads]--→ /Clean/Items.ion │ │  │
│ │   │ script: 68001        │        │ 68000 --[imports]--→ 68001          │ │  │
│ │   │ file: /Clean/Items   │        │ 68000 --[writes]--→ /Export/Out.ion │ │  │
│ │   │ table: Items         │        │ 68000 --[defines]--→ 68000::func::X │ │  │
│ │   │ function: MyFunc     │        └─────────────────────────────────────┘ │  │
│ │   │ var: myConst         │                                                 │  │
│ │   └──────────────────────┘                                                 │  │
│ └───────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                   │
│                              ▼                                                   │
│              ┌──────────────────────────────────┐                               │
│              │  data/network/network.json       │                               │
│              │  data/network/metadata.json      │                               │
│              └──────────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Étape 2 : Indexation Vectorielle

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: INDEXATION (code_rag)                                                  │
│                                                                                  │
│  ┌──────────────────────┐    ┌────────────────────────────────────────────────┐ │
│  │  network.json        │───▶│              GraphChunker                       │ │
│  │  (Nodes + Edges)     │    │                                                 │ │
│  └──────────────────────┘    │  ┌───────────────────────────────────────────┐ │ │
│                              │  │  1. BUILD NEIGHBORHOOD MAP                 │ │ │
│                              │  │     For each script, collect:              │ │ │
│                              │  │     • Incoming edges (who imports me?)     │ │ │
│                              │  │     • Outgoing edges (what do I import?)   │ │ │
│                              │  └───────────────────────────────────────────┘ │ │
│                              │                     │                          │ │
│                              │                     ▼                          │ │
│                              │  ┌───────────────────────────────────────────┐ │ │
│                              │  │  2. GENERATE CONTEXT HEADER                │ │ │
│                              │  │                                            │ │ │
│                              │  │  [Script: MyScript | Path: /Clean/Proc]   │ │ │
│                              │  │  [Reads: /Data/Items.ion]                 │ │ │
│                              │  │  [Imports: /Lib/Utils.nvn]                │ │ │
│                              │  └───────────────────────────────────────────┘ │ │
│                              │                     │                          │ │
│                              │                     ▼                          │ │
│                              │  ┌───────────────────────────────────────────┐ │ │
│                              │  │  3. SLIDING WINDOW CHUNKING                │ │ │
│                              │  │                                            │ │ │
│                              │  │  chunk_size: 512 tokens                   │ │ │
│                              │  │  overlap: 50 tokens (5 lines kept)        │ │ │
│                              │  │  header: prepended to each chunk          │ │ │
│                              │  └───────────────────────────────────────────┘ │ │
│                              └─────────────────────────────────────────────────┘ │
│                                              │                                   │
│                                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                              GraphIndexer                                    ││
│  │                                                                              ││
│  │  ┌──────────────────┐    ┌───────────────────┐    ┌────────────────────┐   ││
│  │  │ SentenceTransfor│───▶│   EMBEDDINGS       │───▶│    FAISS Index     │   ││
│  │  │ mers (MiniLM)    │    │   (384 dims)       │    │  (IndexFlatL2)     │   ││
│  │  └──────────────────┘    └───────────────────┘    └────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                              │                                   │
│                                              ▼                                   │
│              ┌──────────────────────────────────────────────────────────┐       │
│              │  data/vector_store/faiss.index                            │       │
│              │  data/vector_store/metadata.json (chunks + sources)       │       │
│              └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Étape 3 : Requêtage & Agent

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: QUERY (envision_copilot)                                               │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  USER QUERY: "Comment est calculé le Reorder Point ?"                     │   │
│  └────────────────────────────────────┬─────────────────────────────────────┘   │
│                                       │                                          │
│                                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                        EnvisionAgent (LangGraph)                          │   │
│  │                                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │  │  ReAct Loop (Reason → Act → Observe → Reason...)                    │ │   │
│  │  │                                                                      │ │   │
│  │  │  ITERATION 1:                                                        │ │   │
│  │  │  ┌─────────────────────────────────────────────────────────────────┐│ │   │
│  │  │  │ Thought: Je dois chercher "Reorder Point" dans le code.         ││ │   │
│  │  │  │          C'est une question sémantique → semantic_search.       ││ │   │
│  │  │  │                                                                  ││ │   │
│  │  │  │ Action: semantic_search("Reorder Point calculation")            ││ │   │
│  │  │  │                                                                  ││ │   │
│  │  │  │ Observation: Found 3 results:                                   ││ │   │
│  │  │  │   - /Clean/Inventory.nvn (lines 120-150, score: 0.89)          ││ │   │
│  │  │  │   - /Clean/Replenish.nvn (lines 45-80, score: 0.76)            ││ │   │
│  │  │  └─────────────────────────────────────────────────────────────────┘│ │   │
│  │  │                                                                      │ │   │
│  │  │  ITERATION 2:                                                        │ │   │
│  │  │  ┌─────────────────────────────────────────────────────────────────┐│ │   │
│  │  │  │ Thought: Le meilleur résultat est Inventory.nvn. Je vais lire   ││ │   │
│  │  │  │          les lignes 100-170 pour avoir le contexte.            ││ │   │
│  │  │  │                                                                  ││ │   │
│  │  │  │ Action: read_file("/Clean/Inventory.nvn", 100, 170)             ││ │   │
│  │  │  │                                                                  ││ │   │
│  │  │  │ Observation: [Code showing ReorderPoint = LeadTime * ...]       ││ │   │
│  │  │  └─────────────────────────────────────────────────────────────────┘│ │   │
│  │  │                                                                      │ │   │
│  │  │  ITERATION 3:                                                        │ │   │
│  │  │  ┌─────────────────────────────────────────────────────────────────┐│ │   │
│  │  │  │ Thought: J'ai trouvé la formule. Je peux répondre.              ││ │   │
│  │  │  │                                                                  ││ │   │
│  │  │  │ Final Answer: Le Reorder Point est calculé dans                 ││ │   │
│  │  │  │   /Clean/Inventory.nvn à la ligne 135:                          ││ │   │
│  │  │  │   ReorderPoint = LeadTime * AvgDailyDemand + SafetyStock        ││ │   │
│  │  │  └─────────────────────────────────────────────────────────────────┘│ │   │
│  │  └─────────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Modules et Packages

### 1. `envision_preprocess` — 👁️ The Eyes

**Rôle** : Parser les scripts Envision et construire le graphe de dépendances.

| Fichier | Description |
|---------|-------------|
| `typedefs.py` | Dataclasses `Node`, `Edge`, `Network` avec types énumérés |
| `builder.py` | `NetworkBuilder` : parsing regex, résolution placeholders, construction graphe |
| `network.py` | CLI pour `uv run network --build/--stats/--query` |
| `api.py` | `EnvisionGraphAPI` : interface programmatique pour requêter le graphe |
| `utils.py` | `ConfigLoader` : chargement config YAML |

**Patterns Regex Utilisés** :

```python
read_pattern   = r'read\s+["\']([^"\']+)["\']'          # read "path"
write_pattern  = r'write\s+\w+\s+as\s+["\']([^"\']+)["\']'  # write T as "path"
import_pattern = r'import\s+["\']([^"\']+)["\']'        # import "path"
table_pattern  = r'table\s+(\w+)\s*='                   # table Name =
func_pattern   = r'(?:process|def)\s+([^{=(]+)'        # def/process Func
const_pattern  = r'const\s+(\w+)\s*=\s*"(.*)"'         # const X = "val"
```

---

### 2. `code_rag` — 🧠 The Memory

**Rôle** : Indexer le code en chunks contextualisés et fournir la recherche sémantique.

| Fichier | Description |
|---------|-------------|
| `chunker.py` | `GraphChunker` : découpage en chunks avec header de contexte structurel |
| `indexer.py` | `GraphIndexer` : embedding via SentenceTransformers + FAISS index |
| `retriever.py` | `GraphRetriever` : recherche (recall + reranking) |
| `reranker.py` | `Reranker` : Cross-Encoder pour affiner les résultats |
| `index.py` | CLI pour `uv run index --build` |

**Paramètres Chunker** :

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `chunk_size` | 512 | Nombre max de tokens (~mots) par chunk |
| `overlap` | 50 | Tokens de chevauchement entre chunks |
| `block_keywords` | `[read, write, def, process, ...]` | Mots-clés DSL pour détection de blocs |

**Pipeline Retrieval** :

```
Query → [Bi-Encoder] → Top-50 Candidates → [Cross-Encoder] → Top-5 Results
           ↓                                    ↓
    all-MiniLM-L6-v2                   ms-marco-MiniLM-L-6-v2
```

---

### 3. `envision_copilot` — 🤖 The Brain

**Rôle** : Agent orchestrateur utilisant LangGraph pour raisonner et exécuter des outils.

| Fichier | Description |
|---------|-------------|
| `agent.py` | `EnvisionAgent` : StateGraph LangGraph avec boucle ReAct |
| `main.py` | CLI `uv run copilot -i/-q` |
| `config.yaml` | Prompts système, instructions, descriptions outils |
| `tools/structural.py` | `StructuralTools` : exploration du graphe |
| `tools/semantic.py` | `SemanticTools` : recherche RAG |
| `tools/grep.py` | `GrepTools` : recherche regex dans le contenu |
| `tools/read_code.py` | `CodeReader` : lecture de fichiers |

**Outils de l'Agent** :

| Outil | Usage | Arguments |
|-------|-------|-----------|
| `semantic_search` | Questions de sens ("Comment fonctionne X?") | `query: str, top_k: int` |
| `structural_explorer` | Questions structurelles ("Qui importe X?") | `action: str, node_id: str, relation_type: str` |
| `read_file` | Lire un script spécifique | `path: str, start_line: int, end_line: int` |
| `grep_search` | Recherche regex quand RAG échoue | `pattern: str, node_type: str` |

**State Machine (LangGraph)** :

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  START  │────▶│ REASON  │────▶│   ACT   │
└─────────┘     └────┬────┘     └────┬────┘
                     │               │
                     │    Loop       │
                     ◀───────────────┘
                     │
                     │ Final Answer?
                     ▼
                ┌─────────┐
                │   END   │
                └─────────┘
```

---

### 4. `llms` — 🔌 The Interface

**Rôle** : Abstraction unifiée pour différents providers LLM.

| Fichier | Description |
|---------|-------------|
| `__init__.py` | Factory `get_llm(provider)` |
| `interface.py` | Classe abstraite `LLM` |
| `mistral.py` | `MistralLLM` via API Mistral |
| `gemini.py` | `GeminiLLM` via Google AI |
| `groq.py` | `GroqLLM` (Mistral accéléré) |

**Usage** :

```python
from llms import get_llm

llm = get_llm("mistral")  # ou "gemini", "groq"
response = llm.generate("Explique le code suivant...")
```

---

### 5. `envision_benchmark` — 📊 The Judge

**Rôle** : Évaluation automatisée de l'agent via LLM-as-a-Judge.

| Fichier | Description |
|---------|-------------|
| `runner.py` | `BenchmarkRunner` : exécution des tests + scoring |
| `main.py` | CLI `uv run benchmark` |
| `questions.json` | Dataset de questions avec topics attendus |

**Format Questions** :

```json
{
  "id": 1,
  "question": "Comment est calculé le stock disponible?",
  "expected_topics": ["StockOnHand", "Inventory", "calculation"]
}
```

---

## 📐 Structures de Données

### Node (Nœud du Graphe)

```python
@dataclass
class Node:
    id: str              # Identifiant unique (ex: "68000" ou "/Clean/Items.ion")
    type: NodeType       # SCRIPT | FILE | TABLE | VAR | FUNCTION
    name: str            # Nom court (ex: "Items.nvn")
    path: str            # Chemin logique (ex: "/Clean/Processing")
    content: str         # Contenu complet (pour scripts/fonctions)
    start_line: int      # Ligne de début (pour fonctions/vars)
    end_line: int        # Ligne de fin
    metadata: Dict       # Métadonnées additionnelles (docs, qualifiers...)
```

**NodeType Enum** :

| Type | Description | Exemple ID |
|------|-------------|------------|
| `SCRIPT` | Script Envision | `68000` |
| `FILE` | Fichier de données | `/Clean/Items.ion` |
| `TABLE` | Table définie | `68000::table::Items` |
| `VAR` | Variable/Constante | `68000::const::exportPath` |
| `FUNCTION` | Fonction définie | `68000::func::MyProcess` |

---

### Edge (Arête du Graphe)

```python
@dataclass
class Edge:
    source: str          # ID du nœud source
    target: str          # ID du nœud cible
    type: EdgeType       # READS | WRITES | IMPORTS | DEFINES | USES | EXPORT
    metadata: Dict       # count, occurrences, raw path...
```

**EdgeType Enum** :

| Type | Description | Exemple |
|------|-------------|---------|
| `READS` | Script lit un fichier | `68000 --[reads]→ /Clean/Items.ion` |
| `WRITES` | Script écrit un fichier | `68000 --[writes]→ /Export/Out.ion` |
| `IMPORTS` | Script importe un autre | `68000 --[imports]→ 68001` |
| `DEFINES` | Script définit une entité | `68000 --[defines]→ 68000::func::X` |
| `EXPORT` | Script exporte un schéma | `68000 --[export]→ /Schema/S.ion` |

---

### Chunk (Unité pour RAG)

```python
{
    "id": "68000_40_80",           # Format: nodeId_startLine_endLine
    "source_id": "68000",          # ID du script source
    "source": "/Clean/Processing", # Chemin logique complet
    "text": "...",                 # Texte complet (header + body)
    "content": "...",              # Corps seul (pour affichage)
    "context": "[Script: ...]\n[Reads: ...]",  # Header contextuel
    "lines": "40-80"               # Lignes couvertes
}
```

---

### AgentState (État de l'Agent)

```python
class AgentState(TypedDict):
    question: str        # Question utilisateur
    messages: List[str]  # Historique des réponses
    scratchpad: str      # Mémoire de travail (trace ReAct)
    final_answer: str    # Réponse finale
    step_count: int      # Compteur d'itérations
    facts: List[Any]     # Observations collectées
    plan: List[str]      # Plan de sous-tâches
```

---

## 🚀 Installation et Prise en Main

### Prérequis

- **Linux/MacOS** (Windows via WSL)
- **Python 3.10+**
- **[uv](https://github.com/astral-sh/uv)** (gestionnaire de packages rapide)

### Installation

```bash
# 1. Cloner le repository
git clone <repository-url>
cd llm-DSL-info-extraction0

# 2. Installer les dépendances (via uv)
uv sync --reinstall

# 3. Configurer les clés API
cp src/envision_copilot/.env.example src/envision_copilot/.env
# Éditer .env avec vos clés MISTRAL_API_KEY, GROQ_API_KEY, etc.
```

### Vérification de l'Installation

```bash
# Vérifier que les CLI sont disponibles
uv run network --help
uv run index --help
uv run copilot --help
uv run benchmark --help
```

---

## 💻 Utilisation Détaillée

### 1. Construction de la Base de Connaissances

```bash
# Étape 1: Construire le graphe de dépendances
uv run network --build

# Output attendu:
# 🔍 NetworkBuilder: Scanning 60 files in scripts...
# ✨ Resolved 5 glob patterns to concrete files.
# 🔄 Resolved 42 placeholder cascades.
# ✅ Network saved to data/network/network.json
# ✅ Metadata saved to data/network/metadata.json

# Étape 2: Construire l'index vectoriel
uv run index --build

# Output attendu:
# 📦 Loading Embedding Model: sentence-transformers/all-MiniLM-L6-v2
# 📂 Loading Graph from data/network/network.json...
# 🔗 Building Neighborhood Map...
# 🔄 Generating Graph-Aware Chunks...
# 📊 Generated 320 chunks from 60 scripts.
# 🧠 Computing Embeddings...
# 🗂️ Building FAISS Index (Dim: 384)...
# ✅ Index saved.
```

### 2. Inspection du Graphe

```bash
# Afficher les statistiques globales
uv run network --stats

# Lister les scripts
uv run network --stats --type script

# Lister les edges de type "imports"
uv run network --stats --edge-type imports

# Inspecter un nœud spécifique
uv run network --query 68000

# Rechercher un nœud
uv run network --query "Items" --find

# Voir les globs résolus
uv run network --globs

# Voir les cascades de placeholders
uv run network --cascades
```

### 3. Interagir avec l'Agent

```bash
# Mode interactif (chat)
uv run copilot -i

# Query one-shot
uv run copilot -q "Comment est calculé le stock disponible?"

# Mode verbose (affiche les thoughts/actions)
uv run copilot -q "Où est défini ReorderPoint?" --verbose
```

### 4. Évaluation / Benchmark

```bash
# Exécuter le benchmark (5 questions par défaut)
uv run benchmark

# Résultats sauvegardés dans:
# data/logs/benchmark_report.json
```

---

## 🔬 Simulation de Fonctionnement

### Scénario : "Comment est calculé le Reorder Point ?"

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SIMULATION COMPLÈTE                                       │
│                                                                                  │
│  User: "Comment est calculé le Reorder Point ?"                                  │
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║ ITERATION 1                                                                ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║ 🧠 THOUGHT:                                                                ║  │
│  ║   Cette question porte sur la SIGNIFICATION d'un calcul.                   ║  │
│  ║   Je dois utiliser semantic_search pour trouver le code pertinent.        ║  │
│  ║   Query: "Reorder Point calculation inventory replenishment"               ║  │
│  ║                                                                            ║  │
│  ║ 🛠️  ACTION:                                                                 ║  │
│  ║   semantic_search("Reorder Point calculation inventory", top_k=5)         ║  │
│  ║                                                                            ║  │
│  ║ 👀 OBSERVATION:                                                            ║  │
│  ║   ┌────────────────────────────────────────────────────────────────────┐  ║  │
│  ║   │ Results:                                                            │  ║  │
│  ║   │   1. source: /Clean/Inventory.nvn, lines: 120-150, score: 0.89     │  ║  │
│  ║   │      context: [Script: Inventory | Reads: /Data/Items.ion]         │  ║  │
│  ║   │   2. source: /Clean/Replenish.nvn, lines: 45-80, score: 0.76       │  ║  │
│  ║   │   3. source: /Optim/Reorder.nvn, lines: 200-230, score: 0.71       │  ║  │
│  ║   └────────────────────────────────────────────────────────────────────┘  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║ ITERATION 2                                                                ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║ 🧠 THOUGHT:                                                                ║  │
│  ║   Le meilleur résultat est /Clean/Inventory.nvn (score 0.89).             ║  │
│  ║   Je dois lire les lignes 100-170 pour avoir le contexte complet.        ║  │
│  ║   (Règle: étendre de ±20 lignes autour du résultat)                       ║  │
│  ║                                                                            ║  │
│  ║ 🛠️  ACTION:                                                                 ║  │
│  ║   read_file("/Clean/Inventory.nvn", 100, 170)                              ║  │
│  ║                                                                            ║  │
│  ║ 👀 OBSERVATION:                                                            ║  │
│  ║   ┌────────────────────────────────────────────────────────────────────┐  ║  │
│  ║   │ /// Calcul du Reorder Point                                        │  ║  │
│  ║   │ // Le Reorder Point détermine quand déclencher une commande       │  ║  │
│  ║   │                                                                     │  ║  │
│  ║   │ ReorderPoint = LeadTime * AvgDailyDemand + SafetyStock             │  ║  │
│  ║   │                                                                     │  ║  │
│  ║   │ // Où:                                                              │  ║  │
│  ║   │ //   LeadTime = délai fournisseur (jours)                          │  ║  │
│  ║   │ //   AvgDailyDemand = demande moyenne journalière                  │  ║  │
│  ║   │ //   SafetyStock = stock de sécurité                               │  ║  │
│  ║   └────────────────────────────────────────────────────────────────────┘  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║ ITERATION 3                                                                ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════╣  │
│  ║ 🧠 THOUGHT:                                                                ║  │
│  ║   J'ai trouvé la formule exacte et sa documentation.                       ║  │
│  ║   Je peux maintenant fournir une réponse complète.                         ║  │
│  ║                                                                            ║  │
│  ║ ✅ FINAL ANSWER:                                                           ║  │
│  ║   Le **Reorder Point** est calculé dans `/Clean/Inventory.nvn` à la       ║  │
│  ║   ligne 135 avec la formule :                                              ║  │
│  ║                                                                            ║  │
│  ║   ```                                                                      ║  │
│  ║   ReorderPoint = LeadTime * AvgDailyDemand + SafetyStock                   ║  │
│  ║   ```                                                                      ║  │
│  ║                                                                            ║  │
│  ║   **Composants:**                                                          ║  │
│  ║   - `LeadTime`: Délai fournisseur en jours                                ║  │
│  ║   - `AvgDailyDemand`: Demande moyenne journalière                         ║  │
│  ║   - `SafetyStock`: Stock de sécurité pour absorber les variations         ║  │
│  ║                                                                            ║  │
│  ║   Ce calcul est utilisé pour déterminer le seuil de déclenchement         ║  │
│  ║   des commandes de réapprovisionnement.                                    ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration Avancée

### Configuration du Parsing (`envision_preprocess/config.yaml`)

```yaml
parsing:
  script_dir: "scripts"           # Répertoire des scripts .nvn
  script_ext: "nvn"               # Extension des fichiers
  mapping_file: "src/envision_preprocess/mapping.txt"  # Mapping ID → Path
  recursion_limit: 10             # Profondeur max résolution placeholders

output:
  network_file: "data/network/network.json"
  metadata_file: "data/network/metadata.json"
  snippet_lines: 10               # Lignes pour preview CLI
```

### Configuration du RAG (`code_rag/config.yaml`)

```yaml
indexing:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"  # Modèle embedding
  chunk_size: 512                 # Tokens par chunk
  overlap: 50                     # Chevauchement entre chunks
  block_keywords:                 # Mots-clés DSL
    - "read"
    - "write"
    - "def"
    - "process"
    - "table"

retrieval:
  reranker_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k_recall: 50                # Candidats initiaux (FAISS)
  top_k_final: 5                  # Résultats après reranking
  use_reranker: true              # Activer le reranking

input:
  network_file: "data/network/network.json"

output:
  index_file: "data/vector_store/faiss.index"
  metadata_file: "data/vector_store/metadata.json"
```

### Configuration de l'Agent (`envision_copilot/config.yaml`)

```yaml
agent:
  main_model: "mistral"           # mistral | gemini | groq
  constraints:
    max_iterations: 10            # Stop forcé après N itérations
    max_depth: 7                  # Profondeur Tree of Thoughts
    max_branches: 2               # Branches parallèles

prompts:
  system: |
    You are Envision Copilot, an expert pair-programmer...
  
  instructions: |
    ### CRITICAL: Choose the Right Tool FIRST
    - STRUCTURAL Questions → structural_explorer
    - SEMANTIC Questions → semantic_search
    - DATA FILE PROTOCOL: Never read .ion files directly!
```

---

## 📜 Protocoles et Conventions

### Data File Protocol (Fichiers `.ion`, `.csv`)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  DATA FILE PROTOCOL                                                             │
│                                                                                 │
│  Les fichiers de données (.ion, .csv) sont des "BOÎTES NOIRES".                │
│                                                                                 │
│  ❌ L'agent NE DOIT JAMAIS:                                                     │
│     - Lire directement un fichier .ion/.csv                                    │
│     - Vérifier si un fichier de données existe                                 │
│     - Analyser le contenu brut des données                                     │
│                                                                                 │
│  ✅ L'agent DOIT:                                                               │
│     - Trouver les SCRIPTS qui lisent ces fichiers                              │
│     - Analyser les statements `read "file.ion" with ...` pour comprendre       │
│       la structure des données                                                  │
│     - FAIRE CONFIANCE aux références (si un script lit X.ion, X.ion existe)   │
│                                                                                 │
│  RAISON: Les fichiers de données sont volumineux et tokeniseraient             │
│          inutilement le contexte LLM. La logique est dans les SCRIPTS.          │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Convention de Nommage des Nodes

| Type | Format ID | Exemple |
|------|-----------|---------|
| Script | `<file_id>` | `68000` |
| File (data) | `<logical_path>` | `/Clean/Items.ion` |
| Table | `<script_id>::table::<name>` | `68000::table::Catalog` |
| Function | `<script_id>::func::<name>` | `68000::func::ProcessBalance` |
| Variable | `<script_id>::const::<name>` | `68000::const::exportPath` |

### Résolution des Placeholders

```
Entrée:  read "\{filestorage}Catalog/Catalog*"
          où filestorage = "\{testStorage}/Input/"
          où testStorage = ""

Étape 1: "\{testStorage}/Input/" → "/Input/"
Étape 2: "\{filestorage}Catalog/Catalog*" → "/Input/Catalog/Catalog*"
Étape 3: Glob resolution → ["/Input/Catalog/Catalog2024.ion", ...]
```

---

## 👥 Contribuer

1. **Explorer le code** : Commencer par `src/envision_copilot/agent.py` pour comprendre la boucle agent
2. **Tester localement** : `uv run copilot -i --verbose` pour voir les traces
3. **Ajouter des outils** : Créer un fichier dans `src/envision_copilot/tools/`
4. **Améliorer les prompts** : Éditer `src/envision_copilot/config.yaml`

---

## 📄 License

Internal / Proprietary — Lokad R&D Project.

---

<div align="center">

**Built with 🧠 by the Lokad AI Team**

*"Understanding code is understanding the supply chain."*

</div>
