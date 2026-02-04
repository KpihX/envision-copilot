# Envision Copilot (`src/envision_copilot`)

**Le "Cerveau" du système — Assistant IA Agentique.**

Ce module implémente un assistant capable d'investiguer, expliquer et débugger la codebase Envision DSL via un pipeline multi-agents avec scaffolding cognitif.

---

## 🚀 Quick Start

```bash
# Mode interactif (recommandé) - conversation continue
uv run copilot -i

# Query unique (one-shot)
uv run copilot -q "Où est défini le ReDispatchCycle ?"

# Avec détails du raisonnement
uv run copilot -v -q "Combien de scripts lisent Items.ion ?"
```

---

## 📋 CLI Reference

```
uv run copilot [OPTIONS]

Options:
  -q, --query TEXT      Question unique (one-shot mode)
  -i, --interactive     Mode conversation continue
  -v, --verbose         Afficher le processus de raisonnement
  --debug               Afficher les prompts bruts et réponses LLM
  --max-depth INT       Limite d'itérations (défaut: 10)
```

### Exemples de Questions

```bash
# Questions structurelles (graphe)
uv run copilot -q "Quels scripts lisent /Clean/Items.ion ?"
uv run copilot -q "Quels modules sont importés par le script 68006 ?"

# Questions sémantiques (RAG)
uv run copilot -q "Comment est calculé le stock de sécurité ?"
uv run copilot -q "Où paramétrer le fill rate cible ?"

# Questions hybrides
uv run copilot -q "Quelle est la différence entre Standard et Long Term Purchase ?"
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
agent:
  llm_type: "qwen"              # Provider: mistral, qwen, gemini, groq
  llm_model: "qwen3-max"        # Modèle spécifique
  constraints:
    max_depth: 10               # Itérations max du Thinker
    max_branches: 2             # Actions parallèles par itération
    recursion_limit: 50         # Limite LangGraph

presentation:
  max_lines: 200                # Lignes max affichées par outil
  max_items: 30                 # Items max dans les listes
  debug: true                   # Afficher contexte debug
```

### Mode Lite API (économie de tokens)

Configuré dans `envision_preprocess/config.yaml` :
```yaml
api:
  mode: "lite"    # Réponses optimisées pour LLM (-80% tokens)
```

---

## 🏗️ Architecture

```
┌─────────┐     ┌─────────┐     ┌─────────────┐
│ Starter │ ──▶ │ Thinker │ ──▶ │ Synthesizer │
│ (Gate)  │     │ (Loop)  │     │  (Answer)   │
└─────────┘     └────┬────┘     └─────────────┘
                     │ ▲
                     ▼ │
                ┌─────────┐
                │  Tools  │
                └─────────┘
```

### Agents

| Agent | Rôle | Innovation Clé |
|-------|------|----------------|
| **Starter** | Triage & traduction | Filtre les hors-sujet, normalise en anglais |
| **Thinker** | Raisonne, planifie, explore | Reçoit `last_thought_process` anti-boucle |
| **Synthesizer** | Génère la réponse finale | Construit depuis la mémoire curée uniquement |

---

## 🛠️ Outils Disponibles

### `graph` — Navigation Structurelle
```json
{"action": "tree", "domain": "scripts"}
{"action": "neighbors", "node_id": "68006", "direction": "outgoing", "relation_type": "reads"}
```

| Action | Usage |
|--------|-------|
| `tree` | Arborescence des dossiers |
| `node` | Détails d'un nœud par ID |
| `neighbors` | Voisins (incoming/outgoing/siblings) |
| `search` | Recherche par nom |

### `rag` — Recherche Sémantique
```json
{"query": "comment calculer le stock"}
```
Retourne des **Matches** avec contexte et chemin source.

### `reader` — Lecture de Code
```json
{"node_id": "68006"}
```
⚠️ Utilise l'**ID numérique** du script, pas le chemin !

### `grep` — Recherche Pattern
```json
{"pattern": "StockEvol", "node_types": ["script", "function"]}
```
Pour trouver des constantes, variables locales, strings exactes.

---

## 🔄 Mécanismes Anti-Échec

### 1. Anti-Loop (injection `last_thought_process`)
Le Thinker voit son raisonnement précédent pour éviter de répéter les mêmes actions.

### 2. Stopping Rule (Checklist Stratégique)
Avant de s'arrêter, le Thinker doit avoir essayé :

**Approche STRUCTURELLE** :
- [ ] Explorer l'arbre avec `graph.tree`
- [ ] Lire les scripts pertinents avec `reader`
- [ ] Tracer les dépendances avec `graph.neighbors`

**Approche SÉMANTIQUE** :
- [ ] Chercher avec `rag` (termes originaux)
- [ ] Chercher avec `rag` (synonymes)
- [ ] Utiliser `grep` pour patterns exacts

### 3. Memory Protocol (Sélection Granulaire)
Le Thinker sélectionne explicitement quels résultats garder en mémoire :
```json
{
  "add_result_indices": {
    "0": [1, 3]    // Tool 0, garder chunks 1 et 3
  }
}
```

---

## 📁 Structure du Package

```
envision_copilot/
├── main.py              # CLI entry point
├── config.yaml          # Prompts & configuration
├── core/
│   ├── main.py          # EnvisionCopilot orchestrator
│   └── agents/
│       ├── starter.py   # Agent gatekeeper
│       ├── thinker.py   # Agent raisonneur
│       └── synthesizer.py
├── tools/
│   ├── definitions.py   # Documentation outils (avec tables direction)
│   ├── graph.py         # Tool graph
│   ├── rag.py           # Tool semantic search
│   ├── reader.py        # Tool code reader
│   └── grep.py          # Tool pattern search
└── utils/
    └── prompt_loader.py # Gestion des prompts
```

---

## 💡 Philosophie

> **"Prompt engineering appliqué à l'architecture d'agents"**
> **"Tout raisonneur peut performer si l'environnement est bien scaffoldé"**

Le Copilot maximise la qualité de raisonnement avec des LLMs modestes via :
- **Documentation riche des outils** avec tables et exemples
- **Injection anti-boucle** du raisonnement précédent
- **Stopping rule explicite** pour éviter l'abandon prématuré
- **Mode lite** pour économiser les tokens

---

## 📚 Pour Aller Plus Loin

- **Architecture détaillée** : voir [GEMINI.md](../../GEMINI.md)
- **Graphe de dépendances** : voir [envision_preprocess/README.md](../envision_preprocess/README.md)
- **Index sémantique** : voir [code_rag/README.md](../code_rag/README.md)


