## 🏗️ Architecture générale

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         🧠 ARCHITECTURE AGENTIQUE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│     ┌──────────────┐                                                        │
│     │  Question    │                                                        │
│     └──────┬───────┘                                                        │
│            ▼                                                                │
│     ┌──────────────┐     "Quelle info me manque ?"                         │
│     │   PLANNER    │◄──────────────────────────────┐                        │
│     │   (LLM)      │                               │                        │
│     └──────┬───────┘                               │                        │
│            │ Choix d'outil                         │                        │
│            ▼                                       │                        │
│     ┌──────────────────────────────────────┐      │                        │
│     │         TOOL ROUTER                   │      │                        │
│     └─────┬──────┬──────┬──────┬───────────┘      │                        │
│           │      │      │      │                   │                        │
│           ▼      ▼      ▼      ▼                   │                        │
│     ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐               │                        │
│     │ RAG │ │GREP │ │FIND │ │REGEN│               │                        │
│     └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘               │                        │
│        │       │       │       │                   │                        │
│        └───────┴───────┴───────┘                   │                        │
│                    │                               │                        │
│                    ▼                               │                        │
│           ┌────────────────┐                       │                        │
│           │  DISTILLATION  │  "Résumer ce que     │                        │
│           │     (LLM)      │   j'ai trouvé"       │                        │
│           └───────┬────────┘                       │                        │
│                   │                                │                        │
│                   ▼                                │                        │
│           ┌────────────────┐                       │                        │
│           │ KNOWLEDGE BANK │ ◄───── Faits vérifiés│                        │
│           └───────┬────────┘                       │                        │
│                   │                                │                        │
│                   ▼                                │  Boucle si             │
│           ┌────────────────┐                       │  insuffisant           │
│           │    SOLVER      │───────────────────────┘                        │
│           │    (LLM)       │                                                │
│           └───────┬────────┘                                                │
│                   │                                                         │
│                   ▼                                                         │
│           ┌────────────────┐                                                │
│           │ Réponse Finale │                                                │
│           └────────────────┘                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🏗️ Architecture en 4 Couches

┌─────────────────────────────────────────────────────────────┐
│  🎮 INTERFACE (main.py)                                     │
│  Mode interactif | Query unique | Benchmark                 │
├─────────────────────────────────────────────────────────────┤
│  🕸️ WORKFLOW (langgraph_base.py)                           │
│  Router → Retrieve → Engineer → Generate → Grade            │
├─────────────────────────────────────────────────────────────┤
│  🔄 RAG PIPELINE (rag/)                                     │
│  Parser → Chunker → Embedder → Retriever                    │
├─────────────────────────────────────────────────────────────┤
│  🤖 AGENTS (agents/)                                        │
│  Gemini | GPT | Mistral | Groq | Llama3                     │
└─────────────────────────────────────────────────────────────┘



## 🔄 Flux d'Exécution Détaillé

```
# Le Planner reçoit la question et l'historique
# Il retourne un choix structuré en XML :

<thought>
Je dois trouver où est définie la variable "Sales.Total".
Un grep serait plus efficace qu'une recherche sémantique.
</thought>
<tool>grep_tool</tool>
<parameter>Sales\.Total\s*=</parameter>
```


┌─────────────────────────────────────────────────────────────┐
│                    🛠️ OUTILS DISPONIBLES                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📚 RAG_TOOL                                                │
│  └─ Recherche sémantique via embeddings FAISS              │
│  └─ Usage: Concepts, logique métier, documentation         │
│  └─ Exemple: "comment fonctionne le calcul de prévision"   │
│                                                             │
│  🔍 GREP_TOOL                                               │
│  └─ Recherche par regex dans le code                       │
│  └─ Usage: Variables, fichiers, patterns exacts            │
│  └─ Exemple: "read \"/Clean/Items.ion\""                   │
│  └─ Option: `<sources>`file1.nvn, file2.nvn `</sources>`        │
│                                                             │
│  📄 SCRIPT_FINDER_TOOL                                      │
│  └─ Lit le contenu complet d'un fichier                    │
│  └─ Usage: Analyse approfondie d'un script spécifique      │
│  └─ Exemple: "forecasting.nvn, inventory.nvn"              │
│                                                             │
│  🔄 SIMPLE_REGENERATION_TOOL                                │
│  └─ Relance la réflexion sans nouvel outil                 │
│  └─ Usage: Corriger une erreur de raisonnement             │
│                                                             │
│  ✅ GRADE_ANSWER                                            │
│  └─ Termine le processus, la réponse est satisfaisante     │
│                                                             │
└─────────────────────────────────────────────────────────────┘


```python
# Entrée : 5 chunks de code récupérés
items_to_distill = [
    ("Sales.Total = Qty * Price where ...", "/scripts/sales.nvn"),
    ("Sales.Total used in report ...", "/scripts/report.nvn"),
]

# Sortie : Faits concis
distilled_facts = [
    ("Sales.Total est calculé comme Qty * Price", "/scripts/sales.nvn"),
    ("Sales.Total est affiché dans le rapport principal", "/scripts/report.nvn"),
]


```


┌─────────────────────────────────────────────────────────────┐
│                   📚 KNOWLEDGE BANK                         │
│              (Mémoire persistante de l'agent)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Étape 1 (RAG): "Le système de prévision utilise          │
│                  un modèle ARIMA" [Source: forecast.nvn]   │
│                                                             │
│  Étape 2 (GREP): "Le fichier Items.ion est lu par         │
│                   27 scripts" [Source: All Sources]        │
│                                                             │
│  Étape 3 (Script): "Le script main.nvn initialise         │
│                     les paramètres globaux" [Source: main] │
│                                                             │
└─────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                         BOUCLE ReAct                                     │
│                                                                          │
│   Itération 1                    Itération 2                Itération N  │
│  ┌──────────┐                   ┌──────────┐               ┌──────────┐  │
│  │ THOUGHT  │ ─── "Chercher" ──►│ THOUGHT  │ ─"Approfondir"│ THOUGHT  │  │
│  │ + TOOL   │                   │ + TOOL   │               │ GRADE    │  │
│  └────┬─────┘                   └────┬─────┘               └────┬─────┘  │
│       │                              │                          │        │
│       ▼                              ▼                          ▼        │
│  ┌─────────┐                   ┌─────────┐               ┌──────────┐   │
│  │ Execute │                   │ Execute │               │  FINISH  │   │
│  │ RAG     │                   │ GREP    │               │          │   │
│  └────┬────┘                   └────┬────┘               └──────────┘   │
│       │                              │                                   │
│       ▼                              ▼                                   │
│  Knowledge: [F1]              Knowledge: [F1, F2, F3]                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘


class WorkflowState(TypedDict):
    pipeline_state: AgentGraphState  # État global (question, knowledge_bank, history)
    regenerate: bool                  # Faut-il continuer la boucle ?
    current_thought: str              # Raisonnement actuel du Planner
    tool: str                         # Outil sélectionné
    tool_parameter: Any               # Paramètre de l'outil
    rewritten_prompt: str             # Prompt enrichi pour le Solver
