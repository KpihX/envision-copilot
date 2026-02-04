# Envision Benchmark (`src/envision_benchmark`)

**Framework d'évaluation automatisé pour le Copilot.**

---

## 🚀 Quick Start

```bash
# Lancer le benchmark complet
uv run benchmark

# Mode verbose (voir les réponses)
uv run benchmark -v

# Question spécifique par ID
uv run benchmark --id 15
```

---

## 📋 CLI Reference

```
uv run benchmark [OPTIONS]

Options:
  -v, --verbose         Afficher les réponses complètes
  --id INT              Tester une question spécifique
  --answered-only       Ne tester que les questions avec réponse attendue
  --unanswered-only     Ne tester que les questions sans réponse
  -o, --output PATH     Fichier de sortie pour le rapport
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
judge:
  model: "mistral"              # LLM pour évaluer les réponses

input:
  questions_file: "src/envision_benchmark/questions.json"

output:
  report_dir: "datas/benchmark"  # Rapports générés
```

---

## 📊 Structure des Questions (`questions.json`)

```json
{
  "answered": [
    {
      "id": 15,
      "question": "Combien de profils de saisonnalité sont appris ?",
      "deterministic": true,
      "answers": [10],
      "appendix": ["10 profils de saisonnalité"]
    }
  ],
  "unanswered": [
    {
      "id": 33,
      "question": "Où est définie la logique d'actualisation du stock ?",
      "deterministic": true
    }
  ]
}
```

| Champ | Description |
|-------|-------------|
| `id` | Identifiant unique |
| `question` | Question en français |
| `deterministic` | `true` = réponse unique attendue |
| `answers` | Liste des réponses valides |
| `appendix` | Preuves/contexte additionnel |

---

## 🏗️ Architecture

```
envision_benchmark/
├── main.py         # CLI entry point
├── runner.py       # Exécute les questions via Copilot
├── utils.py        # Parsing et formatage
├── questions.json  # Dataset de test
└── config.yaml     # Configuration
```

### Pipeline d'Évaluation

```
questions.json → Runner → Copilot → Response → Judge (LLM) → Score
```

1. **Runner** : Soumet chaque question au Copilot
2. **Copilot** : Génère une réponse avec son pipeline agentic
3. **Judge** : LLM évalue la correspondance réponse/attendu
4. **Rapport** : JSON avec scores et détails

---

## 📈 Métriques

| Métrique | Description |
|----------|-------------|
| **Accuracy** | % de réponses correctes |
| **Partial Match** | % de réponses partiellement correctes |
| **Coverage** | % de questions avec réponse trouvée |

---

## 📁 Outputs

Les rapports sont générés dans `datas/benchmark/` :

```
datas/benchmark/
├── 2026-02-04_14-30-00_abc123.json  # Rapport complet
└── latest.json                       # Lien vers le dernier
```
