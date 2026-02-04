# LLMs (`src/llms`)

**Couche d'abstraction unifiée pour les fournisseurs LLM.**

---

## 🚀 Quick Start

```bash
# Query directe
uv run llm -q "Explique ce code" -m mistral

# Mode interactif
uv run llm -i

# Modèle spécifique
uv run llm -q "Hello" -m gemini --model gemini-1.5-pro
```

---

## 📋 CLI Reference

```
uv run llm [OPTIONS]

Options:
  -q, --query TEXT      Question à poser au LLM
  -m, --model TEXT      Provider: mistral, gemini, groq, qwen, ollama
  --model-name TEXT     Nom spécifique du modèle (override config)
  -i, --interactive     Mode conversation continue
  -t, --temperature     Température (0.0-1.0)
  --max-tokens INT      Limite de tokens en sortie
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
defaults:
  model: "mistral"      # Provider par défaut
  temperature: 0.0      # Déterministe
  max_tokens: 4096

providers:
  mistral:
    model_name: "mistral-large-latest"
  gemini:
    model_name: "gemini-1.5-pro-latest"
  groq:
    model_name: "llama3-70b-8192"
  qwen:
    model_name: "qwen-plus"
    region: "singapore"
  ollama:
    model_name: "qwen2.5-coder:latest"
    base_url: "http://localhost:11434"
```

---

## 🔑 API Keys (`.env`)

```bash
# Copier le template
cp .env.example .env

# Configurer les clés nécessaires
MISTRAL_API_KEY=your_key
GOOGLE_API_KEY=your_key
GROQ_API_KEY=your_key
DASHSCOPE_API_KEY=your_key  # Pour Qwen
```

---

## 🏗️ Architecture

```
llms/
├── base.py       # Interface abstraite LLM
├── mistral.py    # Provider Mistral
├── gemini.py     # Provider Google Gemini
├── groq.py       # Provider Groq
├── main.py       # CLI entry point
├── utils.py      # Factory get_llm()
└── config.yaml   # Configuration
```

### Usage Programmatique

```python
from llms.utils import get_llm

# Factory pattern
llm = get_llm(provider="mistral")
response = llm.query("Explique le code")

# Avec config personnalisée
llm = get_llm(provider="gemini", temperature=0.7)
```

---

## 📦 Providers Supportés

| Provider | Modèles | Notes |
|----------|---------|-------|
| **Mistral** | mistral-large-latest, codestral | Recommandé pour code |
| **Gemini** | gemini-1.5-pro, gemini-1.5-flash | Long context |
| **Groq** | llama3-70b-8192 | Très rapide |
| **Qwen** | qwen-plus, qwen-max | Multilingue |
| **Ollama** | Tout modèle local | Self-hosted |
