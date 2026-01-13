# 🧠 Envision RAG

> **Hybrid RAG System for Envision DSL Codebase Analysis**

A production-grade Retrieval-Augmented Generation system that combines **structural graph analysis** with **semantic vector search** to answer complex questions about Envision DSL codebases.

---

## 🎯 Why This Project?

Supply Chain Scientists working with [Lokad's Envision DSL](https://docs.lokad.com/) face a unique challenge:

```
"Which scripts read /Clean/Items.ion?"
"Where is the ReDispatchCycle variable defined?"
"What's the impact if I modify this file?"
```

Traditional RAG systems fail here because:
- **Code is structural**: Dependencies matter more than keywords
- **Envision is domain-specific**: Generic embeddings miss DSL semantics
- **Codebases are graphs**: Files `read`, `write`, and `import` each other

**This system solves it with a Hybrid approach.**

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
│            "Combien de scripts lisent /Clean/Items.ion?"        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      🧠 ReAct AGENT                             │
│                    (LangGraph Workflow)                         │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│   │  Reason  │───▶│   Act    │───▶│ Observe  │──┐              │
│   └──────────┘    └──────────┘    └──────────┘  │              │
│        ▲                                         │              │
│        └─────────────────────────────────────────┘              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │  📊 GRAPH TOOLS   │   │  🔍 VECTOR TOOLS  │
        │  (Deterministic)  │   │    (Semantic)     │
        ├───────────────────┤   ├───────────────────┤
        │ • scan_references │   │ • search_code     │
        │ • describe_impact │   │ • grep_code       │
        │ • read_code       │   │ • read_code       │
        └─────────┬─────────┘   └─────────┬─────────┘
                  │                       │
                  ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │   NetworkX Graph  │   │    FAISS Index    │
        │   (462 edges)     │   │   (Embeddings)    │
        └───────────────────┘   └───────────────────┘
```

---

## 🔬 How It Works: A Simulation

Let's trace through a real query step-by-step:

### Query: *"Combien de scripts lisent /Clean/Items.ion?"*

#### Step 1: Agent Reasoning
```
🧠 Thought: The question asks about scripts reading a specific file.
   I must use PRECISION-FIRST strategy with the exact path.
   
   Plan: ["Use scan_references with exact path", "Count results", "Verify no noise"]
```

#### Step 2: Tool Execution
```
🛠️ Action: scan_references("read /Clean/Items.ion")

👀 Observation: 
{
  "count": 28,
  "results": [
    {"source_script": "/3. Inspectors/2 - Sales Analysis", "target_file": "/Clean/Items.ion"},
    {"source_script": "/4. Optimization workflow/01 Inventory...", "target_file": "/Clean/Items.ion"},
    ... (26 more)
  ],
  "unique_targets": ["/Clean/Items.ion"],  ← No noise!
  "unique_targets_count": 1
}
```

#### Step 3: Final Answer
```
✅ Final Answer: 28 scripts lisent /Clean/Items.ion.

📎 Appendix:
• /1. utilities/1. populating dataset/03 - Manual Inputs
• /3. Inspectors/1 - Item Inspector
• /4. Optimization workflow/05. Standard Purchase Suggestions
... (full list)
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/envision-rag.git
cd envision-rag

# Install with uv (recommended)
uv sync

# Configure API keys in .env
cp .env.example .env
# Edit .env with your API keys (MISTRAL_API_KEY, GOOGLE_API_KEY, etc.)
```

---

## 🚀 Quick Start

### 1. Build the Graph Index
```bash
uv run envision-rag --rebuild -q "INIT"
```

### 2. Build the Vector Index
```bash
uv run python build_index_v2.py
```

### 3. Query the System
```bash
# Single query (verbose trace)
uv run envision-rag -v -q "Combien de scripts lisent /Clean/Items.ion?"

# Interactive mode
uv run envision-rag -i
```

---

## 📊 Benchmark & Evaluation

The system includes an **LLM-as-Judge** benchmark:

```bash
# Run default benchmark (questions 1-5)
uv run envision-benchmark

# Test specific questions
uv run envision-benchmark --ids 1 3 5 8

# Test a range
uv run envision-benchmark --from 1 --to 10
```

### Sample Results
```
📊 BENCHMARK RESULTS
 Total Questions  10
 Passed           9
 Failed           1
 Accuracy         90.0%
 Average Score    0.95
```

---

## 📜 Log Replay

All sessions are persisted for debugging and analysis:

```bash
# View last main query
uv run envision-logs -t main -n 1

# View last benchmark
uv run envision-logs -t benchmark -n 1

# List available logs
uv run envision-logs -t main --list
```

---

## 🗂️ Project Structure

```
envision-rag/
├── src/envision_rag/
│   ├── workflow/
│   │   └── agent.py          # 🧠 LangGraph ReAct Agent
│   ├── graph/
│   │   ├── builder.py        # 📊 Parses .nvn files → NetworkX
│   │   └── graph_types.py    # Node/Edge definitions
│   ├── index/
│   │   ├── chunker.py        # Semantic code chunking
│   │   └── vector_tools.py   # FAISS retrieval
│   ├── tools/
│   │   ├── graph_tools.py    # scan_references, describe_impact
│   │   └── search_tools.py   # grep_code, read_code
│   ├── agents/
│   │   ├── mistral_agent.py  # Mistral API wrapper
│   │   ├── gemini_agent.py   # Gemini API wrapper
│   │   └── ...
│   ├── benchmark/
│   │   ├── runner.py         # Benchmark orchestration
│   │   └── judge.py          # LLM-as-Judge evaluation
│   └── logging/
│       └── session_logger.py # Session persistence
├── config.yaml               # Configuration
├── questions.json            # Benchmark questions (32 answered)
└── env_scripts/              # Your Envision codebase (gitignored)
```

---

## ⚙️ Configuration

```yaml
# config.yaml
agent:
  main_model: "mistral"       # Options: mistral, gemini, gpt, llama3
  max_iterations: 10          # Prevent infinite loops

benchmark:
  judge_model: "mistral"      # Model for LLM-as-Judge
  questions_file: "questions.json"

logging:
  enabled: true
  log_dir: "data/logs"
```

---

## 🔧 Key Design Decisions

### 1. **Precision-First Query Strategy**
The agent always starts with exact paths, then broadens only if needed:
```
GOOD: scan_references("read /Clean/Items.ion")
BAD:  scan_references("read Items.ion")  ← matches multiple files!
```

### 2. **Graph vs. Vector: When to Use What?**

| Question Type | Best Tool | Why |
|--------------|-----------|-----|
| "Who reads X?" | `scan_references` | Deterministic graph lookup |
| "Where is X defined?" | `grep_code` | Exact pattern match |
| "How does X work?" | `search_code` | Semantic understanding |
| "What's the impact?" | `describe_impact` | Graph traversal |

### 3. **ReAct Loop with Self-Correction**
The agent can:
- Recognize noisy results and refine queries
- Read code to verify before answering
- Update its plan mid-execution

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Graph Nodes | 243 |
| Graph Edges | 462 |
| Avg. Query Time | ~15s |
| Benchmark Accuracy | 90%+ |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run the benchmark to ensure no regression
4. Submit a PR

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for Supply Chain Scientists
  <br>
  <a href="https://lokad.com">Lokad</a> • <a href="https://docs.lokad.com/">Envision Docs</a>
</p>
