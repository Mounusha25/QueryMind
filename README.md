# Enterprise Query Assistant — Multi-Agent AI System

A self-correcting multi-agent AI assistant built with **LangGraph**, **LangChain**, and **RAG** that accepts complex natural language business questions and returns verified, grounded answers by coordinating four specialised agents.

---

## Architecture

```
User Query
    │
    ▼
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Planner │───▶│ Research │───▶│ Executor │───▶│ Reviewer │
│ GPT-4o  │    │ RAG/FAISS│    │GPT-4o-mini    │GPT-4o-mini
└─────────┘    └──────────┘    └──────────┘    └──────┬───┘
                    ▲                                  │
                    │           confidence < 0.7       │
                    └──────────────────────────────────┘
                                                       │
                                    confidence ≥ 0.7   ▼
                                                      END
                                               (Final Answer)
```

| Layer | Tool | Purpose |
|-------|------|---------|
| UI | Streamlit | Query input, live agent steps, output display |
| Orchestration | LangGraph + LangChain | State machine, conditional routing |
| LLM | GPT-4o / GPT-4o-mini | Planning, reasoning, generation |
| RAG | FAISS + `text-embedding-3-small` | Vector search, chunk retrieval, similarity scoring |
| Tools | Python REPL, LangChain tools | Data analysis, transformations |
| Observability | LangSmith | Full trace per query, latency per agent, eval dashboards |

---

## Project Structure

```
GenAI_PM/
├── app.py                        # Streamlit UI (Phase 6)
├── requirements.txt
├── .env.example                 
├── .streamlit/config.toml        # UI theme
│
├── src/
│   ├── rag/
│   │   ├── loader.py             # PDF + TXT document loading
│   │   ├── chunker.py            # RecursiveCharacterTextSplitter
│   │   ├── embeddings.py         # FAISS + text-embedding-3-small
│   │   └── retriever.py          # Top-k retrieval with similarity scores
│   │
│   ├── agents/
│   │   ├── state.py              # AgentState TypedDict + Pydantic schemas
│   │   ├── planner.py            # GPT-4o: query decomposition → TaskPlan
│   │   ├── research.py           # RAG retrieval → context text
│   │   ├── executor.py           # GPT-4o-mini: analysis + structured answer
│   │   └── reviewer.py           # GPT-4o-mini: scores completeness + groundedness
│   │
│   ├── graph/
│   │   └── workflow.py           # LangGraph StateGraph wiring + conditional edges
│   │
│   └── evaluation/
│       ├── questions.py          # 30 fixed test questions with ground truth
│       └── harness.py            # LLM-as-judge evaluation harness
│
└── data/
    ├── sample_docs/              
    │   ├── q3_complaint_trends_report.txt
    │   ├── customer_support_policy.txt
    │   └── product_improvement_recommendations.txt
    ├── faiss_index/              
    └── eval_results.csv          
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY=sk-...
#   LANGCHAIN_API_KEY=ls__...
```

### 3. Add your documents

Drop PDFs or TXT files into `data/sample_docs/`.  
Three sample documents are provided for Q3 complaints use case.

### 4. Launch the Streamlit UI

```bash
streamlit run app.py
```

### 5. Run the evaluation harness (optional)

```python
from dotenv import load_dotenv
load_dotenv()

from src.graph.workflow import build_rag_pipeline
from src.evaluation.harness import run_comparison_eval

retrieve = build_rag_pipeline()
df = run_comparison_eval(retrieve, run_baseline=True)
```

---

## The Four Agents

### 1. Planner (`gpt-4o`)
- Input: raw user query  
- Output: `TaskPlan` with sub-tasks and optimised retrieval query  
- Uses Pydantic structured output for type safety

### 2. Research (`text-embedding-3-small` + FAISS)
- Input: retrieval query from planner + iteration number  
- Output: top-k chunks with similarity scores, assembled context text  
- Retrieves more chunks on re-runs (`k += iteration * 2`)

### 3. Executor (`gpt-4o-mini`)
- Input: task plan + retrieved context  
- Output: structured, cited analysis grounded in source documents

### 4. Reviewer (`gpt-4o-mini`)
- Scores: **completeness** (0–1) + **groundedness** (0–1)  
- `confidence = 0.4 × completeness + 0.6 × groundedness`  
- If `confidence < 0.7` → re-routes to Research  
- If `confidence ≥ 0.7` or `iteration ≥ 3` → approves and ends

---

## Evaluation Methodology

30 fixed test questions across 6 categories: complaints, product, policy, strategy, operations, synthesis.

| Metric | Description |
|--------|-------------|
| `answer_correctness` | LLM-as-judge 0–1: key facts from ground truth in answer |
| `hallucination_rate` | Fraction of answers with invented facts |
| `latency_seconds` | Wall-clock time per query end-to-end |

Run against two systems:
- **Baseline**: single GPT-4o call, no RAG, no agents  
- **Multi-Agent**: full LangGraph pipeline

---

## LangSmith Tracing

With `LANGCHAIN_TRACING_V2=true` set, every query automatically produces a full execution trace at [smith.langchain.com](https://smith.langchain.com):

- Which agent ran and when  
- Token usage and latency per agent  
- Conditional routing decisions  
- Full prompt/response at each step

---

## Tech Stack Summary

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph `StateGraph` |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Embeddings | `text-embedding-3-small` |
| Vector Store | FAISS (local) or Chroma (persistent) |
| Document Loading | LangChain `PyPDFLoader`, `DirectoryLoader` |
| Chunking | `RecursiveCharacterTextSplitter` |
| UI | Streamlit |
| Observability | LangSmith |
| Evaluation | LLM-as-judge (GPT-4o-mini) |
| Configuration | `python-dotenv` |
