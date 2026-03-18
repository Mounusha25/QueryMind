"""
Orchestration Layer – LangGraph Workflow
Wires the four agents into a stateful graph with a conditional re-route loop.

Graph topology:
    planner → research → executor → reviewer
                  ↑                      |
                  └── (confidence < 0.7) ┘
                              ↓ (confidence >= 0.7)
                             END

The conditional re-route is what makes this a reasoning system:
when the reviewer is not satisfied, it sends control back to research
with an incremented iteration counter so the research agent broadens its retrieval.
"""
import os
from typing import Callable

from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.planner import planner_fn
from src.agents.research import make_research_fn
from src.agents.executor import executor_fn
from src.agents.reviewer import reviewer_fn
from src.rag.embeddings import build_or_load_vectorstore
from src.rag.retriever import build_retriever
from src.rag.loader import load_documents
from src.rag.chunker import chunk_documents


MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 3))


def _routing_decision(state: AgentState) -> str:
    """Conditional edge: re-route to research if confidence is low."""
    confidence = state.get("confidence", 0.0)
    iteration = state.get("iteration", 0)
    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))

    if confidence >= threshold or iteration >= MAX_ITERATIONS:
        return END
    return "research"


def build_graph(retrieve: Callable) -> StateGraph:
    """Construct and compile the LangGraph workflow.

    Args:
        retrieve: The retrieval callable from build_retriever().

    Returns:
        A compiled LangGraph application ready to invoke.
    """
    research_fn = make_research_fn(retrieve)

    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("planner",  planner_fn)
    graph.add_node("research", research_fn)
    graph.add_node("executor", executor_fn)
    graph.add_node("reviewer", reviewer_fn)

    # ── Wire edges ────────────────────────────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner",  "research")
    graph.add_edge("research", "executor")
    graph.add_edge("executor", "reviewer")

    # Conditional edge: reviewer → research (re-loop) or END
    graph.add_conditional_edges(
        "reviewer",
        _routing_decision,
        {
            "research": "research",  # low confidence → retry
            END: END,                # high confidence  → done
        },
    )

    return graph.compile()


def build_rag_pipeline(docs_dir: str | None = None, force_rebuild: bool = False):
    """Convenience function: load docs → chunk → embed → build retriever.

    Returns:
        retrieve callable
    """
    source = docs_dir or os.getenv("DOCS_DIR", "data/sample_docs")
    docs = load_documents(source)
    chunks = chunk_documents(docs)
    vectorstore = build_or_load_vectorstore(chunks=chunks, force_rebuild=force_rebuild)
    return build_retriever(vectorstore)


def run_query(query: str, retrieve: Callable) -> AgentState:
    """Run a single query through the compiled multi-agent graph.

    Args:
        query:    The user's natural language question.
        retrieve: Bound retrieval function.

    Returns:
        Final AgentState with final_answer, sources, confidence, agent_steps.
    """
    app = build_graph(retrieve)

    initial_state: AgentState = {
        "query": query,
        "iteration": 0,
        "confidence": 0.0,
        "agent_steps": [],
    }

    final_state = app.invoke(initial_state)
    return final_state
