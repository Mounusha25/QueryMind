"""
Research Agent
Runs RAG retrieval using the planner's retrieval_query.
Appends retrieved chunks and assembled context text to the state.
"""
import os
from typing import Callable, List

from .state import AgentState
from src.rag.retriever import RetrievedChunk


def make_research_fn(retrieve: Callable):
    """Factory that binds the retrieval function and returns a LangGraph node.

    Args:
        retrieve: Callable returned by src.rag.retriever.build_retriever()

    Returns:
        research_fn compatible with LangGraph add_node()
    """
    def research_fn(state: AgentState) -> AgentState:
        """LangGraph node: Research / RAG agent."""
        steps: list = list(state.get("agent_steps", []))
        plan = state.get("task_plan")
        iteration = state.get("iteration", 0)

        if plan is None:
            steps.append("⚠️  Research: No task plan found, using raw query.")
            search_query = state["query"]
        else:
            search_query = plan.retrieval_query

        steps.append(f"🔍  Research (iter {iteration + 1}): Retrieving for '{search_query[:60]}…'")

        top_k = int(os.getenv("RETRIEVER_TOP_K", 5))
        # On re-runs retrieve more chunks to broaden context
        effective_k = top_k + (iteration * 2)

        try:
            chunks: List[RetrievedChunk] = retrieve(search_query, k=effective_k)

            # Serialise for state storage
            serialised = [
                {
                    "content": c.content,
                    "source": c.source,
                    "page": c.page,
                    "score": c.score,
                }
                for c in chunks
            ]

            # Build context string for the LLM
            context_parts = []
            for i, c in enumerate(chunks, 1):
                page_info = f" (page {c.page})" if c.page is not None else ""
                context_parts.append(
                    f"[Source {i}: {c.source}{page_info}, score={c.score:.3f}]\n{c.content}"
                )
            context_text = "\n\n---\n\n".join(context_parts)

            avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0
            steps.append(
                f"✅  Research: Retrieved {len(chunks)} chunks "
                f"(avg score={avg_score:.3f})"
            )

            return {
                **state,
                "retrieved_chunks": serialised,
                "context_text": context_text,
                "agent_steps": steps,
            }

        except Exception as e:
            steps.append(f"❌  Research error: {e}")
            return {**state, "agent_steps": steps, "error": str(e)}

    return research_fn


# Placeholder for when the graph is constructed without a retriever (testing)
def research_fn(state: AgentState) -> AgentState:  # noqa: F811
    raise RuntimeError(
        "research_fn is not bound to a retriever. "
        "Use make_research_fn(retrieve) to create a bound version."
    )
