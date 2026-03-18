"""
Reviewer Agent
Scores the executor's answer on completeness, groundedness, and confidence.
Sets the routing flag that drives the conditional re-loop in LangGraph.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .state import AgentState, ReviewResult

_SYSTEM = """You are a rigorous quality-assurance reviewer for an AI answer system.
Evaluate the proposed answer against the original query and the retrieved source documents.

Score each dimension 0.0 – 1.0:

completeness  : Does the answer fully address all aspects of the query?
                0 = completely misses the point, 1 = every sub-task answered thoroughly.

groundedness  : Are the claims in the answer supported by the retrieved sources?
                0 = answer contradicts/ignores sources (hallucination),
                1 = every claim is cited and traceable to a source.

confidence    : Weighted overall score = 0.4 * completeness + 0.6 * groundedness
                (groundedness is weighted higher because hallucination is more dangerous).

Rules:
- Lower groundedness aggressively if the answer introduces facts NOT in the sources.
- Lower completeness if sub-tasks from the plan are skipped.
- Set approved = true only if confidence >= {threshold}.
- Write constructive feedback that tells the research/executor what to improve.
"""

_HUMAN = """Original query: {query}

Task plan intent: {intent}

Retrieved sources (summary):
{sources_summary}

Proposed answer:
{answer}

Output your review as JSON only."""


def reviewer_fn(state: AgentState) -> AgentState:
    """LangGraph node: Reviewer agent."""
    steps: list = list(state.get("agent_steps", []))
    steps.append("🔎  Reviewer: Evaluating answer quality …")

    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
    query = state.get("query", "")
    plan = state.get("task_plan")
    answer = state.get("structured_answer", "")
    chunks = state.get("retrieved_chunks", [])

    intent = plan.intent if plan else "Not specified"

    # Build a short sources summary for the reviewer prompt
    if chunks:
        sources_lines = [
            f"[Source {i+1}] {c['source']} (score={c['score']:.3f}): "
            f"{c['content'][:120]}…"
            for i, c in enumerate(chunks[:5])
        ]
        sources_summary = "\n".join(sources_lines)
    else:
        sources_summary = "No sources retrieved."

    model_name = os.getenv("REVIEWER_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    parser = PydanticOutputParser(pydantic_object=ReviewResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM.format(threshold=threshold)
         + "\n\nOutput format:\n{format_instructions}"),
        ("human", _HUMAN),
    ])

    chain = prompt | llm | parser

    try:
        review: ReviewResult = chain.invoke({
            "query": query,
            "intent": intent,
            "sources_summary": sources_summary,
            "answer": answer,
            "format_instructions": parser.get_format_instructions(),
        })

        emoji = "✅" if review.approved else "🔄"
        steps.append(
            f"{emoji}  Reviewer: confidence={review.confidence:.2f} "
            f"(completeness={review.completeness:.2f}, "
            f"groundedness={review.groundedness:.2f}) "
            f"{'APPROVED' if review.approved else 'RE-ROUTING → Research'}"
        )
        if not review.approved:
            steps.append(f"   Feedback: {review.feedback}")

        # If approved, set the final answer and collect source citations
        final_answer = answer if review.approved else state.get("final_answer")
        sources = (
            list({c["source"] for c in chunks}) if review.approved
            else state.get("sources", [])
        )

        return {
            **state,
            "review": review,
            "confidence": review.confidence,
            "iteration": state.get("iteration", 0) + 1,
            "final_answer": final_answer,
            "sources": sources,
            "agent_steps": steps,
        }

    except Exception as e:
        steps.append(f"❌  Reviewer error: {e}")
        # On error, pass through with low confidence to avoid infinite loop
        return {
            **state,
            "confidence": 0.0,
            "iteration": state.get("iteration", 0) + 1,
            "agent_steps": steps,
            "error": str(e),
        }
