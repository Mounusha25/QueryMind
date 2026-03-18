"""
Executor Agent
Uses the task plan and retrieved context to produce a structured answer.
Runs on GPT-4o-mini for speed and cost efficiency.
Supports Python REPL-style analysis for structured data tasks.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState

_SYSTEM = """You are an expert business analyst AI.
You have been given:
1. A user query and a structured task plan.
2. Relevant document excerpts retrieved from a knowledge base.

Your job is to produce a thorough, well-structured answer that:
- Directly addresses every sub-task in the plan.
- Grounds every claim in the provided source documents (cite them as [Source N]).
- Identifies patterns, trends, or insights from the evidence.
- If the task involves data analysis, summarise numeric findings clearly.
- Ends with a "Recommendations" section if the query asks for suggestions.

Be factual. Do not invent information not found in the sources.
If the sources are insufficient, state which sub-tasks could not be answered.
"""

_HUMAN = """User query: {query}

Task plan:
{task_plan}

Retrieved document context:
{context}

Produce the structured answer now."""


def executor_fn(state: AgentState) -> AgentState:
    """LangGraph node: Executor agent."""
    steps: list = list(state.get("agent_steps", []))
    steps.append("⚙️  Executor: Analysing retrieved context and generating answer …")

    query = state.get("query", "")
    plan = state.get("task_plan")
    context = state.get("context_text", "No context retrieved.")

    # Format task plan for the prompt
    if plan:
        plan_text = f"Intent: {plan.intent}\nSub-tasks:\n"
        for t in plan.sub_tasks:
            plan_text += f"  {t.id}. {t.description}\n"
    else:
        plan_text = "No structured plan available. Answer the query directly."

    model_name = os.getenv("EXECUTOR_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "query": query,
            "task_plan": plan_text,
            "context": context,
        })
        answer = response.content

        steps.append(f"✅  Executor: Answer generated ({len(answer)} chars)")

        return {
            **state,
            "structured_answer": answer,
            "agent_steps": steps,
        }

    except Exception as e:
        steps.append(f"❌  Executor error: {e}")
        return {**state, "agent_steps": steps, "error": str(e)}
