"""
Planner Agent
Takes the raw user query and produces a structured TaskPlan via GPT-4o
with structured output (Pydantic).
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .state import AgentState, TaskPlan

_SYSTEM = """You are a query planning agent in an enterprise AI assistant.
Your job is to decompose a complex business question into an ordered list of sub-tasks
that a team of specialised agents will execute.

Rules:
- Be concise and specific.
- The retrieval_query should be a dense, keyword-rich search string for semantic search.
- Mark sub-tasks that require document retrieval with requires_retrieval=true.
- Always output valid JSON matching the schema.
"""

_HUMAN = """User query:
\"\"\"{query}\"\"\"

Decompose this into a structured task plan. Output JSON only."""


def planner_fn(state: AgentState) -> AgentState:
    """LangGraph node: Planner agent."""
    query = state["query"]
    steps: list = state.get("agent_steps", [])
    steps.append("🗂️  Planner: Decomposing query into sub-tasks …")

    model_name = os.getenv("PLANNER_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model_name, temperature=0)

    parser = PydanticOutputParser(pydantic_object=TaskPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM + "\n\nOutput format:\n{format_instructions}"),
        ("human", _HUMAN),
    ])

    chain = prompt | llm | parser

    try:
        plan: TaskPlan = chain.invoke({
            "query": query,
            "format_instructions": parser.get_format_instructions(),
        })
        steps.append(
            f"✅  Planner: Found {len(plan.sub_tasks)} sub-tasks. "
            f"Retrieval query: '{plan.retrieval_query}'"
        )
        return {
            **state,
            "task_plan": plan,
            "iteration": state.get("iteration", 0),
            "agent_steps": steps,
        }
    except Exception as e:
        steps.append(f"❌  Planner error: {e}")
        return {**state, "agent_steps": steps, "error": str(e)}
