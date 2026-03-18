"""
Shared state schema for the LangGraph multi-agent workflow.
Every agent reads from and writes back to this TypedDict.
"""
from __future__ import annotations

from typing import Any, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ── Planner output schema ──────────────────────────────────────────────────────
class SubTask(BaseModel):
    id: int = Field(description="Sequential task number")
    description: str = Field(description="What needs to be done")
    requires_retrieval: bool = Field(
        default=True, description="Whether this sub-task needs document retrieval"
    )


class TaskPlan(BaseModel):
    original_query: str = Field(description="The verbatim user query")
    intent: str = Field(description="One-line summary of what the user needs")
    sub_tasks: List[SubTask] = Field(description="Ordered list of sub-tasks")
    retrieval_query: str = Field(
        description="Optimised search string to use for vector retrieval"
    )


# ── Reviewer output schema ─────────────────────────────────────────────────────
class ReviewResult(BaseModel):
    completeness: float = Field(ge=0, le=1, description="0-1 how fully the question is answered")
    groundedness: float = Field(ge=0, le=1, description="0-1 how well answer is supported by sources")
    confidence: float = Field(ge=0, le=1, description="Overall confidence score (weighted average)")
    feedback: str = Field(description="Short reviewer feedback for the next loop iteration")
    approved: bool = Field(description="True if confidence >= threshold")


# ── LangGraph state dict ───────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    # Input
    query: str

    # Planner
    task_plan: Optional[TaskPlan]

    # Research
    retrieved_chunks: Optional[List[dict]]   # serialised RetrievedChunk dicts
    context_text: Optional[str]              # concatenated context for the LLM

    # Executor
    analysis_result: Optional[str]
    structured_answer: Optional[str]

    # Reviewer
    review: Optional[ReviewResult]
    confidence: float
    iteration: int

    # Final
    final_answer: Optional[str]
    sources: Optional[List[str]]

    # Streaming / UI metadata
    agent_steps: List[str]
    error: Optional[str]
