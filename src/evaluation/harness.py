"""
Evaluation Harness – Phase 5
Runs 30 fixed test questions through:
    (A) Vanilla single-call GPT-4o baseline
    (B) Multi-agent LangGraph system

Measures:
    - answer_correctness  : LLM-as-judge (0-1)
    - hallucination_rate  : fraction of answers contradicting sources
    - latency_seconds     : wall-clock time per query

Outputs a pandas DataFrame and saves results to data/eval_results.csv
"""
import os
import time
import json
from datetime import datetime
from typing import Callable, List, Dict, Any

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.graph.workflow import run_query
from .questions import TEST_QUESTIONS


# ── LLM-as-judge schema ────────────────────────────────────────────────────────
class JudgeScore(BaseModel):
    correctness: float = Field(ge=0, le=1, description="How correct is the answer vs ground truth?")
    hallucination: bool = Field(description="True if the answer contains facts not in ground truth / sources")
    reasoning: str = Field(description="One sentence explaining the scores")


_JUDGE_SYSTEM = """You are an impartial answer quality judge.
Compare the system answer to the ground truth and score:
- correctness 0-1: how many key facts from the ground truth appear correctly in the answer.
- hallucination: true if the answer contains clearly invented facts NOT in the ground truth.
Output JSON only."""

_JUDGE_HUMAN = """Question: {question}
Ground truth: {ground_truth}
System answer: {answer}

Output your judgment as JSON."""


def _judge_answer(question: str, ground_truth: str, answer: str) -> JudgeScore:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = PydanticOutputParser(pydantic_object=JudgeScore)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _JUDGE_SYSTEM + "\n\nFormat:\n{format_instructions}"),
        ("human", _JUDGE_HUMAN),
    ])
    chain = prompt | llm | parser
    return chain.invoke({
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "format_instructions": parser.get_format_instructions(),
    })


def _baseline_answer(question: str) -> tuple[str, float]:
    """Single-call GPT-4o baseline — no RAG, no agents."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    start = time.time()
    response = llm.invoke(question)
    latency = time.time() - start
    return response.content, latency


# ── Main harness ───────────────────────────────────────────────────────────────
class EvaluationHarness:
    def __init__(self, retrieve: Callable, questions: List[Dict] | None = None):
        self.retrieve = retrieve
        self.questions = questions or TEST_QUESTIONS
        self.results: List[Dict[str, Any]] = []

    def run(self, run_baseline: bool = True, verbose: bool = True) -> pd.DataFrame:
        """Run all test questions through both systems.

        Args:
            run_baseline: If False, skip the expensive baseline GPT-4o calls.
            verbose:      Print progress to stdout.

        Returns:
            DataFrame with per-question metrics.
        """
        rows = []
        total = len(self.questions)

        for i, q in enumerate(self.questions, 1):
            qid = q["id"]
            question = q["question"]
            ground_truth = q["ground_truth"]
            category = q["category"]

            if verbose:
                print(f"\n[{i}/{total}] Q{qid} ({category}): {question[:80]}…")

            row: Dict[str, Any] = {
                "id": qid,
                "category": category,
                "question": question,
                "ground_truth": ground_truth,
            }

            # ── Multi-agent system ─────────────────────────────────────────
            try:
                start = time.time()
                state = run_query(question, self.retrieve)
                ma_latency = time.time() - start
                ma_answer = state.get("final_answer") or state.get("structured_answer", "")
                ma_confidence = state.get("confidence", 0.0)
                ma_iterations = state.get("iteration", 1)

                ma_judge = _judge_answer(question, ground_truth, ma_answer)

                row.update({
                    "ma_answer": ma_answer[:300],
                    "ma_correctness": ma_judge.correctness,
                    "ma_hallucination": int(ma_judge.hallucination),
                    "ma_confidence": ma_confidence,
                    "ma_iterations": ma_iterations,
                    "ma_latency": round(ma_latency, 2),
                    "ma_judge_reasoning": ma_judge.reasoning,
                })
                if verbose:
                    print(f"  ✅ Multi-agent: correct={ma_judge.correctness:.2f}, "
                          f"hallucination={ma_judge.hallucination}, "
                          f"latency={ma_latency:.1f}s, confidence={ma_confidence:.2f}")
            except Exception as e:
                row.update({
                    "ma_answer": f"ERROR: {e}", "ma_correctness": 0,
                    "ma_hallucination": 1, "ma_confidence": 0,
                    "ma_iterations": 0, "ma_latency": 0, "ma_judge_reasoning": str(e),
                })
                if verbose:
                    print(f"  ❌ Multi-agent error: {e}")

            # ── Baseline system ────────────────────────────────────────────
            if run_baseline:
                try:
                    bl_answer, bl_latency = _baseline_answer(question)
                    bl_judge = _judge_answer(question, ground_truth, bl_answer)
                    row.update({
                        "bl_answer": bl_answer[:300],
                        "bl_correctness": bl_judge.correctness,
                        "bl_hallucination": int(bl_judge.hallucination),
                        "bl_latency": round(bl_latency, 2),
                        "bl_judge_reasoning": bl_judge.reasoning,
                    })
                    if verbose:
                        print(f"  ✅ Baseline:     correct={bl_judge.correctness:.2f}, "
                              f"hallucination={bl_judge.hallucination}, "
                              f"latency={bl_latency:.1f}s")
                except Exception as e:
                    row.update({
                        "bl_answer": f"ERROR: {e}", "bl_correctness": 0,
                        "bl_hallucination": 1, "bl_latency": 0,
                        "bl_judge_reasoning": str(e),
                    })

            rows.append(row)

        df = pd.DataFrame(rows)
        self.results = rows

        # Save to CSV
        out_path = "data/eval_results.csv"
        os.makedirs("data", exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n[Eval] Results saved to {out_path}")

        self._print_summary(df, run_baseline)
        return df

    def _print_summary(self, df: pd.DataFrame, run_baseline: bool):
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        ma_correct = df["ma_correctness"].mean()
        ma_hallu = df["ma_hallucination"].mean() * 100
        ma_lat = df["ma_latency"].mean()
        print(f"Multi-Agent System:")
        print(f"  Answer Correctness : {ma_correct:.2f}")
        print(f"  Hallucination Rate : {ma_hallu:.1f}%")
        print(f"  Avg Latency        : {ma_lat:.1f}s")

        if run_baseline and "bl_correctness" in df.columns:
            bl_correct = df["bl_correctness"].mean()
            bl_hallu = df["bl_hallucination"].mean() * 100
            bl_lat = df["bl_latency"].mean()
            print(f"\nBaseline (GPT-4o single call):")
            print(f"  Answer Correctness : {bl_correct:.2f}")
            print(f"  Hallucination Rate : {bl_hallu:.1f}%")
            print(f"  Avg Latency        : {bl_lat:.1f}s")

            hallu_reduction = bl_hallu - ma_hallu
            correct_delta = (ma_correct - bl_correct) * 100
            print(f"\nDelta (Multi-Agent vs Baseline):")
            print(f"  Hallucination Reduction : {hallu_reduction:+.1f}pp")
            print(f"  Correctness Improvement : {correct_delta:+.1f}pp")
        print("="*60)


def run_comparison_eval(retrieve: Callable, run_baseline: bool = True) -> pd.DataFrame:
    """Convenience wrapper to run the full evaluation."""
    harness = EvaluationHarness(retrieve=retrieve)
    return harness.run(run_baseline=run_baseline)
