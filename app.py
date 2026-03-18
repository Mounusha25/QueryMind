"""
Streamlit UI – Enterprise Query Assistant
Phase 6: Full multi-agent interface with live agent steps, retrieved chunks,
confidence score, and final grounded answer.

Run:
    streamlit run app.py
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Query Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Agent step item */
.step-item {
    padding: 6px 12px;
    border-left: 3px solid #6C63FF;
    margin: 4px 0;
    background: rgba(108, 99, 255, 0.08);
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
    font-family: monospace;
}
/* Confidence bar */
.conf-bar-wrap { background: #1A1D2E; border-radius: 8px; height: 14px; width: 100%; }
.conf-bar      { height: 14px; border-radius: 8px; }
/* Source chip */
.source-chip {
    display: inline-block;
    background: rgba(108,99,255,0.2);
    border: 1px solid #6C63FF;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=64)
    st.title("⚙️ Configuration")

    docs_dir = st.text_input(
        "Documents directory",
        value=str(ROOT / "data" / "sample_docs"),
        help="Path to your PDF/TXT knowledge base",
    )
    force_rebuild = st.checkbox("Force re-index documents", value=False)
    top_k = st.slider("Retriever top-k chunks", min_value=3, max_value=10, value=5)
    threshold = st.slider("Confidence threshold (re-route if below)", 0.0, 1.0, 0.7, 0.05)

    os.environ["RETRIEVER_TOP_K"] = str(top_k)
    os.environ["CONFIDENCE_THRESHOLD"] = str(threshold)

    st.divider()
    st.markdown("**Models**")
    st.caption(f"Planner: `{os.getenv('PLANNER_MODEL', 'gpt-4o')}`")
    st.caption(f"Executor: `{os.getenv('EXECUTOR_MODEL', 'gpt-4o-mini')}`")
    st.caption(f"Reviewer: `{os.getenv('REVIEWER_MODEL', 'gpt-4o-mini')}`")
    st.caption(f"Embeddings: `{os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}`")

    st.divider()
    run_eval = st.button("🧪 Run Evaluation Harness", use_container_width=True)
    st.caption("Runs 30 test questions and saves results to data/eval_results.csv")


# ── Cached pipeline ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Building RAG index …")
def get_retrieve(docs_dir: str, _force: bool):
    from src.graph.workflow import build_rag_pipeline
    return build_rag_pipeline(docs_dir=docs_dir, force_rebuild=_force)


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🤖 Enterprise Query Assistant")
st.caption(
    "Multi-agent AI powered by **LangGraph** · RAG over your documents · "
    "Self-correcting via confidence threshold · Traced by **LangSmith**"
)

# Example queries
EXAMPLES = [
    "Summarise Q3 customer complaint trends and suggest the top 3 product improvements.",
    "Which product areas should engineering prioritise in Q4 to reduce complaints by 15%?",
    "What is the first-contact resolution rate and how does it compare to industry benchmark?",
    "Draft an executive summary of Q3 support performance with risks and recommended actions.",
]

with st.expander("💡 Example queries", expanded=False):
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state["query_input"] = ex

query = st.text_area(
    "Your question",
    value=st.session_state.get("query_input", ""),
    height=80,
    placeholder="Ask a complex business question about your document knowledge base …",
    key="query_area",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("▶ Run", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑 Clear", use_container_width=True)

if clear_btn:
    for key in ["query_input", "last_result"]:
        st.session_state.pop(key, None)
    st.rerun()

# ── Run query ─────────────────────────────────────────────────────────────────
if run_btn and query.strip():
    st.session_state["query_input"] = query

    with st.spinner("🚀 Starting multi-agent pipeline …"):
        try:
            retrieve = get_retrieve(docs_dir, force_rebuild)
        except Exception as e:
            st.error(f"Failed to build RAG index: {e}")
            st.stop()

    # Live agent-step panel
    st.divider()
    steps_placeholder = st.empty()

    with st.spinner("🤖 Agents working …"):
        try:
            from src.graph.workflow import run_query

            # We run the query and then display the result
            # (LangGraph is synchronous; for true streaming use async + callbacks)
            start_time = time.time()
            result = run_query(query, retrieve)
            elapsed = time.time() - start_time

            st.session_state["last_result"] = result
            st.session_state["last_elapsed"] = elapsed
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

# ── Display results ───────────────────────────────────────────────────────────
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    elapsed = st.session_state.get("last_elapsed", 0)

    agent_steps = result.get("agent_steps", [])
    chunks = result.get("retrieved_chunks", [])
    confidence = result.get("confidence", 0.0)
    final_answer = result.get("final_answer") or result.get("structured_answer", "No answer generated.")
    sources = result.get("sources", [])
    iterations = result.get("iteration", 1)
    plan = result.get("task_plan")

    # Top metrics row
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{confidence:.0%}")
    m2.metric("Iterations", iterations)
    m3.metric("Chunks retrieved", len(chunks))
    m4.metric("Elapsed", f"{elapsed:.1f}s")

    # Confidence colour
    bar_color = "#2ECC71" if confidence >= 0.7 else "#E74C3C"
    st.markdown(
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar" style="width:{confidence*100:.0f}%;background:{bar_color};"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Three-column layout: Agent Steps | Chunks | Answer
    col_steps, col_chunks, col_answer = st.columns([1.2, 1.2, 2.6], gap="medium")

    with col_steps:
        st.subheader("🗂️ Agent Steps")
        for step in agent_steps:
            st.markdown(f'<div class="step-item">{step}</div>', unsafe_allow_html=True)

        if plan:
            st.markdown("**Task Plan**")
            st.caption(f"Intent: {plan.intent}")
            for t in plan.sub_tasks:
                st.caption(f"{t.id}. {t.description}")

    with col_chunks:
        st.subheader("📄 Retrieved Chunks")
        if chunks:
            for i, c in enumerate(chunks[:5], 1):
                with st.expander(f"[{i}] {Path(c['source']).name} — score {c['score']:.3f}"):
                    if c.get("page") is not None:
                        st.caption(f"Page {c['page']}")
                    st.write(c["content"][:500] + ("…" if len(c["content"]) > 500 else ""))
        else:
            st.caption("No chunks retrieved.")

    with col_answer:
        st.subheader("✅ Final Answer")
        review = result.get("review")
        if review:
            rcol1, rcol2 = st.columns(2)
            rcol1.metric("Completeness", f"{review.completeness:.0%}")
            rcol2.metric("Groundedness", f"{review.groundedness:.0%}")
            if not review.approved:
                st.warning(f"⚠️ Max iterations reached. Last reviewer feedback: {review.feedback}")

        st.markdown(final_answer)

        if sources:
            st.divider()
            st.caption("**Sources cited:**")
            chips = " ".join(
                f'<span class="source-chip">{Path(s).name}</span>' for s in sources
            )
            st.markdown(chips, unsafe_allow_html=True)

    # LangSmith link hint
    project = os.getenv("LANGCHAIN_PROJECT", "GenAI_PM_MultiAgent")
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        st.divider()
        st.info(
            f"🔭 **LangSmith tracing is ON.** "
            f"View full execution trace at "
            f"[smith.langchain.com](https://smith.langchain.com) → project `{project}`"
        )


# ── Evaluation Harness ────────────────────────────────────────────────────────
if run_eval:
    st.divider()
    st.subheader("🧪 Evaluation Harness")
    with st.spinner("Running 30 test questions … this may take several minutes"):
        try:
            retrieve = get_retrieve(docs_dir, False)
            from src.evaluation.harness import EvaluationHarness
            harness = EvaluationHarness(retrieve=retrieve)
            df = harness.run(run_baseline=False, verbose=False)

            st.success("Evaluation complete!")
            ma_correct = df["ma_correctness"].mean()
            ma_hallu = df["ma_hallucination"].mean() * 100
            ma_lat = df["ma_latency"].mean()

            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Avg Correctness", f"{ma_correct:.0%}")
            ec2.metric("Hallucination Rate", f"{ma_hallu:.1f}%")
            ec3.metric("Avg Latency", f"{ma_lat:.1f}s")

            st.dataframe(df[["id", "category", "question", "ma_correctness",
                              "ma_hallucination", "ma_confidence", "ma_latency"]])

            csv = df.to_csv(index=False)
            st.download_button("⬇ Download CSV", csv, "eval_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
