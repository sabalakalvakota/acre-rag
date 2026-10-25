"""Ask questions about ACRE's Q2 2026 10-Q, answered with citations by a local LLM.

Run:  streamlit run app.py   (Ollama must be running, with the model pulled)
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import graph
import pipeline

EXAMPLES = [
    "What was the net interest margin this quarter?",
    "Which loans are on non-accrual status?",
    "How large is the current expected credit loss (CECL) reserve?",
    "What dividend per share was declared for the quarter?",
]

st.set_page_config(page_title="ACRE 10-Q Q&A", page_icon="📄", layout="wide")
st.title("📄 ACRE 10-Q Q&A")
st.caption("Ares Commercial Real Estate Corp. · Form 10-Q for the quarter ended June 30, 2026 · "
           f"answers by `{pipeline.LLM_MODEL}`, running locally in Ollama")

with st.sidebar:
    st.header("Settings")
    strategy = st.radio("Chunks", pipeline.STRATEGIES, help="How the filing was cut into passages.")
    prompt = st.radio("Prompt", list(pipeline.PROMPTS), index=list(pipeline.PROMPTS).index(pipeline.DEFAULT_PROMPT),
                      help="v1 (default) answers with inline citations and scored best in testing. "
                           "v2 quotes its evidence first, which code checks word for word, "
                           "but the small model often paraphrases the quote.")
    st.divider()
    st.markdown("**Try:**")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state.question = example
    st.divider()
    st.caption("Answers come only from the filing, may be wrong, and are not investment advice. "
               "Always check the quoted source.")


@st.cache_resource(show_spinner="Loading models (first time only)…")
def warm_up():
    """Load the embedding model, reranker and Chroma index once per app session."""
    for name in pipeline.STRATEGIES:
        pipeline.store(name)
    pipeline.reranker()


warm_up()

question = st.text_input("Your question", key="question", placeholder="e.g. What was the net interest margin this quarter?")
if not question:
    st.stop()

try:
    with st.spinner("Searching the filing and writing an answer (about 10–60 seconds)…"):
        state = graph.ask(question, strategy=strategy, prompt=prompt)
except Exception as error:  # usually Ollama not running, or the model not pulled
    st.error(f"Couldn't get an answer from Ollama: {error}\n\n"
             f"Check that Ollama is running and that `ollama pull {pipeline.LLM_MODEL}` has been done.")
    st.stop()

result = state["result"]

st.subheader("Answer")
if result["not_found"]:
    st.warning(result["answer"])
else:
    st.markdown(result["answer"])
if result["quote"] and result["quote"].lower() != "none":
    st.markdown(f"> {result['quote']}")
if result["sources"]:
    st.caption("Source: " + ", ".join(result["sources"]))

if result["passed"]:
    st.success(f"Checks passed · {state['attempts']} attempt(s) · {state['seconds']} s")
else:
    st.error("Checks found problems, so verify this answer against the excerpts below:\n\n- "
             + "\n- ".join(result["problems"]))

with st.expander(f"The {len(state['chunks'])} excerpts the model read"):
    for score, doc in state["chunks"]:
        cited = " ✅ cited" if doc.metadata["chunk_id"] in result["sources"] else ""
        st.markdown(f"**{doc.metadata['chunk_id']}**{cited} · rerank score {score:.2f} · {doc.metadata['section']}")
        st.text(doc.page_content)
        st.divider()

if state["attempts"] > 1:
    with st.expander("First attempt (before the retry)"):
        first = state["history"][0]
        st.text(first["reply"])
        st.caption("Problems: " + "; ".join(first["problems"]))
