"""Answer questions about the ACRE 10-Q: retrieve, rerank, answer with a local LLM, verify.

Shared by the notebooks, the evaluation and the Streamlit app, so they all behave the same.
Everything runs locally: Chroma on disk, Hugging Face models from the cache, the LLM in Ollama.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

# Hugging Face warns that Windows can't use symlinks in its model cache; caching still works.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.getenv("ACRE_LLM_MODEL", "phi4-mini")
STRATEGIES = ["fixed", "semantic"]
RETRIEVE_K = 20
KEEP = 5

# A "Not found" answer is suspicious when the reranker scored a chunk above this
# (positive scores mean the reranker thinks the chunk is relevant).
NOT_FOUND_SUSPICIOUS_SCORE = 0.0

NOT_FOUND = "Not found in the provided excerpts."

PROMPTS = {
    # v1: free-form answer with inline citations.
    "v1": """You answer questions about the Form 10-Q of Ares Commercial Real Estate Corporation (ACRE) \
for the quarter ended June 30, 2026, using ONLY the excerpts provided.

Rules:
- "This quarter" means the three months ended June 30, 2026, not the six months.
- After every fact, cite the excerpt it came from with its id in square brackets, e.g. [fixed-0063].
- Copy numbers exactly as written in the excerpts. Do not calculate or round.
- If the excerpts do not contain the answer, reply exactly: Not found in the provided excerpts.
- Answer in 2 to 4 sentences.""",
    # v2: quote the evidence first, in a fixed format that code can check.
    "v2": """You answer questions about the Form 10-Q of Ares Commercial Real Estate Corporation (ACRE) \
for the quarter ended June 30, 2026. Use only the excerpts provided. "This quarter" means the three months \
ended June 30, 2026.

Reply in exactly this format:
QUOTE: <the sentence or table row from the excerpts that answers the question, copied word for word>
SOURCE: <the id of that excerpt, e.g. [fixed-0063]>
ANSWER: <one or two sentences answering the question, with numbers exactly as they appear in the quote>

Only if no excerpt is about the question at all, reply:
QUOTE: none
SOURCE: none
ANSWER: Not found in the provided excerpts.""",
}
DEFAULT_PROMPT = "v1"  # best in 07_evaluate (with the retry); v2's quotes often fail the strict quote check

CHUNK_ID = r"(?:fixed|semantic)-\d{4}"
FIELDS = r"(QUOTE|SOURCE|ANSWER):\s*(.*?)(?=\n\s*(?:QUOTE|SOURCE|ANSWER):|\Z)"


# ---------- models (each loaded once per process) ----------

@lru_cache(maxsize=None)
def embeddings():
    # Must be the same model that built the index, or question and chunk vectors won't be comparable.
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


@lru_cache(maxsize=None)
def store(strategy):
    return Chroma(collection_name=f"acre_{strategy}", embedding_function=embeddings(),
                  persist_directory=str(CHROMA_DIR))


@lru_cache(maxsize=None)
def reranker():
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=None)
def llm(model=LLM_MODEL):
    # temperature=0: the same input always gives the same answer.
    # num_ctx: tokens the model reads at once; 5 chunks + instructions fit well under 4096.
    return ChatOllama(model=model, temperature=0, num_ctx=4096)


# ---------- the steps ----------

def retrieve(question, strategy="fixed"):
    """Top RETRIEVE_K chunks by vector distance, reranked; returns the best KEEP as (score, doc)."""
    hits = store(strategy).similarity_search(question, k=RETRIEVE_K)
    scores = reranker().predict([(question, doc.page_content) for doc in hits])
    ranked = sorted(zip((float(s) for s in scores), hits), key=lambda pair: pair[0], reverse=True)
    return ranked[:KEEP]


def answer(question, chunks, prompt=DEFAULT_PROMPT, model=LLM_MODEL, retry_of=None, problems=None):
    """Ask the LLM. On a retry, show it its previous reply and what was wrong with it."""
    excerpts = "\n\n".join(
        f"[{doc.metadata['chunk_id']}] ({doc.metadata['section']})\n{doc.page_content}" for _, doc in chunks
    )
    messages = [
        ("system", PROMPTS[prompt]),
        ("human", f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"),
    ]
    if retry_of is not None:
        messages += [
            ("ai", retry_of),
            ("human", f"Your reply had these problems: {'; '.join(problems)}. Read the excerpts again and "
                      "reply once more, following the instructions exactly."),
        ]
    return llm(model).invoke(messages).content.strip()


# ---------- checking ----------

def numbers_in(text):
    """Numbers like 287.3, 0.15 or 1,234 (commas removed), skipping chunk ids inside [...]."""
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def normalize(text):
    """Lower-case, no spaces or quote marks: so a quote still matches if spacing or ’ vs ' differ."""
    return re.sub(r"[\s\"'“”‘’]+", "", text).lower().rstrip(".…")


def parse(reply, prompt=DEFAULT_PROMPT):
    """Split a reply into answer, quote and cited chunk ids."""
    if prompt == "v1":
        return {"answer": reply, "quote": "", "sources": re.findall(CHUNK_ID, reply), "format_ok": True}
    found = dict(re.findall(FIELDS, reply, flags=re.S))
    parts = {key: found.get(key, "").strip() for key in ("QUOTE", "SOURCE", "ANSWER")}
    return {
        "answer": parts["ANSWER"] or reply,  # whole reply if the model ignored the format
        "quote": parts["QUOTE"],
        "sources": re.findall(CHUNK_ID, parts["SOURCE"]),
        "format_ok": all(parts.values()),
    }


def verify(reply, chunks, prompt=DEFAULT_PROMPT):
    """Parse the reply and list anything that code can prove is wrong. No problems = passed."""
    result = parse(reply, prompt)
    by_id = {doc.metadata["chunk_id"]: doc.page_content for _, doc in chunks}
    top_score = max((score for score, _ in chunks), default=float("-inf"))
    result["not_found"] = result["answer"].startswith(NOT_FOUND)

    problems = []
    if result["not_found"]:
        if top_score > NOT_FOUND_SUSPICIOUS_SCORE:
            problems.append(f"said not found, but an excerpt looks relevant (rerank score {top_score:.1f})")
    else:
        if not result["format_ok"]:
            problems.append("did not follow the QUOTE / SOURCE / ANSWER format")
        if not result["sources"]:
            problems.append("no excerpt id cited")
        unknown = sorted(set(result["sources"]) - set(by_id))
        if unknown:
            problems.append(f"cited excerpts that were not given: {unknown}")
        if prompt != "v1":
            cited_text = "\n".join(by_id.get(i, "") for i in result["sources"])
            if not cited_text or normalize(result["quote"]) not in normalize(cited_text):
                problems.append("quote not found word for word in the cited excerpt")
        invented = sorted(numbers_in(result["answer"]) - numbers_in("\n".join(by_id.values())))
        if invented:
            problems.append(f"numbers not in the excerpts: {invented}")

    result["problems"] = problems
    result["passed"] = not problems
    return result
