"""The question-answering flow as a LangGraph graph.

    retrieve ──▶ answer ──▶ verify ──▶ done
                   ▲           │
                   └── retry ──┘   (once, if the checks found problems)

Each step is a small function that reads the shared state and returns what it adds.
"""

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import pipeline

MAX_ATTEMPTS = 2  # the first answer plus one retry


class State(TypedDict, total=False):
    question: str
    strategy: str
    prompt: str
    model: str
    chunks: list  # [(rerank score, Document)], best first
    reply: str  # the LLM's raw reply
    result: dict  # pipeline.verify() output: answer, quote, sources, problems, passed
    attempts: int
    history: list  # every reply and its problems, oldest first


def retrieve_step(state: State) -> State:
    return {"chunks": pipeline.retrieve(state["question"], state["strategy"]), "attempts": 0, "history": []}


def answer_step(state: State) -> State:
    previous = state.get("result")
    reply = pipeline.answer(
        state["question"], state["chunks"], state["prompt"], state["model"],
        retry_of=state.get("reply") if previous else None,
        problems=previous["problems"] if previous else None,
    )
    return {"reply": reply, "attempts": state["attempts"] + 1}


def verify_step(state: State) -> State:
    result = pipeline.verify(state["reply"], state["chunks"], state["prompt"])
    return {"result": result, "history": state["history"] + [{"reply": state["reply"], "problems": result["problems"]}]}


def retry_or_finish(state: State) -> str:
    if state["result"]["passed"] or state["attempts"] >= MAX_ATTEMPTS:
        return "finish"
    return "retry"


def build():
    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve_step)
    graph.add_node("answer", answer_step)
    graph.add_node("verify", verify_step)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "verify")
    graph.add_conditional_edges("verify", retry_or_finish, {"retry": "answer", "finish": END})
    return graph.compile()


_graph = None


def ask(question, strategy="fixed", prompt=pipeline.DEFAULT_PROMPT, model=pipeline.LLM_MODEL):
    """Run the whole flow for one question; returns the final state plus the time taken."""
    global _graph
    _graph = _graph or build()
    start = time.perf_counter()
    state = _graph.invoke({"question": question, "strategy": strategy, "prompt": prompt, "model": model})
    state["seconds"] = round(time.perf_counter() - start, 1)
    return state
