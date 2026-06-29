"""
PROOF OF CONCEPT - confirm LangGraph runs with your local Ollama (qwen).

This does NOT touch your real system. It only verifies that:
  1. langgraph + langchain-ollama import correctly
  2. a LangGraph node can call your local qwen2.5 model
  3. the graph compiles and runs end to end

If this prints a sensible answer from qwen, the framework works with your
setup and a full LangGraph rebuild is feasible. If it errors, we fix the
integration BEFORE porting anything.

Run:  python poc_langgraph.py
"""

from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


# 1. The shared state passed between nodes (LangGraph is state-based).
class State(TypedDict):
    question: str
    answer: str


# 2. A model handle pointing at your LOCAL Ollama (no API key, no cloud).
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.2)


# 3. A single node: takes the question from state, asks qwen, writes the answer.
def answer_node(state: State) -> State:
    resp = llm.invoke(state["question"])
    return {"answer": resp.content}


# 4. Build the graph: START -> answer_node -> END
builder = StateGraph(State)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)
graph = builder.compile()


if __name__ == "__main__":
    print("Invoking LangGraph (one node) with local qwen...\n")
    result = graph.invoke(
        {"question": "In one sentence, what is NVIDIA known for?", "answer": ""}
    )
    print("qwen answered:\n ", result["answer"])
    print("\nIf you see a sensible sentence above, LangGraph + Ollama WORKS.")