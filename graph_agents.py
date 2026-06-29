"""
LangGraph multi-agent system for the NVIDIA CEO Agent.

This is the orchestration layer rebuilt as a LangGraph graph of separate agents.
It REUSES the existing task functions (scrapers, clean, index, analyze,
recommend, sentiment, validate, briefing) - those remain the working engines.
LangGraph provides the multi-agent structure and routing on top.

Agent design (honest mix):
  TRUE AGENTS (own reasoning / decisions):
    - supervisor : decides which agent runs next, given the goal + state
    - analysis   : RAG + self-evaluation loop (reuses agent.py)
    - validation : rule gate + LLM groundedness judge (reuses validate.py)
  TASK NODES (deterministic execution, no fake agency):
    - collection, indexing, recommendation, sentiment, briefing

Build is staged. THIS FILE currently wires the supervisor + a few nodes to
prove the pattern. More nodes are added once this runs.

Run:  python graph_agents.py
"""

import json
import os
import subprocess
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

import sys

LLM_MODEL = "qwen2.5:7b-instruct"
llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

# Collection (scraping) runs only if raw data is missing OR --scrape is passed.
# This prevents a live scrape (and possible rate-limiting) on every run while
# still allowing a full from-scratch pipeline on demand.
ALLOW_SCRAPE = "--scrape" in sys.argv

# The agents whose outputs define "goal met". Data-prep agents (collection,
# cleaning, indexing) are prerequisites, not goal outputs themselves.
FINAL_GOAL_AGENTS = ["recommendation", "sentiment", "validation", "briefing"]


# ---------------------------------------------------------------------------
# SHARED STATE  (LangGraph passes this dict between agents)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    goal: str
    completed: list      # names of agents that have finished
    next_agent: str      # supervisor's decision
    log: list            # human-readable trace of decisions


# ---------------------------------------------------------------------------
# Helper: what outputs exist on disk (the agents' shared memory of progress)
# ---------------------------------------------------------------------------
ARTIFACTS = {
    "collection": "data/raw",
    "cleaning": "data/clean/docs.json",
    "indexing": "chroma_db",
    "analysis": "data/analysis.json",
    "recommendation": "data/recommendations.json",
    "sentiment": "data/sentiment.json",
    "validation": "data/recommendations_validated.json",
    "briefing": "data/briefing.json",
}
# Prereqs: an agent can only run once these artifacts exist.
PREREQS = {
    "collection": [],
    "cleaning": ["data/raw"],
    "indexing": ["data/clean/docs.json"],
    "analysis": ["chroma_db"],
    "recommendation": ["data/analysis.json"],
    "sentiment": ["data/clean/docs.json"],
    "validation": ["data/recommendations.json"],
    "briefing": ["data/analysis.json", "data/recommendations.json"],
}
# The agents whose outputs together satisfy the goal. Collection/cleaning/
# indexing are means to that end; they are eligible only when their output is
# missing, so the supervisor naturally routes through them when data is absent.
GOAL_AGENTS = ["collection", "cleaning", "indexing", "analysis",
               "recommendation", "sentiment", "validation", "briefing"]


def _exists(p):
    return os.path.exists(p)


def _run(cmd):
    return subprocess.run(cmd).returncode == 0


# ---------------------------------------------------------------------------
# SUPERVISOR AGENT  (true agent - decides the next agent via the LLM)
# ---------------------------------------------------------------------------
def supervisor(state: AgentState) -> AgentState:
    done = [a for a in ARTIFACTS if _exists(ARTIFACTS[a])]

    eligible = []
    for a in GOAL_AGENTS:
        if _exists(ARTIFACTS[a]):
            continue  # already produced its output
        if not all(_exists(p) for p in PREREQS[a]):
            continue  # prerequisites not ready
        if a == "collection" and not ALLOW_SCRAPE:
            continue  # don't scrape unless permitted
        eligible.append(a)

    # Goal is met when the FINAL goal artifacts exist (data-prep is a means).
    if all(_exists(ARTIFACTS[a]) for a in FINAL_GOAL_AGENTS):
        decision, reason = "FINISH", "All goal artifacts exist."
    elif not eligible:
        decision, reason = "FINISH", "No eligible agent can run."
    else:
        prompt = f"""You are the SUPERVISOR agent coordinating a team of agents.

GOAL: {state['goal']}

Agents that have completed: {done}
Agents eligible to run now (prerequisites met): {eligible}

Choose the single next agent to run from the eligible list.
Return STRICT JSON: {{"next_agent": "<name>", "reason": "<one sentence>"}}
Output ONLY the JSON."""
        try:
            raw = llm.invoke(prompt).content
            raw = raw.replace("```json", "").replace("```", "").strip()
            s, e = raw.find("{"), raw.rfind("}")
            parsed = json.loads(raw[s:e + 1])
            decision = parsed.get("next_agent", eligible[0])
            reason = parsed.get("reason", "")
            if decision not in eligible:
                decision, reason = eligible[0], "fallback: LLM picked invalid agent."
        except Exception as ex:
            decision, reason = eligible[0], f"fallback ({type(ex).__name__})."

    log = state["log"] + [f"SUPERVISOR -> {decision} ({reason})"]
    print(f"SUPERVISOR decides: {decision} - {reason}")
    return {**state, "next_agent": decision, "log": log}


# ---------------------------------------------------------------------------
# TASK / AGENT NODES  (each reuses the existing working script)
# ---------------------------------------------------------------------------
def collection_node(state: AgentState) -> AgentState:
    print("  [collection] scraping sources (RSS)...")
    _run(["python", "scrapers/reddit_scraper.py"])
    _run(["python", "scrapers/news_scraper.py"])
    _run(["python", "scrapers/nvidia_ir_scraper.py"])
    return {**state, "completed": state["completed"] + ["collection"]}


def cleaning_node(state: AgentState) -> AgentState:
    print("  [cleaning] dedup + normalize (clean.py)...")
    _run(["python", "clean.py"])
    return {**state, "completed": state["completed"] + ["cleaning"]}


def indexing_node(state: AgentState) -> AgentState:
    print("  [indexing] embed -> ChromaDB (index.py)...")
    _run(["python", "index.py"])
    return {**state, "completed": state["completed"] + ["indexing"]}


def analysis_agent(state: AgentState) -> AgentState:
    # TRUE AGENT: agent.py runs the self-evaluating RAG loop.
    print("  [analysis agent] running self-evaluating RAG (agent.py)...")
    _run(["python", "agent.py"])
    return {**state, "completed": state["completed"] + ["analysis"]}


def recommendation_node(state: AgentState) -> AgentState:
    print("  [recommendation] synthesizing (recommend.py)...")
    _run(["python", "recommend.py"])
    return {**state, "completed": state["completed"] + ["recommendation"]}


def sentiment_node(state: AgentState) -> AgentState:
    print("  [sentiment] VADER scoring (sentiment.py)...")
    _run(["python", "sentiment.py"])
    return {**state, "completed": state["completed"] + ["sentiment"]}


def validation_agent(state: AgentState) -> AgentState:
    # TRUE AGENT: validate.py judges groundedness.
    print("  [validation agent] rule gate + LLM groundedness (validate.py)...")
    _run(["python", "validate.py"])
    return {**state, "completed": state["completed"] + ["validation"]}


def briefing_node(state: AgentState) -> AgentState:
    print("  [briefing] CEO summary (briefing.py)...")
    _run(["python", "briefing.py"])
    return {**state, "completed": state["completed"] + ["briefing"]}


# ---------------------------------------------------------------------------
# ROUTING  (supervisor's decision -> which node to go to)
# ---------------------------------------------------------------------------
def route(state: AgentState) -> str:
    return state["next_agent"]


# ---------------------------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor)
    g.add_node("collection", collection_node)
    g.add_node("cleaning", cleaning_node)
    g.add_node("indexing", indexing_node)
    g.add_node("analysis", analysis_agent)
    g.add_node("recommendation", recommendation_node)
    g.add_node("sentiment", sentiment_node)
    g.add_node("validation", validation_agent)
    g.add_node("briefing", briefing_node)

    g.add_edge(START, "supervisor")

    # Supervisor routes to whichever agent it chose (or ends).
    g.add_conditional_edges("supervisor", route, {
        "collection": "collection",
        "cleaning": "cleaning",
        "indexing": "indexing",
        "analysis": "analysis",
        "recommendation": "recommendation",
        "sentiment": "sentiment",
        "validation": "validation",
        "briefing": "briefing",
        "FINISH": END,
    })

    # After each agent finishes, control returns to the supervisor to re-decide.
    for node in ["collection", "cleaning", "indexing", "analysis",
                 "recommendation", "sentiment", "validation", "briefing"]:
        g.add_edge(node, "supervisor")

    return g.compile()


def main():
    graph = build_graph()
    initial = {
        "goal": ("Produce validated, evidence-based NVIDIA recommendations, "
                 "a sentiment picture, and a CEO briefing."),
        "completed": [],
        "next_agent": "",
        "log": [],
    }
    print("=" * 60)
    print("LANGGRAPH MULTI-AGENT SYSTEM")
    print("=" * 60)
    # recursion_limit lets the supervisor loop run enough times.
    final = graph.invoke(initial, {"recursion_limit": 50})
    print("\n" + "=" * 60)
    print("AGENT DECISION LOG:")
    for line in final["log"]:
        print("  " + line)
    print("=" * 60)
    print("Launch dashboard:  streamlit run app.py")


if __name__ == "__main__":
    main()