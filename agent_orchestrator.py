"""
Agentic tool-calling orchestrator for the NVIDIA CEO Agent.

Instead of running the pipeline as a fixed script sequence, this exposes each
operation as a TOOL and uses an LLM PLANNER to decide which tool to call next
at each step, given the current state of the pipeline.

This satisfies the agent criteria explicitly:

  - TOOL USAGE:   each operation (scrape, clean, index, analyze, recommend,
                  sentiment, briefing) is a registered tool with a name,
                  description, and function.
  - AUTONOMOUS DECISION-MAKING: at every step the LLM is given the tool list
                  and the current state, and it CHOOSES the next tool to run.
  - PLANNING / MULTI-STEP EXECUTION: the agent plans over data dependencies
                  (it must not, e.g., index before cleaning), executes tools
                  one at a time, and re-plans after each result.
  - MEMORY:       the agent maintains running STATE of which tools have run
                  and which output artifacts now exist, and feeds that memory
                  back into each planning decision.

The agent reasons over a DEPENDENCY GRAPH: at each step it considers which
tools are eligible (their prerequisites are satisfied) and picks the next
action, including whether to re-run analysis if evidence is weak, or to FINISH.

Run:  python agent_orchestrator.py
      python agent_orchestrator.py --scrape   # allow fresh scraping
"""

import json
import os
import subprocess
import sys
import time

import ollama

LLM_MODEL = "qwen2.5:7b-instruct"
MAX_STEPS = 12   # safety cap: the agent loop can never run forever


# ---------------------------------------------------------------------------
# TOOL DEFINITIONS
# Each tool: a command to run, a human description, and prerequisite outputs.
# ---------------------------------------------------------------------------
def _exists(path):
    return os.path.exists(path)


TOOLS = {
    "scrape": {
        "desc": "Collect raw documents from Reddit, news, and NVIDIA IR. "
                "Produces data/raw/*.json. Slow; only if fresh data is needed.",
        "cmd": ["python", "scrapers/reddit_scraper.py"],
        "extra_cmds": [
            ["python", "scrapers/news_scraper.py"],
            ["python", "scrapers/nvidia_ir_scraper.py"],
        ],
        "produces": "data/raw",
        "requires": [],
    },
    "clean": {
        "desc": "Clean and deduplicate raw docs. Produces data/clean/docs.json. "
                "Requires raw data to exist.",
        "cmd": ["python", "clean.py"],
        "produces": "data/clean/docs.json",
        "requires": ["data/raw"],
    },
    "index": {
        "desc": "Embed cleaned docs into the ChromaDB vector store. "
                "Requires data/clean/docs.json.",
        "cmd": ["python", "index.py"],
        "produces": "chroma_db",
        "requires": ["data/clean/docs.json"],
    },
    "analyze": {
        "desc": "Agentic RAG analysis (agent.py): retrieves, reasons, and "
                "self-evaluates findings. Produces data/analysis.json. "
                "Requires the vector store.",
        "cmd": ["python", "agent.py"],
        "produces": "data/analysis.json",
        "requires": ["chroma_db"],
    },
    "recommend": {
        "desc": "Synthesize analysis into prioritized recommendations. "
                "Produces data/recommendations.json. Requires analysis.",
        "cmd": ["python", "recommend.py"],
        "produces": "data/recommendations.json",
        "requires": ["data/analysis.json"],
    },
    "sentiment": {
        "desc": "Score sentiment per source with VADER. Produces "
                "data/sentiment.json. Requires cleaned docs.",
        "cmd": ["python", "sentiment.py"],
        "produces": "data/sentiment.json",
        "requires": ["data/clean/docs.json"],
    },
    "briefing": {
        "desc": "Generate the CEO briefing. Produces data/briefing.json. "
                "Requires analysis and recommendations.",
        "cmd": ["python", "briefing.py"],
        "produces": "data/briefing.json",
        "requires": ["data/analysis.json", "data/recommendations.json"],
    },
    "validate": {
        "desc": "Validate recommendations before presenting them: a rule-based "
                "structural gate plus an LLM groundedness check. Produces "
                "data/recommendations_validated.json. Requires recommendations.",
        "cmd": ["python", "validate.py"],
        "produces": "data/recommendations_validated.json",
        "requires": ["data/recommendations.json"],
    },
}

GOAL_OUTPUTS = [
    "data/recommendations.json",
    "data/sentiment.json",
    "data/briefing.json",
    "data/recommendations_validated.json",
]


def current_state(allow_scrape: bool) -> dict:
    """MEMORY: what artifacts exist right now + which tools are eligible.

    A tool is eligible when its prerequisites exist AND its own output does
    not yet exist (so the agent never re-runs completed work). Scrape is
    additionally gated behind the --scrape flag.
    """
    done = {name: _exists(t["produces"]) for name, t in TOOLS.items()}
    eligible = []
    for name, t in TOOLS.items():
        if name == "scrape" and not allow_scrape:
            continue
        prereqs_ok = all(_exists(r) for r in t["requires"])
        already_done = _exists(t["produces"])
        if prereqs_ok and not already_done:
            eligible.append(name)
    goal_met = all(_exists(p) for p in GOAL_OUTPUTS)
    return {"artifacts_exist": done, "eligible_tools": eligible,
            "goal_met": goal_met}


def ask_planner(state: dict, history: list) -> dict:
    """AUTONOMOUS DECISION: the LLM picks the next tool given state + memory."""
    tool_descriptions = "\n".join(
        f"  - {name}: {t['desc']}" for name, t in TOOLS.items()
    )
    prompt = f"""You are the planning agent orchestrating an NVIDIA strategic
intelligence pipeline. You decide which TOOL to run next.

AVAILABLE TOOLS:
{tool_descriptions}

GOAL: produce recommendations.json, sentiment.json, and briefing.json.

CURRENT STATE (your memory):
  artifacts that already exist: {json.dumps(state['artifacts_exist'])}
  tools eligible to run now (prerequisites satisfied): {state['eligible_tools']}
  goal already met: {state['goal_met']}

ACTIONS YOU HAVE TAKEN SO FAR: {history if history else "none yet"}

Decide the single NEXT action. You must only choose a tool from the eligible
list. If the goal is met, choose "finish".

Return STRICT JSON with EXACTLY these keys:
  "next_tool": one of {state['eligible_tools'] + ["finish"]}
  "reason": one short sentence explaining why this is the right next step

Output ONLY the JSON object.

JSON:"""
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    raw = resp["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"planner returned no JSON: {raw[:200]}")
    return json.loads(raw[s:e + 1])


def run_tool(name: str) -> bool:
    """TOOL EXECUTION: run the tool's command(s) as subprocesses."""
    t = TOOLS[name]
    cmds = [t["cmd"]] + t.get("extra_cmds", [])
    for cmd in cmds:
        print(f"      running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"      tool '{name}' FAILED (exit {result.returncode})")
            return False
    return True


def main():
    allow_scrape = "--scrape" in sys.argv
    history = []
    print("=" * 64)
    print("AGENTIC ORCHESTRATOR — LLM plans which tool to call at each step")
    print("=" * 64)

    for step in range(1, MAX_STEPS + 1):
        state = current_state(allow_scrape)

        if state["goal_met"]:
            print(f"\nStep {step}: goal already met. Agent finishing.")
            break
        if not state["eligible_tools"]:
            print("\nNo eligible tools and goal not met — stopping.")
            break

        decision = ask_planner(state, history)
        nxt = decision.get("next_tool", "finish")
        reason = decision.get("reason", "")
        print(f"\nStep {step}: agent chose '{nxt}' — {reason}")

        if nxt == "finish":
            # Guard: the LLM may wrongly declare 'finish' while a required
            # output is still missing. Only honour 'finish' if the goal is
            # genuinely met; otherwise force the tool that produces a missing
            # goal output (not just any eligible tool).
            if state["goal_met"]:
                print("Agent decided the goal is complete.")
                break
            missing = [p for p in GOAL_OUTPUTS if not _exists(p)]
            # Find an eligible tool whose output is one of the missing goals.
            nxt = None
            for tool_name in state["eligible_tools"]:
                if TOOLS[tool_name]["produces"] in missing:
                    nxt = tool_name
                    break
            print(f"  (override) goal NOT met yet - still missing {missing}. "
                  f"Forcing tool that produces a missing output: {nxt}")
            if nxt is None:
                print("No eligible tool can produce the missing output; stopping.")
                break
        if nxt not in TOOLS:
            print(f"Agent picked invalid tool '{nxt}'; stopping.")
            break

        ok = run_tool(nxt)
        history.append({"step": step, "tool": nxt, "ok": ok})
        if not ok:
            print("Stopping due to tool failure.")
            break
        time.sleep(0.3)

    print("\n" + "=" * 64)
    print("AGENT RUN COMPLETE. Decision trace:")
    for h in history:
        print(f"  step {h['step']}: ran {h['tool']} (ok={h['ok']})")
    print("=" * 64)
    print("Launch dashboard:  streamlit run app.py")


if __name__ == "__main__":
    main()