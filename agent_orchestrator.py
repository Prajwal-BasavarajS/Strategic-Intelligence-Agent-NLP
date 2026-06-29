"""
Goal-decomposition orchestrator for the NVIDIA CEO Agent.

This is the PLANNING agent. Instead of following a fixed script, it is given a
high-level GOAL and the set of available TOOLS, and it autonomously:

  1. DECOMPOSES the goal into the sub-tasks it judges necessary, deciding for
     each tool whether to INCLUDE or SKIP it - with a stated reason.
     (Freedom to skip, but never silently: every skip is reported and justified.)

  2. EXECUTES the included tasks in an order that respects data dependencies
     (it cannot, e.g., index before cleaning), running each as a tool.

  3. RE-PLANS at a checkpoint after analysis: given what it actually found, it
     decides whether the current plan is still adequate or whether an extra
     step is needed - the hallmark of an agent that reasons about its own state.

This demonstrates:
  - Planning before execution        (the decomposition phase)
  - Autonomous decision-making       (include/skip per task, with reasons)
  - Multi-step task execution        (dependency-ordered tool runs)
  - Memory                           (tracks plan, completed tasks, artifacts)
  - Tool usage beyond the LLM        (scrapers, embeddings, ChromaDB, VADER)

"""

import json
import os
import subprocess
import sys
import time

import ollama

LLM_MODEL = "qwen2.5:7b-instruct"
MAX_STEPS = 15

GOAL = ("Produce an executive intelligence briefing on NVIDIA's strategic "
        "position: validated, evidence-based recommendations, a sentiment "
        "picture, and a CEO briefing - all grounded in freshly analyzed data.")


def _exists(path):
    return os.path.exists(path)


# TOOL REGISTRY

TOOLS = {
    "scrape": {
        "desc": "Collect raw documents from Reddit, news, and NVIDIA IR (RSS). "
                "Needed only if fresh data is required and raw data is absent.",
        "cmd": ["python", "scrapers/reddit_scraper.py"],
        "extra_cmds": [["python", "scrapers/news_scraper.py"],
                       ["python", "scrapers/nvidia_ir_scraper.py"]],
        "produces": "data/raw",
        "requires": [],
    },
    "clean": {
        "desc": "Clean and deduplicate raw documents into data/clean/docs.json.",
        "cmd": ["python", "clean.py"],
        "produces": "data/clean/docs.json",
        "requires": ["data/raw"],
    },
    "index": {
        "desc": "Embed cleaned docs into the ChromaDB vector store.",
        "cmd": ["python", "index.py"],
        "produces": "chroma_db",
        "requires": ["data/clean/docs.json"],
    },
    "analyze": {
        "desc": "Agentic RAG analysis (agent.py): retrieve, reason, self-evaluate "
                "findings for opportunities/risks/trends.",
        "cmd": ["python", "agent.py"],
        "produces": "data/analysis.json",
        "requires": ["chroma_db"],
    },
    "recommend": {
        "desc": "Synthesize analysis into prioritized recommendations.",
        "cmd": ["python", "recommend.py"],
        "produces": "data/recommendations.json",
        "requires": ["data/analysis.json"],
    },
    "sentiment": {
        "desc": "Score sentiment per source with VADER.",
        "cmd": ["python", "sentiment.py"],
        "produces": "data/sentiment.json",
        "requires": ["data/clean/docs.json"],
    },
    "briefing": {
        "desc": "Generate the CEO briefing from analysis + recommendations.",
        "cmd": ["python", "briefing.py"],
        "produces": "data/briefing.json",
        "requires": ["data/analysis.json", "data/recommendations.json"],
    },
    "validate": {
        "desc": "Validate recommendations before presenting (rule gate + LLM "
                "groundedness check).",
        "cmd": ["python", "validate.py"],
        "produces": "data/recommendations_validated.json",
        "requires": ["data/recommendations.json"],
    },
}



# PHASE 1: DECOMPOSE THE GOAL

def decompose_goal(allow_scrape: bool) -> dict:
    """Ask the LLM to decide, for each tool, whether the goal requires it.

    Returns {"plan": [{"task","decision","reason"}...]}.
    """
    tool_lines = "\n".join(f"  - {n}: {t['desc']}" for n, t in TOOLS.items())
    scrape_note = ("Fresh scraping IS permitted this run."
                   if allow_scrape else
                   "Fresh scraping is NOT permitted this run; if raw data is "
                   "missing the goal cannot use new collection - rely on existing data.")
    raw_state = "present" if _exists("data/raw") else "absent"

    prompt = f"""You are the planning agent for a strategic-intelligence system.

GOAL:
{GOAL}

AVAILABLE TOOLS:
{tool_lines}

CONTEXT:
- Raw collected data is currently {raw_state}.
- {scrape_note}

Decompose the goal: decide, for EACH tool, whether the goal REQUIRES it
(INCLUDE) or whether it can be skipped (SKIP). You have genuine freedom to skip
a tool if it is not needed for this goal - but you must justify every decision.

Return STRICT JSON: a list, one object per tool, EXACTLY these keys:
  "task": the tool name
  "decision": "include" or "skip"
  "reason": one short sentence justifying the decision

Output ONLY the JSON array.

JSON:"""
    resp = ollama.chat(model=LLM_MODEL,
                       messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0.2})
    raw = resp["message"]["content"].replace("```json", "").replace("```", "").strip()
    s, e = raw.find("["), raw.rfind("]")
    if s == -1 or e == -1:
        raise ValueError(f"planner returned no JSON plan:\n{raw[:300]}")
    plan = json.loads(raw[s:e + 1])
    return {"plan": plan}


# PHASE 3: RE-PLAN CHECKPOINT (after analysis)

def replan_after_analysis(included: set) -> dict:
    """After analysis, the agent reasons about whether the plan still holds.

    Returns {"adjust": bool, "reason": str}. (Kept conservative: it can flag
    that more work is warranted, which we record in the trace.)
    """
    analysis = {}
    if _exists("data/analysis.json"):
        with open("data/analysis.json") as f:
            analysis = json.load(f)
    # Summarize what analysis produced for the agent to reflect on.
    summary = {cat: {"findings": len(blk.get("findings", [])),
                     "low_confidence": blk.get("low_confidence", False)}
               for cat, blk in analysis.items()}

    prompt = f"""You are the planning agent. Analysis has completed. Reflect on
whether your plan is still adequate.

ANALYSIS RESULT SUMMARY (per category):
{json.dumps(summary, indent=2)}

If any category has low_confidence or too few findings, note it. You cannot
re-run analysis in this checkpoint, but you should record whether the briefing
should treat some findings as provisional.

Return STRICT JSON with EXACTLY:
  "adjust": true or false   (true if the result is weaker than ideal)
  "reason": one short sentence

Output ONLY the JSON object.

JSON:"""
    try:
        resp = ollama.chat(model=LLM_MODEL,
                           messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.2})
        raw = resp["message"]["content"].replace("```json", "").replace("```", "").strip()
        s, e = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[s:e + 1])
        return {"adjust": bool(parsed.get("adjust", False)),
                "reason": parsed.get("reason", "")}
    except Exception:
        return {"adjust": False, "reason": "checkpoint skipped (parse issue)"}



# EXECUTION

def run_tool(name: str) -> bool:
    t = TOOLS[name]
    for cmd in [t["cmd"]] + t.get("extra_cmds", []):
        print(f"      running: {' '.join(cmd)}")
        if subprocess.run(cmd).returncode != 0:
            print(f"      tool '{name}' FAILED")
            return False
    return True


def ready(name: str) -> bool:
    return all(_exists(r) for r in TOOLS[name]["requires"])


def main():
    allow_scrape = "--scrape" in sys.argv
    print("=" * 66)
    print("GOAL-DECOMPOSITION AGENT")
    print("=" * 66)
    print(f"\nGOAL: {GOAL}\n")

    #  PHASE 1: decompose 
    print("PHASE 1 - Decomposing goal into sub-tasks...\n")
    plan = decompose_goal(allow_scrape)["plan"]

    included, skipped = [], []
    decided = set()
    for item in plan:
        task = item.get("task")
        if task not in TOOLS:
            continue
        decided.add(task)
        if item.get("decision") == "include":
            included.append(task)
            print(f"  INCLUDE  {task:<10} - {item.get('reason','')}")
        else:
            skipped.append((task, item.get("reason", "")))
            print(f"  SKIP     {task:<10} - {item.get('reason','')}")

    # Completeness guarantee: every registered tool must appear as a reported
    # decision - no tool is silently dropped. Fill in any the LLM omitted.
    for task in TOOLS:
        if task in decided:
            continue
        if task == "scrape":
            if allow_scrape and not _exists("data/raw"):
                # genuinely needed and permitted -> include it
                included.insert(0, task)
                reason = "fresh collection requested and no raw data present."
                print(f"  INCLUDE  {task:<10} - {reason}")
            else:
                reason = ("fresh collection not requested; existing raw data is "
                          "sufficient for this goal."
                          if _exists("data/raw")
                          else "fresh collection not permitted this run (no --scrape).")
                skipped.append((task, reason))
                print(f"  SKIP     {task:<10} - {reason}")
        else:
            # Any other omitted tool: default to skip, reported.
            reason = "not selected by the planner for this goal."
            skipped.append((task, reason))
            print(f"  SKIP     {task:<10} - {reason}")

    # Safety: if downstream tasks need data that won't exist, say so plainly.
    if not _exists("data/raw") and "scrape" not in included:
        print("\n  NOTE: raw data is absent and scrape is not included; "
              "tasks needing collected data may be unable to run.")

    #  PHASE 2: execute included tasks in dependency order 
    print("\nPHASE 2 - Executing included tasks (dependency-ordered)...")
    history = []
    did_analysis = False
    for step in range(1, MAX_STEPS + 1):
        # pick the next included task whose prereqs are met and output missing
        nxt = None
        for task in included:
            if _exists(TOOLS[task]["produces"]):
                continue
            if ready(task):
                nxt = task
                break
        if nxt is None:
            break

        print(f"\nStep {step}: running '{nxt}'")
        ok = run_tool(nxt)
        history.append({"step": step, "tool": nxt, "ok": ok})
        if not ok:
            print("Stopping due to tool failure.")
            break

        #  PHASE 3: re-plan checkpoint, once, right after analysis 
        if nxt == "analyze" and not did_analysis:
            did_analysis = True
            print("\n  CHECKPOINT - re-planning after analysis...")
            chk = replan_after_analysis(set(included))
            print(f"  re-plan: adjust={chk['adjust']} - {chk['reason']}")
        time.sleep(0.2)

    #  SUMMARY 
    print("\n" + "=" * 66)
    print("AGENT RUN COMPLETE")
    print("=" * 66)
    print("Plan decisions:")
    for task in included:
        print(f"  - included: {task}")
    for task, reason in skipped:
        print(f"  - SKIPPED:  {task} ({reason})")
    print("\nExecution trace:")
    for h in history:
        print(f"  step {h['step']}: ran {h['tool']} (ok={h['ok']})")
    print("\nLaunch dashboard:  streamlit run app.py")


if __name__ == "__main__":
    main()