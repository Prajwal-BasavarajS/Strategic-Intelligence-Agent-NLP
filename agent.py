"""
Agentic analysis loop for the NVIDIA CEO Agent.

This wraps the existing deterministic analyze.py with genuine agentic
behaviour, satisfying the four agent criteria:

  - AUTONOMOUS DECISION-MAKING: after each analysis pass, the LLM evaluates
    its own findings and DECIDES whether the evidence is sufficient or whether
    it needs to gather more. The system acts on that decision with no human
    in the loop.
  - MEMORY: the agent remembers which queries it has already tried and which
    document ids it has already seen across iterations, and uses that memory
    to broaden its search instead of repeating itself.
  - PLANNING / MULTI-STEP EXECUTION: based on the self-evaluation, the agent
    reformulates its retrieval (broader query, higher k) and loops, executing
    a multi-step plan until the findings are adequate or a max-iteration cap.

It re-uses the existing, tested building blocks from analyze.py (retrieve,
build_prompt, extract_json) so the working pipeline is untouched. If this
agent fails, analyze.py still runs exactly as before.

Run:  python agent.py
"""

import json
import os

import ollama

# Re-use the existing, tested components. analyze.py is NOT modified.
from analyze import (
    retrieve,
    build_prompt,
    extract_json,
    CATEGORY_CONFIG,
    LLM_MODEL,
)

AGENT_ANALYSIS_PATH = "data/analysis.json"
MAX_ITERS = 3            # safety cap so the loop can never run away
START_K = 6
K_STEP = 4              # how many more docs to pull when the agent decides to


def evaluate_findings(category_noun, findings) -> dict:
    """AUTONOMOUS DECISION step.

    Ask the LLM to judge its OWN findings and decide whether they are
    well-supported or whether more evidence is needed. Returns a dict like
    {"sufficient": bool, "reason": str, "refine_query": str}.
    """
    findings_text = json.dumps(findings, indent=2) if findings else "[]"
    prompt = f"""You are a critical reviewer evaluating an analyst's findings
about NVIDIA's {category_noun}.

FINDINGS:
{findings_text}

Decide whether these findings are sufficient and well-supported, or whether
the analyst should gather MORE evidence and try again.

Return STRICT JSON with EXACTLY these keys:
  "sufficient": true or false
  "reason": one short sentence explaining your decision
  "refine_query": if not sufficient, a broader/different search query string
                  to find more evidence; otherwise an empty string

Output ONLY the JSON object. No prose, no markdown.

JSON:"""
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    raw = resp["message"]["content"]
    # extract_json handles arrays or a single object; take first if a list.
    parsed = extract_json(raw)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    return {
        "sufficient": bool(parsed.get("sufficient", True)),
        "reason": parsed.get("reason", ""),
        "refine_query": parsed.get("refine_query", ""),
    }


def analyze_with_findings(category_noun, docs, guidance="") -> list:
    """One generation pass over a given set of docs (reuses build_prompt)."""
    prompt = build_prompt(category_noun, docs, guidance)
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    findings = extract_json(resp["message"]["content"])

    # Bind evidence ids -> {title, url} from the docs we actually have.
    doc_by_id = {d["id"]: d for d in docs}
    for f in findings:
        ev = []
        for eid in f.get("evidence_ids", []):
            if eid in doc_by_id:
                ev.append({"title": doc_by_id[eid]["title"],
                           "url": doc_by_id[eid]["url"]})
        f["evidence"] = ev
    return findings


def agentic_analyze_category(category: str) -> dict:
    """Run one category as an AGENT: retrieve -> generate -> self-evaluate ->
    decide -> (maybe) gather more -> loop."""
    cfg = CATEGORY_CONFIG[category]
    noun = cfg["noun"]
    guidance = cfg.get("guidance", "")

    # MEMORY: remember queries tried and doc ids already seen.
    tried_queries = []
    seen_doc_ids = set()

    query = cfg["query"]
    k = START_K
    findings = []
    trace = []   # record of the agent's decisions, for transparency/demo

    # Track the final decision so we can flag a cap-out (loop ended without
    # the agent ever judging the evidence sufficient).
    final_sufficient = False
    for iteration in range(1, MAX_ITERS + 1):
        tried_queries.append(query)
        docs = retrieve(query, k=k)

        # MEMORY in action: note which docs are newly seen this iteration.
        new_ids = [d["id"] for d in docs if d["id"] not in seen_doc_ids]
        seen_doc_ids.update(d["id"] for d in docs)

        findings = analyze_with_findings(noun, docs, guidance)

        # AUTONOMOUS DECISION: the agent judges its own work.
        decision = evaluate_findings(noun, findings)
        final_sufficient = decision["sufficient"]
        trace.append({
            "iteration": iteration,
            "query": query,
            "k": k,
            "new_docs_retrieved": len(new_ids),
            "num_findings": len(findings),
            "sufficient": decision["sufficient"],
            "reason": decision["reason"],
        })
        print(f"  [{category}] iter {iteration}: {len(findings)} findings, "
              f"sufficient={decision['sufficient']} ({decision['reason']})")

        if decision["sufficient"]:
            break

        # PLANNING: act on the decision - broaden the search and loop.
        # Use the agent's suggested refined query if it gave one; else widen k.
        refine = decision.get("refine_query", "").strip()
        if refine and refine not in tried_queries:
            query = refine
        k += K_STEP   # pull more documents next time
        print(f"  [{category}] decided to gather more -> "
              f"query='{query[:50]}...' k={k}")

    # If the agent never judged the evidence sufficient, flag the findings as
    # low confidence so downstream and the dashboard can mark them honestly.
    low_confidence = not final_sufficient
    if low_confidence:
        print(f"  [{category}] capped out after {MAX_ITERS} iterations without "
              f"sufficient evidence -> flagged low_confidence")

    return {
        "category": category,
        "findings": findings,
        "iterations": len(trace),
        "agent_trace": trace,        # shows the decisions for the demo
        "docs_seen": len(seen_doc_ids),
        "low_confidence": low_confidence,
        "low_confidence_reason": (
            "Agent could not find sufficient evidence after maximum retrieval "
            "attempts; findings shown are provisional." if low_confidence else ""
        ),
    }


def agentic_analyze_all() -> dict:
    results = {}
    for category in CATEGORY_CONFIG:
        print(f"\nAgent analyzing: {category} ...")
        try:
            results[category] = agentic_analyze_category(category)
        except Exception as e:
            print(f"  [{category}] agent failed: {type(e).__name__}: "
                  f"{str(e)[:120]} -- writing empty findings")
            results[category] = {"category": category, "findings": [],
                                 "iterations": 0, "agent_trace": [],
                                 "error": str(e)}
        n = len(results[category]["findings"])
        iters = results[category].get("iterations", 0)
        print(f"  -> {n} findings after {iters} iteration(s)")

    os.makedirs(os.path.dirname(AGENT_ANALYSIS_PATH), exist_ok=True)
    with open(AGENT_ANALYSIS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved agentic analysis -> {AGENT_ANALYSIS_PATH}")
    return results


if __name__ == "__main__":
    agentic_analyze_all()