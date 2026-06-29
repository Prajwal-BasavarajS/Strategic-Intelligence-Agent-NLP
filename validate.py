"""
Validation agent for the NVIDIA CEO Agent.

Implements the "Validate" stage of the agent workflow:
    Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> VALIDATE

Before recommendations are presented to the user, this agent checks each one
in two layers:

  LAYER 1 - RULE-BASED GATE (deterministic):
      structural integrity - does the recommendation have the required parts?
        * non-empty recommendation text and rationale
        * a valid priority (High/Medium/Low)
        * at least one piece of supporting evidence
        * a complete risk_assessment (financial/operational/strategic)

  LAYER 2 - LLM JUDGMENT (semantic):
      groundedness - does the cited evidence actually support the claim, or is
      the recommendation a stretch / unsupported / hallucinated? An LLM reviews
      the recommendation against its evidence titles and returns a verdict.

Recommendations that fail are flagged (and optionally dropped). The result is
written to data/recommendations_validated.json, each rec carrying a
"validation" block so the dashboard can show only trustworthy recommendations.

"""

import json
import os
import sys

import ollama

RECS_PATH = "data/recommendations.json"
OUT_PATH = "data/recommendations_validated.json"
LLM_MODEL = "qwen2.5:7b-instruct"

VALID_PRIORITIES = {"High", "Medium", "Low"}
RISK_KEYS = {"financial", "operational", "strategic"}

# If True, recommendations that fail validation are removed from the output
# shown to the user. If False, they are kept but marked as not validated.
DROP_FAILED = False


def load_recs(path=RECS_PATH):
    with open(path) as f:
        return json.load(f)


# LAYER 1: rule-based structural gate

def rule_check(rec: dict) -> list:
    """Return a list of structural problems. Empty list == passes the gate."""
    problems = []

    if not rec.get("recommendation", "").strip():
        problems.append("missing recommendation text")
    if not rec.get("rationale", "").strip():
        problems.append("missing rationale")

    if rec.get("priority") not in VALID_PRIORITIES:
        problems.append(f"invalid priority: {rec.get('priority')!r}")

    if not rec.get("evidence"):
        problems.append("no supporting evidence")

    ra = rec.get("risk_assessment", {})
    missing_risk = RISK_KEYS - set(ra.keys())
    if missing_risk:
        problems.append(f"incomplete risk_assessment, missing {sorted(missing_risk)}")

    return problems


# LAYER 2: LLM semantic groundedness judgment

def llm_check(rec: dict) -> dict:
    """Ask the LLM whether the evidence actually supports the recommendation."""
    evidence_titles = "\n".join(f"  - {e['title']}" for e in rec.get("evidence", []))
    prompt = f"""You are a validation reviewer. Decide whether a strategic
recommendation is genuinely SUPPORTED by its cited evidence.

RECOMMENDATION: {rec.get('recommendation', '')}
RATIONALE: {rec.get('rationale', '')}

CITED EVIDENCE (titles):
{evidence_titles or '  (none)'}

Question: Does the cited evidence plausibly support this recommendation, or is
the recommendation unsupported, a stretch, or unrelated to the evidence?

Return STRICT JSON with EXACTLY these keys:
  "supported": true or false
  "reason": one short sentence justifying the verdict

Output ONLY the JSON object.

JSON:"""
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    raw = resp["message"]["content"].replace("```json", "").replace("```", "").strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        # If the judge misbehaves, fail open with a flag rather than crash.
        return {"supported": True, "reason": "validator could not parse LLM output"}
    parsed = json.loads(raw[s:e + 1])
    return {"supported": bool(parsed.get("supported", True)),
            "reason": parsed.get("reason", "")}


# Orchestration of the two layers

def validate_one(rec: dict) -> dict:
    problems = rule_check(rec)
    if problems:
        # Failed the deterministic gate; don't waste an LLM call.
        rec["validation"] = {
            "passed": False,
            "layer": "rule",
            "problems": problems,
            "llm_reason": None,
        }
        return rec

    verdict = llm_check(rec)
    rec["validation"] = {
        "passed": bool(verdict["supported"]),
        "layer": "llm",
        "problems": [] if verdict["supported"] else ["evidence does not support claim"],
        "llm_reason": verdict["reason"],
    }
    return rec


def main():
    # Optional input path: `python validate.py some_file.json`
    # Defaults to the real recommendations. A custom input is treated as a
    # demo run and does NOT overwrite the real validated output.
    in_path = sys.argv[1] if len(sys.argv) > 1 else RECS_PATH
    is_demo = in_path != RECS_PATH

    data = load_recs(in_path)
    recs = data.get("recommendations", [])
    print(f"Validating {len(recs)} recommendations from {in_path} "
          f"(rule gate + LLM judgment)...\n")

    validated = []
    passed_count = 0
    for i, rec in enumerate(recs, 1):
        rec = validate_one(rec)
        v = rec["validation"]
        status = "PASS" if v["passed"] else "FAIL"
        if v["passed"]:
            passed_count += 1
        detail = v["llm_reason"] or "; ".join(v["problems"])
        print(f"  [{status}] ({v['layer']}) rec {i}: {rec.get('recommendation','')[:60]}")
        print(f"         -> {detail}")
        if v["passed"] or not DROP_FAILED:
            validated.append(rec)

    out = {
        "recommendations": validated,
        "validation_summary": {
            "total": len(recs),
            "passed": passed_count,
            "failed": len(recs) - passed_count,
            "dropped_failed": DROP_FAILED,
        },
    }
    if is_demo:
        print(f"\n{passed_count}/{len(recs)} passed validation. "
              f"(demo run - real output not modified)")
        return

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{passed_count}/{len(recs)} passed validation.")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()