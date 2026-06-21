"""
Recommendation engine for the NVIDIA CEO Agent (Phase 7).

Reads data/analysis.json (the opportunities/risks/trends findings) and
synthesizes them into prioritized, executive-grade recommendations.

Each recommendation follows the brief's Task 6 template:
  - recommendation text
  - priority (High / Medium / Low)
  - supporting evidence (can draw across categories)
  - expected impact
  - risk assessment (financial / operational / strategic)

This is the 'agent reasoning over its own analysis' step: it takes the
extracted intelligence and decides what the CEO should DO.

Run:  python recommend.py
"""

import json
import re
import os

import ollama

ANALYSIS_PATH = "data/analysis.json"
RECS_PATH = "data/recommendations.json"
LLM_MODEL = "qwen2.5:7b-instruct"


def load_analysis() -> dict:
    with open(ANALYSIS_PATH) as f:
        return json.load(f)


def build_evidence_pool(analysis: dict) -> list[dict]:
    """Flatten all findings across categories into one evidence pool,
    each tagged with a stable ref id the LLM can cite."""
    pool = []
    for category, block in analysis.items():
        for finding in block.get("findings", []):
            ref = f"{category[:3]}_{len(pool)}"   # e.g. opp_0, ris_3
            pool.append({
                "ref": ref,
                "category": category,
                "title": finding["title"],
                "detail": finding["detail"],
                "evidence": finding.get("evidence", []),
            })
    return pool


def build_prompt(pool: list[dict]) -> str:
    findings_block = "\n".join(
        f"- ref: {p['ref']} [{p['category']}] {p['title']}: {p['detail']}"
        for p in pool
    )
    return f"""You are the strategic advisor to NVIDIA's CEO.

Below are analyzed findings about NVIDIA, grouped as opportunities, risks, and trends.

FINDINGS:
{findings_block}

TASK: Produce 3 to 4 prioritized strategic recommendations for the CEO.
Synthesize ACROSS findings - a strong recommendation may address a risk by
leveraging an opportunity. Cite the finding refs that justify each one.

Return STRICT JSON - a list of recommendation objects, each with EXACTLY these keys:
  "recommendation": a clear action (one sentence, starts with a verb)
  "priority": one of "High", "Medium", "Low"
  "rationale": one or two sentences on why this matters now
  "evidence_refs": list of finding ref strings from above that support this
  "expected_impact": one sentence on the business upside
  "risk_assessment": object with keys "financial", "operational", "strategic",
                     each a short phrase describing the risk of acting

Rules:
- Base recommendations only on the findings above. Do not invent facts.
- Prefer recommendations supported by MULTIPLE findings where possible.
- Use only refs that appear above.
- Output ONLY the JSON array. No prose, no markdown, no code fences.

JSON:"""


def extract_json(raw: str):
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found:\n{raw[:300]}")
    return json.loads(raw[start:end + 1])


def generate(retries: int = 1) -> dict:
    analysis = load_analysis()
    pool = build_evidence_pool(analysis)
    ref_map = {p["ref"]: p for p in pool}
    prompt = build_prompt(pool)

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            recs = extract_json(resp["message"]["content"])
            break
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt + 1} failed: {type(e).__name__}: "
                  f"{str(e)[:120]}")
    else:
        return {"recommendations": [], "error": str(last_err)}

    # Resolve evidence_refs -> the underlying source evidence (title+url).
    for r in recs:
        sources = []
        for ref in r.get("evidence_refs", []):
            if ref in ref_map:
                sources.extend(ref_map[ref]["evidence"])
            else:
                print(f"  warning: ref '{ref}' not found in findings")
        # de-dup sources by url
        seen, unique = set(), []
        for s in sources:
            if s["url"] not in seen:
                seen.add(s["url"])
                unique.append(s)
        r["evidence"] = unique

    return {"recommendations": recs}


def main():
    print("Generating recommendations from analysis.json ...")
    result = generate()
    os.makedirs(os.path.dirname(RECS_PATH), exist_ok=True)
    with open(RECS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    n = len(result.get("recommendations", []))
    print(f"\nSaved {n} recommendations -> {RECS_PATH}")


if __name__ == "__main__":
    main()