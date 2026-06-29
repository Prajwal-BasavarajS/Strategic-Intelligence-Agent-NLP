"""
CEO Briefing generator for the NVIDIA CEO Agent (Phase 9).

Reads analysis.json + recommendations.json and produces the executive
summary that answers the brief's three questions:
  - What happened?
  - Why does it matter?
  - What should management do next?

Writes data/briefing.json, which Dashboard Section 7 reads.
"""

import json
import re
import os

import ollama

ANALYSIS_PATH = "data/analysis.json"
RECS_PATH = "data/recommendations.json"
BRIEFING_PATH = "data/briefing.json"
LLM_MODEL = "qwen2.5:7b-instruct"


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def build_prompt(analysis, recs):
    # Compact summaries so the prompt stays focused.
    def summarize(block):
        return "; ".join(
            f"{f['title']} ({f.get('impact','')})"
            for f in block.get("findings", [])
        )

    opps = summarize(analysis.get("opportunities", {}))
    risks = summarize(analysis.get("risks", {}))
    trends = summarize(analysis.get("trends", {}))
    rec_lines = "; ".join(
        f"{r['recommendation']} [{r.get('priority','')}]"
        for r in recs.get("recommendations", [])
    )

    return f"""You are the chief of staff preparing a briefing for NVIDIA's CEO,
based on this week's collected intelligence.

OPPORTUNITIES: {opps}
RISKS: {risks}
TRENDS: {trends}
RECOMMENDED ACTIONS: {rec_lines}

Write a concise executive briefing as STRICT JSON with EXACTLY these keys:
  "what_happened": 2-3 sentences summarizing the key developments this week
  "why_it_matters": 2-3 sentences on the strategic significance
  "what_next": 2-3 sentences on the prioritized actions management should take

Rules:
- Base it only on the intelligence above. Do not invent specific numbers.
- Write for an executive: clear, direct, no fluff.
- Output ONLY the JSON object. No prose, no markdown, no code fences.

JSON:"""


def extract_json(raw: str):
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found:\n{raw[:300]}")
    return json.loads(raw[start:end + 1])


def main():
    analysis = load(ANALYSIS_PATH, {})
    recs = load(RECS_PATH, {"recommendations": []})

    prompt = build_prompt(analysis, recs)
    print("Generating CEO briefing ...")

    last_err = None
    for attempt in range(2):
        try:
            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            briefing = extract_json(resp["message"]["content"])
            break
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt + 1} failed: {type(e).__name__}: "
                  f"{str(e)[:120]}")
    else:
        briefing = {
            "what_happened": "Briefing generation failed.",
            "why_it_matters": str(last_err),
            "what_next": "Re-run briefing.py.",
        }

    with open(BRIEFING_PATH, "w") as f:
        json.dump(briefing, f, indent=2)

    print(f"\nSaved briefing -> {BRIEFING_PATH}\n")
    for k in ("what_happened", "why_it_matters", "what_next"):
        print(f"{k}:\n  {briefing.get(k, '—')}\n")


if __name__ == "__main__":
    main()