"""
Strategic analysis engine for the NVIDIA CEO Agent (Phase 5/6).

For a given category (opportunities | risks | trends):
  1. Retrieve the top-k most relevant documents from ChromaDB.
  2. Feed them to a local Ollama LLM (qwen2.5) with their ids/titles/urls.
  3. Get back STRICT JSON: a list of findings, each citing the evidence
     doc ids that support it.
  4. Map evidence ids back to {title, url} so the dashboard can show
     clickable supporting evidence.

"""

import json
import os
import re

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

#  Config 
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "nvidia_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:7b-instruct"
TOP_K = 6
ANALYSIS_PATH = "data/analysis.json"

# Retrieval query + instruction framing per category.
CATEGORY_CONFIG = {
    "opportunities": {
        "query": "NVIDIA opportunities new markets partnerships emerging technologies growth",
        "noun": "strategic opportunities",
    },
    "risks": {
        "query": ("NVIDIA threats competitive pressure AMD Intel Amazon custom chips "
                  "regulation export restrictions lawsuit supply shortage customer loss "
                  "declining demand stock decline"),
        "noun": "strategic risks and threats",
        "guidance": ("A RISK is something that could HARM NVIDIA: a competitive threat, "
                     "regulatory or legal danger, supply problem, loss of customers, or "
                     "negative market pressure. Do NOT label NVIDIA's own product launches, "
                     "partnerships, or positive developments as risks. If a document "
                     "describes something good for NVIDIA, it is NOT a risk - omit it. "
                     "Only report genuine threats. If few genuine risks are present, "
                     "report fewer findings rather than reframing positive news as risk."),
    },
    "trends": {
        "query": "NVIDIA technology trends industry shifts customer behavior market direction",
        "noun": "emerging trends management should monitor",
    },
}

# Lazy singletons so importing this module is cheap and the model/DB
# load only once even across multiple category calls.
_embed_model = None
_collection = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Embed the query and return the top-k nearest documents from ChromaDB."""
    model = _get_embed_model()
    collection = _get_collection()
    q_emb = model.encode([query]).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["metadatas", "documents"],
    )
    docs = []
    for doc_id, meta, text in zip(
        res["ids"][0], res["metadatas"][0], res["documents"][0]
    ):
        docs.append({
            "id": doc_id,
            "title": meta["title"],
            "url": meta["url"],
            "source": meta["source"],
            "text": text,
        })
    return docs


def build_prompt(category_noun: str, docs: list[dict], guidance: str = "") -> str:
    """Construct the LLM prompt with the retrieved evidence documents.

    Each document is labelled by its id (not a positional number) so the
    model cites the exact id string, which we can resolve back to a source.

    `guidance` is an optional category-specific instruction (e.g. a stricter
    definition of what counts as a 'risk') injected into the rules.
    """
    evidence_block = "\n".join(
        f"- id: {d['id']}\n  title: {d['title']}\n  content: {d['text'][:400]}"
        for d in docs
    )
    guidance_line = f"\n- {guidance}" if guidance else ""
    return f"""You are a strategic intelligence analyst advising NVIDIA's CEO.

Below are {len(docs)} documents collected from news, community, and company sources.

DOCUMENTS:
{evidence_block}

TASK: Identify the most important {category_noun} for NVIDIA based ONLY on these documents.

Return STRICT JSON - a list of 2 to 4 findings. Each finding must be an object with EXACTLY these keys:
  "title": short title (under 12 words)
  "detail": one or two sentences explaining it
  "impact": one of "High", "Medium", "Low"
  "confidence": a number between 0.0 and 1.0
  "evidence_ids": a list of the document id strings that support this finding (use the exact ids shown above)

Rules:
- Base every finding on the documents. Do not invent facts.
- Only cite a document as evidence if it DIRECTLY supports the finding. If a document is not clearly relevant, do not use it.
- It is better to report fewer findings than to cite weak or unrelated evidence. A finding with no strong supporting document should be omitted entirely.
- The "detail" must describe what the cited documents actually say. Do not attribute a claim to a source that does not contain it.
- Use only document ids that appear above for evidence_ids.{guidance_line}
- Output ONLY the JSON array. No prose, no markdown, no code fences.

JSON:"""


def extract_json(raw: str):
    """Pull JSON out of model output, tolerating stray text and accepting
    either a [...] array or a single {...} object (wrapped into a list)."""
    # Strip code fences if the model added them despite instructions.
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Prefer an array.
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start:end + 1])

    # Fall back: a lone object becomes a one-element list.
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return [json.loads(raw[start:end + 1])]

    raise ValueError(f"No JSON found in model output:\n{raw[:300]}")


def analyze_category(category: str) -> dict:
    """Full pipeline for one category. Returns findings + bound evidence."""
    cfg = CATEGORY_CONFIG[category]
    docs = retrieve(cfg["query"])
    doc_by_id = {d["id"]: d for d in docs}

    prompt = build_prompt(cfg["noun"], docs, cfg.get("guidance", ""))
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},   # low temp = more consistent JSON
    )
    raw = resp["message"]["content"]
    findings = extract_json(raw)

    # Bind evidence ids -> {title, url}; warn on any id not in the retrieved set.
    for f in findings:
        ev = []
        for eid in f.get("evidence_ids", []):
            if eid in doc_by_id:
                ev.append({
                    "title": doc_by_id[eid]["title"],
                    "url": doc_by_id[eid]["url"],
                })
            else:
                print(f"    warning: evidence id '{eid}' not in retrieved docs")
        f["evidence"] = ev

    return {"category": category, "findings": findings, "retrieved": len(docs)}


def analyze_category_safe(category: str, retries: int = 1) -> dict:
    """analyze_category with one retry if the model emits bad JSON."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return analyze_category(category)
        except Exception as e:
            last_err = e
            print(f"    {category} attempt {attempt + 1} failed: "
                  f"{type(e).__name__}: {str(e)[:120]}")
    # Both attempts failed: return empty findings rather than crashing.
    print(f"    {category}: giving up, returning empty findings")
    return {"category": category, "findings": [], "retrieved": 0,
            "error": str(last_err)}


def analyze_all() -> dict:
    """Run all three categories and cache to data/analysis.json."""
    results = {}
    for category in CATEGORY_CONFIG:
        print(f"Analyzing: {category} ...")
        results[category] = analyze_category_safe(category)
        n = len(results[category]["findings"])
        print(f"  -> {n} findings\n")

    os.makedirs(os.path.dirname(ANALYSIS_PATH), exist_ok=True)
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved analysis -> {ANALYSIS_PATH}")
    return results


if __name__ == "__main__":
    analyze_all()