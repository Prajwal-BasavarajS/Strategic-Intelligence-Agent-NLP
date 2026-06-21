"""
Cleaning & deduplication for the NVIDIA CEO Agent (Phase 4).

Reads every JSON in data/raw/, then:
  1. Cleans text  - unescape HTML entities, strip Reddit boilerplate.
  2. Drops junk   - docs with no real body after cleaning.
  3. Deduplicates - same normalized title within the same source.
                    (Cross-source duplicates are KEPT: two sources reporting
                     the same event is corroborating evidence, not noise.)

Writes the survivors to data/clean/docs.json.

Run:  python clean.py
"""

import json
import glob
import os
import re
import html

RAW_DIR = "data/raw"
CLEAN_PATH = "data/clean/docs.json"

# Reddit RSS leaves this boilerplate in the body; strip it.
REDDIT_BOILERPLATE = re.compile(
    r"submitted by.*?\[link\].*?\[comments\]", re.IGNORECASE | re.DOTALL
)


def clean_text(raw: str) -> str:
    """Unescape entities, remove Reddit boilerplate, collapse whitespace."""
    if not raw:
        return ""
    text = html.unescape(raw)                      # &amp; -> &, &#32; -> space
    text = REDDIT_BOILERPLATE.sub("", text)
    text = re.sub(r"\u200b|\u200c|\u200d", "", text)  # zero-width chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title(title: str) -> str:
    """For dedup: lowercase, drop trailing ' - Outlet' suffix, strip punct."""
    t = html.unescape(title or "").lower()
    t = re.sub(r"\s+-\s+[^-]+$", "", t)            # drop ' - Yahoo Finance'
    t = re.sub(r"[^a-z0-9 ]", "", t)               # strip punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_all() -> list[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.json"))):
        with open(path) as f:
            batch = json.load(f)
        print(f"  loaded {len(batch):>4} from {os.path.basename(path)}")
        docs.extend(batch)
    return docs


def main():
    print("Loading raw documents...")
    docs = load_all()
    print(f"  total raw: {len(docs)}\n")

    cleaned = []
    dropped_empty = 0
    dropped_dupe = 0
    seen = set()   # (source, normalized_title)

    for d in docs:
        title = html.unescape(d.get("title", "")).strip()
        body = clean_text(d.get("text", ""))

        # Drop docs with no usable content (empty body AND empty title).
        if not body and not title:
            dropped_empty += 1
            continue

        # Dedup: same normalized title within the same source.
        key = (d["source"], normalize_title(title))
        if key in seen:
            dropped_dupe += 1
            continue
        seen.add(key)

        d["title"] = title
        d["text"] = body if body else title
        cleaned.append(d)

    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    with open(CLEAN_PATH, "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"  dropped {dropped_empty} empty")
    print(f"  dropped {dropped_dupe} same-source title duplicates")
    print(f"\nSaved {len(cleaned)} clean documents -> {CLEAN_PATH}")

    # Quick source breakdown so you can sanity-check the mix.
    by_source = {}
    for d in cleaned:
        by_source[d["source"]] = by_source.get(d["source"], 0) + 1
    print("\n  by source:")
    for s, n in sorted(by_source.items()):
        print(f"    {s:<12} {n}")


if __name__ == "__main__":
    main()