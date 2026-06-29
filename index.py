"""
Indexing for the NVIDIA CEO Agent (Phase 3).

Loads every JSON file in data/raw/, embeds each document with
all-MiniLM-L6-v2, and stores it in a persistent ChromaDB collection with
full provenance in metadata. One document = one vector (no chunking).

Embed once; the dashboard and RAG layer read this same store.
"""

import json
import glob
import os

import chromadb
from sentence_transformers import SentenceTransformer

# Config 
RAW_DIR = "data/clean"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "nvidia_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_all_docs() -> list[dict]:
    """Read every *.json file in data/raw/ into one combined list."""
    docs = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.json"))):
        with open(path) as f:
            batch = json.load(f)
        print(f"  loaded {len(batch):>4} from {os.path.basename(path)}")
        docs.extend(batch)
    return docs


def main():
    # 1. Load 
    print("Loading raw documents...")
    docs = load_all_docs()
    print(f"  total: {len(docs)} documents\n")

    # Guard against duplicate ids across files (Chroma requires unique ids).
    # Keep first occurrence; report how many dupes we dropped.
    seen = set()
    unique = []
    for d in docs:
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        unique.append(d)
    if len(unique) < len(docs):
        print(f"  dropped {len(docs) - len(unique)} duplicate ids "
              f"(exact id collisions only)\n")
    docs = unique

    # 2. Embed 
    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    # Embed title + text so the headline contributes to the match.
    texts = [f"{d['title']} {d['text']}".strip() for d in docs]
    print(f"Embedding {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True,
                              convert_to_numpy=True)

    # 3. Store in ChromaDB 
    print("\nBuilding ChromaDB collection...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh build each run: delete any existing collection of this name.
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  removed existing '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    # Chroma stores: ids, embeddings, documents (the text), metadatas.
    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{
            "source": d["source"],
            "source_detail": d["source_detail"],
            "title": d["title"],
            "url": d["url"],
            "date": d["date"],
        } for d in docs],
    )

    print(f"\nIndexed {collection.count()} documents -> {CHROMA_DIR}")
    print(f"Collection: '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()