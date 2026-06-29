"""
Retrieval smoke test (Phase 3 verification).

Queries the ChromaDB store with a few strategic questions and prints the
top matches with source + similarity. Purpose: prove retrieval returns
RELEVANT documents before we build the RAG/LLM layer on top of it.

"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "nvidia_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

QUERIES = [
    "competitors building rival AI chips to challenge NVIDIA",
    "NVIDIA partnerships and AI factory infrastructure deals",
    "risks and threats to NVIDIA's business",
]


def main():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Collection has {collection.count()} documents\n")

    for q in QUERIES:
        print("=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        q_emb = model.encode([q]).tolist()
        res = collection.query(
            query_embeddings=q_emb,
            n_results=4,
            include=["metadatas", "distances"],
        )
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            sim = 1 - dist          # cosine distance -> similarity
            print(f"\n  [{sim:.3f}] ({meta['source']}/{meta['source_detail']})")
            print(f"  {meta['title']}")
            print(f"  {meta['url'][:80]}")
        print()


if __name__ == "__main__":
    main()