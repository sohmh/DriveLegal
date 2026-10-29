"""
DriveLegal – scripts/embed.py
Embeds chunks using a FREE local sentence-transformers model.
No external API key or credits required.

Model: all-MiniLM-L6-v2 (~80MB, downloads once, then runs offline)

Run:  python scripts/embed.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

CHUNKS_FILE = Path("data/processed/chunks.json")
INDEX_PATH  = Path("data/faiss_index")
BATCH_SIZE  = 100


def load_chunks() -> list:
    if not CHUNKS_FILE.exists():
        print(f"[ERROR] {CHUNKS_FILE} not found. Run scripts/preprocess.py first.")
        sys.exit(1)
    raw  = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    docs = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in raw if c.get("text", "").strip()
    ]
    print(f"Loaded {len(docs)} chunks.")
    return docs


def build_index(docs: list) -> FAISS:
    print("Loading local embedding model (all-MiniLM-L6-v2)...")
    print("First run downloads ~80MB — subsequent runs are instant.")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"Embedding {len(docs)} chunks...")
    vectorstore = FAISS.from_documents(docs[:BATCH_SIZE], embeddings)
    for start in range(BATCH_SIZE, len(docs), BATCH_SIZE):
        vectorstore.add_documents(docs[start : start + BATCH_SIZE])
        print(f"  Batch {start // BATCH_SIZE + 1} done")
    return vectorstore


def test_retrieval(vectorstore: FAISS) -> None:
    print("\n-- Retrieval test --")
    results = vectorstore.similarity_search(
        "fine for jumping a red light in Pune on a bike", k=3
    )
    for i, r in enumerate(results, 1):
        print(f"  Result {i} | region={r.metadata.get('region')} | {r.page_content[:80]}...")


def main():
    docs = load_chunks()
    vectorstore = build_index(docs)
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"\nFAISS index saved -> {INDEX_PATH}/")
    test_retrieval(vectorstore)
    print("\nEmbedding complete! No external API credits used.")


if __name__ == "__main__":
    main()