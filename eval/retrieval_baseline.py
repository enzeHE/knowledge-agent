import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["HF_HUB_OFFLINE"] = "1"

from app.rag.retriever import retriever

test_queries = [
    "How do you define a path parameter in FastAPI?",
    "How do you handle file uploads in FastAPI?",
    "What is dependency injection in FastAPI?",
    "How do you add CORS middleware in FastAPI?",
    "How do you use async database queries in FastAPI?",
]

print("=== Current Configuration ===")
print("chunk_size: 512, overlap: 50, top_k: 10→3, RRF k: 60\n")
print("Loading embedding model and building BM25 index (first query may take 1-2 min)...\n")

for i, query in enumerate(test_queries, 1):
    print(f"\n[Query {i}] {query}")
    results = retriever.hybrid_search(query, top_k=3)

    for j, doc in enumerate(results, 1):
        source = doc['metadata'].get('source', 'unknown')
        text_preview = doc['text'][:150].replace('\n', ' ')
        print(f"  [{j}] score={doc['score']:.4f} | {source}")
        print(f"      {text_preview}...")
