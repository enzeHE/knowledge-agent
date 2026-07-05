import sys
sys.path.insert(0, "D:/knowledge-agent")

from app.rag.embedder import embedder
from qdrant_client import QdrantClient

c = QdrantClient(url="http://localhost:6333")
q = embedder.embed_text("fastapi query parameters")
results = c.search("knowledge_docs", query_vector=q, limit=3)

for r in results:
    print(r.payload["text"][:200])
    print("---")
