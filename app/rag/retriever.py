from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from app.core.config import settings
import uuid


class HybridRetriever:
    def __init__(self):
        self.client = None
        self.collection_name = settings.qdrant_collection
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_metadata = []

    def _get_client(self):
        if self.client is None:
            self.client = QdrantClient(url=settings.qdrant_url)
        return self.client

    def _get_embedder(self):
        from app.rag.embedder import embedder
        return embedder

    def create_collection(self, vector_size: int = 1024):
        client = self._get_client()
        collections = client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        embedder = self._get_embedder()
        embeddings = embedder.embed_batch(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"text": text, **meta}
            )
            for text, emb, meta in zip(texts, embeddings, metadatas)
        ]

        client = self._get_client()
        client.upsert(collection_name=self.collection_name, points=points)

        # 构建 BM25 索引
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = texts
        self.bm25_metadata = metadatas

    def vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        embedder = self._get_embedder()
        query_vector = embedder.embed_text(query)
        client = self._get_client()
        results = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [{"text": r.payload["text"], "score": r.score, "metadata": {k: v for k, v in r.payload.items() if k != "text"}} for r in results]

    def rebuild_if_needed(self):
        """Check if BM25 index is stale compared to Qdrant and rebuild if needed."""
        if not self.bm25_index:
            return
        try:
            client = self._get_client()
            count_result = client.count(collection_name=self.collection_name)
            qdrant_count = count_result.count
            if qdrant_count != len(self.bm25_corpus):
                print(f"BM25 index stale ({len(self.bm25_corpus)} vs {qdrant_count} in Qdrant), rebuilding...")
                self._rebuild_bm25_from_qdrant()
        except Exception:
            pass

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.bm25_index:
            print("Building BM25 index from Qdrant...")
            self._rebuild_bm25_from_qdrant()
            if not self.bm25_index:
                return []

        # 检测上传新文档后 BM25 索引是否过期
        self.rebuild_if_needed()

        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {"text": self.bm25_corpus[i], "score": float(scores[i]), "metadata": self.bm25_metadata[i]}
            for i in top_indices
        ]

    def _rebuild_bm25_from_qdrant(self):
        client = self._get_client()
        offset = None
        all_docs = []
        while True:
            results = client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
            )
            points, offset = results
            if not points:
                break
            for p in points:
                all_docs.append({
                    "text": p.payload["text"],
                    "metadata": {k: v for k, v in p.payload.items() if k != "text"}
                })
            if offset is None:
                break

        if all_docs:
            self.bm25_corpus = [d["text"] for d in all_docs]
            self.bm25_metadata = [d["metadata"] for d in all_docs]
            tokenized_corpus = [text.lower().split() for text in self.bm25_corpus]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            print(f"BM25 index built with {len(all_docs)} documents.")

    def rrf_fusion(self, results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
        scores = {}
        for results in results_list:
            for rank, doc in enumerate(results):
                text = doc["text"]
                if text not in scores:
                    scores[text] = {"score": 0, "metadata": doc["metadata"], "text": text}
                scores[text]["score"] += 1 / (k + rank + 1)

        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_docs

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        vector_results = self.vector_search(query, top_k=10)
        bm25_results = self.bm25_search(query, top_k=10)

        fused = self.rrf_fusion([vector_results, bm25_results])
        return fused[:top_k]

    def multi_query_search(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        """多个 query 分别检索后 RRF 融合结果"""
        if len(queries) == 1:
            return self.hybrid_search(queries[0], top_k=top_k)

        all_results = []
        for q in queries:
            results = self.hybrid_search(q, top_k=top_k * 2)
            if results:
                all_results.append(results)

        if not all_results:
            return []
        if len(all_results) == 1:
            return all_results[0][:top_k]

        # 多个子查询的结果再做跨查询 RRF 融合
        fused = self.rrf_fusion(all_results, k=60)
        return fused[:top_k]

    def search_with_rerank(
        self, queries: List[str], top_k: int = 5, rerank_top_k: int = 20
    ) -> List[Dict]:
        """多查询检索 → RRF 粗排 → Cross-Encoder 精排

        先初召更多结果（rerank_top_k），再用 cross-encoder 重打分后截取 top_k。
        """
        # Stage 1: 多查询检索 + RRF 粗排
        raw_results = self.multi_query_search(queries, top_k=rerank_top_k)

        if not raw_results:
            return []

        # Stage 2: Cross-Encoder 精排
        from app.rag.reranker import reranker

        reranked = reranker.rerank(queries[0], raw_results, top_k=top_k)
        return reranked


retriever = HybridRetriever()
