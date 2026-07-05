"""Document search tools for the knowledge agent.

Full retrieval pipeline:
  Query Rewrite (短 query 改写) → BM25 + 向量 → RRF 粗排 → Cross-Encoder 精排
"""

from langchain.tools import tool
from app.rag.retriever import retriever
from app.rag.query_rewrite import rewriter


@tool
async def search_docs(query: str) -> str:
    """Search documents in the knowledge base for relevant information.

    Uses query rewriting for vague/short queries, hybrid search, and
    cross-encoder reranking to find the most relevant results.

    Args:
        query: The search query

    Returns:
        Relevant document chunks with source metadata
    """
    # 1. Query Rewrite: 对模糊短 query 生成多角度子查询
    queries = await rewriter.rewrite(query)
    notes = []
    if len(queries) > 1:
        notes.append(f"rewrote to {len(queries)} sub-queries")

    # 2. 多路检索 + RRF 粗排 + Cross-Encoder 精排
    results = retriever.search_with_rerank(queries, top_k=5, rerank_top_k=20)

    if not results:
        return "No relevant documentation found."

    # 检查是否成功做了 rerank
    has_rerank = any("rerank_score" in r for r in results)
    if has_rerank:
        notes.append("reranked")

    if notes:
        header = f"[Pipeline] {' + '.join(notes)}\n"
    else:
        header = ""

    output = [header] if header else []
    for i, doc in enumerate(results, 1):
        text_preview = doc["text"][:300].replace("\n", " ")
        score_info = ""
        if "rerank_score" in doc:
            score_info = f" (rerank={doc['rerank_score']:.2f})"
        output.append(f"[{i}]{score_info} {text_preview}...")
        source = doc["metadata"].get("source", "unknown")
        output.append(f"Source: {source}\n")

    return "\n".join(output)
