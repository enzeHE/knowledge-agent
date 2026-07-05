from langchain.tools import tool
from app.rag.retriever import retriever


@tool
def search_docs(query: str) -> str:
    """Search documents in the knowledge base for relevant information.

    Args:
        query: The search query

    Returns:
        Relevant document chunks
    """
    results = retriever.hybrid_search(query, top_k=5)

    if not results:
        return "No relevant documentation found."

    output = []
    for i, doc in enumerate(results, 1):
        output.append(f"[{i}] {doc['text'][:300]}...")
        output.append(f"Source: {doc['metadata'].get('source', 'unknown')}\n")

    return "\n".join(output)
