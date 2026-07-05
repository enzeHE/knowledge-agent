"""Query Rewrite: 对模糊短 query 做 LLM 改写，生成多角度子查询"""

from app.core.config import settings

# 明确的技术关键词：包含这些词说明 query 已经比较具体，可跳过 rewrite
_TECH_KEYWORDS = [
    "fastapi", "langchain", "langgraph", "api", "function", "class",
    "method", "route", "endpoint", "middleware", "dependency", "model",
    "query", "parameter", "request", "response", "schema", "async",
    "decorator", "router", "exception", "websocket", "security",
    "header", "cookie", "body", "form", "file", "upload",
]


def _extract_text(response) -> str:
    """从 LLM 响应中提取文本，兼容 thinking 模式"""
    if isinstance(response.content, str):
        return response.content
    # thinking 模式下 content 是 list[dict]，如 [{"type":"thinking",...}, {"type":"text","text":"..."}]
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


class QueryRewriter:
    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                temperature=0.3,
            )
        return self._llm

    def should_rewrite(self, query: str) -> bool:
        """判断是否需要 rewrite

        短 query（≤10 字）或不含技术关键词的模糊 query 需要改写；
        已经明确具体的 query 跳过以节省延迟。
        """
        if len(query) <= 10:
            return True
        if not any(kw in query.lower() for kw in _TECH_KEYWORDS):
            return True
        return False

    async def rewrite(self, query: str) -> list[str]:
        """对 query 做改写，返回多个子查询

        如果 query 已经明确，直接返回 [query] 不调用 LLM。
        """
        if not self.should_rewrite(query):
            return [query]

        system = (
            "You are a search query rewriter for a technical documentation QA system. "
            "The knowledge base covers FastAPI, LangChain, and LangGraph documentation. "
            "Your job is to rewrite vague user queries into specific, search-engine-friendly sub-queries. "
            "Return ONLY the sub-queries, one per line, no explanation, no thinking."
        )

        prompt = (
            f"Rewrite this vague query into 2-3 specific sub-queries "
            f"for searching technical documentation:\n\n"
            f"Original: \"{query}\"\n\n"
            f"Rules:\n"
            f"- Each sub-query should focus on a different aspect\n"
            f"- Use specific technical terminology\n"
            f"- Stay within the original scope, don't expand beyond what was asked\n"
            f"- One sub-query per line, no numbering, no prefix\n\n"
            f"Sub-queries:"
        )

        response = await self.llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ])
        content = _extract_text(response)
        queries = [
            q.strip().strip("\"'").strip("'")
            for q in content.strip().split("\n")
            if q.strip()
        ]

        # 始终包含原始 query，放在首位
        if query not in queries:
            queries.insert(0, query)

        return queries[:4]


# 单例
rewriter = QueryRewriter()
