"""LangGraph Agent — 四节点 DAG 编排

流程：
  IntentRouter（意图路由）→ DocRetriever（检索）→ AnswerGenerator（生成）
                     ↙ ↓ ↘
              clarify  general  retrieve（走检索）
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.config import settings
from app.rag.retriever import retriever
from app.rag.query_rewrite import rewriter

# ── Prompt 模板 ──────────────────────────────────────────────

INTENT_SYSTEM = (
    "You are an intent classifier for a documentation QA system. "
    "The knowledge base covers FastAPI, LangChain, and LangGraph documentation.\n\n"
    "Classify the user's query into exactly one category:\n"
    "- retrieve: technical question that needs documentation search\n"
    "- clarify: too vague or lacks context to search (e.g. '怎么用', 'how to use', 'what is this')\n"
    "- general: greeting, thanks, or chat that doesn't need docs\n\n"
    "Reply with ONLY the category word: retrieve, clarify, or general."
)

RETRIEVE_SYSTEM = (
    "You are a helpful documentation assistant. Answer based on the provided context.\n\n"
    "Rules:\n"
    "1. Base your answer ONLY on the provided context\n"
    "2. If the context lacks enough information, say so honestly\n"
    "3. Cite specific source filenames when referencing information\n"
    "4. Format code examples with markdown code blocks\n"
    "5. Keep answers concise but complete"
)

CLARIFIER_SYSTEM = (
    "The user's question is too vague for a documentation search.\n"
    "Ask a clarifying question that narrows down what they need.\n\n"
    "Rules:\n"
    "- Be specific about what information you need\n"
    "- Suggest possible topics (e.g. 'Are you asking about routing, dependencies, or middleware?')\n"
    "- Keep your response to 1-2 sentences"
)

GENERAL_SYSTEM = (
    "You are a knowledgeable documentation assistant. "
    "Respond briefly and politely, then offer to help with technical documentation questions."
)

# ── State ────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（MemorySaver 自动管理）
    query: str                                # 当前用户 query
    intent: str                               # 意图分类结果
    retrieved_docs: list                      # 检索到的文档片段
    response: str                             # 最终回答
    guard_triggered: bool                     # 输入护栏是否触发
    guard_message: str                        # 护栏拦截提示
    context_summary: str                      # 历史对话摘要（滑动窗口压缩后）


# ── Graph 构建 ──────────────────────────────────────────────


def create_agent():
    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    # ── 节点 ──

    async def input_guard(state: AgentState) -> dict:
        """输入护栏：空 query → 长度 → 敏感词"""
        from app.agent.guardrails import InputGuard

        result = InputGuard.check(state["query"])
        if not result.passed:
            return {
                "guard_triggered": True,
                "guard_message": result.reason,
                "response": result.reason,
            }
        if result.sanitized:
            return {
                "query": result.sanitized,
                "guard_message": result.reason,
            }
        return {"guard_triggered": False}

    async def intent_router(state: AgentState) -> dict:
        """意图路由：判断用户意图"""
        response = await llm.ainvoke([
            SystemMessage(content=INTENT_SYSTEM),
            HumanMessage(content=f"Query: {state['query']}\n\nCategory:"),
        ])
        raw = _extract_text(response).strip().lower().rstrip(".")
        # 归一化
        if "clarify" in raw:
            intent = "clarify"
        elif "general" in raw:
            intent = "general"
        else:
            intent = "retrieve"
        return {"intent": intent}

    async def doc_retriever(state: AgentState) -> dict:
        """文档检索：Query Rewrite → 多路召回 → Rerank"""
        queries = await rewriter.rewrite(state["query"])
        docs = retriever.search_with_rerank(queries, top_k=5, rerank_top_k=20)
        return {"retrieved_docs": docs}

    async def answer_generator(state: AgentState) -> dict:
        """答案生成：组装 context → LLM 生成 → 输出护栏校验"""
        query = state["query"]
        docs = state.get("retrieved_docs", [])

        # 历史上下文注入
        summary = state.get("context_summary", "")
        history_hint = f"\nPrevious conversation: {summary}\n" if summary else ""

        # 组装检索 context
        context_parts = []
        for d in docs:
            source = d["metadata"].get("source", "unknown")
            context_parts.append(f"--- Source: {source} ---\n{d['text'][:800]}")
        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

        system = RETRIEVE_SYSTEM + history_hint
        response = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"),
        ])
        answer = _extract_text(response)

        # 输出护栏：来源一致性校验 + 来源链接
        from app.agent.guardrails import OutputGuard

        passed, warning = OutputGuard.verify_sources(answer, docs)
        if not passed:
            answer = f"{answer}\n\n*{warning}*"

        answer = OutputGuard.append_sources(answer, docs)

        return {"response": answer, "messages": [AIMessage(content=answer)]}

    async def clarify_handler(state: AgentState) -> dict:
        """追问澄清：对模糊 query 生成追问"""
        summary = state.get("context_summary", "")
        system = CLARIFIER_SYSTEM
        if summary:
            system += f"\n\nPrevious conversation: {summary}"
        response = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=f"User query: {state['query']}"),
        ])
        clarification = _extract_text(response)
        return {"response": clarification, "messages": [AIMessage(content=clarification)]}

    async def general_handler(state: AgentState) -> dict:
        """通用对话：问候/闲聊"""
        summary = state.get("context_summary", "")
        system = GENERAL_SYSTEM
        if summary:
            system += f"\n\nPrevious conversation: {summary}"
        response = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=state["query"]),
        ])
        reply = _extract_text(response)
        return {"response": reply, "messages": [AIMessage(content=reply)]}

    async def context_manager(state: AgentState) -> dict:
        """滑动窗口：超出 4K token 时压缩历史为摘要"""
        from app.agent.context import update_summary

        messages = state.get("messages", [])
        existing = state.get("context_summary", "")
        new_summary = await update_summary(llm, messages, existing)
        if new_summary != existing:
            return {"context_summary": new_summary}
        return {}

    # ── 路由 ──

    def route_after_guard(state: AgentState) -> Literal["intent_router", "__end__"]:
        """护栏拦截 → END，放行 → intent_router"""
        if state.get("guard_triggered"):
            return END
        return "intent_router"

    def route_after_intent(state: AgentState) -> Literal["doc_retriever", "clarify_handler", "general_handler"]:
        if state["intent"] == "clarify":
            return "clarify_handler"
        elif state["intent"] == "general":
            return "general_handler"
        return "doc_retriever"

    # ── 组合 ──

    workflow = StateGraph(AgentState)

    workflow.add_node("input_guard", input_guard)
    workflow.add_node("intent_router", intent_router)
    workflow.add_node("doc_retriever", doc_retriever)
    workflow.add_node("answer_generator", answer_generator)
    workflow.add_node("clarify_handler", clarify_handler)
    workflow.add_node("general_handler", general_handler)
    workflow.add_node("context_manager", context_manager)

    workflow.set_entry_point("input_guard")
    workflow.add_conditional_edges("input_guard", route_after_guard, {END: END, "intent_router": "intent_router"})
    workflow.add_conditional_edges("intent_router", route_after_intent)
    workflow.add_edge("doc_retriever", "answer_generator")
    workflow.add_edge("answer_generator", "context_manager")
    workflow.add_edge("clarify_handler", "context_manager")
    workflow.add_edge("general_handler", "context_manager")
    workflow.add_edge("context_manager", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


agent_graph = create_agent()


# ── 辅助 ──


def _extract_text(response) -> str:
    """从 LLM 响应提取文本（兼容 thinking 模式）"""
    if isinstance(response.content, str):
        return response.content
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""
