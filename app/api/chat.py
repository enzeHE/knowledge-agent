from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from app.agent.graph import agent_graph
from app.core.config import settings
import json

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


def _extract_text_from_msg(msg) -> str:
    """从 AIMessage 提取文本（兼容 thinking 模式）"""
    if isinstance(msg.content, str):
        return msg.content
    for block in msg.content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


@router.post("/chat")
async def chat(request: ChatRequest):
    # 预加载 embedding 模型（避免 torch vs aiomysql 的 C 扩展冲突）
    from app.rag.embedder import embedder
    embedder._load_model()

    from app.agent.memory import load_long_term_memory, save_long_term_memory

    long_term = await load_long_term_memory(request.thread_id)
    config = {"configurable": {"thread_id": request.thread_id}}

    # 初始状态（新会话全量，已有会话由 checkpoint 恢复 + merge）
    initial_state = {
        "query": request.message,
        "messages": [HumanMessage(content=request.message)],
        "intent": "",
        "retrieved_docs": [],
        "response": "",
        "guard_triggered": False,
        "guard_message": "",
        "context_summary": "",
    }
    # 如果是新会话，把长期记忆注入 messages
    checkpointer = agent_graph.checkpointer
    from langgraph.checkpoint.base import empty_checkpoint
    existing = checkpointer.get(config)
    if not existing or existing == empty_checkpoint():
        if long_term:
            initial_state["messages"].insert(
                0,
                HumanMessage(
                    content=f"[Previous conversation context]\n{long_term}\n\n---\nNew message: {request.message}"
                ),
            )
        else:
            initial_state["messages"] = [HumanMessage(content=request.message)]

    async def generate():
        all_messages = []
        retrieved_contexts = []

        try:
            async for event in agent_graph.astream(
                initial_state, config, stream_mode="updates"
            ):
                for node_name, node_output in event.items():
                    if not isinstance(node_output, dict):
                        continue

                    # 发送节点状态
                    yield f"data: {json.dumps({'type': 'status', 'node': node_name})}\n\n"

                    # 收集检索上下文
                    if node_name == "doc_retriever":
                        docs = node_output.get("retrieved_docs", [])
                        for d in docs:
                            src = d["metadata"].get("source", "unknown")
                            retrieved_contexts.append({"source": src, "text": d["text"][:300]})

                    # 发送文本回答
                    response_text = node_output.get("response", "")
                    if response_text:
                        yield f"data: {json.dumps({'type': 'text', 'content': response_text})}\n\n"

                    # 收集 messages 用于长期记忆
                    new_messages = node_output.get("messages", [])
                    all_messages.extend(new_messages)

            # 发送检索来源
            if retrieved_contexts:
                yield f"data: {json.dumps({'type': 'contexts', 'content': retrieved_contexts})}\n\n"

            yield "data: [DONE]\n\n"

            # 后台异步保存长期记忆
            import asyncio
            asyncio.create_task(save_long_term_memory(request.thread_id, all_messages))

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
