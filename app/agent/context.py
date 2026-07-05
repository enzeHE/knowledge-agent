"""滑动窗口上下文管理

超出 4K token 上限时自动压缩历史对话为摘要，
保持上下文连贯的同时控制 token 消耗。
"""

import tiktoken

_tokenizer = tiktoken.get_encoding("cl100k_base")

# 保留最近 3 轮完整对话 + 当前 query
KEEP_TURNS = 3
MAX_TOKENS = 4000


def count_tokens(messages: list) -> int:
    """统计 messages 的总 token 数"""
    total = 0
    for msg in messages:
        if isinstance(msg.content, str):
            total += len(_tokenizer.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += len(_tokenizer.encode(block.get("text", "")))
    return total


def should_compress(messages: list) -> bool:
    """判断是否需要压缩"""
    return count_tokens(messages) > MAX_TOKENS


def _extract_text(response) -> str:
    """从 LLM 响应提取文本"""
    content = response.content
    if isinstance(content, str):
        return content
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


async def update_summary(
    llm,
    messages: list,
    existing_summary: str = "",
) -> str:
    """超出限制时，调用 LLM 生成/更新对话摘要"""
    if not should_compress(messages) and existing_summary:
        return existing_summary

    # 构建可压缩的消息文本（保留最后 KEEP_TURNS 轮之外的）
    keep = KEEP_TURNS * 2
    to_summarize = messages[:-keep] if len(messages) > keep else messages

    lines = []
    for msg in to_summarize:
        text = msg.content if isinstance(msg.content, str) else ""
        if not text:
            continue
        prefix = "User: " if getattr(msg, "type", "") == "human" else "Assistant: "
        lines.append(f"{prefix}{text[:500]}")
    conversation_text = "\n".join(lines)

    if not conversation_text.strip():
        return existing_summary

    from langchain_core.messages import SystemMessage, HumanMessage

    if existing_summary:
        prompt = (
            f"Previous summary: {existing_summary}\n\n"
            f"New conversation to merge:\n{conversation_text}\n\n"
            f"Update the summary in 2-3 sentences:"
        )
    else:
        prompt = (
            f"Summarize this technical Q&A conversation in 2-3 sentences:\n\n"
            f"{conversation_text}"
        )

    response = await llm.ainvoke([
        SystemMessage(content="You summarize technical conversations concisely, preserving key technical details and user intent."),
        HumanMessage(content=prompt),
    ])
    return _extract_text(response)
