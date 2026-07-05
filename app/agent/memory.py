from sqlalchemy import select
from app.core.db import AsyncSessionLocal
from app.models.models import Conversation


async def load_long_term_memory(session_id: str) -> str:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        conv = result.scalar_one_or_none()
        return conv.summary if conv else ""


async def save_long_term_memory(session_id: str, messages: list) -> None:
    from langchain_anthropic import ChatAnthropic
    from app.core.config import settings

    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    conversation_text = "\n".join([
        f"{m.type}: {m.content}" for m in messages
        if hasattr(m, "content") and isinstance(m.content, str)
    ])

    if not conversation_text.strip():
        return

    summary_response = await llm.ainvoke(
        f"Summarize this conversation in 2-3 sentences for future context:\n\n{conversation_text}"
    )
    summary = summary_response.content

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        conv = result.scalar_one_or_none()

        if conv:
            conv.summary = summary
        else:
            db.add(Conversation(session_id=session_id, summary=summary))

        await db.commit()
