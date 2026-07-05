import sys
sys.path.insert(0, "D:/knowledge-agent")
import asyncio
from app.core.db import engine, Base
from app.models.models import Document, Conversation

async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created")

asyncio.run(init())
