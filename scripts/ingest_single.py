import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
from app.rag.loader import load_document
from app.rag.splitter import split_documents
from app.rag.retriever import retriever
from app.core.db import AsyncSessionLocal
from app.models.models import Document
from sqlalchemy import select


async def run(file_path: str, doc_id: int):
    docs = load_document(file_path)
    chunks = split_documents(docs)
    texts = [c.page_content for c in chunks]
    metadatas = [{"source": Path(file_path).name, "doc_id": doc_id} for _ in chunks]

    retriever.add_documents(texts, metadatas)

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc:
            doc.status = "done"
            doc.chunk_count = len(chunks)
            await db.commit()

    print(f"Done: {len(chunks)} chunks ingested for doc_id={doc_id}")


if __name__ == "__main__":
    file_path = sys.argv[1]
    doc_id = int(sys.argv[2])
    asyncio.run(run(file_path, doc_id))
