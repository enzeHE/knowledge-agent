from fastapi import APIRouter, UploadFile, File
from sqlalchemy import select, desc
from app.core.db import AsyncSessionLocal
from app.models.models import Document
from app.tasks.ingest import ingest_document_task
import shutil
import os

router = APIRouter()
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/documents/")
async def list_documents():
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Document).order_by(desc(Document.created_at)).limit(50)
        )
        docs = result.scalars().all()
        return [
            {
                "doc_id": d.id,
                "filename": d.filename,
                "status": d.status or "pending",
                "chunk_count": d.chunk_count or 0,
                "created_at": str(d.created_at) if d.created_at else "",
            }
            for d in docs
        ]


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    async with AsyncSessionLocal() as db:
        doc = Document(filename=file.filename, source=file_path, status="pending")
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

    ingest_document_task.delay(file_path, doc.id)

    return {"doc_id": doc.id, "filename": file.filename, "status": "processing"}


@router.get("/documents/{doc_id}")
async def get_document_status(doc_id: int):
    from sqlalchemy import select
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return {"error": "not found"}
        return {"doc_id": doc.id, "status": doc.status, "chunk_count": doc.chunk_count}
