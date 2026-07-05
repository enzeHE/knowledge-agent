import os
from app.core.config import settings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.api.document import router as document_router

app = FastAPI(title="Knowledge Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(document_router, prefix="/api", tags=["documents"])


@app.get("/health")
async def health():
    return {"status": "ok"}
