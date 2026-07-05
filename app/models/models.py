from sqlalchemy import Column, Integer, String, Text, DateTime, func
from app.core.db import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    source = Column(String(500))          # 来源路径或URL
    status = Column(String(20), default="pending")  # pending/processing/done/failed
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), nullable=False, index=True)
    summary = Column(Text)               # 长期记忆摘要
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
