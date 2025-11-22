from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.sql import func

from ..core.db import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)

    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    session_token = Column(String, unique=True, nullable=False)

    # We'll store a simple JSON list of {q: "...", a: "..."} pairs as text
    history = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
