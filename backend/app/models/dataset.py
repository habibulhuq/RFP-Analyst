from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

from ..core.db import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    raw_path = Column(String, nullable=False)
    clean_path = Column(String, nullable=True)
    status = Column(String, nullable=False, default="uploaded")  # uploaded, cleaned, processed

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
