from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.sql import func

from ..core.db import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)

    type = Column(String, nullable=False)  # e.g. "cleaning", "eda", "report", "pipeline"
    status = Column(String, nullable=False, default="pending")  # pending, running, done, failed

    log = Column(Text, nullable=True)          # to store messages about what happened
    report_path = Column(String, nullable=True)  # path to report file if this job generates one

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
