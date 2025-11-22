from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import pandas as pd
from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.dataset import Dataset
from ..models.job import Job
from .ingestion_service import load_dataset_raw
from ..agents.data_cleaning_agent import DataCleaningAgent
from ..agents.eda_agent import EDAAgent
from ..agents.report_agent import ReportAgent


def create_job(db: Session, dataset_id: int, job_type: str) -> Job:
    """
    Create a Job row for tracking pipeline steps (cleaning, eda, report, etc.)
    """
    job = Job(
        dataset_id=dataset_id,
        type=job_type,
        status="pending",
        log="Job created.",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_job_status(
    db: Session,
    job: Job,
    status: str,
    log: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Job:
    """
    Update job status, append to log, and optionally set report_path.
    """
    job.status = status

    if log:
        if job.log:
            job.log = job.log + "\n" + log
        else:
            job.log = log

    if report_path is not None:
        job.report_path = report_path

    db.commit()
    db.refresh(job)
    return job


def run_cleaning_pipeline(db: Session, dataset_id: int) -> Tuple[Dataset, Job]:
    """
    Load raw dataset, run DataCleaningAgent, save cleaned version,
    update Dataset + Job, and return both.
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        raise ValueError("Dataset not found")

    job = create_job(db, dataset_id=dataset_id, job_type="cleaning")
    update_job_status(db, job, "running", log="Starting data cleaning.")

    df_raw = load_dataset_raw(ds)

    agent = DataCleaningAgent()
    result = agent.run(df=df_raw)

    clean_dir: Path = settings.CLEAN_DIR
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_dir / f"dataset_{ds.id}_clean.parquet"
    result.data.to_parquet(clean_path, index=False)

    ds.clean_path = str(clean_path)
    ds.status = "cleaned"
    db.commit()
    db.refresh(ds)

    log_lines = result.metadata.get("log", [])
    log_text = "\n".join(log_lines)
    update_job_status(db, job, status="done", log="Cleaning completed.\n" + log_text)

    return ds, job


def load_clean_dataset(ds: Dataset) -> pd.DataFrame:
    """
    Load the cleaned dataset from parquet into a pandas DataFrame.
    """
    if not ds.clean_path:
        raise ValueError("Dataset has not been cleaned yet.")
    path = Path(ds.clean_path)
    if not path.exists():
        raise ValueError(f"Cleaned dataset file not found: {path}")
    return pd.read_parquet(path)


def run_eda_pipeline(db: Session, dataset_id: int) -> Tuple[Dataset, Job, Dict[str, Any]]:
    """
    Load cleaned dataset, run EDAAgent, save plots, update Job, and return metadata.
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        raise ValueError("Dataset not found")

    if not ds.clean_path:
        raise ValueError("Dataset is not cleaned yet. Run cleaning pipeline first.")

    job = create_job(db, dataset_id=dataset_id, job_type="eda")
    update_job_status(db, job, "running", log="Starting EDA.")

    df_clean = load_clean_dataset(ds)

    eda_dir: Path = settings.EDA_DIR / f"dataset_{ds.id}"
    agent = EDAAgent()
    result = agent.run(df=df_clean, dataset_id=ds.id, output_dir=eda_dir)

    log_lines = result.metadata.get("log", [])
    log_text = "EDA completed.\n" + "\n".join(log_lines)

    update_job_status(
        db,
        job,
        status="done",
        log=log_text,
        report_path=str(eda_dir),
    )

    return ds, job, result.metadata


def run_report_pipeline(db: Session, dataset_id: int) -> Tuple[Dataset, Job, str]:
    """
    Run a report-generation pipeline:
    - Ensure dataset is cleaned
    - Run EDA (or reuse EDA results)
    - Call ReportAgent to generate a markdown/text report
    - Save report to reports directory
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        raise ValueError("Dataset not found")

    if not ds.clean_path:
        raise ValueError("Dataset is not cleaned yet. Run cleaning pipeline first.")

    job = create_job(db, dataset_id=dataset_id, job_type="report")
    update_job_status(db, job, "running", log="Starting report generation.")

    # Load cleaned data
    df_clean = load_clean_dataset(ds)

    # Re-run EDA here to get fresh metadata (or you could reuse stored metadata)
    eda_dir: Path = settings.EDA_DIR / f"dataset_{ds.id}"
    eda_agent = EDAAgent()
    eda_result = eda_agent.run(df=df_clean, dataset_id=ds.id, output_dir=eda_dir)

    # Generate report text using LLM (or fallback)
    report_agent = ReportAgent()
    dataset_name = ds.name or f"Dataset {ds.id}"
    report_result = report_agent.run(df=df_clean, eda_metadata=eda_result.metadata, dataset_name=dataset_name)

    report_text = report_result.data if isinstance(report_result.data, str) else str(report_result.data)

    # Save report to file
    reports_dir: Path = settings.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"dataset_{ds.id}_eda_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    # Update job
    update_job_status(
        db,
        job,
        status="done",
        log="Report generation completed.",
        report_path=str(report_path),
    )

    return ds, job, str(report_path)
