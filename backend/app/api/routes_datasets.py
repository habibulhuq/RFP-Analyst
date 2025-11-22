# backend/app/api/routes_datasets.py

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import pandas as pd
import io

from ..core.db import get_db
from ..core.config import settings
from ..models.dataset import Dataset
from ..models.job import Job

from ..services.ingestion_service import (
    save_uploaded_file,
    register_dataset,
    load_dataset_raw,
    basic_validate_df,
)

from ..services.pipeline_service import run_cleaning_pipeline, run_eda_pipeline


# ------------------------------------------------------
# IMPORTANT: ROUTER MUST BE DEFINED FIRST
# ------------------------------------------------------
router = APIRouter(prefix="/datasets", tags=["datasets"])


# ------------------------------------------------------
# BASIC HEALTH ENDPOINT
# ------------------------------------------------------
@router.get("/health")
def datasets_health():
    return {"status": "datasets-ok"}


# ------------------------------------------------------
# UPLOAD DATASET
# ------------------------------------------------------
@router.post("/upload", response_model=dict)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    raw_path = save_uploaded_file(file, settings.RAW_DIR)
    ds = register_dataset(db, name=name, raw_path=raw_path, description=description)

    try:
        df = load_dataset_raw(ds)
    except Exception as e:
        Path(ds.raw_path).unlink(missing_ok=True)
        db.delete(ds)
        db.commit()
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    ok, msg = basic_validate_df(df)
    if not ok:
        Path(ds.raw_path).unlink(missing_ok=True)
        db.delete(ds)
        db.commit()
        raise HTTPException(status_code=400, detail=msg)

    return {
        "dataset_id": ds.id,
        "name": ds.name,
        "raw_path": ds.raw_path,
        "status": ds.status,
        "message": "Upload and basic validation successful.",
    }


# ------------------------------------------------------
# GET DATASET INFO
# ------------------------------------------------------
@router.get("/{dataset_id}", response_model=dict)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "id": ds.id,
        "name": ds.name,
        "description": ds.description,
        "raw_path": ds.raw_path,
        "clean_path": ds.clean_path,
        "status": ds.status,
        "created_at": ds.created_at,
        "updated_at": ds.updated_at,
    }


# ------------------------------------------------------
# RUN BASIC CLEANING PIPELINE (old method)
# ------------------------------------------------------
@router.post("/{dataset_id}/run_cleaning", response_model=dict)
def run_cleaning(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        cleaned_ds, job = run_cleaning_pipeline(db, dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {e}")

    return {
        "job_id": job.id,
        "dataset_id": cleaned_ds.id,
        "dataset_status": cleaned_ds.status,
        "clean_path": cleaned_ds.clean_path,
        "job_status": job.status,
    }


# ------------------------------------------------------
# JOB STATUS
# ------------------------------------------------------
@router.get("/jobs/{job_id}", response_model=dict)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "dataset_id": job.dataset_id,
        "type": job.type,
        "status": job.status,
        "log": job.log,
        "report_path": job.report_path,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


# ------------------------------------------------------
# DOWNLOAD CLEANED CSV
# ------------------------------------------------------
@router.get("/{dataset_id}/download_clean")
def download_cleaned_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not ds.clean_path:
        raise HTTPException(status_code=400, detail="Dataset not cleaned yet.")

    try:
        df = pd.read_parquet(ds.clean_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load cleaned dataset: {e}")

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=cleaned_dataset_{dataset_id}.csv"
        }
    )


# ------------------------------------------------------
# FULL LANGCHAIN MULTI-AGENT PIPELINE ENDPOINT
# ------------------------------------------------------
@router.post("/{dataset_id}/run_pipeline", response_model=dict)
def run_full_pipeline_api(dataset_id: int, db: Session = Depends(get_db)):
    """
    Runs complete multi-agent pipeline:
        1. Cleaning
        2. EDA
        3. Report Generation
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Step 1: Load raw data
    try:
        df = load_dataset_raw(ds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    # Step 2: Run full pipeline
    from ..pipeline.data_pipeline import run_full_pipeline
    try:
        result = run_full_pipeline(
            raw_df=df,
            dataset_id=dataset_id,
            dataset_name=ds.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")

    # Step 3: Save cleaned dataset path
    ds.clean_path = result["clean_path"]
    ds.status = "pipeline_complete"
    db.commit()

    # Step 4: Save report
    report_path = settings.REPORT_DIR / f"dataset_{dataset_id}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result["report_text"])

    # Step 5: Save job
    job = Job(
        dataset_id=dataset_id,
        type="full_pipeline",
        status="done",
        log=str(result.get("clean_log")),
        report_path=str(report_path)
    )
    db.add(job)
    db.commit()

    return {
        "dataset_id": dataset_id,
        "status": "completed",
        "clean_path": result["clean_path"],
        "eda_results": result["eda_results"],
        "report_path": str(report_path),
        "job_id": job.id
    }
