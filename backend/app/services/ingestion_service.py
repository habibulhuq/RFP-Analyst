from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from fastapi import UploadFile
from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.dataset import Dataset


def save_uploaded_file(upload_file: UploadFile, dest_dir: Path) -> Path:
    """
    Save an uploaded file (FastAPI UploadFile) to dest_dir with a unique name.
    Returns the full path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    original_name = upload_file.filename or "uploaded_file"
    ext = Path(original_name).suffix or ".csv"
    # Very simple unique name: original name with timestamp could also be used
    dest_path = dest_dir / original_name

    with dest_path.open("wb") as f:
        content = upload_file.file.read()
        f.write(content)

    return dest_path


def register_dataset(
    db: Session,
    name: str,
    raw_path: Path,
    description: Optional[str] = None
) -> Dataset:
    """
    Create a Dataset row in the database for the uploaded file.
    """
    ds = Dataset(
        name=name,
        description=description,
        raw_path=str(raw_path),
        status="uploaded",
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


def load_dataset_raw(ds: Dataset) -> pd.DataFrame:
    """
    Load the raw dataset from disk into a pandas DataFrame.
    Supports CSV and Excel.
    """
    path = Path(ds.raw_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def basic_validate_df(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Very basic validation: not empty, has at least 1 column.
    You can add more later.
    """
    if df.empty:
        return False, "Dataset is empty."
    if df.shape[1] == 0:
        return False, "Dataset has no columns."
    return True, "OK"
