# backend/app/agents/cleaning_tool.py

from typing import Dict, Any, List
from pathlib import Path

import pandas as pd

from ..core.config import settings


def run_cleaning(df: pd.DataFrame, dataset_id: int) -> Dict[str, Any]:
    """
    Simple but fairly smart cleaning step:
      - drops duplicate rows
      - fills missing numeric values with mean
      - fills missing categorical values with mode
      - saves cleaned data to Parquet in CLEAN_DIR

    Returns a dict with:
      - clean_df_json: cleaned dataframe as list-of-dicts
      - clean_path: path to saved parquet file
      - log: list of text messages describing what was done
    """
    log: List[str] = []

    log.append(f"Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        log.append(f"Dropped {duplicates} duplicate rows. New shape: {df.shape}")
    else:
        log.append("No duplicate rows found.")

    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            missing = df[col].isna().sum()
            if missing > 0:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                log.append(f"Filled {missing} missing values in numeric column '{col}' with mean={mean_val:.4f}.")
        else:
            missing = df[col].isna().sum()
            if missing > 0:
                mode_series = df[col].mode()
                if len(mode_series) > 0:
                    mode_val = mode_series.iloc[0]
                    df[col] = df[col].fillna(mode_val)
                    log.append(f"Filled {missing} missing values in categorical column '{col}' with mode='{mode_val}'.")
                else:
                    log.append(f"Column '{col}' has missing values but no mode; left as-is.")

    # Ensure CLEAN_DIR exists
    clean_dir = Path(settings.CLEAN_DIR)
    clean_dir.mkdir(parents=True, exist_ok=True)

    clean_path = clean_dir / f"dataset_{dataset_id}_clean.parquet"
    df.to_parquet(clean_path, index=False)

    log.append(f"Saved cleaned dataset to: {clean_path}")

    return {
        "clean_df_json": df.to_dict(orient="records"),
        "clean_path": str(clean_path),
        "log": log,
    }
