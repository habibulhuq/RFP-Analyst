# backend/app/agents/eda_tool.py

from typing import Dict, Any
from pathlib import Path
import json

import pandas as pd

from ..core.config import settings


def run_basic_eda(df: pd.DataFrame, dataset_id: int) -> Dict[str, Any]:
    """
    Compute basic EDA stats:
      - shape
      - dtypes
      - missing values per column
      - numeric describe()
      - top values for categorical columns

    Also saves EDA JSON to EDA_DIR.
    """

    results: Dict[str, Any] = {}

    # Shape
    results["shape"] = {"rows": df.shape[0], "cols": df.shape[1]}

    # Dtypes
    results["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    results["missing_per_column"] = df.isna().sum().to_dict()

    # Numeric summary
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        results["numeric_describe"] = num_df.describe().to_dict()

    # Categorical summary
    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        top_values = {}
        for col in cat_df.columns:
            top_values[col] = cat_df[col].value_counts().head(5).to_dict()
        results["categorical_top_values"] = top_values

    # Save EDA JSON to disk
    eda_dir = Path(settings.EDA_DIR)
    eda_dir.mkdir(parents=True, exist_ok=True)

    eda_path = eda_dir / f"dataset_{dataset_id}_eda.json"
    with open(eda_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    results["eda_path"] = str(eda_path)

    return results
