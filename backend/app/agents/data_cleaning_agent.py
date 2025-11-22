from typing import Dict, Any, Optional, List

import pandas as pd

from .base_agent import BaseAgent, AgentResult


class DataCleaningAgent(BaseAgent):
    """
    Advanced cleaning agent:
    - Normalizes column names
    - Infers dtypes
    - Fills missing numeric values with mean
    - Fills missing categorical values with mode or 'UNKNOWN'
    - Logs all operations in metadata['log']
    """

    name = "data_cleaning_agent"

    def run(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> AgentResult:
        config = config or {}
        df_clean = df.copy()
        log_entries: List[str] = []

        # 1. Normalize column names (strip + replace spaces with underscores, lowercased)
        original_cols = list(df_clean.columns)
        new_cols = []
        for c in original_cols:
            if isinstance(c, str):
                nc = c.strip().lower().replace(" ", "_")
            else:
                nc = str(c)
            new_cols.append(nc)
        df_clean.columns = new_cols
        if original_cols != new_cols:
            log_entries.append(
                f"Normalized column names: {dict(zip(original_cols, new_cols))}"
            )

        # 2. Infer better dtypes
        before_dtypes = df_clean.dtypes.astype(str).to_dict()
        df_clean = df_clean.infer_objects()
        after_dtypes = df_clean.dtypes.astype(str).to_dict()
        if before_dtypes != after_dtypes:
            log_entries.append(f"Inferred dtypes. Before: {before_dtypes}, After: {after_dtypes}")

        # 3. Handle missing values (mean for numeric, mode/'UNKNOWN' for non-numeric)
        for col in df_clean.columns:
            missing_count = df_clean[col].isna().sum()
            if missing_count == 0:
                continue

            if pd.api.types.is_numeric_dtype(df_clean[col]):
                mean_val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(mean_val)
                log_entries.append(
                    f"Filled {missing_count} missing values in numeric column '{col}' with mean={mean_val:.4f}."
                )
            else:
                mode_series = df_clean[col].mode(dropna=True)
                if not mode_series.empty:
                    mode_val = mode_series.iloc[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    log_entries.append(
                        f"Filled {missing_count} missing values in categorical column '{col}' with mode='{mode_val}'."
                    )
                else:
                    df_clean[col] = df_clean[col].fillna("UNKNOWN")
                    log_entries.append(
                        f"Filled {missing_count} missing values in column '{col}' with 'UNKNOWN' (no mode available)."
                    )

        # 4. Simple metadata summary
        metadata = {
            "log": log_entries,
            "rows_before": len(df),
            "rows_after": len(df_clean),
            "cols": len(df_clean.columns),
            "missing_before": df.isna().sum().to_dict(),
            "missing_after": df_clean.isna().sum().to_dict(),
        }

        return AgentResult(data=df_clean, metadata=metadata)
