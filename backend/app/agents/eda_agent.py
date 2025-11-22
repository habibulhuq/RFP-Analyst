import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentResult

plt.switch_backend("Agg")  # Ensure plotting works on servers (no GUI required)


class EDAAgent(BaseAgent):
    """
    Performs EDA:
    - Summary statistics
    - Missing value analysis
    - Correlation matrix (numeric only)
    - Histograms for numeric columns
    - Saves plots to /data/eda/dataset_<id>/
    - Returns metadata + paths to plots
    """

    name = "eda_agent"

    def run(self, df: pd.DataFrame, dataset_id: int, output_dir: Path, config: Optional[Dict] = None) -> AgentResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_log: List[str] = []

        # -----------------------------
        # Basic info
        # -----------------------------
        summary_stats = df.describe(include="all").to_dict()
        meta_log.append("Generated summary statistics.")

        # Missing values
        missing = df.isna().sum().to_dict()
        meta_log.append("Computed missing value counts.")

        # -----------------------------
        # Correlation matrix (numeric only)
        # -----------------------------
        numeric_df = df.select_dtypes(include=[np.number])
        corr_path = None
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            corr_path = output_dir / "correlation_matrix.png"

            plt.figure(figsize=(10, 6))
            plt.imshow(corr, cmap="Blues", interpolation="nearest")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
            plt.yticks(range(len(corr.index)), corr.index)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.savefig(corr_path)
            plt.close()
            meta_log.append("Saved correlation matrix heatmap.")
        else:
            meta_log.append("Not enough numeric columns for correlation matrix.")

        # -----------------------------
        # Histograms for numeric features
        # -----------------------------
        hist_paths = []
        for col in numeric_df.columns:
            fig_path = output_dir / f"hist_{col}.png"

            plt.figure(figsize=(7, 4))
            plt.hist(df[col].dropna(), bins=20)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()

            hist_paths.append(str(fig_path))
            meta_log.append(f"Saved histogram for {col}.")

        # -----------------------------
        # Missing value plot
        # -----------------------------
        missing_plot_path = output_dir / "missing_values.png"
        plt.figure(figsize=(10, 4))
        cols = list(missing.keys())
        vals = list(missing.values())
        plt.bar(cols, vals)
        plt.xticks(rotation=45, ha="right")
        plt.title("Missing Values per Column")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(missing_plot_path)
        plt.close()
        meta_log.append("Saved missing value barplot.")

        # -----------------------------
        # Metadata summary
        # -----------------------------
        metadata = {
            "summary_stats": summary_stats,
            "missing_values": missing,
            "histograms": hist_paths,
            "correlation_matrix": str(corr_path) if corr_path else None,
            "missing_plot": str(missing_plot_path),
            "log": meta_log,
        }

        return AgentResult(data=df, metadata=metadata)
