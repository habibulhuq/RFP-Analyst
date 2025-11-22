# backend/app/pipeline/data_pipeline.py

from typing import Dict, Any

import pandas as pd

from ..agents.cleaning_tool import run_cleaning
from ..agents.eda_tool import run_basic_eda
from ..agents.report_agent import generate_report


def run_full_pipeline(raw_df: pd.DataFrame, dataset_id: int, dataset_name: str) -> Dict[str, Any]:
    """
    Runs the entire pipeline in sequence (no LangGraph, no tool .invoke):

        1. Cleaning (Python function)
        2. EDA (Python function)
        3. Report Generation (Gemini via LangChain)

    Returns:
        {
            "clean_path": str,
            "eda_results": dict,
            "report_text": str,
            "clean_log": list[str]
        }
    """

    # -------------------------
    # 1. Cleaning step
    # -------------------------
    print("🧹 [Pipeline] Running cleaning step...")
    clean_result = run_cleaning(raw_df, dataset_id)
    clean_df_json = clean_result["clean_df_json"]
    clean_path = clean_result["clean_path"]
    clean_log = clean_result["log"]

    clean_df = pd.DataFrame(clean_df_json)

    # -------------------------
    # 2. EDA step
    # -------------------------
    print("📊 [Pipeline] Running basic EDA...")
    eda_result = run_basic_eda(clean_df, dataset_id)

    # -------------------------
    # 3. Report generation (Gemini)
    # -------------------------
    print("📝 [Pipeline] Generating report with Gemini...")
    head_str = clean_df.head().to_string(index=False)

    report_text = generate_report(
        df_head_str=head_str,
        eda_results=eda_result,
        dataset_name=dataset_name,
    )

    return {
        "clean_path": clean_path,
        "eda_results": eda_result,
        "report_text": report_text,
        "clean_log": clean_log,
    }
