from textwrap import dedent
from typing import Dict, Any

from ..core.config import settings

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class ReportAgent:
    """
    Takes EDA results + dataset name and produces a human-readable report.

    If LLM_PROVIDER=gemini and GEMINI_API_KEY is set, it will call Gemini.
    Otherwise, it falls back to a plain text report built from the EDA outputs.
    """

    def __init__(self) -> None:
        self.provider = (settings.LLM_PROVIDER or "none").lower()
        self.model_name = settings.LLM_MODEL or "gemini-2.5-flash-lite"

        self.gemini_model = None
        if self.provider == "gemini":
            if genai is None:
                # google-generativeai not installed, we will fall back to plain text
                print("[ReportAgent] google-generativeai not installed, using fallback.")
            elif not settings.GEMINI_API_KEY:
                print("[ReportAgent] GEMINI_API_KEY not set, using fallback.")
            else:
                print(f"[ReportAgent] Initializing Gemini model: {self.model_name}")
                genai.configure(api_key=settings.GEMINI_API_KEY)
                # This class handles the correct endpoint/version for you
                self.gemini_model = genai.GenerativeModel(self.model_name)

    def generate_report(self, dataset_name: str, eda_results: Dict[str, Any]) -> str:
        """
        eda_results is a dict created by the EDA pipeline, expected keys (all markdown strings):
          - overview_markdown
          - summary_stats_markdown
          - column_summary_markdown
          - correlations_markdown
          - plots_markdown
        """
        base_text = self._build_plain_report(dataset_name, eda_results)

        # If Gemini isn't configured properly, just return the plain report
        if self.gemini_model is None:
            return base_text + "\n\n[Note: Gemini not configured, showing plain EDA summary.]"

        prompt = dedent(
            f"""
            You are a senior data analyst. I will give you a rough exploratory data
            analysis (EDA) report in Markdown for a dataset called "{dataset_name}".

            Rewrite it as a clear, concise, business-friendly report with sections:

            - Dataset overview
            - Data quality & cleaning
            - Key statistics and distributions
            - Correlations and relationships
            - Notable insights and recommendations

            Keep important numbers and trends, but you can remove low-level technical details.

            Here is the raw EDA output:

            ```markdown
            {base_text}
            ```
            """
        )

        try:
            response = self.gemini_model.generate_content(prompt)
            text = getattr(response, "text", None) or ""
            if not text.strip():
                return base_text + "\n\n[Gemini returned empty text, showing plain EDA summary.]"
            return text.strip()
        except Exception as e:
            # Never crash the pipeline if Gemini fails – just annotate the fallback.
            return base_text + f"\n\n[LLM (Gemini) call failed: {e}]"

    def _build_plain_report(self, dataset_name: str, eda_results: Dict[str, Any]) -> str:
        """
        Simple concatenation of EDA markdown pieces into a single markdown report.
        """
        g = eda_results.get

        sections = [
            f"# EDA Report for {dataset_name}",
            "",
            "## Overview",
            g("overview_markdown", ""),
            "",
            "## Summary statistics",
            g("summary_stats_markdown", ""),
            "",
            "## Column-wise summary",
            g("column_summary_markdown", ""),
            "",
            "## Correlations",
            g("correlations_markdown", ""),
            "",
            "## Plots summary",
            g("plots_markdown", ""),
        ]

        # Filter out Nones, join into markdown
        return "\n".join(str(s) for s in sections if s is not None)
