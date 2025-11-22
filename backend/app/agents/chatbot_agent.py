# backend/app/agents/chatbot_agent.py

import pandas as pd
from typing import Dict, Any

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from ..llm.gemini_client import get_gemini_llm


class DataChatbotAgent:
    """
    A chatbot that answers questions about a cleaned dataset using
    LangChain's Pandas DataFrame Agent (Python code interpreter).
    """

    def __init__(self, df: pd.DataFrame):
        self.llm = get_gemini_llm()
        self.df = df

        # Create LC Pandas Agent
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Execute natural language query on cleaned dataset.

        Returns:
            {
                "answer": "...",
                "query": "user question",
            }
        """
        try:
            response = self.agent.run(query)
            return {
                "answer": response,
                "query": query,
            }
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
            }
