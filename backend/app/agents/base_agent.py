from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AgentResult:
    def __init__(self, data: Any, metadata: Optional[Dict] = None):
        self.data = data
        self.metadata = metadata or {}


class BaseAgent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def run(self, **kwargs) -> AgentResult:
        """
        Run the agent with given keyword arguments.
        Must return an AgentResult.
        """
        ...
