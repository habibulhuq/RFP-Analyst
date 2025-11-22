from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "RFP Analyst – Data Intelligence Agent"
    DATABASE_URL: str = "sqlite:///./rfp_agent.db"

    # Base directory = project root (…/RFP-Analyst)
    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    # Data directories
    DATA_DIR: Path = BASE_DIR / "backend" / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    CLEAN_DIR: Path = DATA_DIR / "clean"
    REPORTS_DIR: Path = DATA_DIR / "reports"
    EDA_DIR: Path = DATA_DIR / "eda"

    # -------- LLM / Agent settings --------
    # LLM_PROVIDER: "none", "gemini", or "openai"
    LLM_PROVIDER: str = "none"

    # Default model if not given in .env
    LLM_MODEL: str = "gemini-2.5-flash-lite"

    # API keys (read from .env)
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    class Config:
        # Environment file for secrets
        env_file = ".env"
        # Ignore extra env vars instead of crashing
        extra = "ignore"


settings = Settings()
