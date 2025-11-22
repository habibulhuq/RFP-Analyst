# backend/app/llm/gemini_client.py

from typing import Any, List, Optional
import json

import requests
from langchain.llms.base import LLM

from ..core.config import settings


class GeminiRESTLLM(LLM):
    """
    Custom LangChain LLM wrapper that calls the Gemini HTTP API directly
    using `requests`.

    - Uses the model from settings.LLM_MODEL if provided
    - Falls back to 'gemini-pro' automatically if the given model is not found
    - Uses API version v1 (NOT v1beta)
    """

    model: str = "gemini-pro"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_output_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return "gemini-rest"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Core method LangChain calls. We forward to the official Gemini
        REST endpoint: models/{model}:generateContent (v1).
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in settings/.env.")

        # First try with self.model (from settings or default)
        text = self._call_once_with_model(prompt, self.model)

        # Respect stop tokens if provided
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]

        return text

    def _call_once_with_model(self, prompt: str, model_name: str) -> str:
        """
        Call Gemini generateContent once with a specific model name on v1.

        If the model is not found (404), and model_name is not 'gemini-pro',
        we automatically retry with 'gemini-pro'.
        """
        url = (
            f"https://generativelanguage.googleapis.com/v1/models/"
            f"{model_name}:generateContent?key={self.api_key}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        resp = requests.post(url, json=payload, timeout=60)

        # If model not found, retry with 'gemini-pro'
        if resp.status_code == 404 and model_name != "gemini-pro":
            fallback_model = "gemini-pro"
            fallback_url = (
                f"https://generativelanguage.googleapis.com/v1/models/"
                f"{fallback_model}:generateContent?key={self.api_key}"
            )
            fallback_resp = requests.post(fallback_url, json=payload, timeout=60)

            if fallback_resp.status_code != 200:
                raise RuntimeError(
                    f"Gemini API error {fallback_resp.status_code} with fallback model "
                    f"'gemini-pro': {fallback_resp.text}"
                )

            data = fallback_resp.json()
        else:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Gemini API error {resp.status_code}: {resp.text}"
                )
            data = resp.json()

        # Parse the returned text from candidates[0].content.parts[*].text
        text = ""
        try:
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                pieces: List[str] = []
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        pieces.append(part["text"])
                text = "".join(pieces)
        except Exception as e:
            raise RuntimeError(f"Failed to parse Gemini response: {e}; raw={data}")

        if not text:
            text = json.dumps(data)

        return text


def get_gemini_llm() -> GeminiRESTLLM:
    """
    Factory to create the Gemini LLM using values from settings (.env).
    If LLM_MODEL is missing in .env, defaults to 'gemini-pro'.
    """
    model_name = settings.LLM_MODEL or "gemini-pro"
    return GeminiRESTLLM(
        model=model_name,
        api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
        max_output_tokens=2048,
    )
