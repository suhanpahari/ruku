"""Environment-driven configuration.

The provider is auto-selected: if a GROQ_API_KEY is present we use Groq, otherwise
the agent transparently runs on the deterministic offline provider. This lets the
whole pipeline work end-to-end with zero setup while still upgrading to a real LLM
when a key is available.
"""
import os

from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


class Settings:
    def __init__(self) -> None:
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        self.groq_base_url: str = os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        ).strip()
        self.llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "45"))
        self.llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.max_request_chars: int = int(os.getenv("MAX_REQUEST_CHARS", "4000"))
        self.output_dir: str = os.getenv("OUTPUT_DIR", os.path.join(_ROOT, "generated"))

    @property
    def provider_name(self) -> str:
        return "groq" if self.groq_api_key else "offline"


settings = Settings()
