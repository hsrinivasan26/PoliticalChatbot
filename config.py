import os
from dotenv import load_dotenv
load_dotenv()


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"'{key}' is not set.")
    return value
OPENAI_API_KEY: str = _require("OPENAI_API_KEY")
TAVILY_API_KEY: str = _require("TAVILY_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

