import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4-mini"

load_dotenv()


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Return the shared OpenAI client for this project."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to the .env file.")

    return OpenAI(api_key=api_key)