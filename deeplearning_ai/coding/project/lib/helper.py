"""Environment loading utilities."""

import os
from dotenv import load_dotenv, find_dotenv


def load_env():
    """Load environment variables from .env file."""
    _ = load_dotenv(find_dotenv())


def get_openai_api_key():
    """Get OpenAI API key from environment."""
    load_env()
    return os.getenv("OPENAI_API_KEY")
