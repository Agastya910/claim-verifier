import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Try st.secrets first (Streamlit Cloud), then os.environ / .env (local dev)."""
    try:
        import streamlit as st
        val = st.secrets.get(key, None)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


# Database
DATABASE_URL = _get_secret("DATABASE_URL", "postgresql+psycopg://app:password@localhost:5432/claimverifier")

# Ollama Configuration
OLLAMA_BASE_URL = _get_secret("OLLAMA_BASE_URL")
OLLAMA_API_KEY = _get_secret("OLLAMA_API_KEY")
OLLAMA_MODEL = _get_secret("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud")

# Validation function to be called by consumers
def validate_ollama_config():
    missing = []
    if not OLLAMA_BASE_URL:
        missing.append("OLLAMA_BASE_URL")
    if not OLLAMA_API_KEY:
        missing.append("OLLAMA_API_KEY")
    
    if missing:
        raise ValueError(f"Missing required Ollama configuration: {', '.join(missing)}. Please set these in your .env file.")

# Model Configurations
MODEL_CONFIGS = {
    "default": f"ollama_chat/{OLLAMA_MODEL}", # Use configurable model via Ollama
        "groq_backup": "groq/llama-3.3-70b-versatile",
        "premium_claude": "anthropic/claude-3-5-sonnet-20240620",
        "premium_openai": "openai/gpt-4o",
        "local_qwq": "ollama/qwq:32b",
        "local_small": "ollama/deepseek-r1:7b"
}

# Companies
COMPANIES = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "JNJ", "WMT", "NVDA"]

# Fixed Quarters for Testing (2024 has complete data)
def get_last_n_quarters(n=4):
    # Use fixed quarters from 2024 to ensure complete data is available
    return ["2024Q4", "2024Q3", "2024Q2", "2024Q1"]

QUARTERS = get_last_n_quarters(4)

def parse_quarter_string(q_str: str) -> tuple[int, int]:
    """Parse '2025Q3' -> (2025, 3)"""
    year = int(q_str[:4])
    quarter = int(q_str[-1])
    return (year, quarter)

QUARTERS_TUPLES = [parse_quarter_string(q) for q in QUARTERS]

# Entity Types for GLiNER
FINANCIAL_ENTITY_TYPES = [
    "financial metric", 
    "percentage value", 
    "dollar amount", 
    "growth rate", 
    "earnings per share", 
    "profit margin", 
    "revenue figure", 
    "year over year comparison", 
    "quarter over quarter comparison", 
    "non-GAAP indicator"
]

# API Keys (loaded via st.secrets → os.environ → .env)
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
FINNHUB_API_KEY = _get_secret("FINNHUB_API_KEY")
SEC_IDENTITY_EMAIL = _get_secret("SEC_IDENTITY_EMAIL")

ACTIVE_MODEL_TIER = _get_secret("ACTIVE_MODEL_TIER", "default")

# Data Protection
# CRITICAL: This flag controls whether destructive database operations are allowed.
# The data for 10 companies has been fully ingested and verified. It must NEVER be deleted.
# Set to True ONLY in development when you explicitly need to clear and re-ingest data.
ALLOW_DESTRUCTIVE_OPERATIONS = os.getenv("ALLOW_DESTRUCTIVE_OPERATIONS", "false").lower() == "true"
