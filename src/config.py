import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://app:password@localhost:5432/claimverifier")

# Model Configurations
MODEL_CONFIGS = {
    "default": "ollama_chat/deepseek-v3.1:671b-cloud", # Use DeepSeek-V3.1 via Ollama
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

# API Keys (loaded via dotenv)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
SEC_IDENTITY_EMAIL = os.getenv("SEC_IDENTITY_EMAIL")

ACTIVE_MODEL_TIER = os.getenv("ACTIVE_MODEL_TIER", "default")

# Data Protection
# CRITICAL: This flag controls whether destructive database operations are allowed.
# The data for 10 companies has been fully ingested and verified. It must NEVER be deleted.
# Set to True ONLY in development when you explicitly need to clear and re-ingest data.
ALLOW_DESTRUCTIVE_OPERATIONS = os.getenv("ALLOW_DESTRUCTIVE_OPERATIONS", "false").lower() == "true"
