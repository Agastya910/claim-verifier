import logging
import os
import pytest
from src.config import QUARTERS_TUPLES
from scripts.ingest_all import ingest_company
from src.db.connection import SessionLocal

"""
Integration Test: Ingestion Full Flow (Real API)
This test runs the full ingestion process for a single company/quarter, hitting real APIs (Simulated/Real).
Requires:
- API Keys (Finnhub, etc.)
- Network connection
- RUN_INTEGRATION_TESTS=1 environment variable
When to use it:
- Run this to verify the ingestion pipeline is working end-to-end.
- Run when updating ingestion logic or API clients.
"""

if not os.getenv("RUN_INTEGRATION_TESTS"):
    pytest.skip("Integration tests require RUN_INTEGRATION_TESTS=1", allow_module_level=True)

if not os.getenv("ALLOW_DB_WRITES"):
    pytest.skip("Skipping ingestion test because ALLOW_DB_WRITES is not set. Ingestion inherently modifies the DB.", allow_module_level=True)

from src.db.schema import TranscriptRecord, FinancialData, DocumentChunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_ingest")

def check_counts(ticker):
    db = SessionLocal()
    try:
        t_count = db.query(TranscriptRecord).filter(TranscriptRecord.ticker == ticker).count()
        f_count = db.query(FinancialData).filter(FinancialData.ticker == ticker).count()
        d_count = db.query(DocumentChunk).filter(DocumentChunk.ticker == ticker).count()
        return t_count, f_count, d_count
    finally:
        db.close()

if __name__ == "__main__":
    ticker = "AAPL"
    logger.info(f"Testing ingestion for {ticker}...")
    
    # Check initial state
    t0, f0, d0 = check_counts(ticker)
    logger.info(f"Initial counts: Transcripts={t0}, Financials={f0}, Chunks={d0}")
    
    # Run ingestion for 2024 Q3 (guaranteed to exist)
    quarters = [(2024, 3)]
    ingest_company(ticker, quarters)
    
    # Check final state
    t1, f1, d1 = check_counts(ticker)
    logger.info(f"Final counts: Transcripts={t1}, Financials={f1}, Chunks={d1}")
    
    if t1 > t0 or f1 > f0 or d1 > d0:
        logger.info("SUCCESS: Data ingestion increased counts.")
    else:
        logger.warning(f"WARNING: Counts did not increase (data might already exist).")
