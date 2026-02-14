import logging
import sys
import os
import pytest
import unittest.mock
from src.verifier.pipeline import verify_company
from src.db.connection import SessionLocal

"""
Integration Test: Verification End-to-End (Real DB)
This test runs the full verification pipeline on data already in the database.
Requires:
- Populated database for the target company/quarter
- LLM API Key (if not mocked)
- RUN_INTEGRATION_TESTS=1 environment variable
When to use it:
- Run this to verify the core value prop of the application: verifying claims.
- Run when changing verification logic or prompts.
"""

if not os.getenv("RUN_INTEGRATION_TESTS"):
    pytest.skip("Integration tests require RUN_INTEGRATION_TESTS=1", allow_module_level=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_verify")

if __name__ == "__main__":
    ticker = "AAPL"
    # Use recent quarters that have data (e.g., 2025 Q3, Q2, Q1, 2024 Q4)
    # The ingestion script ingested specific quarters.
    # We'll use the same ones.
    from src.config import QUARTERS_TUPLES
    
    logger.info(f"Testing verification for {ticker}...")
    
    db = SessionLocal()
    try:
        # Patch save_verdicts to prevent writing to DB
        with unittest.mock.patch("src.verifier.llm_verifier.save_verdicts") as mock_save:
            # Run verification (should reuse ingested data)
            quarters = [(2024, 3)]
            result = verify_company(ticker, quarters, db, model_tier="auto")
            
            logger.info(f"Verification completed for {ticker}")
            logger.info(f"Summary stats: {result.summary_stats}")
            # ... assertions ...
            logger.info(f"Total claims: {len(result.claims)}")
            logger.info(f"Total verdicts: {len(result.verdicts)}")
            
            if result.verdicts:
                logger.info("Sample verdict:")
                v = result.verdicts[0]
                logger.info(f"Claim: {v.claim.raw_text}")
                logger.info(f"Verdict: {v.verdict}")
                logger.info(f"Explanation: {v.explanation}")
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
