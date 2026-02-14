import logging
import os
import pytest
from src.config import COMPANIES, QUARTERS_TUPLES
from src.data_ingest.financials import fetch_financial_statements

"""
Integration Test: Data Availability on Real DB
This test checks the state of the database to ensure all required transcripts and financial data are present.
Requires:
- A populated database
- Network access (fetches financials if missing)
- RUN_INTEGRATION_TESTS=1 environment variable
When to use it:
- Run this to diagnose missing data issues.
- Run after a large ingestion job to verify completeness.
"""

if not os.getenv("RUN_INTEGRATION_TESTS"):
    pytest.skip("Integration tests require RUN_INTEGRATION_TESTS=1", allow_module_level=True)

from src.data_ingest.transcripts import fetch_transcript
from src.db.connection import SessionLocal
from src.db.schema import TranscriptRecord, FinancialData, DocumentChunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_availability_test")

def test_data_availability():
    """Test data availability for all companies and quarters"""
    db = SessionLocal()
    
    logger.info("=== Data Availability Test ===")
    logger.info(f"Companies: {', '.join(COMPANIES)}")
    logger.info(f"Quarters: {[f'{y}Q{q}' for y, q in QUARTERS_TUPLES]}")
    logger.info("=" * 50)
    
    results = {}
    
    for ticker in COMPANIES:
        logger.info(f"\nTesting {ticker}...")
        ticker_results = {
            'transcripts': {},
            'financials': {},
            'chunks': {}
        }
        
        # Check DB records
        t_count = db.query(TranscriptRecord).filter(TranscriptRecord.ticker == ticker).count()
        f_count = db.query(FinancialData).filter(FinancialData.ticker == ticker).count()
        d_count = db.query(DocumentChunk).filter(DocumentChunk.ticker == ticker).count()
        
        logger.info(f"DB Records: Transcripts={t_count}, Financials={f_count}, Chunks={d_count}")
        
        # Test financial data (force refresh to get complete 2024 data)
        financial_data = fetch_financial_statements(ticker, force_refresh=True)
        available_quarters = list(financial_data.keys())
        logger.info(f"Financial quarters available: {available_quarters}")
        
        # Check required quarters
        for year, quarter in QUARTERS_TUPLES:
            quarter_str = f"{year}Q{quarter}"
            ticker_results['financials'][quarter_str] = quarter_str in available_quarters
            
            if quarter_str in financial_data:
                metrics = list(financial_data[quarter_str]['metrics'].keys())
                logger.info(f"  {quarter_str}: {len(metrics)} metrics available")
        
        # Test transcript availability
        for year, quarter in QUARTERS_TUPLES:
            quarter_str = f"{year}Q{quarter}"
            transcript = fetch_transcript(ticker, year, quarter, db)
            ticker_results['transcripts'][quarter_str] = transcript is not None
            
            if transcript:
                logger.info(f"  {quarter_str}: Transcript found ({len(transcript.segments)} segments)")
        
        results[ticker] = ticker_results
    
    logger.info("\n" + "=" * 50)
    logger.info("=== Summary Report ===")
    
    # Summary statistics
    total_transcripts = 0
    total_financials = 0
    required = len(COMPANIES) * len(QUARTERS_TUPLES)
    
    for ticker in COMPANIES:
        t_available = sum(results[ticker]['transcripts'].values())
        f_available = sum(results[ticker]['financials'].values())
        total_transcripts += t_available
        total_financials += f_available
        
        logger.info(f"{ticker}: {t_available}/{len(QUARTERS_TUPLES)} transcripts, {f_available}/{len(QUARTERS_TUPLES)} financial periods")
    
    logger.info(f"\nOverall: {total_transcripts}/{required} transcripts, {total_financials}/{required} financial periods")
    
    # Check if all data is available
    if total_transcripts == required and total_financials == required:
        logger.info("\n✅ All required data is available!")
    else:
        logger.warning("\n⚠️  Some data is missing!")
    
    db.close()
    return results

if __name__ == "__main__":
    test_data_availability()
