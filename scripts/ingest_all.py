"""
Batch ingestion script for all 10 target companies.
Ingests transcripts (HuggingFace/Finnhub) and financial data (SEC EDGAR)
into the database and indexes for RAG.

Usage: .venv/Scripts/python.exe -m scripts.ingest_all
"""
import logging
import time
from typing import List, Tuple
from src.db.connection import SessionLocal
from src.data_ingest.transcripts import fetch_transcript
from src.data_ingest.financials import fetch_financial_statements
from src.rag.indexer import index_company
from src.db.migrations import init_db
from src.config import COMPANIES, QUARTERS_TUPLES
from src.db.schema import TranscriptRecord, FinancialData, DocumentChunk

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ingest_all")


def check_existing_data(db, ticker: str) -> dict:
    """Check what data already exists for this company."""
    transcripts = db.query(TranscriptRecord).filter(TranscriptRecord.ticker == ticker).count()
    financials = db.query(FinancialData).filter(FinancialData.ticker == ticker).count()
    chunks = db.query(DocumentChunk).filter(DocumentChunk.ticker == ticker).count()
    return {"transcripts": transcripts, "financials": financials, "chunks": chunks}


def ingest_company(ticker: str, quarters: List[Tuple[int, int]]):
    """Ingest financial data and transcripts for a single company."""
    logger.info(f"--- Starting ingestion for {ticker} ---")
    db = SessionLocal()
    try:
        # Check existing data
        existing = check_existing_data(db, ticker)
        logger.info(f"Existing data for {ticker}: {existing}")
        
        if existing["transcripts"] > 0 and existing["financials"] > 0 and existing["chunks"] > 0:
            logger.info(f"Data already exists for {ticker}, skipping ingestion")
            return
        
        # Fetch financial statements (stores to DB internally and returns organized dict)
        logger.info(f"Fetching financial statements for {ticker}...")
        financials = fetch_financial_statements(ticker, n_quarters=len(quarters) + 2)
        logger.info(f"Got {len(financials)} periods of financial data for {ticker}")
        
        transcripts = []
        for year, q in quarters:
            logger.info(f"Fetching transcript for {ticker} {year} Q{q}...")
            try:
                transcript = fetch_transcript(ticker, year, q, db=db)
                if transcript:
                    transcripts.append(transcript)
                    logger.info(f"Successfully fetched transcript for {year} Q{q}")
                else:
                    logger.warning(f"No transcript found for {year} Q{q}")
            except Exception as e:
                logger.error(f"Failed to fetch transcript for {year} Q{q}: {e}")
        
        if not transcripts and not financials:
            logger.warning(f"No data found for {ticker}, skipping indexing.")
            return

        # Only index if not already indexed
        if existing["chunks"] == 0:
            logger.info(f"Indexing company data for {ticker}...")
            index_company(ticker, transcripts, financials, db=db)
            logger.info(f"Successfully indexed {ticker}")
        else:
            logger.info(f"Index already exists for {ticker}, skipping")
        
    except Exception as e:
        logger.error(f"Error during ingestion for {ticker}: {e}")
    finally:
        db.close()

def main():
    logger.info("Initializing database...")
    init_db()
    
    logger.info(f"Target companies: {COMPANIES}")
    logger.info(f"Target quarters: {QUARTERS_TUPLES}")
    
    start_time = time.time()
    logger.info(f"Starting ingestion for {len(COMPANIES)} companies...")
    
    for i, ticker in enumerate(COMPANIES):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i+1}/{len(COMPANIES)}] Processing {ticker}")
            logger.info(f"{'='*60}")
            ingest_company(ticker, QUARTERS_TUPLES)
        except Exception as e:
            logger.error(f"Fatal error ingesting {ticker}: {e}")
            continue
            
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Ingestion complete! Total time: {duration/60:.2f} minutes")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
