#!/usr/bin/env python3
"""
Script to index all document chunks for selected companies from the database.

This script will:
1. Clear existing chunks from the document_chunks table
2. Query financial and transcript data from the database
3. Chunk the data using docling's hybrid chunker
4. Generate embeddings (BGE-small dense + SPLADE sparse)
5. Store the chunks in the document_chunks table

Usage:
    uv run python -m scripts.index_document_chunks
"""

import logging
import sys
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, '.')

# Import modules
from src.db.connection import SessionLocal
from src.rag.indexer import (
    index_all_companies_from_db,
    clear_existing_chunks,
    get_document_chunks_stats
)

# Configuration
COMPANIES = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "JNJ", "WMT", "NVDA"]
QUARTERS = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]


def main():
    logger.info("=== Starting Document Chunks Indexing ===")
    
    # Initialize database connection
    db_session: Session = SessionLocal()
    
    try:
        # Get initial stats
        initial_stats = get_document_chunks_stats(db_session)
        logger.info(f"Initial document chunks stats: {initial_stats}")
        
        # Clear existing chunks (optional but recommended for fresh indexing)
        # CRITICAL: This is DISABLED in production to protect verified data
        from src.config import ALLOW_DESTRUCTIVE_OPERATIONS
        
        if ALLOW_DESTRUCTIVE_OPERATIONS:
            logger.info("Clearing existing document chunks...")
            for ticker in COMPANIES:
                clear_existing_chunks(ticker, db_session)
        else:
            logger.warning(
                "SKIPPING chunk clearing: ALLOW_DESTRUCTIVE_OPERATIONS is False. "
                "The existing indexed data will be preserved. "
                "Set ALLOW_DESTRUCTIVE_OPERATIONS=true in .env ONLY if you are in development "
                "and explicitly need to re-index from scratch."
            )
            
        # Index all companies
        logger.info(f"Indexing {len(COMPANIES)} companies: {', '.join(COMPANIES)}")
        index_all_companies_from_db(COMPANIES, QUARTERS, db_session)
        
        # Get final stats
        final_stats = get_document_chunks_stats(db_session)
        logger.info(f"Final document chunks stats: {final_stats}")
        
        logger.info("=== Indexing Completed ===")
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        db_session.rollback()
        sys.exit(1)
    finally:
        db_session.close()


if __name__ == "__main__":
    main()
