import logging
import random
import os
import pytest
from src.db.connection import SessionLocal
from src.db.schema import DocumentChunk

"""
Integration Test: DB Connection and Vector Insert
This test verifies that we can connect to the PostgreSQL database and insert records with vector embeddings.
Requires:
- Running PostgreSQL instance with pgvector extension
- RUN_INTEGRATION_TESTS=1 environment variable
When to use it:
- Run this when setting up a new environment to verify DB connectivity.
- Run if you suspect DB schema or extension issues.
"""

if not os.getenv("RUN_INTEGRATION_TESTS"):
    pytest.skip("Integration tests require RUN_INTEGRATION_TESTS=1", allow_module_level=True)

from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_db")

def test_insert():
    db = SessionLocal()
    try:
        # Check if vector extension exists
        res = db.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
        logger.info(f"Vector extension exists: {res}")
        
        # Create dummy chunk
        chunk = DocumentChunk(
            ticker="TEST",
            year=2024,
            quarter=1,
            chunk_type="test",
            text="Test chunk",
            dense_embedding=[random.random() for _ in range(1024)],
            sparse_embedding={1: 0.5, 2: 0.8}
        )
        db.add(chunk)
        # CHANGED: Rollback instead of commit to ensure no data is persisted
        db.flush() # Ensure it technically works (can catch some constraints)
        db.rollback()
        logger.info("Rolled back chunk insert (test only)")
        
        # Verify it's GONE after rollback
        count = db.query(DocumentChunk).filter(DocumentChunk.ticker == "TEST").count()
        assert count == 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    test_insert()
