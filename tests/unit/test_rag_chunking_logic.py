import logging
import pandas as pd
from unittest.mock import MagicMock
from src.rag.indexer import chunk_financial_data, chunk_transcript_data, index_documents
from src.models import Transcript, TranscriptSegment

"""
Unit Test: RAG Chunking Logic
This test verifies that transcripts and financial data are chunked correctly before indexing.
Requires:
- No external dependencies (uses mocks for DB)
When to use it:
- Run this when modifying the chunking strategy or prompt context.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chunking():
    ticker = "AAPL"
    
    # Mock financial data
    df_income = pd.DataFrame([
        {"label": "Revenue", "value": 94836000000},
        {"label": "Net Income", "value": 21448000000}
    ])
    df_income.name = "Income Statement"
    
    financials = {
        "2025Q2": {
            "income": df_income,
            "source": "10-Q"
        }
    }
    
    fin_chunks = chunk_financial_data(ticker, financials)
    assert len(fin_chunks) == 2
    assert fin_chunks[0]["ticker"] == "AAPL"
    assert "Revenue: $94,836,000,000" in fin_chunks[0]["text"]
    assert fin_chunks[0]["metric_type"] == "Revenue"
    
    # Mock transcript
    transcript = Transcript(
        ticker=ticker,
        year=2025,
        quarter=2,
        date="2025-05-01",
        segments=[
            TranscriptSegment(speaker="Tim Cook", role="CEO", text="We had a record quarter."),
            TranscriptSegment(speaker="Luca Maestri", role="CFO", text="Revenue grew 5%.")
        ]
    )
    
    tr_chunks = chunk_transcript_data(ticker, transcript)
    assert len(tr_chunks) == 2
    assert "Speaker: Tim Cook" in tr_chunks[0]["text"]
    assert "We had a record quarter." in tr_chunks[0]["text"]
    
    logger.info("Chunking tests passed!")

def test_indexing():
    # Mock DB session
    db_session = MagicMock()
    
    chunks = [
        {
            "ticker": "AAPL",
            "year": 2025,
            "quarter": 2,
            "chunk_type": "financial",
            "text": "Company: AAPL | Period: Q2 2025 | Form: 10-Q\nRevenue: $94,836,000,000"
        }
    ]
    
    # This will actually call fastembed, so it might take a moment
    index_documents(chunks, db_session)
    
    # Verify insert was called
    assert db_session.execute.called
    assert db_session.commit.called
    
    logger.info("Indexing tests passed (with mocks)!")

if __name__ == "__main__":
    test_chunking()
    test_indexing()
