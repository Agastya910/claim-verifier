import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.rag.indexer import chunk_transcript_data, index_documents
from src.rag.pipeline import retrieve_for_claim
from src.models import Transcript, TranscriptSegment, Claim

"""
Unit Test: RAG Indexing and Retrieval (Mocked)
This test verifies the RAG, indexing, and pipeline logic using mocks for embeddings and DB.
Requires:
- No external dependencies
When to use it:
- Run this to verify the RAG pipeline logic without hitting a vector DB or embedding model.
"""

@pytest.fixture
def mock_db():
    return MagicMock()

def create_test_claim(id="1", **kwargs):
    defaults = {
        "id": id,
        "ticker": "AAPL",
        "year": 2024,
        "quarter": 2,
        "speaker": "CEO",
        "metric": "revenue",
        "value": 15.0,
        "unit": "%",
        "period": "YoY",
        "is_gaap": True,
        "is_forward_looking": False,
        "hedging_language": "false",
        "raw_text": "Revenue grew 15% YoY",
        "extraction_method": "test",
        "confidence": 1.0,
        "context": "test context"
    }
    defaults.update(kwargs)
    return Claim(**defaults)

def test_chunking_logic():
    from datetime import date
    transcript = Transcript(
        ticker="AAPL", year=2024, quarter=2, date=date(2024, 7, 28),
        segments=[
            TranscriptSegment(speaker="Tim Cook", role="CEO", text="Revenue grew 10%."),
            TranscriptSegment(speaker="Luca Maestri", role="CFO", text="EPS was record high.")
        ]
    )
    
    chunks = chunk_transcript_data("AAPL", transcript)
    
    assert len(chunks) == 2
    assert "Tim Cook" in chunks[0]["text"]
    assert "AAPL" in chunks[1]["text"]

@patch("src.rag.indexer.dense_model")
@patch("src.rag.indexer.sparse_model")
def test_indexing_calls(mock_sparse, mock_dense, mock_db):
    # Mocking embedding models
    mock_dense.embed.return_value = [np.array([0.1, 0.2])]
    
    # Create a mock object that has .indices.tolist() and .values.tolist()
    class MockSparseEmb:
        def __init__(self):
            self.indices = np.array([1, 2])
            self.values = np.array([0.5, 0.6])
            
    mock_sparse.embed.return_value = [MockSparseEmb()]
    
    chunks = [{
        "ticker": "AAPL", "year": 2024, "quarter": 2, 
        "chunk_type": "transcript", "text": "Test chunk"
    }]
    
    index_documents(chunks, mock_db)
    
    assert mock_db.execute.called
    assert mock_db.commit.called

@patch("src.rag.pipeline.hybrid_search")
@patch("src.rag.pipeline.rerank")
def test_retrieval_pipeline(mock_rerank, mock_hybrid, mock_db):
    claim = create_test_claim(raw_text="Revenue grew 15% YoY")
    
    # Mock hybrid search results
    mock_hybrid.return_value = [{"id": "chunk1", "text": "Result 1", "score": 0.8}]
    mock_rerank.return_value = [{"text": "Result 1", "score": 0.9}]
    
    # Mock deterministic results (empty)
    mock_db.query.return_value.filter.return_value.all.return_value = []
    
    results = retrieve_for_claim(claim, mock_db)
    
    assert len(results) > 0
    # Should have called hybrid search twice (one for current, one for prior year due to YoY)
    assert mock_hybrid.call_count == 2
    assert mock_rerank.called

def test_deterministic_priority(mock_db):
    claim = create_test_claim(raw_text="Revenue grew 15% YoY")
    
    # Mock a gold source in financial_data
    mock_record = MagicMock()
    mock_record.ticker = "AAPL"
    mock_record.year = 2024
    mock_record.quarter = 2
    mock_record.metric = "revenue"
    mock_record.value = 94836.0
    mock_record.unit = "M"
    mock_record.source = "10-Q"
    
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_record]
    
    with patch("src.rag.pipeline.hybrid_search") as mock_hybrid:
        mock_hybrid.return_value = []
        results = retrieve_for_claim(claim, mock_db)
        
        assert len(results) == 1
        assert "GOLD SOURCE" in results[0]["text"]
        assert results[0]["metadata"]["is_gold"] is True
