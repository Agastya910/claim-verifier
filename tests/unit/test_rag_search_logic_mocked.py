import logging
from unittest.mock import MagicMock, patch
from src.models import Claim
from src.rag.retriever import hybrid_search
from src.rag.reranker import rerank
from src.rag.pipeline import retrieve_for_claim, build_verification_context

"""
Unit Test: RAG Search Logic (Mocked)
This test verifies the hybrid search and reranking orchestration logic using mocks.
Requires:
- No external dependencies
When to use it:
- Run this to verify how search results are processed and combined.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_search():
    db_session = MagicMock()
    # Mock return value for db_session.execute().mappings().all()
    mock_results = [
        {
            "id": 1,
            "text": "AAPL Q2 2025 Revenue was $94.8B",
            "ticker": "AAPL",
            "year": 2025,
            "quarter": 2,
            "chunk_type": "financial",
            "metric_type": "Revenue",
            "source_type": "10-Q",
            "rrf_score": 0.033
        }
    ]
    db_session.execute.return_value.mappings.return_value.all.return_value = mock_results
    
    results = hybrid_search("test query", db_session, ticker="AAPL", year=2025, quarter=2)
    assert len(results) == 1
    assert results[0]["text"] == "AAPL Q2 2025 Revenue was $94.8B"
    assert results[0]["metadata"]["ticker"] == "AAPL"
    
    logger.info("Hybrid search test passed!")

def test_rerank():
    query = "What was the revenue?"
    candidates = [
        {"text": "Revenue was $100", "id": 1},
        {"text": "The weather is nice", "id": 2},
        {"text": "Net income grew", "id": 3}
    ]
    
    # This will actually call the model if not mocked
    results = rerank(query, candidates, top_k=2)
    assert len(results) == 2
    # "Revenue was $100" should have higher score than "The weather is nice"
    assert results[0]["text"] == "Revenue was $100"
    
    logger.info("Rerank test passed!")

def test_pipeline_end_to_end():
    db_session = MagicMock()
    
    # Mock deterministic lookup (empty)
    db_session.query.return_value.filter.return_value.all.return_value = []
    
    # Mock hybrid search
    mock_search_results = [
        {
            "id": 10,
            "text": "AAPL Q2 2025 Revenue: $94.8B",
            "score": 0.05,
            "metadata": {"ticker": "AAPL", "year": 2025, "quarter": 2}
        }
    ]
    
    with patch("src.rag.pipeline.hybrid_search", return_value=mock_search_results):
        # We don't necessarily need to patch rerank if it's fast enough, but let's be safe
        with patch("src.rag.pipeline.rerank", return_value=mock_search_results):
            claim = Claim(
                id="claim-123",
                ticker="AAPL",
                quarter=2,
                year=2025,
                speaker="Tim Cook",
                metric="Revenue",
                value=94.8e9,
                unit="USD",
                period="Q2 2025",
                is_gaap=True,
                is_forward_looking=False,
                hedging_language="",
                raw_text="Revenue was $94.8 billion in Q2.",
                extraction_method="llm",
                confidence=0.9,
                context="Tim Cook said..."
            )
            
            results = retrieve_for_claim(claim, db_session)
            assert len(results) > 0
            assert results[0]["text"] == "AAPL Q2 2025 Revenue: $94.8B"
            
            context = build_verification_context(claim, results)
            assert "AAPL Q2 2025 Revenue: $94.8B" in context
            
    logger.info("Pipeline end-to-end test passed!")

if __name__ == "__main__":
    test_hybrid_search()
    test_rerank()
    test_pipeline_end_to_end()
