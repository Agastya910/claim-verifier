import pytest
from unittest.mock import MagicMock, patch
from src.models import Claim, Verdict, Transcript
from src.verifier.pipeline import verify_claim, verify_company

"""
Unit Test: Verification Pipeline (Mocked)
This test verifies the orchestration of the verification pipeline (Deterministic -> RAG -> LLM).
Requires:
- No external dependencies
When to use it:
- Run this to verify the fallback logic (e.g., if deterministic fails, try LLM).
"""

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def sample_claim():
    return Claim(
        id="p1", ticker="NVDA", year=2024, quarter=1, speaker="Jensen Huang",
        metric="gaming revenue", value=2.5, unit="B", period="quarter",
        is_gaap=True, is_forward_looking=False, hedging_language="false",
        raw_text="Gaming revenue was $2.5 billion", extraction_method="llm", confidence=0.9, context=""
    )

def test_verify_claim_deterministic_first(mock_db, sample_claim):
    # Setup deterministic to succeed
    mock_verdict = Verdict(
        claim_id="p1", verdict="VERIFIED", actual_value=2.5, claimed_value=2.5,
        difference=0.0, explanation="Matched perfectly", confidence=1.0,
        data_sources=["DET"]
    )
    
    with patch("src.verifier.pipeline.verify_deterministic", return_value=mock_verdict) as mock_det, \
         patch("src.verifier.pipeline.detect_cherry_picking", return_value=[]), \
         patch("src.verifier.pipeline.compute_metric", return_value=2.5), \
         patch("src.verifier.pipeline.retrieve_for_claim") as mock_rag:
        
        verdict = verify_claim(sample_claim, mock_db)
        
        assert verdict.verdict == "VERIFIED"
        assert mock_det.called
        assert not mock_rag.called # Should NOT fallback to RAG if deterministic works

def test_verify_claim_llm_fallback(mock_db, sample_claim):
    # Setup deterministic to fail (None)
    mock_llm_verdict = Verdict(
        claim_id="p1", verdict="VERIFIED", actual_value=2.5, claimed_value=2.5,
        difference=0.0, explanation="LLM verified this", confidence=1.0,
        data_sources=["LLM"]
    )
    
    with patch("src.verifier.pipeline.verify_deterministic", return_value=None), \
         patch("src.verifier.pipeline.detect_cherry_picking", return_value=[]), \
         patch("src.verifier.pipeline.compute_metric", return_value=2.5), \
         patch("src.verifier.pipeline.retrieve_for_claim", return_value=[]), \
         patch("src.verifier.pipeline.build_verification_context", return_value="Context"), \
         patch("src.verifier.pipeline.verify_with_llm", return_value=mock_llm_verdict) as mock_llm:
        
        verdict = verify_claim(sample_claim, mock_db)
        
        assert verdict.verdict == "VERIFIED"
        assert mock_llm.called

def test_verify_company_pipeline(mock_db):
    ticker = "NVDA"
    quarters = [(2024, 1)]
    
    # Mock all external calls
    mock_transcript = MagicMock(spec=Transcript)
    mock_transcript.ticker = ticker
    mock_transcript.year = 2024
    mock_transcript.quarter = 1
    mock_transcript.segments = []
    
    mock_claim = Claim(
        id="c1", ticker=ticker, year=2024, quarter=1, speaker="X",
        metric="revenue", value=100.0, unit="M", period="quarter",
        is_gaap=True, is_forward_looking=False, hedging_language="false",
        raw_text="...", extraction_method="llm", confidence=0.9, context="..."
    )
    
    mock_verdict = Verdict(
        claim_id="c1", verdict="VERIFIED", actual_value=100.0, claimed_value=100.0,
        difference=0.0, explanation="Accurate", confidence=1.0,
        data_sources=["DET"]
    )

    with patch("src.verifier.pipeline.fetch_transcript", return_value=mock_transcript), \
         patch("src.verifier.pipeline.fetch_financial_statements", return_value={}), \
         patch("src.verifier.pipeline.extract_all_claims", return_value=[mock_claim]), \
         patch("src.verifier.pipeline.index_company"), \
         patch("src.verifier.pipeline.verify_claim", return_value=mock_verdict):
        
        result = verify_company(ticker, quarters, mock_db, "default")
        
        assert result.company == ticker
        assert result.summary_stats["total_claims"] == 1
        assert result.summary_stats["verified_count"] == 1
