import json
import pytest
from unittest.mock import MagicMock, patch
from src.models import Claim
from src.verifier.llm_verifier import verify_with_llm

"""
Unit Test: Verification LLM Logic (Mocked)
This test verifies the LLM verifier's handling of various LLM responses (success, retry, failure) using mocks.
Requires:
- No external dependencies
When to use it:
- Run this to verify that the verifier robustly handles LLM outputs.
"""

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def sample_claim():
    return Claim(
        id="c_test_1", ticker="AAPL", year=2023, quarter=3, speaker="CEO",
        metric="revenue", value=11.0, unit="%", period="YoY",
        is_gaap=True, is_forward_looking=False, hedging_language="false",
        raw_text="Revenue grew 11% YoY", extraction_method="llm", confidence=0.9, context="Context"
    )

def test_verify_with_llm_success(mock_db, sample_claim):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "verdict": "VERIFIED",
            "actual_value": 11.0,
            "claimed_value": 11.0,
            "difference": 0.0,
            "explanation": "Calculated 11% correctly.",
            "misleading_flags": [],
            "confidence": "high",
            "data_sources_used": ["SEC 10-Q"]
        })))
    ]

    with patch("litellm.completion", return_value=mock_response), \
         patch("src.verifier.llm_verifier.save_verdicts") as mock_save:
        
        verdict = verify_with_llm(sample_claim, "Context data", mock_db)
        
        assert verdict.verdict == "VERIFIED"
        assert verdict.actual_value == 11.0
        assert mock_save.called

def test_verify_with_llm_retry_success(mock_db, sample_claim):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "verdict": "VERIFIED",
            "actual_value": 11.0,
            "claimed_value": 11.0,
            "explanation": "Success after retry",
            "confidence": "high"
        })))
    ]

    # First call fails, second succeeds
    with patch("litellm.completion", side_effect=[Exception("API Error"), mock_response]), \
         patch("src.verifier.llm_verifier.save_verdicts"), \
         patch("time.sleep"):
        
        verdict = verify_with_llm(sample_claim, "Context data", mock_db)
        assert verdict.verdict == "VERIFIED"
        assert "Success after retry" in verdict.explanation

def test_verify_with_llm_persistent_failure(mock_db, sample_claim):
    with patch("litellm.completion", side_effect=Exception("API Error")), \
         patch("src.verifier.llm_verifier.save_verdicts") as mock_save, \
         patch("time.sleep"):
        
        verdict = verify_with_llm(sample_claim, "Context data", mock_db)
        assert verdict.verdict == "UNVERIFIABLE"
        assert "failed after 3 retries" in verdict.explanation
        assert mock_save.called

def test_verify_with_llm_markdown_json(mock_db, sample_claim):
    # LLM sometimes wraps JSON in markdown blocks
    markdown_content = "Here is the response:\n```json\n{\"verdict\": \"FALSE\", \"explanation\": \"Bad math\"}\n```"
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=markdown_content))
    ]

    with patch("litellm.completion", return_value=mock_response), \
         patch("src.verifier.llm_verifier.save_verdicts"):
        
        verdict = verify_with_llm(sample_claim, "Context data", mock_db)
        assert verdict.verdict == "FALSE"
        assert verdict.explanation == "Bad math"
