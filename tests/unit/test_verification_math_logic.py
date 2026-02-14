import pytest
from unittest.mock import patch, MagicMock
from src.models import Claim
from src.verifier.deterministic import verify_deterministic

"""
Unit Test: Verification Math Logic
This test verifies the mathematical correctness of verification (e.g. 15% vs 0.15) using mocks.
Requires:
- No external dependencies
When to use it:
- Run this to ensure the core math capabilities of the verifier are correct.
"""

@pytest.fixture
def db_session():
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

def mock_compute_logic(ticker, metric, year, quarter, db):
    # Default values for verification testing
    if metric == "revenue":
        if year == 2024 and quarter == 2: return 115.0
        if year == 2023 and quarter == 2: return 100.0
        if year == 2024 and quarter == 1: return 110.0
    if metric == "net_income":
        if year == 2024 and quarter == 2: return 20.0
        if year == 2023 and quarter == 2: return 15.0
    if metric == "eps":
        if year == 2024 and quarter == 2: return 1.52
    return 0.0

def test_verify_revenue_success(db_session):
    claim = create_test_claim()
    with patch("src.verifier.deterministic.compute_metric", side_effect=mock_compute_logic):
        verdict = verify_deterministic(claim, db_session)
        assert verdict.verdict == "VERIFIED"
        assert abs(verdict.actual_value - 0.15) < 0.001

def test_verify_approximate(db_session):
    claim = create_test_claim()
    def mock_approx(ticker, metric, year, quarter, db):
        if metric == "revenue":
            if year == 2024 and quarter == 2: return 114.8
            if year == 2023 and quarter == 2: return 100.0
            return 110.0
        return 20.0
        
    with patch("src.verifier.deterministic.compute_metric", side_effect=mock_approx):
        verdict = verify_deterministic(claim, db_session)
        assert verdict.verdict == "APPROXIMATELY_TRUE"

def test_verify_false(db_session):
    claim = create_test_claim()
    def mock_false(ticker, metric, year, quarter, db):
        if metric == "revenue":
            if year == 2024 and quarter == 2: return 110.2
            if year == 2023 and quarter == 2: return 100.0
            return 110.0
        return 20.0
        
    with patch("src.verifier.deterministic.compute_metric", side_effect=mock_false):
        verdict = verify_deterministic(claim, db_session)
        assert verdict.verdict == "FALSE"

def test_verify_absolute_verified(db_session):
    claim = create_test_claim(
        metric="revenue", value=94.8, unit="dollars_billions", period="quarterly"
    )
    def mock_abs(ticker, metric, year, quarter, db):
        if metric == "revenue": return 94.836
        return 0.0
        
    with patch("src.verifier.deterministic.compute_metric", side_effect=mock_abs):
        verdict = verify_deterministic(claim, db_session)
        assert verdict.verdict == "VERIFIED"

def test_cherry_picking_misleading(db_session):
    claim = create_test_claim(value=10.0)
    def mock_cherry(ticker, metric, year, quarter, db):
        if metric == "revenue":
            if year == 2024 and quarter == 2: return 110.0
            if year == 2023 and quarter == 2: return 100.0
        if metric == "net_income":
            if year == 2024 and quarter == 2: return 50.0  # Fell from 100
            if year == 2023 and quarter == 2: return 100.0
        return 100.0
        
    with patch("src.verifier.deterministic.compute_metric", side_effect=mock_cherry):
        verdict = verify_deterministic(claim, db_session)
        assert verdict.verdict == "MISLEADING"
        assert "Revenue is growing YoY, but Net Income is declining." in verdict.explanation
