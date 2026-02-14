import pytest
from unittest.mock import MagicMock, patch
from src.models import Claim
from src.db.schema import FinancialData
from src.verifier.deterministic import verify_deterministic, compute_metric, detect_cherry_picking

"""
Unit Test: Verification Deterministic Logic
This test verifies the deterministic logic for verifying claims (Growth, Margin, EPS) using mocks.
Requires:
- No external dependencies
When to use it:
- Run this to verify math and logic for deterministic verification rules.
"""

@pytest.fixture
def mock_db():
    return MagicMock()

def test_compute_metric_direct(mock_db):
    # Setup mock to return a value for a specific metric
    mock_data = MagicMock(spec=FinancialData)
    mock_data.value = 100.0
    
    with patch("src.verifier.deterministic.load_financial_data", return_value=mock_data):
        val = compute_metric("AAPL", "revenue", 2023, 3, mock_db)
        assert val == 100.0

def test_compute_metric_alias(mock_db):
    # Setup mock to return None for 'revenue' but value for 'SalesRevenueNet'
    mock_data = MagicMock(spec=FinancialData)
    mock_data.value = 500.0
    
    def side_effect(db, ticker, metric, year, quarter):
        if metric == "SalesRevenueNet":
            return mock_data
        return None

    with patch("src.verifier.deterministic.load_financial_data", side_effect=side_effect):
        val = compute_metric("AAPL", "revenue", 2023, 3, mock_db)
        assert val == 500.0

def test_verify_growth_success(mock_db):
    # YoY Growth: (120 - 100) / 100 = 20%
    curr_data = MagicMock(value=120.0)
    prev_data = MagicMock(value=100.0)
    
    def side_effect(db, ticker, metric, year, quarter):
        if year == 2023: return curr_data
        if year == 2022: return prev_data
        return None

    claim = Claim(
        id="c1", ticker="AAPL", year=2023, quarter=3, speaker="CEO",
        metric="revenue", value=20.0, unit="%", period="YoY",
        is_gaap=True, is_forward_looking=False, hedging_language="false",
        raw_text="Revenue grew 20% YoY", extraction_method="llm", confidence=0.9, context=""
    )

    with patch("src.verifier.deterministic.load_financial_data", side_effect=side_effect):
        verdict = verify_deterministic(claim, mock_db)
        assert verdict.verdict == "VERIFIED"
        assert verdict.actual_value == 0.2

def test_verify_growth_approx(mock_db):
    # YoY Growth: (121 - 100) / 100 = 21%
    curr_data = MagicMock(value=121.0)
    prev_data = MagicMock(value=100.0)
    
    def side_effect(db, ticker, metric, year, quarter):
        if year == 2023: return curr_data
        if year == 2022: return prev_data
        return None

    claim = Claim(
        id="c2", ticker="AAPL", year=2023, quarter=3, speaker="CEO",
        metric="revenue", value=20.0, unit="%", period="YoY",
        is_gaap=True, is_forward_looking=False, hedging_language="true", # Hedged
        raw_text="Revenue grew about 20% YoY", extraction_method="llm", confidence=0.9, context=""
    )

    with patch("src.verifier.deterministic.load_financial_data", side_effect=side_effect):
        verdict = verify_deterministic(claim, mock_db)
        # 21% actual vs 20% claimed. Diff = 1%. 
        # Hedged threshold is 2% for growth. So it should be approx true.
        assert verdict.verdict == "APPROXIMATELY_TRUE"

def test_verify_eps_precision(mock_db):
    actual_eps = MagicMock(value=1.254) # Rounds to 1.25
    
    claim = Claim(
        id="c3", ticker="AAPL", year=2023, quarter=3, speaker="CFO",
        metric="eps", value=1.25, unit="$", period="quarter",
        is_gaap=True, is_forward_looking=False, hedging_language="false",
        raw_text="EPS was 1.25", extraction_method="llm", confidence=0.9, context=""
    )

    with patch("src.verifier.deterministic.load_financial_data", return_value=actual_eps):
        verdict = verify_deterministic(claim, mock_db)
        assert verdict.verdict == "VERIFIED"

def test_cherry_picking(mock_db):
    # Revenue up, Net Income down
    # 2023: Rev 120, NI 10
    # 2022: Rev 100, NI 15
    def side_effect(db, ticker, metric, year, quarter):
        if year == 2023:
            if metric == "revenue": return MagicMock(value=120.0)
            if metric == "net_income": return MagicMock(value=10.0)
        if year == 2022:
            if metric == "revenue": return MagicMock(value=100.0)
            if metric == "net_income": return MagicMock(value=15.0)
        return None

    claim = Claim(
        id="c4", ticker="AAPL", year=2023, quarter=3, speaker="CEO",
        metric="revenue", value=120.0, unit="M", period="quarter",
        is_gaap=True, is_forward_looking=False, hedging_language="false",
        raw_text="Revenue was 120M", extraction_method="llm", confidence=0.9, context=""
    )

    with patch("src.verifier.deterministic.load_financial_data", side_effect=side_effect):
        verdict = verify_deterministic(claim, mock_db)
        assert verdict.verdict == "MISLEADING"
        assert "Net Income is declining" in verdict.explanation
