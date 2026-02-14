"""
Test script for the refactored /api/ask endpoint.

This script demonstrates the lightweight keyword-based search functionality
that queries the claims table directly without using embeddings or RAG.
"""

import httpx
import json
from typing import Dict, Any


def test_ask_endpoint(
    ticker: str,
    question: str,
    backend_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Test the /api/ask endpoint with a question.
    
    Args:
        ticker: Company ticker symbol
        question: Natural language question
        backend_url: Backend API URL
        
    Returns:
        Response from the API
    """
    endpoint = f"{backend_url}/api/ask"
    payload = {
        "ticker": ticker,
        "question": question
    }
    
    print(f"\n{'='*80}")
    print(f"Testing /api/ask endpoint")
    print(f"{'='*80}")
    print(f"Ticker: {ticker}")
    print(f"Question: {question}")
    print(f"{'='*80}\n")
    
    try:
        response = httpx.post(endpoint, json=payload, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        
        print("✅ SUCCESS\n")
        print(f"Answer:\n{result.get('answer', 'No answer')}\n")
        print(f"Claims Used: {result.get('num_claims_used', 0)}")
        print(f"Claim IDs: {result.get('claim_ids', [])}\n")
        
        return result
        
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_keyword_matching():
    """Test keyword-based search with specific terms."""
    test_ask_endpoint(
        ticker="AAPL",
        question="What was the revenue in Q4 2024?"
    )


def test_fallback_behavior():
    """Test fallback to recent claims when no keywords match."""
    test_ask_endpoint(
        ticker="AAPL",
        question="Tell me about the company"
    )


def test_specific_metric():
    """Test search for a specific metric."""
    test_ask_endpoint(
        ticker="AAPL",
        question="What did they say about gross margin?"
    )


def test_verdict_search():
    """Test search that should match verdict explanations."""
    test_ask_endpoint(
        ticker="AAPL",
        question="Were there any false or misleading claims?"
    )


def test_no_data():
    """Test with a company that has no data."""
    test_ask_endpoint(
        ticker="INVALID",
        question="What is the revenue?"
    )


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     /api/ask Endpoint Test Suite                            ║
║                                                                              ║
║  This script tests the refactored lightweight keyword-based search          ║
║  implementation that queries claims directly without RAG/embeddings.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if backend is running
    try:
        health = httpx.get("http://localhost:8000/api/health", timeout=5.0)
        if health.status_code == 200:
            print("✅ Backend is running\n")
        else:
            print("⚠️  Backend returned unexpected status\n")
    except Exception:
        print("❌ Backend is not running. Please start it with:")
        print("   uv run uvicorn src.api.routes:app --reload\n")
        exit(1)
    
    # Run tests
    print("\n" + "="*80)
    print("TEST 1: Keyword Matching")
    print("="*80)
    test_keyword_matching()
    
    print("\n" + "="*80)
    print("TEST 2: Fallback Behavior")
    print("="*80)
    test_fallback_behavior()
    
    print("\n" + "="*80)
    print("TEST 3: Specific Metric Search")
    print("="*80)
    test_specific_metric()
    
    print("\n" + "="*80)
    print("TEST 4: Verdict Search")
    print("="*80)
    test_verdict_search()
    
    print("\n" + "="*80)
    print("TEST 5: No Data Available")
    print("="*80)
    test_no_data()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80 + "\n")
