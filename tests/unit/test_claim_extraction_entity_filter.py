import sys
import os
from datetime import date
import logging

"""
Unit Test: Entity Filter Logic
This test verifies that the entity filter correctly identifies financial sentences using GLiNER logic (functional test).
Requires:
- GLiNER model (local or cached)
When to use it:
- Run this when tuning the entity extraction threshold or logic.
"""

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Transcript, TranscriptSegment
from src.claim_extraction.entity_filter import filter_financial_sentences

# Configure logging to see the output
logging.basicConfig(level=logging.INFO)

def test_filter():
    # Create sample segments
    segments = [
        TranscriptSegment(
            speaker="Tim Cook",
            role="CEO",
            text="Welcome to the call. We had a great quarter. Revenue grew by 15% year over year."
        ),
        TranscriptSegment(
            speaker="Luca Maestri",
            role="CFO",
            text="Our earnings per share was $2.10. This is a record high for the company."
        ),
        TranscriptSegment(
            speaker="Analyst 1",
            role="Analyst",
            text="Can you comment on the growth in China?"
        ),
        TranscriptSegment(
            speaker="Tim Cook",
            role="CEO",
            text="China was strong. We saw 10% growth there. The weather was also nice today."
        )
    ]
    
    transcript = Transcript(
        ticker="AAPL",
        year=2023,
        quarter=4,
        date=date(2023, 10, 27),
        segments=segments
    )
    
    print("Running filter...")
    results = filter_financial_sentences(transcript)
    
    print(f"\nFiltered Results ({len(results)} sentences):")
    for res in results:
        print(f"[{res['speaker']} ({res['role']})]: {res['sentence']}")
        print(f"Entities: {res['entities']}")
        print("-" * 20)

    # Verification
    sentences_text = [r['sentence'] for r in results]
    
    # 1. Should have financial sentences
    assert any("Revenue grew by 15% year over year" in s for s in sentences_text)
    assert any("Our earnings per share was $2.10" in s for s in sentences_text)
    assert any("We saw 10% growth there" in s for s in sentences_text)
    
    # 2. Should NOT have non-financial sentences (ideally, though GLiNER might be sensitive)
    # "Welcome to the call" is unlikely to have entities
    assert not any("Welcome to the call" in s for s in sentences_text)
    
    # 3. Should NOT have analyst segments
    assert not any("Analyst 1" == r['speaker'] for r in results)
    
    print("\nVerification successful!")

if __name__ == "__main__":
    test_filter()
