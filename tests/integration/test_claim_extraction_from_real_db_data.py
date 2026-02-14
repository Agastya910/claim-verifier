import logging
import os
import pytest

"""
Integration Test: Claim Extraction from Real DB Data
This test verifies that claim extraction works correctly on real transcript data stored in the database.
Requires:
- A populated database with transcript records
- RUN_INTEGRATION_TESTS=1 environment variable
When to use it:
- Run this when you want to verify extraction against real-world examples, not just mocks.
- Run locally or in CI only when the DB is seeded.
"""

if not os.getenv("RUN_INTEGRATION_TESTS"):
    pytest.skip("Integration tests require RUN_INTEGRATION_TESTS=1", allow_module_level=True)

from src.db.connection import SessionLocal
from src.db.schema import TranscriptRecord
from src.models import Transcript, TranscriptSegment
from src.claim_extraction.pipeline import extract_all_claims

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("actual_claim_extraction")

def test_actual_claim_extraction():
    """Test claim extraction on real data from the database"""
    db = SessionLocal()
    
    # Get first transcript from the database
    transcript_record = db.query(TranscriptRecord).first()
    
    if not transcript_record:
        logger.warning("No transcripts found in database")
        return
    
    logger.info(f"Testing claim extraction for {transcript_record.ticker} {transcript_record.year}Q{transcript_record.quarter}")
    
    # Convert DB record to Transcript object
    segments = []
    if transcript_record.segments:
        for s in transcript_record.segments:
            segments.append(TranscriptSegment(
                speaker=s.get("speaker", "Unknown"),
                role=s.get("role", "Unknown"),
                text=s.get("text", "")
            ))
    
    transcript = Transcript(
        ticker=transcript_record.ticker,
        year=transcript_record.year,
        quarter=transcript_record.quarter,
        date=transcript_record.date or None,
        segments=segments
    )
    
    # Test claim extraction
    logger.info(f"Extracting claims from {len(segments)} segments")
    claims = extract_all_claims(transcript)
    
    logger.info(f"Successfully extracted {len(claims)} claims")
    
    if claims:
        logger.info("\n=== Sample Claims ===")
        for i, claim in enumerate(claims[:3]):
            logger.info(f"{i+1}. {claim.metric}: {claim.value} {claim.unit} ({claim.confidence:.2f})")
            logger.info(f"   '{claim.raw_text}'")
    
    db.close()
    return claims

if __name__ == "__main__":
    test_actual_claim_extraction()
