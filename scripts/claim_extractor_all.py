import logging
from src.db.connection import SessionLocal
from src.db.schema import TranscriptRecord
from src.models import Transcript, TranscriptSegment
from src.claim_extraction.pipeline import extract_all_claims
from src.config import COMPANIES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claim_extraction_all")

def test_claim_extraction_all():
    """Test claim extraction for all companies and quarters in the database"""
    db = SessionLocal()
    
    logger.info("=== Testing Claim Extraction for All Companies ===")
    
    # Get all transcripts from the database
    all_transcripts = db.query(TranscriptRecord).all()
    
    if not all_transcripts:
        logger.warning("No transcripts found in database")
        return
    
    logger.info(f"Found {len(all_transcripts)} transcripts")
    
    # Group by company
    company_transcripts = {}
    for rec in all_transcripts:
        if rec.ticker not in company_transcripts:
            company_transcripts[rec.ticker] = []
        company_transcripts[rec.ticker].append(rec)
    
    # Process each company's transcripts
    results = {}
    for ticker in COMPANIES:
        logger.info(f"\n=== Processing {ticker} ===")
        
        if ticker not in company_transcripts:
            logger.warning(f"No transcripts found for {ticker}")
            continue
        
        ticker_claims = []
        
        for transcript_record in company_transcripts[ticker]:
            logger.info(f"  {ticker} {transcript_record.year}Q{transcript_record.quarter}")
            
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
            
            # Extract claims
            claims = extract_all_claims(transcript)
            ticker_claims.extend(claims)
            logger.info(f"    Extracted {len(claims)} claims")
        
        results[ticker] = ticker_claims
    
    logger.info("\n=== Summary ===")
    total_claims = 0
    for ticker in COMPANIES:
        if ticker in results:
            logger.info(f"{ticker}: {len(results[ticker])} claims")
            total_claims += len(results[ticker])
    
    logger.info(f"\nTotal: {total_claims} claims extracted")
    
    db.close()
    return results

if __name__ == "__main__":
    test_claim_extraction_all()
