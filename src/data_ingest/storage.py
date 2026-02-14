import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from src.db.schema import TranscriptRecord, FinancialData, ClaimRecord, VerdictRecord
from src.models import Transcript, Claim, Verdict

logger = logging.getLogger(__name__)

def save_transcript(db: Session, transcript: Transcript, source: str = "finnhub"):
    """Saves a transcript to the database. Skips if already exists (immutable data)."""
    try:
        # Check if already exists
        existing = db.query(TranscriptRecord).filter(
            TranscriptRecord.ticker == transcript.ticker,
            TranscriptRecord.year == transcript.year,
            TranscriptRecord.quarter == transcript.quarter
        ).first()
        
        if existing:
            logger.info(
                f"Duplicate detected — skipping insert for transcript {transcript.ticker} "
                f"{transcript.year}Q{transcript.quarter}. Data is immutable and already exists."
            )
            return
        
        segments_data = [s.model_dump() for s in transcript.segments]
        full_text = "\n".join([f"{s.speaker}: {s.text}" for s in transcript.segments])
        
        new_record = TranscriptRecord(
            ticker=transcript.ticker,
            year=transcript.year,
            quarter=transcript.quarter,
            date=transcript.date,
            source=source,
            full_text=full_text,
            segments=segments_data
        )
        db.add(new_record)
        logger.info(f"Saved new transcript for {transcript.ticker} {transcript.year}Q{transcript.quarter}")
        
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving transcript: {e}")
        raise

def save_claims(db: Session, claims: List[Claim]):
    """Saves multiple claims to the database."""
    try:
        for claim in claims:
            existing = db.query(ClaimRecord).filter(ClaimRecord.id == claim.id).first()
            if existing:
                # Update logic if needed, or skip
                continue
            
            new_claim = ClaimRecord(**claim.model_dump())
            db.add(new_claim)
        
        db.commit()
        logger.info(f"Saved {len(claims)} claims")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving claims: {e}")
        raise

def save_verdicts(db: Session, verdicts: List[Verdict]):
    """Saves multiple verdicts to the database."""
    try:
        for verdict in verdicts:
            new_verdict = VerdictRecord(**verdict.model_dump())
            db.add(new_verdict)
        
        db.commit()
        logger.info(f"Saved {len(verdicts)} verdicts")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving verdicts: {e}")
        raise

def save_financial_data(db: Session, data: List[FinancialData]):
    """Saves multiple financial metrics to the database. Skips if already exists (immutable data)."""
    try:
        skipped_count = 0
        inserted_count = 0
        
        for item in data:
            # Check for existing record
            existing = db.query(FinancialData).filter(
                FinancialData.ticker == item.ticker,
                FinancialData.metric == item.metric,
                FinancialData.year == item.year,
                FinancialData.quarter == item.quarter,
                FinancialData.is_gaap == item.is_gaap
            ).first()
            
            if existing:
                skipped_count += 1
                continue
            
            db.add(item)
            inserted_count += 1
        
        db.commit()
        logger.info(
            f"Saved {inserted_count} new financial data records. "
            f"Skipped {skipped_count} duplicates (data is immutable)."
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving financial data: {e}")
        raise

def load_financial_data(db: Session, ticker: str, metric: str, year: int, quarter: int) -> Optional[FinancialData]:
    """Loads a specific financial metric from the database."""
    return db.query(FinancialData).filter(
        FinancialData.ticker == ticker,
        FinancialData.metric == metric,
        FinancialData.year == year,
        FinancialData.quarter == quarter
    ).first()
