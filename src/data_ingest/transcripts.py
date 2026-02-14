import logging
import time
from typing import List, Optional, Dict
from datetime import date

import finnhub
from datasets import load_dataset
from sqlalchemy.orm import Session

from src.config import FINNHUB_API_KEY
from src.models import Transcript, TranscriptSegment
from src.data_ingest.storage import save_transcript
from src.db.connection import SessionLocal
from src.db.schema import TranscriptRecord

logger = logging.getLogger(__name__)

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


def load_transcript_from_db(db: Session, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
    """Check if transcript already exists in DB and return it."""
    record = db.query(TranscriptRecord).filter(
        TranscriptRecord.ticker == ticker.upper(),
        TranscriptRecord.year == year,
        TranscriptRecord.quarter == quarter
    ).first()
    
    if not record or not record.segments:
        return None
    
    segments = []
    for s in record.segments:
        segments.append(TranscriptSegment(
            speaker=s.get("speaker", "Unknown"),
            role=s.get("role", "Unknown"),
            text=s.get("text", "")
        ))
    
    if not segments:
        return None
    
    logger.info(f"Loaded transcript from DB for {ticker} {year}Q{quarter} ({len(segments)} segments)")
    return Transcript(
        ticker=record.ticker,
        year=record.year,
        quarter=record.quarter,
        date=record.date or date.today(),
        segments=segments
    )


def fetch_transcript_finnhub(ticker: str, year: int, quarter: int) -> Optional[Transcript]:
    """Fetches transcript from Finnhub API with short timeout."""
    try:
        logger.info(f"Fetching transcripts list for {ticker} from Finnhub")
        transcripts_list = finnhub_client.transcripts_list(ticker)
        
        # Find matching year/quarter
        target_entry = None
        for entry in transcripts_list.get("transcripts", []):
            if entry.get("year") == year and entry.get("quarter") == quarter:
                target_entry = entry
                break
        
        if not target_entry:
            logger.warning(f"No Finnhub transcript found for {ticker} {year}Q{quarter}")
            return None
        
        transcript_id = target_entry.get("id")
        logger.info(f"Fetching full transcript content for ID {transcript_id}")
        content = finnhub_client.transcripts(transcript_id)
        
        segments = []
        for s in content.get("transcript", []):
            segments.append(TranscriptSegment(
                speaker=s.get("name", "Unknown"),
                role=s.get("role", "Unknown"),
                text=s.get("speech", "")
            ))
        
        if not segments:
            logger.warning(f"Finnhub transcript for {ticker} {year}Q{quarter} had no segments")
            return None
        
        transcript_date = date.today()
        
        return Transcript(
            ticker=ticker,
            year=year,
            quarter=quarter,
            date=transcript_date,
            segments=segments
        )
    except Exception as e:
        logger.warning(f"Finnhub fetch failed for {ticker} {year}Q{quarter}: {e}")
        return None

def fetch_transcript_huggingface(ticker: str, year: int, quarter: int) -> Optional[Transcript]:
    """Fetches transcript from HuggingFace Dataset fallback."""
    try:
        logger.info(f"Fetching {ticker} {year}Q{quarter} from HuggingFace dataset")
        ds = load_dataset("Bose345/sp500_earnings_transcripts", split="train")
        
        filtered = ds.filter(lambda x: x["symbol"] == ticker and x["year"] == year and x["quarter"] == quarter)
        
        if len(filtered) == 0:
            logger.warning(f"No HuggingFace transcript found for {ticker} {year}Q{quarter}")
            return None
        
        record = filtered[0]
        raw_segments = record.get("structured_content", [])
        
        # If structured_content is empty, try to parse from 'content' field  
        if not raw_segments and record.get("content"):
            logger.info(f"No structured_content, using raw content for {ticker} {year}Q{quarter}")
            raw_text = record["content"]
            # Create a single segment from raw content
            segments = [TranscriptSegment(
                speaker="Transcript",
                role="Full Text",
                text=raw_text[:50000]  # Limit size
            )]
        else:
            segments = []
            for s in raw_segments:
                text = s.get("text", "")
                if text.strip():
                    segments.append(TranscriptSegment(
                        speaker=s.get("speaker", "Unknown"),
                        role="Unknown",
                        text=text
                    ))
        
        if not segments:
            logger.warning(f"HuggingFace transcript for {ticker} {year}Q{quarter} had no usable segments")
            return None
        
        logger.info(f"Got {len(segments)} segments from HuggingFace for {ticker} {year}Q{quarter}")
        return Transcript(
            ticker=ticker,
            year=year,
            quarter=quarter,
            date=date.today(),
            segments=segments
        )
    except Exception as e:
        logger.error(f"Error fetching from HuggingFace: {e}")
        return None

def fetch_transcript(ticker: str, year: int, quarter: int, db: Optional[Session] = None) -> Optional[Transcript]:
    """Orchestrates transcript fetching: DB cache -> Finnhub -> HuggingFace."""
    # Step 0: Check DB cache first
    if db:
        cached = load_transcript_from_db(db, ticker, year, quarter)
        if cached:
            return cached
    
    # Step 1: Try Finnhub (quick fail if unavailable)
    transcript = fetch_transcript_finnhub(ticker, year, quarter)
    source = "finnhub"
    
    # Step 2: Fallback to HuggingFace
    if not transcript:
        logger.info(f"Finnhub failed for {ticker} {year}Q{quarter}, trying HuggingFace fallback")
        transcript = fetch_transcript_huggingface(ticker, year, quarter)
        source = "huggingface"
        
    # Step 3: Store if we got something
    if transcript and db:
        save_transcript(db, transcript, source=source)
        logger.info(f"Successfully fetched and stored transcript from {source}")
        
    return transcript

def fetch_all_transcripts(companies: List[str], quarters: List[tuple[int, int]]) -> Dict[str, List[Transcript]]:
    """Batch fetch transcripts for multiple companies and quarters."""
    results = {}
    db = SessionLocal()
    try:
        for ticker in companies:
            results[ticker] = []
            for year, q in quarters:
                transcript = fetch_transcript(ticker, year, q, db=db)
                if transcript:
                    results[ticker].append(transcript)
                
                # Rate limiting for Finnhub
                time.sleep(0.5)
    finally:
        db.close()
    return results
