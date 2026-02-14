import logging
from typing import List
from src.models import Transcript, Claim
from src.claim_extraction.entity_filter import filter_financial_sentences
from src.claim_extraction.llm_extractor import extract_claims_llm
from src.claim_extraction.normalizer import normalize_claims, enrich_context
from src.data_ingest.storage import save_claims
from src.db.connection import SessionLocal

logger = logging.getLogger(__name__)

def extract_all_claims(transcript: Transcript, model_tier: str = "default") -> List[Claim]:
    """
    End-to-end pipeline: Filter -> Extract -> Normalize -> Enrich -> Store.
    """
    logger.info(f"Starting claim extraction for {transcript.ticker} {transcript.year}Q{transcript.quarter}")
    
    # Step 1: GLiNER entity pre-filter
    filtered_sentences = filter_financial_sentences(transcript)
    total_sentences = sum(len(segment.text.split('.')) for segment in transcript.segments) # Rough estimate
    
    # Step 2: LLM extraction
    raw_claims = extract_claims_llm(
        filtered_sentences, 
        transcript.ticker, 
        transcript.quarter, 
        transcript.year, 
        model_tier
    )
    
    # Step 3: Normalization and deduplication
    normalized_claims = normalize_claims(raw_claims)
    
    # Step 4: Context enrichment
    full_transcript_text = "\n".join([f"{s.speaker}: {s.text}" for s in transcript.segments])
    enriched_claims = [enrich_context(c, full_transcript_text) for c in normalized_claims]
    
    # Step 5: Storage
    db = SessionLocal()
    try:
        save_claims(db, enriched_claims)
        logger.info(f"Stored {len(enriched_claims)} claims in database")
    except Exception as e:
        logger.error(f"Failed to store claims: {e}")
    finally:
        db.close()
        
    # Log statistics
    high_conf = len([c for c in enriched_claims if c.confidence >= 0.8])
    low_conf = len([c for c in enriched_claims if c.confidence < 0.8])
    
    logger.info(
        f"Extraction Complete: Filtered {len(filtered_sentences)} sentences. "
        f"Found {len(enriched_claims)} claims ({high_conf} high conf, {low_conf} low conf)."
    )
    
    return enriched_claims

def extract_claims_for_company(ticker: str, transcripts: List[Transcript], model_tier: str) -> dict[str, List[Claim]]:
    """
    Run extraction across all provided transcripts for a company.
    """
    results = {}
    for transcript in transcripts:
        quarter_key = f"{transcript.year}Q{transcript.quarter}"
        claims = extract_all_claims(transcript, model_tier)
        results[quarter_key] = claims
    return results
