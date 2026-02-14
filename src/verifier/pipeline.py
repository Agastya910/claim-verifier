import logging
import time
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from src.models import Claim, Verdict, VerificationResult
from src.verifier.deterministic import verify_deterministic, detect_cherry_picking, compute_metric
from src.verifier.llm_verifier import verify_with_llm
from src.rag.pipeline import retrieve_for_claim, build_verification_context
from src.data_ingest.transcripts import fetch_transcript
from src.data_ingest.financials import fetch_financial_statements
from src.rag.indexer import index_company
from src.claim_extraction.pipeline import extract_all_claims
from src.db.schema import ClaimRecord, VerdictRecord, TranscriptRecord, DocumentChunk

logger = logging.getLogger(__name__)


def _load_cached_results(ticker: str, quarters: List[tuple[int, int]], db_session: Session, force_rerun: bool = False) -> Optional[VerificationResult]:
    """Check if results already exist in DB for this company/quarters."""
    if force_rerun:
        return None
        
    all_claims = []
    all_verdicts = []
    
    for year, q in quarters:
        claims = db_session.query(ClaimRecord).filter(
            ClaimRecord.ticker == ticker.upper(),
            ClaimRecord.year == year,
            ClaimRecord.quarter == q
        ).all()
        
        if claims:
            verdicts = db_session.query(VerdictRecord).filter(
                VerdictRecord.claim_id.in_([c.id for c in claims])
            ).all()
            all_claims.extend(claims)
            all_verdicts.extend(verdicts)
    
    # Only return cached results if we have BOTH claims AND verdicts
    if all_claims and all_verdicts:
        logger.info(f"Found {len(all_claims)} cached claims and {len(all_verdicts)} verdicts for {ticker}")
        
        # Convert DB records to model objects
        claims_out = []
        for c in all_claims:
            claims_out.append(Claim(
                id=c.id, ticker=c.ticker, quarter=c.quarter, year=c.year,
                speaker=c.speaker or "", metric=c.metric or "", value=c.value or 0.0,
                unit=c.unit or "", period=c.period or "", is_gaap=c.is_gaap if c.is_gaap is not None else True,
                is_forward_looking=c.is_forward_looking or False,
                hedging_language=c.hedging_language or "",
                raw_text=c.raw_text or "", extraction_method=c.extraction_method or "",
                confidence=c.confidence or 0.0, context=c.context or ""
            ))
        
        verdicts_out = []
        for v in all_verdicts:
            verdicts_out.append(Verdict(
                claim_id=v.claim_id, verdict=v.verdict or "UNVERIFIABLE",
                actual_value=v.actual_value, claimed_value=v.claimed_value or 0.0,
                difference=v.difference, explanation=v.explanation or "",
                misleading_flags=v.misleading_flags or [],
                confidence=v.confidence or 0.0, data_sources=v.data_sources or []
            ))
        
        total = len(verdicts_out)
        if total > 0:
            verified = len([v for v in verdicts_out if v.verdict == "VERIFIED"])
            approx = len([v for v in verdicts_out if v.verdict == "APPROXIMATELY_TRUE"])
            false_claims = len([v for v in verdicts_out if v.verdict == "FALSE"])
            misleading = len([v for v in verdicts_out if v.verdict == "MISLEADING"])
            
            summary_stats = {
                "total_claims": total,
                "accuracy_score": (verified + approx) / total,
                "verified_count": verified,
                "approx_true_count": approx,
                "false_count": false_claims,
                "misleading_count": misleading,
                "unverifiable_count": len([v for v in verdicts_out if v.verdict == "UNVERIFIABLE"])
            }
        else:
            summary_stats = {"total_claims": 0}
        
        return VerificationResult(
            company=ticker,
            quarter=f"{quarters[0][0]}Q{quarters[0][1]}",
            claims=claims_out,
            verdicts=verdicts_out,
            summary_stats=summary_stats
        )
    
    return None


def _has_indexed_data(ticker: str, db_session: Session) -> bool:
    """Check if document chunks exist for this company (i.e., data has been indexed for RAG)."""
    count = db_session.query(DocumentChunk).filter(DocumentChunk.ticker == ticker.upper()).count()
    return count > 0


def verify_claim(claim: Claim, db_session: Session, model_tier: str = "default") -> Verdict:
    """
    Tiered verification:
    1. Deterministic check (highest confidence, cheapest)
    2. RAG + LLM fallback (general knowledge, context-rich)
    """
    logger.info(f"Verifying claim {claim.id} for {claim.ticker} {claim.year}Q{claim.quarter}")
    
    # STEP 1: Try deterministic verification first
    verdict = verify_deterministic(claim, db_session)
    
    # STEP 2: Fallback to LLM if deterministic couldn't verify (None or UNVERIFIABLE)
    if not verdict or verdict.verdict == "UNVERIFIABLE":
        logger.info(f"Deterministic verification failed or inconclusive for {claim.id}. falling back to RAG+LLM.")
        
        # Build context through RAG
        retrieved_docs = retrieve_for_claim(claim, db_session)
        context = build_verification_context(claim, retrieved_docs)
        
        # Verify with LLM
        verdict = verify_with_llm(claim, context, db_session, model_tier)
    
    # STEP 3: Post-processing
    new_flags = detect_cherry_picking(claim.ticker, claim.year, claim.quarter, claim.metric.lower(), db_session)
    for flag in new_flags:
        if flag not in verdict.misleading_flags:
            verdict.misleading_flags.append(flag)
            verdict.verdict = "MISLEADING"
            
    # Compute alternative time comparison
    if claim.period == "YoY":
        prev_q_year = claim.year
        prev_q = claim.quarter - 1
        if prev_q == 0:
            prev_q = 4
            prev_q_year = claim.year - 1
            
        curr_val = compute_metric(claim.ticker, claim.metric.lower(), claim.year, claim.quarter, db_session)
        prev_val = compute_metric(claim.ticker, claim.metric.lower(), prev_q_year, prev_q, db_session)
        
        if curr_val is not None and prev_val is not None and prev_val != 0:
            qoq_growth = (curr_val - prev_val) / prev_val
            verdict.explanation += f" Context: QoQ growth was {qoq_growth:.2%}."

    return verdict

def verify_all_claims(claims: List[Claim], db_session: Session, model_tier: str) -> List[Verdict]:
    """Processes multiple claims with rate limiting."""
    verdicts = []
    total = len(claims)
    for i, claim in enumerate(claims):
        logger.info(f"[{i+1}/{total}] Verifying claim...")
        verdict = verify_claim(claim, db_session, model_tier)
        verdicts.append(verdict)
        
        # Simple rate limiting for LLM fallback protection
        time.sleep(1.0) 
        
    return verdicts

def verify_company(ticker: str, quarters: List[tuple[int, int]], db_session: Session, model_tier: str, force_rerun: bool = False) -> VerificationResult:
    """
    Full end-to-end pipeline: Ingest -> Index -> Extract -> Verify.
    
    CACHING: Returns cached results from DB if they already exist, unless force_rerun=True.
    Only re-processes if no claims/verdicts are found for this company.
    """
    logger.info(f"Starting E2E verification for {ticker} across {len(quarters)} quarters (force_rerun={force_rerun})")
    
    # STEP 0: Check for cached results
    cached = _load_cached_results(ticker, quarters, db_session, force_rerun)
    if cached:
        logger.info(f"Returning cached results for {ticker}: {cached.summary_stats}")
        return cached
    
    transcripts = []
    all_claims = []
    
    # 1. Ingest & Extract
    financials = fetch_financial_statements(ticker, n_quarters=len(quarters) + 1)
    
    for year, q in quarters:
        # Fetch transcript (will use DB cache if available)
        transcript = fetch_transcript(ticker, year, q, db_session)
        if transcript:
            transcripts.append(transcript)
            
            # Extract claims (only if not already extracted)
            existing_claims = db_session.query(ClaimRecord).filter(
                ClaimRecord.ticker == ticker.upper(),
                ClaimRecord.year == year,
                ClaimRecord.quarter == q
            ).all()
            
            if existing_claims:
                logger.info(f"Using {len(existing_claims)} cached claims for {ticker} {year}Q{q}")
                for c in existing_claims:
                    all_claims.append(Claim(
                        id=c.id, ticker=c.ticker, quarter=c.quarter, year=c.year,
                        speaker=c.speaker or "", metric=c.metric or "", value=c.value or 0.0,
                        unit=c.unit or "", period=c.period or "", is_gaap=c.is_gaap if c.is_gaap is not None else True,
                        is_forward_looking=c.is_forward_looking or False,
                        hedging_language=c.hedging_language or "",
                        raw_text=c.raw_text or "", extraction_method=c.extraction_method or "",
                        confidence=c.confidence or 0.0, context=c.context or ""
                    ))
            else:
                claims = extract_all_claims(transcript, model_tier)
                all_claims.extend(claims)

    # 2. Index for RAG (only if not already indexed)
    if not _has_indexed_data(ticker, db_session):
        index_company(ticker, transcripts, financials, db=db_session)
    else:
        logger.info(f"RAG index already exists for {ticker}, skipping indexing")
    
    # 3. Verify
    verdicts = verify_all_claims(all_claims, db_session, model_tier)
    
    # 4. Compute Summary Stats
    total = len(verdicts)
    if total > 0:
        verified = len([v for v in verdicts if v.verdict == "VERIFIED"])
        approx = len([v for v in verdicts if v.verdict == "APPROXIMATELY_TRUE"])
        false_claims = len([v for v in verdicts if v.verdict == "FALSE"])
        misleading = len([v for v in verdicts if v.verdict == "MISLEADING"])
        
        summary_stats = {
            "total_claims": total,
            "accuracy_score": (verified + approx) / total,
            "verified_count": verified,
            "approx_true_count": approx,
            "false_count": false_claims,
            "misleading_count": misleading,
            "unverifiable_count": len([v for v in verdicts if v.verdict == "UNVERIFIABLE"])
        }
    else:
        summary_stats = {"total_claims": 0}

    return VerificationResult(
        company=ticker,
        quarter=f"{quarters[0][0]}Q{quarters[0][1]}",
        claims=all_claims,
        verdicts=verdicts,
        summary_stats=summary_stats
    )

def verify_all_companies(companies: List[str], quarters: List[tuple[int, int]], db_session: Session, model_tier: str) -> List[VerificationResult]:
    """Runs batch verification across multiple companies."""
    results = []
    for ticker in companies:
        try:
            result = verify_company(ticker, quarters, db_session, model_tier)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to verify company {ticker}: {e}")
            
    return results
