import logging
import time
import uuid
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import select, distinct, func, String

from src.db.connection import get_db, SessionLocal
from src.db.migrations import init_db
from src.db.schema import ClaimRecord, VerdictRecord, TranscriptRecord, FinancialData, JobRecord
from src.models import Claim, Verdict, VerificationResult
from src.data_ingest.transcripts import fetch_transcript
from src.data_ingest.financials import fetch_financial_statements
from src.rag.indexer import index_company
from src.claim_extraction.pipeline import extract_all_claims
from src.verifier.pipeline import verify_company, verify_all_companies
from src.config import COMPANIES, QUARTERS_TUPLES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests
from pydantic import BaseModel

class IngestRequest(BaseModel):
    ticker: str
    quarters: List[List[int]]  # List of [year, quarter]

class ExtractRequest(BaseModel):
    ticker: str
    year: int
    quarter: int

class VerifyRequest(BaseModel):
    ticker: str
    year: int
    quarter: int
    model_tier: str = "default"

class VerifyAllRequest(BaseModel):
    model_tier: str = "default"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Starting up: Initializing database...")
    try:
        init_db()
        logger.info("Database initialization complete.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Claim Verifier API",
    description="API for ingesting financial data, extracting claims, and verifying them.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Method: {request.method} Path: {request.url.path} Status: {response.status_code} Duration: {process_time:.4f}s")
    return response

@app.get("/")
async def root():
    return {"message": "Claim Verifier API is running", "docs": "/docs"}

@app.get("/api/health")
async def health(db: Session = Depends(get_db)):
    """System health check (DB connection, etc.)"""
    health_status = {
        "status": "healthy",
        "database": "connected",
        "timestamp": time.time()
    }
    try:
        # Simple query to check DB
        db.execute(select(1))
    except Exception as e:
        logger.error(f"Health check failed: Database connection error: {e}")
        health_status["status"] = "unhealthy"
        health_status["database"] = f"error: {str(e)}"
        
    return health_status

@app.get("/api/companies")
async def list_companies(db: Session = Depends(get_db)):
    """List unique companies (tickers) available in the system."""
    try:
        # Get unique tickers from both transcripts and financial data
        tickers_stmt = select(distinct(TranscriptRecord.ticker))
        tickers = db.execute(tickers_stmt).scalars().all()
        
        # Also check financial_data in case some are ingested but no transcripts yet
        fin_tickers_stmt = select(distinct(FinancialData.ticker))
        fin_tickers = db.execute(fin_tickers_stmt).scalars().all()
        
        unique_tickers = sorted(list(set(tickers) | set(fin_tickers)))
        return {"companies": unique_tickers}
    except Exception as e:
        logger.error(f"Error listing companies: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve companies list.")

@app.get("/api/companies/{ticker}/quarters")
async def list_quarters(ticker: str, db: Session = Depends(get_db)):
    """List available years and quarters for a specific company."""
    try:
        # Get quarters from transcripts
        trans_stmt = select(TranscriptRecord.year, TranscriptRecord.quarter).where(TranscriptRecord.ticker == ticker.upper())
        trans_rows = db.execute(trans_stmt).all()
        
        # Get quarters from financial data
        fin_stmt = select(FinancialData.year, FinancialData.quarter).where(FinancialData.ticker == ticker.upper())
        fin_rows = db.execute(fin_stmt).all()
        
        all_quarters = set()
        for y, q in trans_rows + fin_rows:
            all_quarters.add((y, q))
            
        result = [{"year": y, "quarter": q} for y, q in sorted(list(all_quarters), reverse=True)]
        return {"ticker": ticker.upper(), "available_quarters": result}
    except Exception as e:
        logger.error(f"Error listing quarters for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve quarters for {ticker}.")

def update_job_status(job_id: str, status: str, progress: float, message: str = None, result_metadata: dict = None):
    db = SessionLocal()
    try:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job:
            job.status = status
            job.progress = progress
            if message:
                job.message = message
            if result_metadata:
                job.result_metadata = result_metadata
            db.commit()
    except Exception as e:
        logger.error(f"Failed to update job status for {job_id}: {e}")
    finally:
        db.close()

# Background task functions
def run_ingest(job_id: str, ticker: str, quarters: List[List[int]]):
    update_job_status(job_id, "RUNNING", 0.1, f"Starting ingestion for {ticker}")
    db = SessionLocal()
    try:
        financials = fetch_financial_statements(ticker, n_quarters=len(quarters) + 1)
        update_job_status(job_id, "RUNNING", 0.4, f"Financials fetched for {ticker}")
        
        transcripts = []
        total_q = len(quarters)
        for i, (year, q) in enumerate(quarters):
            transcript = fetch_transcript(ticker, year, q, db=db)
            if transcript:
                transcripts.append(transcript)
            update_job_status(job_id, "RUNNING", 0.4 + (0.3 * (i+1)/total_q), f"Fetched transcript {year}Q{q}")
        
        index_company(ticker, transcripts, financials, db=db)
        update_job_status(job_id, "COMPLETED", 1.0, f"Ingestion and indexing complete for {ticker}")
    except Exception as e:
        logger.error(f"Background: Ingestion failed for {ticker}: {e}")
        update_job_status(job_id, "FAILED", 1.0, f"Error: {str(e)}")
    finally:
        db.close()

def run_extraction(job_id: str, ticker: str, year: int, quarter: int):
    update_job_status(job_id, "RUNNING", 0.1, f"Starting extraction for {ticker} {year}Q{quarter}")
    db = SessionLocal()
    try:
        transcript_record = db.query(TranscriptRecord).filter(
            TranscriptRecord.ticker == ticker.upper(),
            TranscriptRecord.year == year,
            TranscriptRecord.quarter == quarter
        ).first()
        
        if not transcript_record:
            update_job_status(job_id, "FAILED", 1.0, "No transcript found in database.")
            return

        from src.models import Transcript, TranscriptSegment
        from datetime import date
        transcript = Transcript(
            ticker=transcript_record.ticker,
            year=transcript_record.year,
            quarter=transcript_record.quarter,
            date=transcript_record.date or date.today(),
            segments=[TranscriptSegment(**s) for s in transcript_record.segments]
        )
        
        extract_all_claims(transcript)
        update_job_status(job_id, "COMPLETED", 1.0, f"Extraction complete for {ticker} {year}Q{quarter}")
    except Exception as e:
        logger.error(f"Background: Extraction failed for {ticker} {year}Q{quarter}: {e}")
        update_job_status(job_id, "FAILED", 1.0, f"Error: {str(e)}")
    finally:
        db.close()

def run_verification(job_id: str, ticker: str, quarters: List[tuple[int, int]], model_tier: str):
    update_job_status(job_id, "RUNNING", 0.1, f"Starting verification for {ticker}")
    try:
        db = SessionLocal()
        result = verify_company(ticker, quarters, db, model_tier)
        db.close()
        update_job_status(job_id, "COMPLETED", 1.0, f"Verification complete for {ticker}", result_metadata=result.summary_stats)
    except Exception as e:
        logger.error(f"Background: Verification failed for {ticker}: {e}")
        update_job_status(job_id, "FAILED", 1.0, f"Error: {str(e)}")

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get the status and progress of a background job."""
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/api/ingest")
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    job_id = str(uuid.uuid4())
    job = JobRecord(id=job_id, status="PENDING", message=f"Queued ingestion for {request.ticker.upper()}")
    db.add(job)
    db.commit()
    background_tasks.add_task(run_ingest, job_id, request.ticker.upper(), request.quarters)
    return {"message": "Ingestion triggered", "job_id": job_id, "ticker": request.ticker.upper()}

@app.post("/api/extract-claims")
async def extract_claims_endpoint(request: ExtractRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    job_id = str(uuid.uuid4())
    job = JobRecord(id=job_id, status="PENDING", message=f"Queued extraction for {request.ticker.upper()}")
    db.add(job)
    db.commit()
    background_tasks.add_task(run_extraction, job_id, request.ticker.upper(), request.year, request.quarter)
    return {"message": "Claim extraction triggered", "job_id": job_id}

@app.post("/api/verify")
async def verify_endpoint(request: VerifyRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    job_id = str(uuid.uuid4())
    job = JobRecord(id=job_id, status="PENDING", message=f"Queued verification for {request.ticker.upper()}")
    db.add(job)
    db.commit()
    background_tasks.add_task(run_verification, job_id, request.ticker.upper(), [(request.year, request.quarter)], request.model_tier)
    return {"message": "Verification triggered", "job_id": job_id}

@app.post("/api/verify-all")
async def verify_all_endpoint(request: VerifyAllRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # Simple version: trigger multiple jobs
    stmt = select(TranscriptRecord.ticker, TranscriptRecord.year, TranscriptRecord.quarter)
    rows = db.execute(stmt).all()
    
    companies_map = {}
    for ticker, year, quarter in rows:
        if ticker not in companies_map:
            companies_map[ticker] = []
        companies_map[ticker].append((year, quarter))
        
    job_ids = []
    for ticker, quarters in companies_map.items():
        job_id = str(uuid.uuid4())
        job = JobRecord(id=job_id, status="PENDING", message=f"Queued batch verification for {ticker}")
        db.add(job)
        db.commit()
        background_tasks.add_task(run_verification, job_id, ticker, quarters, request.model_tier)
        job_ids.append(job_id)
        
    return {"message": f"Verification triggered for {len(companies_map)} companies", "job_ids": job_ids}

@app.get("/api/results/{ticker}")
async def get_results(ticker: str, db: Session = Depends(get_db)):
    """Get all cached results (claims and verdicts) for a specific company."""
    ticker = ticker.upper()
    try:
        # Query claims and their verdicts
        claims = db.query(ClaimRecord).filter(ClaimRecord.ticker == ticker).all()
        verdicts = db.query(VerdictRecord).join(ClaimRecord).filter(ClaimRecord.ticker == ticker).all()
        
        # Format results
        return {
            "ticker": ticker,
            "total_claims": len(claims),
            "claims": claims,
            "verdicts": verdicts
        }
    except Exception as e:
        logger.error(f"Error retrieving results for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results for {ticker}.")

@app.get("/api/results/{ticker}/{year}/{quarter}")
async def get_quarter_results(ticker: str, year: int, quarter: int, db: Session = Depends(get_db)):
    """Get results for a specific company quarter."""
    ticker = ticker.upper()
    try:
        claims = db.query(ClaimRecord).filter(
            ClaimRecord.ticker == ticker,
            ClaimRecord.year == year,
            ClaimRecord.quarter == quarter
        ).all()
        
        verdicts = db.query(VerdictRecord).join(ClaimRecord).filter(
            ClaimRecord.ticker == ticker,
            ClaimRecord.year == year,
            ClaimRecord.quarter == quarter
        ).all()
        
        return {
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "total_claims": len(claims),
            "claims": claims,
            "verdicts": verdicts
        }
    except Exception as e:
        logger.error(f"Error retrieving quarter results for {ticker} {year}Q{quarter}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve quarter results.")

@app.get("/api/dashboard")
async def get_dashboard(db: Session = Depends(get_db)):
    """Get aggregate dashboard data for all companies with pre-computed results."""
    try:
        # Get all companies with data
        all_tickers = db.execute(select(distinct(ClaimRecord.ticker))).scalars().all()
        
        dashboard = {
            "companies": [],
            "totals": {"claims": 0, "verified": 0, "false": 0, "misleading": 0, "unverifiable": 0, "approx_true": 0},
            "target_companies": COMPANIES,
            "has_precomputed_data": len(all_tickers) > 0
        }
        
        for ticker in sorted(all_tickers):
            claims = db.query(ClaimRecord).filter(ClaimRecord.ticker == ticker).all()
            claim_ids = [c.id for c in claims]
            
            verdicts = []
            if claim_ids:
                verdicts = db.query(VerdictRecord).filter(VerdictRecord.claim_id.in_(claim_ids)).all()
            
            v_counts = {}
            for v in verdicts:
                v_counts[v.verdict] = v_counts.get(v.verdict, 0) + 1
            
            company_data = {
                "ticker": ticker,
                "total_claims": len(claims),
                "total_verdicts": len(verdicts),
                "verified": v_counts.get("VERIFIED", 0),
                "false": v_counts.get("FALSE", 0),
                "misleading": v_counts.get("MISLEADING", 0),
                "approx_true": v_counts.get("APPROXIMATELY_TRUE", 0),
                "unverifiable": v_counts.get("UNVERIFIABLE", 0),
                "quarters": sorted(list(set(f"{c.year}Q{c.quarter}" for c in claims)), reverse=True)
            }
            dashboard["companies"].append(company_data)
            
            dashboard["totals"]["claims"] += len(claims)
            dashboard["totals"]["verified"] += company_data["verified"]
            dashboard["totals"]["false"] += company_data["false"]
            dashboard["totals"]["misleading"] += company_data["misleading"]
            dashboard["totals"]["unverifiable"] += company_data["unverifiable"]
            dashboard["totals"]["approx_true"] += company_data["approx_true"]
        
        return dashboard
    except Exception as e:
        logger.error(f"Error building dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to build dashboard.")

@app.get("/api/all-results")
async def get_all_results(db: Session = Depends(get_db)):
    """Get results for ALL companies that have been processed."""
    try:
        tickers = db.execute(select(distinct(ClaimRecord.ticker))).scalars().all()
        results = []
        for ticker in sorted(tickers):
            claims = db.query(ClaimRecord).filter(ClaimRecord.ticker == ticker).all()
            claim_ids = [c.id for c in claims]
            verdicts = db.query(VerdictRecord).filter(VerdictRecord.claim_id.in_(claim_ids)).all() if claim_ids else []
            results.append({
                "ticker": ticker,
                "total_claims": len(claims),
                "claims": claims,
                "verdicts": verdicts
            })
        return {"results": results, "total_companies": len(results)}
    except Exception as e:
        logger.error(f"Error fetching all results: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch all results.")

@app.get("/api/transcripts/{ticker}/{year}/{quarter}")
async def get_transcript_raw(ticker: str, year: int, quarter: int, db: Session = Depends(get_db)):
    """Get raw transcript for a specific company quarter."""
    record = db.query(TranscriptRecord).filter(
        TranscriptRecord.ticker == ticker.upper(),
        TranscriptRecord.year == year,
        TranscriptRecord.quarter == quarter
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return record

@app.get("/api/financials/{ticker}/{year}/{quarter}")
async def get_financials_raw(ticker: str, year: int, quarter: int, db: Session = Depends(get_db)):
    """Get raw financial data for a specific company quarter."""
    records = db.query(FinancialData).filter(
        FinancialData.ticker == ticker.upper(),
        FinancialData.year == year,
        FinancialData.quarter == quarter
    ).all()
    if not records:
        raise HTTPException(status_code=404, detail="Financial data not found")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# RAG-powered Q&A endpoint — uses EXISTING stored data only (no re-ingestion)
# ──────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    ticker: str
    question: str

@app.post("/api/ask")
async def ask_question(request: AskRequest, db: Session = Depends(get_db)):
    """
    Answer a natural-language question about a company using ONLY the existing
    verified claims stored in the database. Uses lightweight keyword-based search
    on claims and verdicts. Does NOT use embeddings, pgvector, or RAG.
    """
    from src.config import MODEL_CONFIGS, ACTIVE_MODEL_TIER, OLLAMA_BASE_URL, OLLAMA_API_KEY, validate_ollama_config
    import litellm
    import re

    ticker = request.ticker.upper()
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Extract keywords from the question (simple tokenization)
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'is', 'was', 'were', 'are', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                      'could', 'may', 'might', 'must', 'can', 'about', 'in', 'on', 'at',
                      'to', 'for', 'of', 'with', 'by', 'from', 'up', 'down', 'out', 'off',
                      'over', 'under', 'again', 'further', 'then', 'once'}
        
        # Tokenize and filter keywords
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Query claims with keyword-based search
        # Join claims with verdicts to get verification reasoning and evidence
        query = db.query(ClaimRecord, VerdictRecord).join(
            VerdictRecord, ClaimRecord.id == VerdictRecord.claim_id, isouter=True
        ).filter(ClaimRecord.ticker == ticker)
        
        # If we have keywords, perform ILIKE search
        matched_claims = []
        if keywords:
            for keyword in keywords[:5]:  # Limit to top 5 keywords to avoid overly complex queries
                keyword_pattern = f"%{keyword}%"
                keyword_query = query.filter(
                    (ClaimRecord.raw_text.ilike(keyword_pattern)) |
                    (ClaimRecord.metric.ilike(keyword_pattern)) |
                    (VerdictRecord.explanation.ilike(keyword_pattern)) |
                    (func.cast(VerdictRecord.evidence, String).ilike(keyword_pattern))
                )
                matched_claims.extend(keyword_query.all())
        
        # Remove duplicates while preserving order
        seen_texts = set()
        unique_matches = []
        for claim, verdict in matched_claims:
            if claim.raw_text not in seen_texts:
                seen_texts.add(claim.raw_text)
                unique_matches.append((claim, verdict))
        
        # If no keyword matches, fallback to most recent claims
        if not unique_matches:
            logger.info(f"No keyword matches for {ticker}, falling back to recent claims")
            unique_matches = query.order_by(
                ClaimRecord.year.desc(),
                ClaimRecord.quarter.desc()
            ).limit(10).all()
        else:
            # Sort matches by most recent quarter first
            unique_matches.sort(key=lambda x: (x[0].year, x[0].quarter), reverse=True)
            unique_matches = unique_matches[:10]  # Limit to top 10
        
        if not unique_matches:
            return {
                "answer": f"No verified claims found for {ticker}. Please make sure the company has been analyzed.",
                "claim_texts": [],
            }
        
        # Build structured context from retrieved claims
        context_blocks = []
        
        for claim, verdict in unique_matches:
            
            # Build structured context block
            context_block = f"Claim: {claim.raw_text}\n"
            context_block += f"Metric: {claim.metric} = {claim.value} {claim.unit or ''}\n"
            context_block += f"Quarter: {claim.year} Q{claim.quarter}\n"
            context_block += f"Speaker: {claim.speaker}\n"
            
            if verdict:
                context_block += f"Verdict: {verdict.verdict}\n"
                context_block += f"Reasoning: {verdict.explanation}\n"
                
                if verdict.evidence:
                    evidence_list = verdict.evidence if isinstance(verdict.evidence, list) else []
                    if evidence_list:
                        context_block += f"Evidence: {'; '.join(evidence_list)}\n"
            else:
                context_block += "Verdict: NOT YET VERIFIED\n"
            
            context_blocks.append(context_block)
        
        context_str = "\n---\n\n".join(context_blocks)
        
        # Send to LLM with strict system instruction
        model = MODEL_CONFIGS.get(ACTIVE_MODEL_TIER, MODEL_CONFIGS["default"])
        
        system_message = (
            "You are a financial analysis assistant. Answer the user's question using ONLY "
            "the provided verified claims and evidence. If the answer cannot be derived from "
            "the provided context, say 'The available verified claims do not contain enough "
            "information to answer this question.' Do not fabricate data."
        )
        
        user_message = (
            f"Company: {ticker}\n\n"
            f"Question: {question}\n\n"
            f"--- VERIFIED CLAIMS ---\n{context_str}\n--- END VERIFIED CLAIMS ---\n\n"
            f"Please answer the question based solely on the verified claims above."
        )
        
        # Prepare request arguments
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
            "timeout": 300,  # 300s (5min) to match other Ollama operations text generation
        }

        # Inject Ollama configuration if needed
        if "ollama" in model:
            validate_ollama_config()
            kwargs["api_base"] = OLLAMA_BASE_URL
            kwargs["api_key"] = OLLAMA_API_KEY

        response = litellm.completion(**kwargs)
        
        # Access the content (works with both object and dict response types)
        if hasattr(response.choices[0], 'message'):
            answer = response.choices[0].message.content.strip()
        else:
            answer = response['choices'][0]['message']['content'].strip()
        
        return {
            "answer": answer,
            "claim_texts": [c.raw_text for c, _ in unique_matches],
            "num_claims_used": len(unique_matches)
        }
    
    except Exception as e:
        logger.error(f"Error answering question for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")
