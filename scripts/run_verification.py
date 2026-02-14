import logging
import json
import os
import argparse
from typing import List
from src.db.connection import SessionLocal
from src.db.schema import TranscriptRecord, VerdictRecord, ClaimRecord
from src.verifier.pipeline import verify_company
from sqlalchemy import select, text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_verification")

def generate_report(results: List[dict], output_path: str = "verification_summary.json"):
    """Save verification results to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Verification report saved to {output_path}")
    
    # Print a text summary
    print("\n--- Verification Summary Report ---")
    print(f"{'Ticker':<8} | {'Total':<6} | {'Verified':<8} | {'False':<6} | {'Misleading':<10}")
    print("-" * 50)
    for res in results:
        stats = res.get("summary_stats", {})
        print(f"{res['ticker']:<8} | {stats.get('total_claims', 0):<6} | "
              f"{stats.get('VERIFIED', 0):<8} | {stats.get('FALSE', 0):<6} | "
              f"{stats.get('MISLEADING', 0):<10}")
    print("-" * 50)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run claim verification pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume verification using existing results (don't clear verdicts or force rerun)")
    args = parser.parse_args()
    
    db = SessionLocal()
    try:
        # Get all ticker/year/quarter combos that have transcripts
        # (This implies they've been ingested and are ready for verification)
        stmt = select(TranscriptRecord.ticker, TranscriptRecord.year, TranscriptRecord.quarter)
        rows = db.execute(stmt).all()
        
        if not rows:
            logger.warning("No ingested data found in the database. Run ingest_all.py first.")
            return
            
        # Group by ticker
        company_data = {}
        for ticker, year, quarter in rows:
            if ticker not in company_data:
                company_data[ticker] = []
            company_data[ticker].append((year, quarter))
            
        all_results = []
        logger.info(f"Starting verification for {len(company_data)} companies...")
        
        for ticker, quarters in company_data.items():
            logger.info(f"Verifying {ticker} for {len(quarters)} quarters...")
            # Create a fresh session for each company to ensure isolation
            company_db = SessionLocal()
            
            # Clear existing verdicts only if not in resume mode
            if not args.resume:
                from src.config import ALLOW_DESTRUCTIVE_OPERATIONS
                
                if not ALLOW_DESTRUCTIVE_OPERATIONS:
                    raise RuntimeError(
                        f"DESTRUCTIVE OPERATION BLOCKED: Cannot delete verdicts for {ticker}. "
                        f"The verification data has been computed and stored. "
                        f"Use --resume flag to skip deletion and resume from existing verdicts, "
                        f"OR set ALLOW_DESTRUCTIVE_OPERATIONS=true in .env if you are in development "
                        f"and explicitly need to re-run verification from scratch."
                    )
                
                try:
                    company_db.execute(
                        text("DELETE FROM verdicts WHERE claim_id IN (SELECT id FROM claims WHERE ticker = :ticker)"),
                        {"ticker": ticker}
                    )
                    company_db.commit()
                    logger.info(f"Cleared existing verdicts for {ticker}")
                except Exception as e:
                    logger.warning(f"Failed to clear verdicts for {ticker}: {e}")
                    company_db.rollback()
                
            try:
                # Run full verification pipeline for this company
                result = verify_company(ticker, quarters, company_db, model_tier="default", force_rerun=not args.resume)
                
                # We need to serialize the result for the report
                # result is likely a CompanyVerificationResult object
                summary = {
                    "ticker": ticker,
                    "summary_stats": result.summary_stats,
                    "quarters_processed": quarters
                }
                all_results.append(summary)
                logger.info(f"Completed verification for {ticker}")
                
            except Exception as e:
                logger.error(f"Error verifying {ticker}: {e}")
                company_db.rollback()
            finally:
                company_db.close()
                
        if all_results:
            generate_report(all_results)
        else:
            logger.warning("No verification results were generated.")
            
    finally:
        db.close()

if __name__ == "__main__":
    main()
