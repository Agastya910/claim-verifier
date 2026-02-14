"""
Full pipeline script: Ingest -> Extract -> Verify for all 10 target companies.
This runs everything end-to-end and stores results in the database.

Usage: .venv/Scripts/python.exe -m scripts.run_full_pipeline
"""
import logging
import time
from src.db.connection import SessionLocal
from src.db.migrations import init_db
from src.config import COMPANIES, QUARTERS_TUPLES
from src.verifier.pipeline import verify_company
from src.db.schema import ClaimRecord, VerdictRecord

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("full_pipeline")


def run_company(ticker: str, quarters: list, model_tier: str = "default"):
    """Run the full pipeline for a single company."""
    db = SessionLocal()
    try:
        logger.info(f"Running full pipeline for {ticker}...")
        result = verify_company(ticker, quarters, db, model_tier)
        
        logger.info(f"  {ticker} Results:")
        logger.info(f"    Total claims: {result.summary_stats.get('total_claims', 0)}")
        logger.info(f"    Verified: {result.summary_stats.get('verified_count', 0)}")
        logger.info(f"    Approx True: {result.summary_stats.get('approx_true_count', 0)}")
        logger.info(f"    False: {result.summary_stats.get('false_count', 0)}")
        logger.info(f"    Misleading: {result.summary_stats.get('misleading_count', 0)}")
        logger.info(f"    Unverifiable: {result.summary_stats.get('unverifiable_count', 0)}")
        
        return result
    except Exception as e:
        logger.error(f"Failed pipeline for {ticker}: {e}")
        return None
    finally:
        db.close()


def print_summary(results):
    """Print a summary table of all results."""
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"{'Company':<10} {'Claims':<10} {'Verified':<10} {'Approx':<10} {'False':<10} {'Misleading':<12} {'Unverify':<10}")
    print("-" * 80)
    
    totals = {"claims": 0, "verified": 0, "approx": 0, "false": 0, "misleading": 0, "unverifiable": 0}
    
    for r in results:
        if r is None:
            continue
        s = r.summary_stats
        claims = s.get("total_claims", 0)
        verified = s.get("verified_count", 0)
        approx = s.get("approx_true_count", 0)
        false_c = s.get("false_count", 0)
        misleading = s.get("misleading_count", 0)
        unverify = s.get("unverifiable_count", 0)
        
        totals["claims"] += claims
        totals["verified"] += verified
        totals["approx"] += approx
        totals["false"] += false_c
        totals["misleading"] += misleading
        totals["unverifiable"] += unverify
        
        print(f"{r.company:<10} {claims:<10} {verified:<10} {approx:<10} {false_c:<10} {misleading:<12} {unverify:<10}")
    
    print("-" * 80)
    print(f"{'TOTAL':<10} {totals['claims']:<10} {totals['verified']:<10} {totals['approx']:<10} {totals['false']:<10} {totals['misleading']:<12} {totals['unverifiable']:<10}")
    print("=" * 80)
    
    if totals["claims"] > 0:
        accuracy = (totals["verified"] + totals["approx"]) / totals["claims"] * 100
        print(f"\nOverall Accuracy Score: {accuracy:.1f}%")


def main():
    logger.info("Initializing database...")
    init_db()
    
    logger.info(f"Target companies: {COMPANIES}")
    logger.info(f"Target quarters: {QUARTERS_TUPLES}")
    
    start_time = time.time()
    results = []
    
    for i, ticker in enumerate(COMPANIES):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(COMPANIES)}] {ticker}")
        logger.info(f"{'='*60}")
        
        result = run_company(ticker, QUARTERS_TUPLES)
        results.append(result)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print_summary(results)
    logger.info(f"\nTotal pipeline time: {duration/60:.2f} minutes")


if __name__ == "__main__":
    main()
