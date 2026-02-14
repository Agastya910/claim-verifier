import logging
from src.db.connection import get_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_duplicate_claims():
    """Check if there are any duplicate claims in the database"""
    db = next(get_db())
    
    try:
        # Check for duplicate IDs
        duplicate_ids = db.execute("""
            SELECT id, COUNT(*) as count
            FROM claims
            GROUP BY id
            HAVING COUNT(*) > 1
        """).fetchall()
        
        if duplicate_ids:
            logger.warning(f"Found {len(duplicate_ids)} duplicate claim IDs:")
            for claim_id, count in duplicate_ids:
                logger.warning(f"  Claim ID: {claim_id}, Count: {count}")
        else:
            logger.info("✅ No duplicate claim IDs found")
        
        # Check for duplicate content
        duplicate_content = db.execute("""
            SELECT raw_text, COUNT(*) as count
            FROM claims
            GROUP BY raw_text
            HAVING COUNT(*) > 1
        """).fetchall()
        
        if duplicate_content:
            logger.warning(f"Found {len(duplicate_content)} duplicate claim texts:")
            for text, count in duplicate_content:
                logger.warning(f"  Text: '{text[:100]}...', Count: {count}")
        else:
            logger.info("✅ No duplicate claim content found")
            
    finally:
        db.close()

if __name__ == "__main__":
    check_duplicate_claims()
