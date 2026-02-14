import logging
from sqlalchemy import text
from src.db.connection import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")

def migrate():
    db = SessionLocal()
    try:
        logger.info("checking if 'evidence' column exists in 'verdicts' table...")
        # Check if column exists
        result = db.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name='verdicts' AND column_name='evidence'"
        )).fetchone()
        
        if not result:
            logger.info("Adding 'evidence' column to 'verdicts' table...")
            db.execute(text("ALTER TABLE verdicts ADD COLUMN evidence JSON"))
            db.commit()
            logger.info("Migration successful.")
        else:
            logger.info("'evidence' column already exists.")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    migrate()
