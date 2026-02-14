import logging
import sys
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import SparseVector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.getcwd())

# Mock DB connection string - adjust if needed, but assuming standard from project
# The user's error log implies they have a working DB, just the query fails.
# We will use the same logic as the retriever to reproduce.

def reproduce_error():
    # 1. Create a dummy SparseVector
    # Represents a vector with 30522 dimensions, having value 1.0 at index 100
    sparse_vec = SparseVector({100: 1.0}, 30522)
    
    # Use project's connection logic
    from src.config import DATABASE_URL
    # Ensure usage of psycopg 3 if not already in the URL
    if "postgresql+psycopg" not in DATABASE_URL and "postgresql" in DATABASE_URL:
         # If it's just postgresql://, sqlalchemy might default to psycopg2 which we don't have.
         # So we force it.
         db_url = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://")
    else:
         db_url = DATABASE_URL

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # 3. Test correct formatting for SparseVector
    # Check attributes
    print(f"SparseVector attributes: {dir(sparse_vec)}")
    # Try formatting
    # Assuming attributes indices and values exist
    try: 
        # Check if to_text method exists and works
        print("Testing to_text()...")
        if hasattr(sparse_vec, "to_text"):
             formatted_sparse = sparse_vec.to_text()
             print(f"to_text() result: {formatted_sparse}")
             
             sql_sparse = text("SELECT CAST(:vec AS sparsevec)")
             session.execute(sql_sparse, {"vec": formatted_sparse})
             print("SUCCESS: Sparse query executed with to_text()")

        # Test empty sparse vector
        print("\nTesting empty SparseVector...")
        empty_sparse = SparseVector({}, 30522)
        if hasattr(empty_sparse, "to_text"):
             formatted_empty = empty_sparse.to_text()
             print(f"Empty to_text() result: {formatted_empty}")
             session.execute(sql_sparse, {"vec": formatted_empty})
             print("SUCCESS: Empty sparse query executed")

        else:
             print("to_text method not found.")
    except Exception as e:
        print("\nCAUGHT ERROR (Sparse to_text):")
        print(e)



    print("\nAttempting to execute query with Dense Vector parameter...")
    # 4. Test Dense Vector (to repro invalid operator error)
    # Dense vector with 3 dimensions (should match column def, but for repro just checking type)
    # The error was "operator does not exist: vector <=> double precision[]"
    # FIX: Cast parameter to vector in SQL using CAST to avoid syntax issues with ::
    sql_dense = text("SELECT '[0.1,0.2,0.3]'::vector <=> CAST(:vec AS vector)")
    try:
        session.execute(sql_dense, {"vec": [0.1, 0.2, 0.3]})
        print("SUCCESS: Dense query executed without error (Fix verified)")
    except Exception as e:
        print("\nCAUGHT ERROR (Dense Vector Fix Failed):")
        print(e)

if __name__ == "__main__":
    reproduce_error()
