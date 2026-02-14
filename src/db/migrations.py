from sqlalchemy import text
from src.db.connection import engine
from src.db.schema import Base

def init_db():
    """Initializes the database by creating all tables and enabling pgvector."""
    with engine.begin() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # Create all tables
        Base.metadata.create_all(bind=conn)
        
        # Create Vector Indexes (HNSW for dense, GIN for sparse)
        # Note: These usually require specific syntax for pgvector
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_dense_hnsw 
            ON document_chunks USING hnsw (dense_embedding vector_cosine_ops);
        """))
        
        # pgvector-python handles sparsevector as well
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_sparse_hnsw 
            ON document_chunks USING hnsw (sparse_embedding sparsevec_l2_ops);
        """))
        
        # Composite B-tree indexes for metadata filtering
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_metadata 
            ON document_chunks (ticker, year, quarter);
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_financial_data_metadata 
            ON financial_data (ticker, year, quarter);
        """))

if __name__ == "__main__":
    init_db()
