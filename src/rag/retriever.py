import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from src.rag.indexer import sparse_model, dense_model
from pgvector.sqlalchemy import SparseVector

logger = logging.getLogger(__name__)

def hybrid_search(
    query: str, 
    db_session: Session, 
    ticker: str = None, 
    year: int = None, 
    quarter: int = None, 
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Execute hybrid search using SQL with RRF fusion.
    Combines dense (BGE) and sparse (SPLADE) embeddings.
    """
    # 1. Generate embeddings
    logger.info(f"Generating query embeddings for: {query}")
    dense_vec = list(dense_model.embed([query]))[0].tolist()
    sparse_emb = list(sparse_model.embed([query]))[0]
    sparse_dict = dict(zip(sparse_emb.indices.tolist(), sparse_emb.values.tolist()))
    sparse_vec = SparseVector(sparse_dict, 30522)

    # 2. Build SQL query with RRF
    sql = text("""
        WITH dense_results AS (
            SELECT id, text, ticker, year, quarter, chunk_type, metric_type, source_type,
                   ROW_NUMBER() OVER (ORDER BY dense_embedding <=> CAST(:query_dense_vec AS vector)) as dense_rank
            FROM document_chunks
            WHERE (CAST(:ticker AS VARCHAR) IS NULL OR ticker = CAST(:ticker AS VARCHAR))
              AND (CAST(:year AS INTEGER) IS NULL OR year = CAST(:year AS INTEGER))
              AND (CAST(:quarter AS INTEGER) IS NULL OR quarter = CAST(:quarter AS INTEGER))
            ORDER BY dense_embedding <=> CAST(:query_dense_vec AS vector)
            LIMIT :top_k
        ),
        sparse_results AS (
            SELECT id, text, ticker, year, quarter, chunk_type, metric_type, source_type,
                   ROW_NUMBER() OVER (ORDER BY sparse_embedding <=> CAST(:query_sparse_vec AS sparsevec)) as sparse_rank
            FROM document_chunks
            WHERE (CAST(:ticker AS VARCHAR) IS NULL OR ticker = CAST(:ticker AS VARCHAR))
              AND (CAST(:year AS INTEGER) IS NULL OR year = CAST(:year AS INTEGER))
              AND (CAST(:quarter AS INTEGER) IS NULL OR quarter = CAST(:quarter AS INTEGER))
            ORDER BY sparse_embedding <=> CAST(:query_sparse_vec AS sparsevec)
            LIMIT :top_k
        )
        SELECT COALESCE(d.id, s.id) as id,
               COALESCE(d.text, s.text) as text,
               COALESCE(d.ticker, s.ticker) as ticker,
               COALESCE(d.year, s.year) as year,
               COALESCE(d.quarter, s.quarter) as quarter,
               COALESCE(d.chunk_type, s.chunk_type) as chunk_type,
               COALESCE(d.metric_type, s.metric_type) as metric_type,
               COALESCE(d.source_type, s.source_type) as source_type,
               (1.0 / (60 + COALESCE(d.dense_rank, :top_k + 1))) +
               (1.0 / (60 + COALESCE(s.sparse_rank, :top_k + 1))) as rrf_score
        FROM dense_results d
        FULL OUTER JOIN sparse_results s ON d.id = s.id
        ORDER BY rrf_score DESC
        LIMIT :top_k;
    """)

    params = {
        "query_dense_vec": dense_vec,
        "query_sparse_vec": sparse_vec.to_text(),
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "top_k": top_k
    }

    try:
        results = db_session.execute(sql, params).mappings().all()
        
        formatted_results = []
        for row in results:
            formatted_results.append({
                "id": row["id"],
                "text": row["text"],
                "score": float(row["rrf_score"]),
                "metadata": {
                    "ticker": row["ticker"],
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "chunk_type": row["chunk_type"],
                    "metric_type": row["metric_type"],
                    "source_type": row["source_type"]
                }
            })
            
        logger.info(f"Hybrid search returned {len(formatted_results)} results.")
        return formatted_results

    except Exception as e:
        logger.error(f"Error executing hybrid search: {e}")
        db_session.rollback()
        return []
