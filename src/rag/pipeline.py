import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from src.models import Claim
from src.db.schema import FinancialData
from src.rag.retriever import hybrid_search
from src.rag.reranker import rerank

logger = logging.getLogger(__name__)

def retrieve_for_claim(claim: Claim, db_session: Session) -> List[Dict[str, Any]]:
    """
    Orchestrate retrieval for a claim.
    1. Deterministic lookup in financial_data table.
    2. RAG retrieval (Hybrid Search + Reranking).
    """
    logger.info(f"Retrieving context for claim: {claim.id} ({claim.metric})")
    
    results = []

    # STEP 0: Deterministic lookup
    # Try to find the exact metric in the financial_data table
    deterministic_records = db_session.query(FinancialData).filter(
        FinancialData.ticker == claim.ticker,
        FinancialData.year == claim.year,
        FinancialData.quarter == claim.quarter,
        # Metric name might need normalization or fuzzy matching in a real scenario
        # For now, exact match on metric name
        FinancialData.metric == claim.metric
    ).all()
    
    for rec in deterministic_records:
        formatted_val = f"${rec.value:,.0f}" if isinstance(rec.value, (int, float)) else str(rec.value)
        text = f"GOLD SOURCE | Company: {rec.ticker} | Period: Q{rec.quarter} {rec.year} | Source: {rec.source}\n{rec.metric}: {formatted_val} {rec.unit}"
        results.append({
            "text": text,
            "score": 2.0, # High score for deterministic source
            "metadata": {
                "source_type": rec.source,
                "is_gold": True
            }
        })

    # STEP 1: Formulate search queries
    # Base query
    base_query = f"{claim.metric} for {claim.ticker} in Q{claim.quarter} {claim.year}"
    queries = [base_query]
    
    # Handle YoY: query both periods if period suggests comparison
    if "year-over-year" in claim.raw_text.lower() or "yoy" in claim.raw_text.lower():
        prior_year_query = f"{claim.metric} for {claim.ticker} in Q{claim.quarter} {claim.year - 1}"
        queries.append(prior_year_query)
        logger.info("Detected YoY comparison, adding prior year query.")

    # STEP 2 & 3: Run hybrid_search and rerank for each query
    rag_candidates = []
    for q in queries:
        # Determine filters for this specific query
        target_year = claim.year if "year - 1" not in q else claim.year - 1
        
        search_results = hybrid_search(
            query=q,
            db_session=db_session,
            ticker=claim.ticker,
            year=target_year,
            quarter=claim.quarter,
            top_k=30
        )
        rag_candidates.extend(search_results)

    # Deduplicate by ID
    seen_ids = set()
    unique_candidates = []
    for c in rag_candidates:
        if c["id"] not in seen_ids:
            unique_candidates.append(c)
            seen_ids.add(c["id"])

    # Rerank
    if unique_candidates:
        reranked_results = rerank(claim.raw_text, unique_candidates, top_k=10)
        results.extend(reranked_results)

    # STEP 4: Return top results with gold source priority
    # (Gold sources are already at the top because we added them first)
    return results[:10]

def build_verification_context(claim: Claim, retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Format retrieved documents into a clean context string for the LLM.
    """
    if not retrieved_docs:
        return "No relevant context found."

    context_blocks = []
    for i, doc in enumerate(retrieved_docs):
        marker = "[GOLD SOURCE]" if doc.get("metadata", {}).get("is_gold") else f"[Source {i+1}]"
        block = f"{marker} {doc['text']}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)
