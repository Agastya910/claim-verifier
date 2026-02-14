import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Initialize CrossEncoder model
# This will download the model on first call
try:
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2", max_length=512)
except Exception as e:
    logger.error(f"Failed to load CrossEncoder model: {e}")
    reranker_model = None

def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Score all (query, candidate_text) pairs and return top_k.
    """
    if not candidates or reranker_model is None:
        return candidates[:top_k]

    logger.info(f"Reranking {len(candidates)} candidates for query: {query}")
    
    # Prepare pairs for scoring
    pairs = [[query, c["text"]] for c in candidates]
    
    # Get scores
    scores = reranker_model.predict(pairs)
    
    # Combine scores with candidates
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])
        
    # Sort descending by rerank_score
    ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    
    logger.info(f"Reranking complete. Top score: {ranked_candidates[0]['rerank_score'] if ranked_candidates else 'N/A'}")
    
    return ranked_candidates[:top_k]
