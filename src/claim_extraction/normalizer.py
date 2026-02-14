import re
import uuid
from typing import List
from src.models import Claim

METRIC_ALIASES = {
    "top line": "revenue",
    "bottom line": "net_income",
    "earnings per share": "eps",
    "profit": "net_income",
    "sales": "revenue",
}

GAAP_KEYWORDS = ["adjusted", "non-gaap", "excluding", "pro forma", "ex-items", "core"]
HEDGING_KEYWORDS = ["approximately", "roughly", "about", "around", "nearly", "~"]

def normalize_claims(claims: List[Claim]) -> List[Claim]:
    """
    Deduplicates, normalizes metric names and periods, and detects GAAP/hedging status.
    """
    normalized_claims = []
    seen_claims = set()

    for claim in claims:
        # 1. Normalize Metric Name
        metric_lower = claim.metric.lower()
        for alias, canonical in METRIC_ALIASES.items():
            if alias in metric_lower:
                claim.metric = canonical
                break

        # 2. Normalize Period
        period_lower = claim.period.lower()
        if any(x in period_lower for x in ["year over year", "year-over-year", "yoy"]):
            claim.period = "YoY"
        elif any(x in period_lower for x in ["sequentially", "quarter over quarter", "qoq"]):
            claim.period = "QoQ"
        elif any(x in period_lower for x in ["full year", "annual"]):
            claim.period = "annual"

        # 3. Detect GAAP/Non-GAAP
        raw_text_lower = claim.raw_text.lower()
        if any(kw in raw_text_lower for kw in GAAP_KEYWORDS):
            claim.is_gaap = False
        else:
            claim.is_gaap = True

        # 4. Detect Hedging
        if any(kw in raw_text_lower for kw in HEDGING_KEYWORDS):
            claim.hedging_language = "true" # Storing as string "true" as per model field if it's Text, or fix model
        else:
            claim.hedging_language = "false"

        # 5. Filter Forward-looking (optional, but requested to note)
        if claim.is_forward_looking:
            # We keep them but they might be skipped by verification engine later
            pass

        # 6. Deduplicate: same (ticker, metric, period, value)
        # Using rounded value for comparison to avoid float precision issues
        dedup_key = (claim.ticker, claim.metric, claim.period, round(claim.value, 4))
        
        if dedup_key not in seen_claims:
            seen_claims.add(dedup_key)
            normalized_claims.append(claim)
        else:
            # If already seen, could potentially update if this one has higher confidence
            # but for now, simple first-one-wins or keep highest confidence
            for existing in normalized_claims:
                if (existing.ticker, existing.metric, existing.period, round(existing.value, 4)) == dedup_key:
                    if claim.confidence > existing.confidence:
                        normalized_claims.remove(existing)
                        normalized_claims.append(claim)
                    break

    return normalized_claims

def enrich_context(claim: Claim, full_transcript_text: str) -> Claim:
    """
    Extracts 2 sentences before and after the raw_text in the full transcript.
    """
    if not claim.raw_text or not full_transcript_text:
        return claim

    # Simple approach: find the raw_text in the full_transcript_text
    # and take a window around it.
    # For more robustness, we'd use sentence splitting.
    
    # Split transcript into sentences roughly
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', full_transcript_text)
    
    try:
        # Find the index of the sentence containing raw_text
        target_idx = -1
        for i, sent in enumerate(sentences):
            if claim.raw_text.strip() in sent.strip() or sent.strip() in claim.raw_text.strip():
                target_idx = i
                break
        
        if target_idx != -1:
            start = max(0, target_idx - 2)
            end = min(len(sentences), target_idx + 3)
            context_sentences = sentences[start:end]
            claim.context = " ".join(context_sentences)
    except Exception:
        # Fallback to current context if something fails
        pass
        
    return claim
