"""
Smart Retrieval Engine for Claim Q&A.

Lightweight, zero-new-dependency retrieval that replaces naive keyword ILIKE
search with intent detection, query decomposition, multi-signal scoring, and
adaptive result sizing.

Works on Streamlit Cloud — uses only stdlib + SQLAlchemy.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from sqlalchemy import func, String
from sqlalchemy.orm import Session

from src.db.schema import ClaimRecord, VerdictRecord

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    claims: List[Tuple[Any, Any]]  # (ClaimRecord, VerdictRecord | None)
    intent: str
    filters_applied: Dict[str, Any]
    system_prompt_hint: str


# ─────────────────────────────────────────────────────────────────────────────
# Intent patterns — executed in order; first match wins
# ─────────────────────────────────────────────────────────────────────────────

VERDICT_PATTERNS: Dict[str, List[str]] = {
    "FALSE": [
        r"\bfalse\b", r"\blie[sd]?\b", r"\blying\b", r"\binaccurate\b",
        r"\bwrong\b", r"\buntrue\b", r"\bfabricated?\b", r"\bfake\b",
    ],
    "VERIFIED": [
        r"\bverified\b", r"\btrue\b", r"\baccurate\b", r"\bcorrect\b",
        r"\bconfirmed?\b", r"\bhonest\b",
    ],
    "MISLEADING": [
        r"\bmisleading\b", r"\bdeceptive\b", r"\bexaggerat",
    ],
    "APPROXIMATELY_TRUE": [
        r"\bapprox", r"\bclose\b", r"\bnearly\b", r"\broughly\b",
    ],
    "UNVERIFIABLE": [
        r"\bunverif", r"\bcannot verify\b", r"\bno data\b",
    ],
}

METRIC_SYNONYMS: Dict[str, List[str]] = {
    "revenue":           [r"revenue", r"sales", r"top[\s\-]?line", r"net[\s\-]?revenue"],
    "eps":               [r"\beps\b", r"earnings[\s\-]?per[\s\-]?share"],
    "gross_margin":      [r"gross[\s\-]?margin", r"gross[\s\-]?profit"],
    "operating_income":  [r"operating[\s\-]?(income|profit|earnings)", r"\bebit\b"],
    "net_income":        [r"net[\s\-]?(income|profit|earnings)", r"bottom[\s\-]?line"],
    "free_cash_flow":    [r"free[\s\-]?cash[\s\-]?flow", r"\bfcf\b"],
    "operating_margin":  [r"operating[\s\-]?margin"],
    "cash":              [r"\bcash\b", r"cash[\s\-]?position"],
    "debt":              [r"\bdebt\b", r"leverage"],
    "capex":             [r"\bcapex\b", r"capital[\s\-]?expenditure"],
    "r_and_d":           [r"\br&d\b", r"\br\s*and\s*d\b", r"research"],
    "dividend":          [r"dividend", r"payout"],
    "buyback":           [r"buyback", r"share[\s\-]?repurchase"],
    "headcount":         [r"headcount", r"employees", r"workforce", r"hiring"],
    "cloud":             [r"\bcloud\b", r"\bazure\b", r"\baws\b", r"\bgcp\b"],
    "ai":                [r"\bai\b", r"\bartificial[\s\-]?intelligence\b", r"\bmachine[\s\-]?learning\b"],
    "guidance":          [r"guidance", r"outlook", r"forecast", r"expect"],
    "growth":            [r"growth", r"grew", r"increase[sd]?", r"gain"],
    "margin":            [r"\bmargin\b"],
    "services":          [r"services"],
    "segment":           [r"segment"],
    "subscribers":       [r"subscriber", r"user base", r"dau", r"mau"],
}

SPEAKER_PATTERNS: Dict[str, List[str]] = {
    "CEO":  [r"\bceo\b", r"\bchief executive\b"],
    "CFO":  [r"\bcfo\b", r"\bchief financial\b", r"\bfinance chief\b"],
    "COO":  [r"\bcoo\b", r"\bchief operating\b"],
    "CTO":  [r"\bcto\b", r"\bchief technology\b"],
}

QUARTER_PATTERN = re.compile(
    r"(?:q(\d)[\s,]*(\d{4}))|(?:(\d{4})[\s,]*q(\d))",
    re.IGNORECASE,
)

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "could", "may", "might", "must", "can", "about", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "up", "down", "out", "off",
    "over", "under", "again", "further", "then", "once", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "or", "and",
    "but", "not", "no", "so", "if", "its", "it", "they", "them", "their",
    "there", "here", "how", "why", "when", "where", "any", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "than",
    "too", "very", "just", "because", "as", "until", "while", "during",
    "before", "after", "above", "below", "between", "same", "own", "into",
    "through", "only", "also", "tell", "give", "show", "me", "please",
    "company", "companies", "claim", "claims", "say", "said", "did",
    "many", "much",
})


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_verdict_intent(question: str) -> Optional[str]:
    """Return the verdict type the user is asking about, or None."""
    q = question.lower()
    for verdict_type, patterns in VERDICT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                return verdict_type
    return None


def _detect_metrics(question: str) -> List[str]:
    """Return list of canonical metric names mentioned in the question."""
    q = question.lower()
    found = []
    for canonical, patterns in METRIC_SYNONYMS.items():
        for pat in patterns:
            if re.search(pat, q):
                found.append(canonical)
                break
    return found


def _detect_quarters(question: str) -> List[Tuple[int, int]]:
    """Return list of (year, quarter) tuples mentioned in the question."""
    quarters = []
    for m in QUARTER_PATTERN.finditer(question):
        if m.group(1) and m.group(2):  # Q4 2024 format
            quarters.append((int(m.group(2)), int(m.group(1))))
        elif m.group(3) and m.group(4):  # 2024 Q4 format
            quarters.append((int(m.group(3)), int(m.group(4))))
    return quarters


def _detect_speaker(question: str) -> Optional[str]:
    """Return role keyword if user is asking about a specific speaker."""
    q = question.lower()
    for role, patterns in SPEAKER_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                return role
    return None


def _is_comparison(question: str) -> bool:
    """Detect if the user wants to compare across periods."""
    q = question.lower()
    return bool(re.search(r"\bcompar|vs\.?|versus|change|differ|trend\b", q))


def _classify_intent(
    verdict: Optional[str],
    metrics: List[str],
    quarters: List[Tuple[int, int]],
    speaker: Optional[str],
    is_comparison: bool,
) -> str:
    if verdict:
        return "VERDICT_FILTER"
    if is_comparison and len(quarters) >= 2:
        return "COMPARISON"
    if speaker:
        return "SPEAKER_FILTER"
    if metrics:
        return "METRIC_LOOKUP"
    return "GENERAL"


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _extract_keywords(question: str) -> List[str]:
    words = re.findall(r"\b\w+\b", question.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def _keyword_score(keywords: List[str], text: str) -> float:
    """Fraction of keywords that appear in the text (case-insensitive)."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits / len(keywords)


def _metric_match_score(
    claim_metric: str,
    detected_metrics: List[str],
) -> float:
    """1.0 if the claim metric matches any detected metric synonym."""
    if not detected_metrics or not claim_metric:
        return 0.0
    cm = claim_metric.lower()
    for canonical in detected_metrics:
        # Check if the canonical name or any of its synonyms match
        patterns = METRIC_SYNONYMS.get(canonical, [canonical])
        for pat in patterns:
            if re.search(pat, cm):
                return 1.0
    return 0.0


def _score_claim(
    claim,
    verdict,
    keywords: List[str],
    detected_metrics: List[str],
    target_verdict: Optional[str],
    target_quarters: List[Tuple[int, int]],
    max_year: int,
    max_quarter: int,
) -> float:
    """Composite relevance score for a (claim, verdict) pair."""

    # 1. Keyword density in raw_text + explanation
    searchable = claim.raw_text or ""
    if claim.metric:
        searchable += " " + claim.metric
    if verdict and verdict.explanation:
        searchable += " " + verdict.explanation
    kw_score = _keyword_score(keywords, searchable)

    # 2. Metric match
    m_score = _metric_match_score(claim.metric or "", detected_metrics)

    # 3. Verdict match
    v_score = 0.0
    if target_verdict and verdict:
        v_score = 1.0 if verdict.verdict == target_verdict else 0.0
    elif target_verdict and not verdict:
        v_score = 0.0

    # 4. Quarter match bonus (if user specified quarters)
    q_score = 0.0
    if target_quarters:
        if (claim.year, claim.quarter) in target_quarters:
            q_score = 1.0
    else:
        # Recency scoring when no specific quarter requested
        # Scale 0-1 based on how close to the most recent quarter
        year_diff = max_year - (claim.year or max_year)
        q_diff = max_quarter - (claim.quarter or max_quarter) + year_diff * 4
        q_score = max(0.0, 1.0 - q_diff * 0.15)

    # 5. Evidence quality
    eq = 0.0
    if verdict:
        eq += 0.5
        if verdict.explanation:
            eq += 0.2
        if verdict.evidence and isinstance(verdict.evidence, list) and verdict.evidence:
            eq += 0.3

    # Weighted composite
    score = (
        0.30 * kw_score
        + 0.25 * m_score
        + 0.20 * v_score
        + 0.10 * q_score
        + 0.15 * eq
    )
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Prompt hints per intent
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(intent: str, filters: Dict[str, Any]) -> str:
    base = (
        "You are a financial analysis assistant. Answer the user's question "
        "using ONLY the provided verified claims and evidence below. "
        "If the answer cannot be derived from the provided context, say so clearly. "
        "Do not fabricate data. Be specific: cite exact numbers, quarters, and verdicts.\n\n"
    )

    if intent == "VERDICT_FILTER":
        vt = filters.get("verdict_type", "flagged")
        base += (
            f"IMPORTANT: The user is asking specifically about {vt} claims. "
            f"List each {vt} claim, its metric, the claimed vs actual value, "
            f"and the verification reasoning. If there are no {vt} claims in the "
            f"context, say so explicitly."
        )
    elif intent == "METRIC_LOOKUP":
        metrics = filters.get("detected_metrics", [])
        base += (
            f"The user is asking about specific financial metric(s): {', '.join(metrics)}. "
            f"Provide the exact values, which quarter they are from, who stated them, "
            f"and their verification status (verified/false/etc)."
        )
    elif intent == "COMPARISON":
        base += (
            "The user wants to compare data across time periods. "
            "Organize your answer by quarter. Highlight changes and trends. "
            "Use exact numbers from the claims."
        )
    elif intent == "SPEAKER_FILTER":
        speaker = filters.get("speaker_role", "")
        base += (
            f"The user is asking about what the {speaker} said. "
            f"Focus on claims attributed to that speaker and their verification status."
        )
    else:
        base += (
            "Provide a comprehensive answer by synthesizing all the relevant "
            "verified claims. Group related claims together and note their "
            "verification status."
        )

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Main retrieval entry point
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_claims(
    db: Session,
    ticker: str,
    question: str,
) -> RetrievalResult:
    """
    Smart retrieval: detect intent, decompose query, score & rank claims.

    Returns a RetrievalResult with the best matching claims, detected intent,
    applied filters, and an intent-specific system prompt hint.
    """
    ticker = ticker.upper()

    # 1. Decompose the question
    target_verdict = _detect_verdict_intent(question)
    detected_metrics = _detect_metrics(question)
    target_quarters = _detect_quarters(question)
    target_speaker = _detect_speaker(question)
    comparison = _is_comparison(question)
    keywords = _extract_keywords(question)

    intent = _classify_intent(
        target_verdict, detected_metrics, target_quarters,
        target_speaker, comparison,
    )

    filters = {
        "intent": intent,
        "verdict_type": target_verdict,
        "detected_metrics": detected_metrics,
        "target_quarters": target_quarters,
        "speaker_role": target_speaker,
        "keywords": keywords,
    }

    logger.info(f"Smart retrieval for {ticker}: intent={intent}, filters={filters}")

    # 2. Build the base query (all claims for this ticker with their verdicts)
    base_query = db.query(ClaimRecord, VerdictRecord).join(
        VerdictRecord, ClaimRecord.id == VerdictRecord.claim_id, isouter=True
    ).filter(ClaimRecord.ticker == ticker)

    # 3. Pre-filter at the SQL level for strong intents to reduce scoring work
    if target_verdict:
        # For verdict filtering, we need claims that HAVE a verdict of the target type.
        # But we'll also retrieve some non-matching ones for context.
        # Strategy: get ALL matching verdict claims + a sample of others.
        verdict_query = base_query.filter(VerdictRecord.verdict == target_verdict)
        verdict_matches = verdict_query.all()

        # Also get a small set of non-matching claims for context diversity
        other_query = base_query.filter(
            (VerdictRecord.verdict != target_verdict) | (VerdictRecord.verdict.is_(None))
        ).limit(20)
        other_matches = other_query.all()

        all_candidates = verdict_matches + other_matches
    elif target_quarters:
        # Pre-filter to requested quarters + surrounding
        q_filters = []
        for year, quarter in target_quarters:
            q_filters.append(
                (ClaimRecord.year == year) & (ClaimRecord.quarter == quarter)
            )
        from sqlalchemy import or_
        quarter_query = base_query.filter(or_(*q_filters))
        quarter_matches = quarter_query.all()

        # Also get some from other quarters for context
        other_query = base_query.limit(30)
        other_matches = other_query.all()

        # Merge, dedup later
        all_candidates = quarter_matches + other_matches
    elif detected_metrics and keywords:
        # Use ILIKE for metric-specific queries — but smarter than before
        # Search metric column specifically (much more targeted than raw_text)
        metric_matches = []
        for canonical in detected_metrics:
            patterns = METRIC_SYNONYMS.get(canonical, [canonical])
            for pat in patterns:
                # Convert regex pattern to SQL ILIKE pattern
                sql_pattern = "%" + pat.replace(r"\b", "").replace(r"[\s\-]?", "%").replace(r"[\s\-]", "%") + "%"
                q = base_query.filter(ClaimRecord.metric.ilike(sql_pattern))
                metric_matches.extend(q.all())

        # Also do keyword search on raw_text for remaining keywords
        kw_matches = []
        for kw in keywords[:3]:
            kw_q = base_query.filter(
                (ClaimRecord.raw_text.ilike(f"%{kw}%")) |
                (ClaimRecord.metric.ilike(f"%{kw}%"))
            )
            kw_matches.extend(kw_q.all())

        all_candidates = metric_matches + kw_matches
    else:
        # General: keyword search + fallback to recent
        kw_matches = []
        if keywords:
            # AND-style: require at least some keywords to match
            for kw in keywords[:5]:
                kw_q = base_query.filter(
                    (ClaimRecord.raw_text.ilike(f"%{kw}%")) |
                    (ClaimRecord.metric.ilike(f"%{kw}%")) |
                    (VerdictRecord.explanation.ilike(f"%{kw}%"))
                )
                kw_matches.extend(kw_q.all())

        # Always include some recent claims as fallback
        recent = base_query.order_by(
            ClaimRecord.year.desc(), ClaimRecord.quarter.desc()
        ).limit(20).all()

        all_candidates = kw_matches + recent

    # 4. Deduplicate by claim ID
    seen_ids = set()
    unique_candidates = []
    for claim, verdict in all_candidates:
        if claim.id not in seen_ids:
            seen_ids.add(claim.id)
            unique_candidates.append((claim, verdict))

    if not unique_candidates:
        return RetrievalResult(
            claims=[],
            intent=intent,
            filters_applied=filters,
            system_prompt_hint=_build_system_prompt(intent, filters),
        )

    # 5. Determine max year/quarter for recency scoring
    max_year = max(c.year or 0 for c, _ in unique_candidates)
    max_quarter = max(
        c.quarter or 0 for c, _ in unique_candidates
        if (c.year or 0) == max_year
    ) if max_year else 1

    # 6. Score and rank
    scored = []
    for claim, verdict in unique_candidates:
        score = _score_claim(
            claim, verdict, keywords, detected_metrics,
            target_verdict, target_quarters, max_year, max_quarter,
        )
        scored.append((score, claim, verdict))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 7. Adaptive result sizing
    if intent == "VERDICT_FILTER":
        # Return all claims that match the verdict, up to 20
        # (verdict matches will score highest due to v_score)
        max_results = 20
    elif intent == "METRIC_LOOKUP":
        max_results = 8
    elif intent == "COMPARISON":
        max_results = 15
    else:
        max_results = 12

    # Take top N but ensure minimum relevance threshold
    results = []
    min_score = 0.05  # Very low threshold to avoid empty results
    for score, claim, verdict in scored[:max_results]:
        if score >= min_score or len(results) < 5:
            results.append((claim, verdict))

    logger.info(
        f"Smart retrieval: {len(unique_candidates)} candidates → "
        f"{len(results)} results (intent={intent}, "
        f"top_score={scored[0][0]:.3f}, min_score={scored[-1][0]:.3f})"
    )

    return RetrievalResult(
        claims=results,
        intent=intent,
        filters_applied=filters,
        system_prompt_hint=_build_system_prompt(intent, filters),
    )
