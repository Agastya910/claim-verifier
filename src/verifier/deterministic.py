import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from src.models import Claim, Verdict
from src.db.schema import FinancialData
from src.data_ingest.financials import METRIC_ALIASES
from src.data_ingest.storage import load_financial_data

logger = logging.getLogger(__name__)

def compute_metric(ticker: str, metric_name: str, year: int, quarter: int, db: Session) -> Optional[float]:
    """Gets a specific metric, handling aliases and computed values."""
    # 1. Resolve canonical metric if it's an alias or computed
    aliases = METRIC_ALIASES.get(metric_name)
    
    # If not in METRIC_ALIASES, try to load it directly as an XBRL tag
    if not aliases:
        cached = load_financial_data(db, ticker, metric_name, year, quarter)
        return cached.value if cached else None

    # 2. Check for computed metrics
    for alias in aliases:
        if alias.startswith("compute:"):
            try:
                if metric_name == "free_cash_flow":
                    # Free Cash Flow = Operating Cash Flow - CapEx
                    op_cash = compute_metric(ticker, "operating_cashflow", year, quarter, db)
                    capex = compute_metric(ticker, "capex", year, quarter, db)
                    if op_cash is not None and capex is not None:
                        return op_cash - capex
                elif metric_name == "operating_margin":
                    # Operating Margin = Operating Income / Revenue
                    op_inc = compute_metric(ticker, "operating_income", year, quarter, db)
                    rev = compute_metric(ticker, "revenue", year, quarter, db)
                    if op_inc is not None and rev is not None and rev != 0:
                        return op_inc / rev
            except Exception as e:
                logger.error(f"Error computing metric {metric_name}: {e}")
            return None

        # 3. Try standard XBRL tags listed as aliases
        cached = load_financial_data(db, ticker, alias, year, quarter)
        if cached:
            return cached.value

    # 4. Fallback to trying the metric name itself
    cached = load_financial_data(db, ticker, metric_name, year, quarter)
    return cached.value if cached else None

def detect_cherry_picking(ticker: str, year: int, quarter: int, highlighted_metric: str, db: Session) -> List[str]:
    """
    Checks if other key metrics tell a different story or if YoY/QoQ trends diverge.
    """
    observations = []
    
    # Key comparisons
    # 1. Revenue vs Net Income
    if highlighted_metric == "revenue":
        rev_curr = compute_metric(ticker, "revenue", year, quarter, db)
        ni_curr = compute_metric(ticker, "net_income", year, quarter, db)
        
        # Get prior year for growth check
        rev_prev = compute_metric(ticker, "revenue", year - 1, quarter, db)
        ni_prev = compute_metric(ticker, "net_income", year - 1, quarter, db)
        
        if rev_curr is not None and ni_curr is not None and rev_prev is not None and ni_prev is not None:
            rev_growth = (rev_curr - rev_prev) / rev_prev if rev_prev != 0 else 0
            ni_growth = (ni_curr - ni_prev) / ni_prev if ni_prev != 0 else 0
            
            if rev_growth > 0 and ni_growth < 0:
                observations.append("Revenue is growing YoY, but Net Income is declining.")

    # 2. YoY vs QoQ
    # Calculate previous quarter
    prev_q_year = year
    prev_q = quarter - 1
    if prev_q == 0:
        prev_q = 4
        prev_q_year = year - 1
        
    val_curr = compute_metric(ticker, highlighted_metric, year, quarter, db)
    val_yoy_prev = compute_metric(ticker, highlighted_metric, year - 1, quarter, db)
    val_qoq_prev = compute_metric(ticker, highlighted_metric, prev_q_year, prev_q, db)
    
    if val_curr is not None and val_yoy_prev is not None and val_qoq_prev is not None:
        yoy_growth = (val_curr - val_yoy_prev) / val_yoy_prev if val_yoy_prev != 0 else 0
        qoq_growth = (val_curr - val_qoq_prev) / val_qoq_prev if val_qoq_prev != 0 else 0
        
        if yoy_growth > 0 and qoq_growth < -0.05: # Significant QoQ drop
            observations.append(f"{highlighted_metric.capitalize()} shows YoY growth, but has declined significantly (>5%) QoQ.")

    return observations

def verify_deterministic(claim: Claim, db: Session) -> Optional[Verdict]:
    """
    Orchestrates deterministic verification for a claim.
    """
    # 1. Resolve Metric
    canonical_metric = claim.metric.lower()
    
    # 2. Fetch Actual Value(s)
    actual_value = compute_metric(claim.ticker, canonical_metric, claim.year, claim.quarter, db)
    
    # 3. Determine Claim Type and Goal
    # We look for "growth" in unit or context, or if period is YoY/QoQ
    is_growth_claim = claim.period in ["YoY", "QoQ"] or "%" in claim.unit
    
    if actual_value is None and not is_growth_claim:
        return None

    claimed_val = claim.value
    actual_comp_val = None
    explanation = ""
    verdict_type = "UNVERIFIABLE"
    
    # 4. Handle Growth Claims
    if is_growth_claim:
        # Determine base period
        base_year, base_quarter = claim.year, claim.quarter
        if claim.period == "YoY":
            base_year -= 1
        else: # QoQ
            base_quarter -= 1
            if base_quarter == 0:
                base_quarter = 4
                base_year -= 1
                
        base_val = compute_metric(claim.ticker, canonical_metric, base_year, base_quarter, db)
        
        if base_val is not None and actual_value is not None:
            actual_comp_val = (actual_value - base_val) / base_val if base_val != 0 else 0
            # Multiplier for comparison (claim might be 15 for 15%)
            if claim.unit == "%" or claimed_val > 1.0: # Heuristic: if claim is > 1.0 but unit is %, it's likely 15.0 for 15%
                compare_claimed = claimed_val / 100.0 if "%" in claim.unit or claimed_val > 0.5 else claimed_val
            else:
                compare_claimed = claimed_val
                
            diff = abs(actual_comp_val - compare_claimed)
            
            # Tolerances
            is_hedged = claim.hedging_language == "true"
            threshold = 0.02 if is_hedged else 0.005
            
            if diff <= threshold:
                verdict_type = "VERIFIED" if diff < 0.001 else "APPROXIMATELY_TRUE"
            else:
                verdict_type = "FALSE"
                
            explanation = f"Calculated {claim.period} {canonical_metric} growth: {actual_comp_val:.2%}. Claimed: {compare_claimed:.2%}."
        else:
            explanation = f"Could not find historical data for {canonical_metric} in {base_year}Q{base_quarter} to verify growth."
            return None

    # 5. Handle Absolute Values
    else:
        actual_comp_val = actual_value
        diff = abs(actual_comp_val - claimed_val)
        
        # EPS Penny Precision
        if canonical_metric == "eps":
            if diff <= 0.011: # Within a cent (plus epsilon for float)
                verdict_type = "VERIFIED"
            else:
                verdict_type = "FALSE"
            explanation = f"Actual EPS: ${actual_comp_val:.2f}. Claimed: ${claimed_val:.2f}."
        else:
            is_hedged = claim.hedging_language == "true"
            # 5% for hedged, 1% for precise
            threshold = 0.05 * actual_comp_val if is_hedged else 0.01 * actual_comp_val
            
            if diff <= threshold:
                verdict_type = "VERIFIED" if diff < (0.001 * actual_comp_val) else "APPROXIMATELY_TRUE"
            else:
                verdict_type = "FALSE"
            explanation = f"Actual {canonical_metric}: {actual_comp_val}. Claimed: {claimed_val}."

    # 6. Cherry-picking Detection
    misleading_flags = detect_cherry_picking(claim.ticker, claim.year, claim.quarter, canonical_metric, db)
    if misleading_flags and verdict_type in ["VERIFIED", "APPROXIMATELY_TRUE"]:
        verdict_type = "MISLEADING"
        explanation += " " + " ".join(misleading_flags)

    return Verdict(
        claim_id=claim.id,
        verdict=verdict_type,
        actual_value=actual_comp_val,
        claimed_value=claimed_val,
        difference=abs(actual_comp_val - claimed_val) if actual_comp_val is not None else None,
        explanation=explanation,
        misleading_flags=misleading_flags,
        confidence=1.0, # Deterministic check is highly confident if data exists
        data_sources=["SEC EDGAR (Deterministic)"],
        evidence=[f"{canonical_metric} ({claim.year}Q{claim.quarter}): {actual_value}"] if actual_value is not None else []
    )
