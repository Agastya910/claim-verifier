import logging
import httpx
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import date
from sqlalchemy.orm import Session

from src.config import SEC_IDENTITY_EMAIL, FINNHUB_API_KEY
from src.db.schema import FinancialData
from src.data_ingest.storage import save_financial_data, load_financial_data

logger = logging.getLogger(__name__)

# Set SEC EDGAR identity for edgartools
try:
    from edgar import Company as EdgarCompany, set_identity
    set_identity(SEC_IDENTITY_EMAIL)
    EDGAR_AVAILABLE = True
except Exception as e:
    logger.warning(f"edgartools not available: {e}")
    EDGAR_AVAILABLE = False

METRIC_ALIASES = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "free_cash_flow": ["compute:operating_cashflow - capex"],
    "operating_margin": ["compute:operating_income / revenue"],
}

# SEC EDGAR XBRL concept tags we want to extract
SEC_XBRL_CONCEPTS = {
    "revenue": "Revenues",
    "revenue_alt": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "net_income": "NetIncomeLoss",
    "eps_diluted": "EarningsPerShareDiluted",
    "eps_basic": "EarningsPerShareBasic",
    "gross_profit": "GrossProfit",
    "operating_income": "OperatingIncomeLoss",
    "total_assets": "Assets",
    "total_liabilities": "Liabilities",
    "stockholders_equity": "StockholdersEquity",
    "cost_of_revenue": "CostOfGoodsAndServicesSold",
    "research_development": "ResearchAndDevelopmentExpense",
    "operating_expenses": "OperatingExpenses",
}

# CIK lookup for our target companies (avoids API call)
TICKER_TO_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "TSLA": "0001318605",
    "JPM": "0000019617",
    "JNJ": "0000200406",
    "WMT": "0000104169",
    "NVDA": "0001045810",
}


def fetch_sec_company_facts(ticker: str) -> Dict[str, Any]:
    """Fetch all XBRL facts for a company from SEC EDGAR companyfacts API."""
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        logger.warning(f"No CIK mapping for {ticker}")
        return {}
    
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": f"{SEC_IDENTITY_EMAIL}"}
    
    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"SEC EDGAR API returned {response.status_code} for {ticker}")
    except Exception as e:
        logger.error(f"Error fetching SEC company facts for {ticker}: {e}")
    
    return {}


def parse_sec_facts(ticker: str, facts_data: Dict[str, Any], n_quarters: int = 6) -> List[Dict[str, Any]]:
    """Parse SEC EDGAR company facts into structured financial records."""
    records = []
    
    us_gaap = facts_data.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        logger.warning(f"No us-gaap facts found for {ticker}")
        return records
    
    for metric_name, xbrl_tag in SEC_XBRL_CONCEPTS.items():
        concept_data = us_gaap.get(xbrl_tag, {})
        if not concept_data:
            continue
        
        units_data = concept_data.get("units", {})
        
        # Determine the right unit key (USD for dollar amounts, USD/shares for EPS)
        unit_key = None
        if "USD" in units_data:
            unit_key = "USD"
            unit = "USD"
        elif "USD/shares" in units_data:
            unit_key = "USD/shares"
            unit = "USD/share"
        elif "shares" in units_data:
            unit_key = "shares"
            unit = "shares"
        else:
            continue
        
        entries = units_data[unit_key]
        
        # Filter entries: 
        # - 10-Q with fp=Q1/Q2/Q3 → quarterly data
        # - 10-K with fp=FY → annual/Q4 data  
        # - Focus on 2024 quarters for complete data
        quarterly_entries = []
        for entry in entries:
            form = entry.get("form", "")
            fp = entry.get("fp", "")
            end_date = entry.get("end", "")
            
            try:
                year = int(end_date[:4]) if end_date else 0
            except (ValueError, IndexError):
                continue
            
            # Focus on 2024 data for this task
            if year != 2024:
                continue
            
            # 10-Q with explicit quarter
            if form == "10-Q" and fp in ("Q1", "Q2", "Q3"):
                entry["_quarter"] = int(fp[1])
                entry["_year"] = year
                quarterly_entries.append(entry)
            # 10-K FY → treat as Q4
            elif form == "10-K" and fp == "FY":
                entry["_quarter"] = 4
                entry["_year"] = year
                quarterly_entries.append(entry)
        
        # Deduplicate: keep last filed per (year, quarter)
        seen = {}
        for entry in quarterly_entries:
            key = (entry["_year"], entry["_quarter"])
            filed = entry.get("filed", "")
            if key not in seen or filed > seen[key].get("filed", ""):
                seen[key] = entry
        
        # Ensure we have all 4 quarters for 2024
        for entry in seen.values():
            try:
                filed_date = entry.get("filed", "")
                val = entry.get("val")
                year = entry["_year"]
                quarter = entry["_quarter"]
                
                if val is None:
                    continue
                
                records.append({
                    "ticker": ticker,
                    "year": year,
                    "quarter": quarter,
                    "metric": metric_name,
                    "value": float(val),
                    "unit": unit,
                    "source": entry.get("form", "SEC"),
                    "is_gaap": True,
                    "filing_date": filed_date,
                })
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(f"Skipping entry for {metric_name}: {e}")
                continue
    
    # Verify we have all 4 quarters
    quarters_found = set()
    for rec in records:
        if rec["year"] == 2024:
            quarters_found.add(rec["quarter"])
    
    logger.debug(f"{ticker} has data for quarters: {sorted(quarters_found)} in 2024")
    
    return records


def load_financials_from_db(ticker: str, db: Session = None) -> Dict[str, Any]:
    """Check if financial data already exists in DB and return it organized by period."""
    close_db = False
    if db is None:
        from src.db.connection import SessionLocal
        db = SessionLocal()
        close_db = True
    
    try:
        records = db.query(FinancialData).filter(
            FinancialData.ticker == ticker.upper()
        ).all()
        
        if not records:
            return {}
        
        organized = {}
        for rec in records:
            period = f"{rec.year}Q{rec.quarter}"
            if period not in organized:
                organized[period] = {"metrics": {}, "source": rec.source or "SEC"}
            organized[period]["metrics"][rec.metric] = rec.value
        
        logger.info(f"Loaded {len(records)} financial records from DB for {ticker} ({len(organized)} periods)")
        return organized
    finally:
        if close_db:
            db.close()


def fetch_financial_statements(ticker: str, n_quarters: int = 6, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetches quarterly financial data from SEC EDGAR companyfacts API.
    Returns a dict keyed by period (e.g., '2024Q1') containing metric dicts.
    Also stores all data into the database.
    
    Returns cached DB data if it already exists for this company.
    Set force_refresh=True to bypass cache.
    """
    # Step 0: Check DB cache first (unless force refresh)
    if not force_refresh:
        cached = load_financials_from_db(ticker)
        if cached:
            # Check if we have all required quarters (2024Q1-Q4)
            required_periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
            missing_periods = [p for p in required_periods if p not in cached]
            
            if not missing_periods:
                logger.info(f"Loaded complete financial data from DB for {ticker}")
                return cached
            else:
                logger.warning(f"DB has incomplete data for {ticker} - missing: {missing_periods}")
    
    logger.info(f"Fetching financial statements for {ticker} from SEC EDGAR")
    
    # Fetch from SEC EDGAR companyfacts API (free, unlimited)
    facts_data = fetch_sec_company_facts(ticker)
    if not facts_data:
        logger.warning(f"No SEC data for {ticker}, trying Finnhub fallback")
        return fetch_basic_metrics_finnhub_structured(ticker)
    
    records = parse_sec_facts(ticker, facts_data, n_quarters)
    logger.info(f"Parsed {len(records)} financial records for {ticker}")
    
    # Store in database
    from src.db.connection import SessionLocal
    db = SessionLocal()
    try:
        store_financials(db, ticker, records)
    except Exception as e:
        logger.error(f"Error storing financials for {ticker}: {e}")
    finally:
        db.close()
    
    # Also organize by period for the indexer
    organized = {}
    for rec in records:
        period = f"{rec['year']}Q{rec['quarter']}"
        if period not in organized:
            organized[period] = {"metrics": {}, "source": rec["source"]}
        organized[period]["metrics"][rec["metric"]] = rec["value"]
    
    return organized


def fetch_basic_metrics_finnhub(ticker: str) -> Dict[str, Any]:
    """Secondary source: Finnhub basic financials."""
    try:
        import finnhub
        client = finnhub.Client(api_key=FINNHUB_API_KEY)
        return client.company_basic_financials(ticker, 'all')
    except Exception as e:
        logger.warning(f"Finnhub basic financials unavailable for {ticker}: {e}")
        return {}


def fetch_basic_metrics_finnhub_structured(ticker: str) -> Dict[str, Any]:
    """Fetch from Finnhub and structure for our pipeline."""
    raw = fetch_basic_metrics_finnhub(ticker)
    if not raw:
        return {}
    
    organized = {}
    series = raw.get("series", {}).get("quarterly", {})
    
    # Map Finnhub metric names to our names
    finnhub_metric_map = {
        "revenuePerShareQuarterly": "revenue_per_share",
        "epsActual": "eps_diluted",
        "grossMarginQuarterly": "gross_margin",
        "operatingMarginQuarterly": "operating_margin",
        "netProfitMarginQuarterly": "net_profit_margin",
    }
    
    from src.db.connection import SessionLocal
    db = SessionLocal()
    all_records = []
    
    try:
        for fh_key, our_key in finnhub_metric_map.items():
            data_points = series.get(fh_key, [])
            for dp in data_points[:8]:  # Last 8 quarters
                period_str = dp.get("period", "")
                val = dp.get("v")
                if not period_str or val is None:
                    continue
                try:
                    year = int(period_str[:4])
                    month = int(period_str[5:7])
                    quarter = (month - 1) // 3 + 1
                    
                    all_records.append({
                        "ticker": ticker,
                        "year": year,
                        "quarter": quarter,
                        "metric": our_key,
                        "value": float(val),
                        "unit": "ratio" if "margin" in our_key.lower() else "USD",
                        "source": "finnhub",
                        "is_gaap": True,
                        "filing_date": period_str,
                    })
                except (ValueError, TypeError):
                    continue
        
        if all_records:
            store_financials(db, ticker, all_records)
    finally:
        db.close()
    
    # Organize by period
    for rec in all_records:
        period = f"{rec['year']}Q{rec['quarter']}"
        if period not in organized:
            organized[period] = {"metrics": {}, "source": "finnhub"}
        organized[period]["metrics"][rec["metric"]] = rec["value"]
    
    return organized


def store_financials(db: Session, ticker: str, records: List[Dict[str, Any]]):
    """Store parsed financial records into the database. Skips if already exists (immutable data)."""
    if not records:
        logger.info(f"No financial records to store for {ticker}")
        return
    
    stored_count = 0
    skipped_count = 0
    
    for rec in records:
        try:
            # Check for existing record to avoid duplicates
            existing = db.query(FinancialData).filter(
                FinancialData.ticker == rec["ticker"],
                FinancialData.metric == rec["metric"],
                FinancialData.year == rec["year"],
                FinancialData.quarter == rec["quarter"],
                FinancialData.is_gaap == rec.get("is_gaap", True)
            ).first()
            
            if existing:
                skipped_count += 1
                continue
            
            filing_date_val = None
            if rec.get("filing_date"):
                try:
                    filing_date_val = date.fromisoformat(rec["filing_date"][:10])
                except (ValueError, TypeError):
                    pass
            
            new_record = FinancialData(
                ticker=rec["ticker"],
                year=rec["year"],
                quarter=rec["quarter"],
                metric=rec["metric"],
                value=rec["value"],
                unit=rec.get("unit", "USD"),
                source=rec.get("source", "SEC"),
                is_gaap=rec.get("is_gaap", True),
                filing_date=filing_date_val,
            )
            db.add(new_record)
            stored_count += 1
        except Exception as e:
            logger.error(f"Error storing record {rec.get('metric')}: {e}")
            continue
    
    try:
        db.commit()
        logger.info(
            f"Stored {stored_count} new financial records for {ticker}. "
            f"Skipped {skipped_count} duplicates (data is immutable)."
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error committing financial records: {e}")
        raise


def get_metric(ticker: str, metric_name: str, year: int, quarter: int, db: Session) -> Optional[float]:
    """Gets a specific metric, handling aliases and computed values."""
    # 1. Check DB cache first
    cached = load_financial_data(db, ticker, metric_name, year, quarter)
    if cached:
        return cached.value
    
    # 2. Resolve aliases or compute
    aliases = METRIC_ALIASES.get(metric_name)
    if not aliases:
        return None
        
    for alias in aliases:
        if alias.startswith("compute:"):
            try:
                if metric_name == "free_cash_flow":
                    op_cash = get_metric(ticker, "operating_cashflow", year, quarter, db)
                    capex = get_metric(ticker, "capex", year, quarter, db)
                    if op_cash is not None and capex is not None:
                        return op_cash - capex
                elif metric_name == "operating_margin":
                    op_inc = get_metric(ticker, "operating_income", year, quarter, db)
                    rev = get_metric(ticker, "revenue", year, quarter, db)
                    if op_inc is not None and rev is not None and rev != 0:
                        return op_inc / rev
            except Exception as e:
                logger.error(f"Error computing metric {metric_name}: {e}")
            continue
            
    return None


def fetch_metric_sec_api(ticker: str, metric_tag: str) -> Dict[str, Any]:
    """Backup: Fetch specific XBRL tag from SEC companyconcept endpoint."""
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        return {}
    
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{metric_tag}.json"
    headers = {"User-Agent": SEC_IDENTITY_EMAIL}
    
    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"Error in direct SEC API fallback for {metric_tag}: {e}")
    return {}
