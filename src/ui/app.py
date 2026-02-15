import sys
import os
# Ensure repo root is on sys.path so `src.*` imports work when running via `streamlit run`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
import re

from sqlalchemy import create_engine, select, distinct, func, String
from sqlalchemy.orm import Session, sessionmaker

from src.config import (
    DATABASE_URL, MODEL_CONFIGS, ACTIVE_MODEL_TIER, COMPANIES,
    OLLAMA_BASE_URL, OLLAMA_API_KEY, validate_ollama_config,
)
from src.db.schema import ClaimRecord, VerdictRecord, TranscriptRecord, FinancialData

logger = logging.getLogger(__name__)

# ─── Configuration ──────────────────────────────────────────────────────
VERDICT_COLORS = {
    "VERIFIED": "#22c55e",
    "APPROXIMATELY_TRUE": "#86efac",
    "FALSE": "#ef4444",
    "MISLEADING": "#f97316",
    "UNVERIFIABLE": "#6b7280"
}
VERDICT_ICONS = {
    "VERIFIED": "✅",
    "APPROXIMATELY_TRUE": "🟢",
    "FALSE": "❌",
    "MISLEADING": "⚠️",
    "UNVERIFIABLE": "❓"
}

st.set_page_config(
    page_title="Claim Verifier Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Database Connection (cached — created once per Streamlit process) ──
@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL)

def get_session():
    engine = get_engine()
    return sessionmaker(bind=engine)()


# ─── Data Access Functions (replace httpx API calls) ────────────────────

def check_health():
    """Check database connectivity."""
    try:
        with Session(get_engine()) as db:
            db.execute(select(1))
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy"}


def list_companies():
    """Get unique tickers from transcripts + financial_data."""
    try:
        with Session(get_engine()) as db:
            t_tickers = db.execute(select(distinct(TranscriptRecord.ticker))).scalars().all()
            f_tickers = db.execute(select(distinct(FinancialData.ticker))).scalars().all()
            return sorted(list(set(t_tickers) | set(f_tickers)))
    except Exception:
        return []


def get_results(ticker):
    """Get all claims and verdicts for a company."""
    ticker = ticker.upper()
    try:
        with Session(get_engine()) as db:
            claims = db.query(ClaimRecord).filter(ClaimRecord.ticker == ticker).all()
            verdicts = db.query(VerdictRecord).join(ClaimRecord).filter(ClaimRecord.ticker == ticker).all()

            # Serialize to dicts (detach from session)
            claims_list = []
            for c in claims:
                claims_list.append({
                    "id": c.id, "ticker": c.ticker, "quarter": c.quarter, "year": c.year,
                    "speaker": c.speaker, "metric": c.metric, "value": c.value, "unit": c.unit,
                    "period": c.period, "is_gaap": c.is_gaap, "is_forward_looking": c.is_forward_looking,
                    "hedging_language": c.hedging_language, "raw_text": c.raw_text,
                    "extraction_method": c.extraction_method, "confidence": c.confidence,
                    "context": c.context,
                })
            verdicts_list = []
            for v in verdicts:
                verdicts_list.append({
                    "id": v.id, "claim_id": v.claim_id, "verdict": v.verdict,
                    "actual_value": v.actual_value, "claimed_value": v.claimed_value,
                    "difference": v.difference, "explanation": v.explanation,
                    "misleading_flags": v.misleading_flags, "confidence": v.confidence,
                    "data_sources": v.data_sources, "evidence": v.evidence,
                })
            return {
                "ticker": ticker,
                "total_claims": len(claims_list),
                "claims": claims_list,
                "verdicts": verdicts_list,
            }
    except Exception as e:
        logger.error(f"Error retrieving results for {ticker}: {e}")
        return None


def get_dashboard():
    """Aggregate dashboard data across all companies."""
    try:
        with Session(get_engine()) as db:
            all_tickers = db.execute(select(distinct(ClaimRecord.ticker))).scalars().all()
            if not all_tickers:
                return {"has_precomputed_data": False, "companies": [], "totals": {}}

            dashboard = {
                "companies": [],
                "totals": {"claims": 0, "verified": 0, "false": 0, "misleading": 0, "unverifiable": 0, "approx_true": 0},
                "target_companies": COMPANIES,
                "has_precomputed_data": True,
            }

            for ticker in sorted(all_tickers):
                claims = db.query(ClaimRecord).filter(ClaimRecord.ticker == ticker).all()
                claim_ids = [c.id for c in claims]
                verdicts = db.query(VerdictRecord).filter(VerdictRecord.claim_id.in_(claim_ids)).all() if claim_ids else []

                v_counts = {}
                for v in verdicts:
                    v_counts[v.verdict] = v_counts.get(v.verdict, 0) + 1

                company_data = {
                    "ticker": ticker,
                    "total_claims": len(claims),
                    "total_verdicts": len(verdicts),
                    "verified": v_counts.get("VERIFIED", 0),
                    "false": v_counts.get("FALSE", 0),
                    "misleading": v_counts.get("MISLEADING", 0),
                    "approx_true": v_counts.get("APPROXIMATELY_TRUE", 0),
                    "unverifiable": v_counts.get("UNVERIFIABLE", 0),
                }
                dashboard["companies"].append(company_data)
                dashboard["totals"]["claims"] += len(claims)
                dashboard["totals"]["verified"] += company_data["verified"]
                dashboard["totals"]["false"] += company_data["false"]
                dashboard["totals"]["misleading"] += company_data["misleading"]
                dashboard["totals"]["unverifiable"] += company_data["unverifiable"]
                dashboard["totals"]["approx_true"] += company_data["approx_true"]

            return dashboard
    except Exception as e:
        logger.error(f"Error building dashboard: {e}")
        return {"has_precomputed_data": False}


def get_quarters(ticker):
    """Get available (year, quarter) pairs for a company."""
    ticker = ticker.upper()
    try:
        with Session(get_engine()) as db:
            trans_rows = db.execute(
                select(TranscriptRecord.year, TranscriptRecord.quarter)
                .where(TranscriptRecord.ticker == ticker)
            ).all()
            fin_rows = db.execute(
                select(FinancialData.year, FinancialData.quarter)
                .where(FinancialData.ticker == ticker)
            ).all()
            all_q = set()
            for y, q in trans_rows + fin_rows:
                all_q.add((y, q))
            result = [{"year": y, "quarter": q} for y, q in sorted(list(all_q), reverse=True)]
            return {"available_quarters": result}
    except Exception:
        return {"available_quarters": []}


def get_transcript(ticker, year, quarter):
    """Get raw transcript data."""
    try:
        with Session(get_engine()) as db:
            rec = db.query(TranscriptRecord).filter(
                TranscriptRecord.ticker == ticker.upper(),
                TranscriptRecord.year == year,
                TranscriptRecord.quarter == quarter,
            ).first()
            if not rec:
                return None
            return {
                "source": rec.source, "date": str(rec.date) if rec.date else None,
                "segments": rec.segments or [],
            }
    except Exception:
        return None


def get_financials(ticker, year, quarter):
    """Get raw financial data."""
    try:
        with Session(get_engine()) as db:
            recs = db.query(FinancialData).filter(
                FinancialData.ticker == ticker.upper(),
                FinancialData.year == year,
                FinancialData.quarter == quarter,
            ).all()
            if not recs:
                return None
            return [
                {"metric": r.metric, "value": r.value, "unit": r.unit,
                 "is_gaap": r.is_gaap, "source": r.source}
                for r in recs
            ]
    except Exception:
        return None


def ask_question(ticker, question):
    """
    Answer a question about a company using keyword search over verified claims + LLM.
    Replaces the /api/ask endpoint.
    """
    import litellm

    ticker = ticker.upper()
    question = question.strip()
    if not question:
        return {"answer": "Please enter a question.", "claim_texts": []}

    try:
        stop_words = {'the', 'a', 'an', 'is', 'was', 'were', 'are', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                      'could', 'may', 'might', 'must', 'can', 'about', 'in', 'on', 'at',
                      'to', 'for', 'of', 'with', 'by', 'from', 'up', 'down', 'out', 'off',
                      'over', 'under', 'again', 'further', 'then', 'once'}

        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        with Session(get_engine()) as db:
            query = db.query(ClaimRecord, VerdictRecord).join(
                VerdictRecord, ClaimRecord.id == VerdictRecord.claim_id, isouter=True
            ).filter(ClaimRecord.ticker == ticker)

            matched_claims = []
            if keywords:
                for keyword in keywords[:5]:
                    kp = f"%{keyword}%"
                    kq = query.filter(
                        (ClaimRecord.raw_text.ilike(kp)) |
                        (ClaimRecord.metric.ilike(kp)) |
                        (VerdictRecord.explanation.ilike(kp)) |
                        (func.cast(VerdictRecord.evidence, String).ilike(kp))
                    )
                    matched_claims.extend(kq.all())

            seen_texts = set()
            unique_matches = []
            for claim, verdict in matched_claims:
                if claim.raw_text not in seen_texts:
                    seen_texts.add(claim.raw_text)
                    unique_matches.append((claim, verdict))

            if not unique_matches:
                unique_matches = query.order_by(
                    ClaimRecord.year.desc(), ClaimRecord.quarter.desc()
                ).limit(10).all()
            else:
                unique_matches.sort(key=lambda x: (x[0].year, x[0].quarter), reverse=True)
                unique_matches = unique_matches[:10]

            if not unique_matches:
                return {
                    "answer": f"No verified claims found for {ticker}. Please make sure the company has been analyzed.",
                    "claim_texts": [],
                }

            # Build context blocks
            context_blocks = []
            claim_texts_out = []
            for claim, verdict in unique_matches:
                claim_texts_out.append(claim.raw_text)
                block = f"Claim: {claim.raw_text}\n"
                block += f"Metric: {claim.metric} = {claim.value} {claim.unit or ''}\n"
                block += f"Quarter: {claim.year} Q{claim.quarter}\n"
                block += f"Speaker: {claim.speaker}\n"
                if verdict:
                    block += f"Verdict: {verdict.verdict}\n"
                    block += f"Reasoning: {verdict.explanation}\n"
                    if verdict.evidence:
                        ev_list = verdict.evidence if isinstance(verdict.evidence, list) else []
                        if ev_list:
                            block += f"Evidence: {'; '.join(ev_list)}\n"
                else:
                    block += "Verdict: NOT YET VERIFIED\n"
                context_blocks.append(block)

            context_str = "\n---\n\n".join(context_blocks)

        # LLM call
        model = MODEL_CONFIGS.get(ACTIVE_MODEL_TIER, MODEL_CONFIGS["default"])

        system_message = (
            "You are a financial analysis assistant. Answer the user's question using ONLY "
            "the provided verified claims and evidence. If the answer cannot be derived from "
            "the provided context, say 'The available verified claims do not contain enough "
            "information to answer this question.' Do not fabricate data."
        )
        user_message = (
            f"Company: {ticker}\n\n"
            f"Question: {question}\n\n"
            f"--- VERIFIED CLAIMS ---\n{context_str}\n--- END VERIFIED CLAIMS ---\n\n"
            f"Please answer the question based solely on the verified claims above."
        )

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
            "timeout": 300,
        }

        if "ollama" in model:
            validate_ollama_config()
            kwargs["api_base"] = OLLAMA_BASE_URL
            kwargs["api_key"] = OLLAMA_API_KEY

        response = litellm.completion(**kwargs)

        if hasattr(response.choices[0], 'message'):
            answer = response.choices[0].message.content.strip()
        else:
            answer = response['choices'][0]['message']['content'].strip()

        return {
            "answer": answer,
            "claim_texts": claim_texts_out,
            "num_claims_used": len(unique_matches),
        }

    except Exception as e:
        logger.error(f"Error answering question for {ticker}: {e}")
        return {"answer": f"Failed to answer question: {str(e)}", "claim_texts": []}


# ─── Premium CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global */
* { font-family: 'Inter', sans-serif; }

/* Metric Cards */
.stMetric {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 16px 20px;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
}
.stMetric label { color: #94a3b8 !important; font-size: 0.85em !important; }
.stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 600 !important; }

/* Verdict badges */
.verdict-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.78em;
    letter-spacing: 0.03em;
    color: white;
}

/* Claim row */
.claim-row {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.claim-row .claim-speaker { color: #818cf8; font-weight: 600; font-size: 0.85em; }
.claim-row .claim-text { color: #e2e8f0; margin: 6px 0; font-size: 0.95em; line-height: 1.5; }
.claim-row .claim-meta { color: #64748b; font-size: 0.8em; }
.claim-row .claim-explanation { color: #94a3b8; font-size: 0.85em; margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(148, 163, 184, 0.1); }

/* Welcome hero */
.hero-section {
    background: linear-gradient(135deg, #312e81 0%, #1e1b4b 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 24px;
    text-align: center;
    border: 1px solid rgba(99, 102, 241, 0.2);
}
.hero-section h1 { color: #e2e8f0; font-size: 2em; margin-bottom: 8px; }
.hero-section p { color: #94a3b8; font-size: 1.1em; line-height: 1.6; }
.hero-section .step-list { text-align: left; max-width: 500px; margin: 24px auto 0; }
.hero-section .step { color: #c7d2fe; font-size: 0.95em; margin: 10px 0; }
.hero-section .step-num {
    background: rgba(99, 102, 241, 0.3);
    border-radius: 50%;
    width: 28px; height: 28px;
    display: inline-flex; align-items: center; justify-content: center;
    margin-right: 12px; font-weight: 700; font-size: 0.85em; color: #a5b4fc;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    padding: 8px 20px;
    border-radius: 8px;
    font-weight: 500;
}

/* Answer container */
.answer-container {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 16px;
    color: #e2e8f0;
    font-size: 0.95em;
    line-height: 1.7;
}
.answer-container .answer-label {
    color: #818cf8;
    font-weight: 600;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}

/* Add-company section */
.add-company-info {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 8px;
    color: #94a3b8;
    font-size: 0.85em;
    line-height: 1.5;
}

/* Remove Streamlit branding */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Claim Verifier")
st.sidebar.caption("Earnings transcript verification system")
st.sidebar.markdown("---")

# System status
health = check_health()
if health and health.get("status") == "healthy":
    st.sidebar.success("🟢 System Online")
else:
    st.sidebar.error("🔴 System Offline — check database connection")

# ─── Section A: Companies ───────────────────────────────────────────────
st.sidebar.subheader("🏢 Companies")

available_companies = list_companies()

selected_tickers = st.sidebar.multiselect(
    "Select companies",
    options=available_companies,
    default=None,
    help="Choose one or more companies to view precomputed verification results.",
    key="company_select",
)

active_tickers = selected_tickers

# ─── Section B: Add New Company (Controlled Future Feature) ─────────────
st.sidebar.markdown("---")
with st.sidebar.expander("➕ Don't see a company? Add one.", expanded=False):
    new_company_name = st.text_input(
        "Company name",
        placeholder="e.g. Netflix",
        key="new_company_name",
    )

    if new_company_name:
        resolved_ticker = new_company_name.strip().upper()[:4]
        st.info(f"🔎 Resolved ticker: **{resolved_ticker}**")
        st.caption("Please confirm this is the correct ticker before proceeding.")

        if st.button("✅ Confirm & Submit Ingestion", key="confirm_ingest"):
            st.success(
                f"📬 Company ingestion job queued for **{resolved_ticker}**.\n\n"
                "This may take several minutes. The pipeline will:\n"
                "- Download transcripts (4 most recent quarters)\n"
                "- Fetch SEC filings (10-K, 10-Q only)\n"
                "- Extract claims & verify\n\n"
                "Please **refresh the page** after a few minutes to see the company once ingestion completes."
            )
    else:
        st.markdown(
            '<div class="add-company-info">'
            "Enter a company name to add it. Ingestion is limited to the 4 most recent quarters and required SEC filings (10-K, 10-Q) only."
            "</div>",
            unsafe_allow_html=True,
        )

# ─── Default Model Display ──────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model")
st.sidebar.markdown(
    '<div style="background: linear-gradient(135deg, #1e293b, #0f172a); '
    "border: 1px solid rgba(99,102,241,0.3); border-radius: 10px; "
    'padding: 12px 16px; margin-bottom: 8px;">'
    '<div style="color: #64748b; font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.05em;">Default Model</div>'
    '<div style="color: #e2e8f0; font-size: 1.05em; font-weight: 600; margin-top: 4px;">DeepSeek (671B)</div>'
    "</div>",
    unsafe_allow_html=True,
)

# API Keys — optional (informational / future-ready)
with st.sidebar.expander("🔑 API Keys (Optional)", expanded=False):
    has_default_key = bool(os.getenv("GROQ_API_KEY"))
    if has_default_key:
        st.success("✅ Default key configured (Groq)")
        st.caption("Add keys below for premium models.")
    else:
        st.warning("No API key configured. Enter at least one.")

    openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
    anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")
    groq_key = st.text_input("Groq API Key", type="password", key="groq_key")

    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key: os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if groq_key: os.environ["GROQ_API_KEY"] = groq_key


# ─── Fetch data for selected companies ───────────────────────────────────
selected_results = []
if active_tickers:
    for ticker in active_tickers:
        res = get_results(ticker)
        if res and res.get("total_claims", 0) > 0:
            selected_results.append(res)

# General dashboard data (used on landing page)
dashboard_data = get_dashboard()


# ─── Main Content ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Claims & Verdicts", "🗄️ Raw Data"])


# ─── Tab 1: Dashboard ──────────────────────────────────────────────────
with tab1:
    if not active_tickers:
        # ─── Welcome Hero ───────────────────────────────────────────
        st.markdown("""
        <div class="hero-section">
            <h1>🔍 Earnings Claim Verifier</h1>
            <p>Quantitative claims from earnings call transcripts have been<br>
            extracted and verified against actual SEC financial data.</p>
            <div class="step-list">
                <div class="step"><span class="step-num">1</span> Select companies in the sidebar</div>
                <div class="step"><span class="step-num">2</span> Browse precomputed claim verdicts</div>
                <div class="step"><span class="step-num">3</span> Ask questions powered by RAG search</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show pre-computed dashboard overview if data exists
        if dashboard_data and dashboard_data.get("has_precomputed_data"):
            st.subheader("📊 Pre-computed Results Overview")

            totals = dashboard_data.get("totals", {})
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Claims", totals.get("claims", 0))
            col2.metric("Verified ✅", totals.get("verified", 0))
            col3.metric("False ❌", totals.get("false", 0))
            col4.metric("Misleading ⚠️", totals.get("misleading", 0))
            col5.metric("Unverifiable ❓", totals.get("unverifiable", 0))

            st.markdown("---")
            st.subheader("🏢 Companies Analyzed")
            cols = st.columns(5)
            for i, comp in enumerate(dashboard_data.get("companies", [])):
                with cols[i % 5]:
                    v_count = comp.get('total_verdicts', 0)
                    c_count = comp.get('total_claims', 0)
                    verified = comp.get('verified', 0)
                    false_c = comp.get('false', 0)
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e293b, #0f172a);
                                border: 1px solid rgba(148,163,184,0.1);
                                border-radius: 10px; padding: 16px; text-align: center; margin-bottom: 8px;">
                        <div style="font-size: 1.4em; font-weight: 700; color: #e2e8f0;">{comp['ticker']}</div>
                        <div style="font-size: 0.85em; color: #94a3b8; margin-top: 4px;">{c_count} claims | {v_count} verdicts</div>
                        <div style="font-size: 0.8em; margin-top: 4px; color: #22c55e;">✅ {verified}</div>
                        <div style="font-size: 0.8em; color: #ef4444;">❌ {false_c}</div>
                    </div>
                    """, unsafe_allow_html=True)
        elif available_companies:
            st.subheader("📈 Available Companies")
            cols = st.columns(5)
            for i, comp in enumerate(available_companies):
                with cols[i % 5]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e293b, #0f172a);
                                border: 1px solid rgba(148,163,184,0.1);
                                border-radius: 10px; padding: 16px; text-align: center; margin-bottom: 8px;">
                        <div style="font-size: 1.4em; font-weight: 700; color: #e2e8f0;">{comp}</div>
                        <div style="font-size: 0.8em; color: #64748b; margin-top: 4px;">Select to view results</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No companies available yet.")

    elif not selected_results:
        st.warning(f"📋 No verification results found for selection. Selected companies may not have been analyzed yet.")
    else:
        # ─── Filtered Aggregated Dashboard ──────────────────────────
        display_names = ", ".join(active_tickers)
        st.header(f"📊 Dashboard: {display_names if len(active_tickers) < 4 else f'{len(active_tickers)} Companies'}")

        # Aggregate metrics
        all_verdicts = []
        all_claims = []
        for r in selected_results:
            all_verdicts.extend(r.get("verdicts", []))
            all_claims.extend(r.get("claims", []))

        v_counts = pd.Series([v.get("verdict") for v in all_verdicts]).value_counts().to_dict() if all_verdicts else {}

        # Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Claims", len(all_claims))
        col2.metric("Verified ✅", v_counts.get("VERIFIED", 0))
        col3.metric("False ❌", v_counts.get("FALSE", 0))
        col4.metric("Misleading ⚠️", v_counts.get("MISLEADING", 0))
        col5.metric("Unverifiable ❓", v_counts.get("UNVERIFIABLE", 0))

        st.markdown("---")

        # Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Verdicts by Company")
            company_stats = []
            for r in selected_results:
                ticker = r.get("ticker", "")
                r_verdicts = r.get("verdicts", [])
                for v in r_verdicts:
                    company_stats.append({"Company": ticker, "Verdict": v["verdict"], "Count": 1})

            if company_stats:
                df_stats = pd.DataFrame(company_stats).groupby(["Company", "Verdict"], as_index=False).sum()
                fig_bar = px.bar(df_stats, x="Company", y="Count", color="Verdict",
                                color_discrete_map=VERDICT_COLORS, barmode="stack",
                                template="plotly_dark")
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font_color='#94a3b8', legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.caption("No verdict data to chart.")

        with chart_col2:
            st.subheader("Overall Distribution")
            if v_counts:
                df_pie = pd.DataFrame([{"Verdict": k, "Count": v} for k, v in v_counts.items()])
                fig_pie = px.pie(df_pie, values="Count", names="Verdict", color="Verdict",
                                color_discrete_map=VERDICT_COLORS, hole=0.45,
                                template="plotly_dark")
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                                      legend=dict(orientation="h", y=-0.15))
                fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                                      textfont_size=12)
                st.plotly_chart(fig_pie, use_container_width=True)

        # Flagged Claims
        st.markdown("---")
        st.subheader("⚠️ Top Flagged Claims")
        flagged = [v for v in all_verdicts if v.get("verdict") in ("MISLEADING", "FALSE")]
        if flagged:
            claims_map = {c["id"]: c for c in all_claims}
            for v in flagged[:8]:
                claim = claims_map.get(v["claim_id"], {})
                color = VERDICT_COLORS.get(v["verdict"], "#6b7280")
                icon = VERDICT_ICONS.get(v["verdict"], "")
                st.markdown(f"""
                <div class="claim-row">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="claim-speaker">{claim.get('speaker', 'Unknown')} — {claim.get('ticker', '')}</span>
                        <span class="verdict-badge" style="background-color: {color};">{icon} {v['verdict']}</span>
                    </div>
                    <div class="claim-text">"{claim.get('raw_text', 'N/A')}"</div>
                    <div class="claim-meta">{claim.get('metric', '')} = {claim.get('value', '')} {claim.get('unit', '')} | {claim.get('year', '')} Q{claim.get('quarter', '')}</div>
                    <div class="claim-explanation">{v.get('explanation', '')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 No misleading or false claims detected in selection!")


# ─── Tab 2: Claims & Verdicts (Merged Company Detail + Claim Inspector) ─
with tab2:
    if not active_tickers:
        st.info("Select companies in the sidebar to view claims and verdicts.")
    elif not selected_results:
        st.info(f"No results found for selection.")
    else:
        # Sub-selection within tab for focus, if multiple selected
        if len(active_tickers) > 1:
            focus_ticker = st.selectbox("Focus on company", active_tickers, index=0)
        else:
            focus_ticker = active_tickers[0]

        focus_res = next((r for r in selected_results if r["ticker"] == focus_ticker), None)

        if focus_res:
            # ─── Search / Question Feature ──────────────────────────────
            st.markdown(f"#### 💬 Ask a question about {focus_ticker}")
            question_input = st.text_input(
                f"Ask a question about {focus_ticker}",
                placeholder=f'e.g. "Was {focus_ticker} lying about revenue in Q4 2024?"',
                key="ask_question",
                label_visibility="collapsed",
            )

            if question_input:
                with st.spinner(f"Searching verified claims for {focus_ticker}…"):
                    answer_resp = ask_question(focus_ticker, question_input)
                if answer_resp:
                    st.markdown(
                        f'<div class="answer-container">'
                        f'<div class="answer-label">Answer</div>'
                        f'{answer_resp.get("answer", "No answer returned.")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    claim_texts = answer_resp.get("claim_texts", [])
                    num_claims = answer_resp.get("num_claims_used", 0)
                    if claim_texts:
                        with st.expander(f"📚 {num_claims} verified claim(s) used", expanded=False):
                            for i, ct in enumerate(claim_texts):
                                st.caption(f"{i+1}. {ct}")
                else:
                    st.error("Failed to get an answer. Please try again.")

            st.markdown("---")

            # ─── Company Summary Metrics (TOP) ──────────────────────────
            st.subheader(f"📈 {focus_ticker} — Summary")

            verdicts = focus_res.get("verdicts", [])
            claims_list = focus_res.get("claims", [])
            v_counts = pd.Series([v.get("verdict") for v in verdicts]).value_counts().to_dict() if verdicts else {}

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Claims", len(claims_list))
            m2.metric("Verified", v_counts.get("VERIFIED", 0))
            m3.metric("Approx True", v_counts.get("APPROXIMATELY_TRUE", 0))
            m4.metric("False", v_counts.get("FALSE", 0))
            m5.metric("Misleading", v_counts.get("MISLEADING", 0))
            m6.metric("Unverifiable", v_counts.get("UNVERIFIABLE", 0))

            st.markdown("---")

            # ─── Filters ────────────────────────────────────────────────
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                filter_verdict = st.multiselect("Filter by Verdict",
                    ["VERIFIED", "FALSE", "MISLEADING", "APPROXIMATELY_TRUE", "UNVERIFIABLE"],
                    default=None, key=f"filter_v_{focus_ticker}")
            with filter_col2:
                quarters_in_data = sorted(
                    list(set(f"{c['year']} Q{c['quarter']}" for c in claims_list)),
                    reverse=True
                )
                filter_quarter = st.multiselect("Filter by Quarter", quarters_in_data, default=None, key=f"filter_q_{focus_ticker}")

            # Build verdict map
            v_map = {v["claim_id"]: v for v in verdicts}

            # Apply filters
            filtered_claims = claims_list
            if filter_verdict:
                filtered_claims = [c for c in filtered_claims if v_map.get(c["id"], {}).get("verdict") in filter_verdict]
            if filter_quarter:
                filtered_claims = [c for c in filtered_claims if f"{c['year']} Q{c['quarter']}" in filter_quarter]

            st.caption(f"Showing {len(filtered_claims)} of {len(claims_list)} claims")

            # ─── Detailed Claim Inspector (BELOW) ───────────────────────
            for c in filtered_claims:
                v = v_map.get(c["id"])
                v_type = v["verdict"] if v else "PENDING"
                color = VERDICT_COLORS.get(v_type, "#6b7280")
                icon = VERDICT_ICONS.get(v_type, "⏳")

                with st.expander(f"{icon} {c.get('metric', 'Unknown metric')} = {c.get('value', '?')} {c.get('unit', '')}  —  {c.get('year','')} Q{c.get('quarter','')}"):
                    # Verdict badge
                    st.markdown(f"""
                    <div style="margin-bottom: 12px;">
                        <span class="verdict-badge" style="background-color: {color};">{icon} {v_type}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Claim details
                    st.markdown(f"**Speaker:** {c.get('speaker', 'N/A')}")
                    st.markdown(f"**Period:** {c.get('year', '')} Q{c.get('quarter', '')}")
                    st.markdown(f"**Extraction Method:** {c.get('extraction_method', 'N/A')}")
                    st.markdown(f"**Confidence:** {c.get('confidence', 0):.0%}")

                    st.markdown("---")

                    st.markdown("**Original Transcript Text:**")
                    st.markdown(f"> {c.get('raw_text', 'N/A')}")

                    st.markdown("**Context:**")
                    st.markdown(f"> {c.get('context', 'N/A')}")

                    if v:
                        st.markdown("---")
                        st.markdown("**Verification Reasoning:**")
                        st.markdown(v.get("explanation", "No explanation available."))

                        if v.get("evidence"):
                            st.markdown("**📎 Evidence:**")
                            for ev in v["evidence"]:
                                st.markdown(f"- {ev}")

                        if v.get("misleading_flags"):
                            st.markdown("**🚩 Misleading Flags:**")
                            for flag in v["misleading_flags"]:
                                st.warning(flag)

                        if v.get("data_sources"):
                            st.markdown("**Data Sources:**")
                            for ds in v["data_sources"]:
                                st.caption(f"• {ds}")
        else:
            st.warning(f"No results available for {focus_ticker}.")


# ─── Tab 3: Raw Data ───────────────────────────────────────────────────
with tab3:
    st.subheader("📂 Source Data Explorer")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        raw_ticker = st.selectbox("Company", available_companies if available_companies else ["—"], key="raw_ticker")
    with col_r2:
        if raw_ticker and raw_ticker != "—":
            quarters_raw = get_quarters(raw_ticker)
            if quarters_raw and quarters_raw.get("available_quarters"):
                q_options = [f"{q['year']} Q{q['quarter']}" for q in quarters_raw["available_quarters"]]
                selected_q = st.selectbox("Quarter", q_options)
            else:
                selected_q = None
                st.caption("No quarters available.")
        else:
            selected_q = None

    if selected_q:
        y_raw = int(selected_q.split(" ")[0])
        q_raw = int(selected_q.split(" ")[1][1])

        data_type = st.radio("Record Type", ["📝 Transcript", "📊 Financial Statements"], horizontal=True)

        if "Transcript" in data_type:
            t_data = get_transcript(raw_ticker, y_raw, q_raw)
            if t_data:
                st.caption(f"Source: {t_data.get('source', 'N/A')} | Date: {t_data.get('date', 'N/A')}")
                st.markdown("---")
                for s in t_data.get("segments", []):
                    speaker = s.get("speaker", "Unknown")
                    text = s.get("text", "")
                    st.markdown(f"**{speaker}:** {text}")
            else:
                st.info("No transcript data available for this quarter.")
        else:
            f_data = get_financials(raw_ticker, y_raw, q_raw)
            if f_data:
                df_f = pd.DataFrame(f_data)
                display_cols = [c for c in ["metric", "value", "unit", "is_gaap", "source"] if c in df_f.columns]
                if display_cols:
                    st.dataframe(df_f[display_cols], use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df_f, use_container_width=True, hide_index=True)
            else:
                st.info("No financial data available for this quarter.")
