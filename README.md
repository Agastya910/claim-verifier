# Earnings Claim Verifier

> 🚨 **IMPORTANT** 🚨
>
> This README is just a canonical summary and deployment instructions.
> **[PLEASE READ THE DESIGN OVERVIEW DOCUMENT](Design%20Overview.pdf)**.

Automated financial claim verification system that cross-references earnings call transcripts against official SEC EDGAR data using LLMs and RAG.

## Architecture

```text
                      ┌────────────────┐
                      │  Data Sources  |
                      └───────┬────────┘
                              |
                  ┌───────────▼───────────┐
                  │                       │
          ┌───────▼───────┐       ┌───────▼───────┐
          │  HuggingFace  │       │   SEC EDGAR   │
          │ (Transcripts) │       │ (Financials)  │
          └───────┬───────┘       └───────┬───────┘
                  │                       │
                  └───────────┬───────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Data Ingestion   │
                    │ (uv sync / edgar) │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Claim Extraction │ (GLiNER + LLM)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Verification RAG  │ (PgVector + Hybrid Search)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │    Streamlit UI   │ (Dashboard / Insights)
                    └───────────────────┘
```

## Quick Start (Local)

1. **Install uv**:
   ```bash
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. **Setup environment**:
   ```bash
   uv sync
   cp .env.example .env
   # Fill in API keys in .env
   ```
3. **Run App**:
   ```bash
   uv run streamlit run src/ui/app.py
   ```

## Deployment

The system is deployed as a single service on **Streamlit Cloud** connected to a **Neon PostgreSQL** database with `pgvector` enabled.

- **Live Demo**: [https://claim-verifier-agastya.streamlit.app/](https://claim-verifier-agastya.streamlit.app/)
- **Database**: Neon (Serverless Postgres)
- **Dependency Management**: `uv` (used internally by Streamlit Cloud via `requirements.txt`)
- **Configuration**: Uses `st.secrets` for cloud deployment and `.env` for local development.

> **Note**: Docker files (`Dockerfile`, `docker-compose.yml`) are included for reference but the primary deployment method is Streamlit Cloud.

## Key Design Decisions

- **GLiNER Pre-filtering**: Used to reduce LLM token costs by identifying financial entities before extraction.
- **Hybrid RAG**: Combines sparse (SPLADE) and dense (BGE-M3) retrieval in PostgreSQL for high-precision metric lookup.
- **Deterministic-First**: Programmatic comparison for standard GAAP metrics (Revenue, EPS) before falling back to LLM reasoning.
- **Explainable AI**: Verification results include "reasoning" steps showing exactly how numbers were calculated.

## Data Sources

- **Transcripts**: HuggingFace (`Bose345/sp500_earnings_transcripts`)
- **Financial Data**: [SEC EDGAR](https://www.sec.gov/edgar/sec-api-documentation) via `edgartools`.

## Future Improvements

- **Fully Automated Ingestion Pipeline**: Allow users to submit arbitrary company tickers and date ranges to trigger a complete end-to-end analysis (transcript fetching, SEC filing retrieval, claim extraction, and verification) without manual intervention. Workers already implemented, not deployed in production due to infra limitations.

- **Comprehensive Benchmarking**: Verify system performance across standard financial datasets (e.g., Fin-Fact) using metrics:
  - **Claim Extraction**: Precision, Recall, and F1-score to measure identification of quantitative claims.
  - **RAG Retrieval**: nDCG@10 and Mean Reciprocal Rank (MRR) to assess evidence retrieval quality.
  - **Verification Accuracy**: Rate of correct verdicts (Supported/Refuted) compared to human auditor ground truth.
  - **Hallucination Rate**: Frequency of generated numbers not present in source text.
- **Global Coverage**: Expand beyond US SEC-filing companies.
- **Multimodal Verification**: Incorporate slide deck (PDF/Image) parsing into the verification pipeline.
- **Advanced Reranking**: Fine-tune cross-encoders specifically on financial claim pairs.
- **Real-time Monitoring**: Alert system for live earnings calls.
