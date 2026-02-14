# Earnings Claim Verifier

Automated financial claim verification system that cross-references earnings call transcripts against official SEC EDGAR data using LLMs and RAG.

## Architecture

```text
                                  ┌────────────────┐
                                  │  Data Sources  │
                                  └───────┬────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  │                       │                       │
          ┌───────▼───────┐       ┌───────▼───────┐       ┌───────▼───────┐
          │   Finnhub     │       │   SEC EDGAR   │       │  HuggingFace  │
          │ (Transcripts) │       │ (Financials)  │       │   (Fallback)  │
          └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
                  │                       │                       │
                  └───────────┬───────────┴───────────────────────┘
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

## Quick Start (with Docker)

1. Clone the repository.
2. Create a `.env` file (see `.env.example`).
3. Start the system:
   ```bash
   docker compose up
   ```
4. Visit the UI: [http://localhost:8501](http://localhost:8501)
5. Backend API: [http://localhost:8000/docs](http://localhost:8000/docs)

## Local Development (without Docker)

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
3. **Run Services**:
   - Backend: `uv run uvicorn src.api.routes:app --reload`
   - UI: `uv run streamlit run src/ui/app.py`

## API Documentation

- `POST /api/ingest`: Trigger ingestion for ticker/quarters.
- `POST /api/extract-claims`: Extract claims from ingested transcripts.
- `POST /api/verify`: Verify specific claims against financial data.
- `GET /api/results/{ticker}`: Get all verification results for a company.

## Key Design Decisions

- **GLiNER Pre-filtering**: Used to reduce LLM token costs by identifying financial entities before extraction.
- **Hybrid RAG**: Combines sparse (SPLADE) and dense (BGE-M3) retrieval in PostgreSQL for high-precision metric lookup.
- **Deterministic-First**: Programmatic comparison for standard GAAP metrics (Revenue, EPS) before falling back to LLM reasoning.
- **Explainable AI**: Verification results include "reasoning" steps showing exactly how numbers were calculated.

## Data Sources

- **Transcripts**: [Finnhub API](https://finnhub.io/)
- **Financial Data**: [SEC EDGAR](https://www.sec.gov/edgar/sec-api-documentation) via `edgartools`.

## Future Improvements

- **Global Coverage**: Expand beyond US SEC-filing companies.
- **Multimodal Verification**: Incorporate slide deck (PDF/Image) parsing into the verification pipeline.
- **Advanced Reranking**: Fine-tune cross-encoders specifically on financial claim pairs.
- **Real-time Monitoring**: Alert system for live earnings calls.

---

Cloud Demo: [https://claim-verifier.railway.app](https://claim-verifier.railway.app)
