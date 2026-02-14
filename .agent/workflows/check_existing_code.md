---
description: Before writing new code, check if existing codebase already has that functionality
---

Before writing any new code, follow these steps to avoid duplicating existing implementations:

1. **Search for existing implementations first**:
   - Use grep to search for relevant function names, class names, or keywords
   - Check the relevant `src/` subdirectory for files that might contain the functionality
   - Look at imports in related files to discover utility functions

2. **Verify correctness of existing code**:
   - If existing code exists, check if it is correct for the current use case
   - Run any existing unit tests related to that code: `python -m pytest tests/ -k <test_name>`
   - If the existing code is correct, use it directly instead of rewriting

3. **Prefer modifying over rewriting**:
   - If existing code needs small changes, modify it rather than deleting and rewriting
   - Use targeted edits (replace_file_content) rather than full file rewrites when possible
   - Document why changes were needed

4. **Key directories to check**:
   - `src/data_ingest/` — Data fetching and storage
   - `src/claim_extraction/` — Claim extraction pipeline
   - `src/verifier/` — Verification logic
   - `src/rag/` — RAG retrieval and indexing
   - `src/db/` — Database schema and connections
   - `src/api/` — API routes
   - `src/ui/` — Streamlit UI
   - `scripts/` — Utility scripts
