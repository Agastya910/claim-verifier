# /api/ask Endpoint - Quick Reference Guide

## Endpoint Details

**URL**: `POST /api/ask`

**Request Body**:

```json
{
  "ticker": "AAPL",
  "question": "What was the revenue in Q4 2024?"
}
```

**Response**:

```json
{
  "answer": "Based on the verified claims...",
  "claim_ids": ["claim_abc123", "claim_def456"],
  "num_claims_used": 2
}
```

## How It Works

### 1. Keyword Extraction

The endpoint extracts meaningful keywords from your question by:

- Tokenizing the question into words
- Removing common stop words (the, a, an, is, was, etc.)
- Keeping only words with length > 2 characters
- Using up to 5 keywords for the search

**Example**:

- Question: "What was the revenue in Q4 2024?"
- Keywords: `["revenue", "2024"]`

### 2. Database Search

Performs ILIKE (case-insensitive) search on:

- `claims.raw_text` - The original claim text from the transcript
- `claims.metric` - The metric name (e.g., "revenue", "gross_margin")
- `verdicts.explanation` - The verification reasoning
- `verdicts.evidence` - The evidence supporting the verdict

### 3. Fallback Mechanism

If no keyword matches are found:

- Returns the 10 most recent claims for the company
- Ordered by year and quarter (most recent first)

### 4. Context Building

Each retrieved claim is formatted as:

```
Claim: <original transcript text>
Metric: <metric_name> = <value> <unit>
Quarter: <year> Q<quarter>
Speaker: <speaker_name>
Verdict: <VERIFIED|FALSE|MISLEADING|etc>
Reasoning: <verification explanation>
Evidence: <evidence_1>; <evidence_2>; ...
```

### 5. LLM Processing

The structured context is sent to the LLM with strict instructions:

- Answer ONLY using the provided verified claims
- Do not fabricate data
- Say "not enough information" if the answer isn't in the context

## Example Usage

### Python (httpx)

```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/ask",
    json={
        "ticker": "AAPL",
        "question": "Was there any misleading information about revenue?"
    },
    timeout=30.0
)

result = response.json()
print(result["answer"])
print(f"Used {result['num_claims_used']} claims")
```

### cURL

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "question": "What did they say about gross margin?"
  }'
```

### JavaScript (fetch)

```javascript
const response = await fetch("http://localhost:8000/api/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    ticker: "AAPL",
    question: "What was the revenue in Q4 2024?",
  }),
});

const result = await response.json();
console.log(result.answer);
console.log(`Used ${result.num_claims_used} claims`);
```

## Question Types That Work Well

### ✅ Specific Metric Queries

- "What was the revenue in Q4 2024?"
- "Did they mention gross margin?"
- "What did they say about operating expenses?"

### ✅ Verification Queries

- "Were there any false claims about revenue?"
- "Was there any misleading information?"
- "What claims were verified?"

### ✅ Speaker Queries

- "What did the CEO say about growth?"
- "What claims did the CFO make?"

### ✅ Time-Based Queries

- "What happened in Q4 2024?"
- "What were the most recent claims?"

### ⚠️ Questions That May Not Work Well

- Very broad questions: "Tell me everything about the company"
  - _Will fallback to 10 most recent claims_
- Questions about data not in claims: "What is the stock price?"
  - _Will respond with "not enough information"_
- Questions requiring calculations: "What is the year-over-year growth?"
  - _May work if claims contain this information_

## Response Interpretation

### Success Response

```json
{
  "answer": "According to the verified claims, Apple reported revenue of $123.4B in Q4 2024...",
  "claim_ids": ["claim_abc123", "claim_def456"],
  "num_claims_used": 2
}
```

### No Data Available

```json
{
  "answer": "No verified claims found for INVALID. Please make sure the company has been analyzed.",
  "claim_ids": [],
  "num_claims_used": 0
}
```

### Insufficient Information

```json
{
  "answer": "The available verified claims do not contain enough information to answer this question.",
  "claim_ids": ["claim_abc123"],
  "num_claims_used": 1
}
```

## Performance Characteristics

- **Speed**: ~100-500ms for keyword search + LLM inference
- **Scalability**: Handles thousands of claims efficiently
- **Determinism**: Same question → same claims (for same data state)
- **Reliability**: No external dependencies beyond database and LLM

## Limitations

1. **Keyword-Based**: Simple tokenization, no semantic understanding
2. **No Ranking**: Results ordered by recency, not relevance
3. **Limited Context**: Maximum 10 claims per query
4. **No Aggregation**: Cannot perform calculations across claims
5. **No Synonyms**: "revenue" won't match "sales" unless both are in the data

## Future Improvements

Consider these enhancements if needed:

- Better keyword extraction (NLP libraries)
- Synonym matching (revenue = sales = turnover)
- TF-IDF ranking for relevance
- Query result caching
- Metric normalization

## Troubleshooting

### "No verified claims found"

- Check if the company ticker is correct
- Verify the company has been ingested and analyzed
- Use `/api/companies` to see available companies

### "Not enough information to answer"

- Try more specific questions with exact metric names
- Check what claims exist using `/api/results/{ticker}`
- Rephrase the question with different keywords

### Slow responses

- Check database indexes on ticker, year, quarter
- Consider reducing max claims from 10 to 5
- Monitor LLM API latency

### Unexpected answers

- Review the claim_ids returned
- Check the actual claims using `/api/results/{ticker}`
- Verify the LLM is using the correct model tier
