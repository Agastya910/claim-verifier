import json
import logging
import time
from typing import Optional

import litellm
from sqlalchemy.orm import Session

from src.config import MODEL_CONFIGS
from src.data_ingest.storage import save_verdicts
from src.models import Claim, Verdict

logger = logging.getLogger(__name__)

def get_litellm_model_string(tier: str) -> str:
    """
    Maps configuration tiers to LiteLLM model strings.
    Same as in llm_extractor.py to ensure consistency.
    """
    mapping = {
        "default": "ollama_chat/deepseek-v3.1:671b-cloud", # Use DeepSeek-V3.1 via Ollama
        "groq_backup": "groq/llama-3.3-70b-versatile",
        "premium_claude": "anthropic/claude-3-5-sonnet-20240620",
        "premium_openai": "openai/gpt-4o",
        "local_qwq": "ollama/qwq:32b",
        "local_small": "ollama/deepseek-r1:7b"
    }
    return mapping.get(tier, MODEL_CONFIGS.get(tier, mapping["default"]))

def verify_with_llm(claim: Claim, context: str, db_session: Session, model_tier: str = "default") -> Verdict:
    """
    Verifies a financial claim using an LLM model and specified context.
    Retries up to 5 times on failure with exponential backoff.
    Uses same configuration as extraction (Ollama) for consistency.
    """
    model_string = get_litellm_model_string(model_tier)
    
    prompt = f"""
    You are a senior financial analyst verifying earnings call claims against official financial data.

    CLAIM TO VERIFY:
    - Text: "{claim.raw_text}"
    - Speaker: {claim.speaker}
    - Company: {claim.ticker}, {claim.quarter} {claim.year}
    - Metric: {claim.metric}
    - Claimed Value: {claim.value} {claim.unit}
    - Period: {claim.period}
    - GAAP: {claim.is_gaap}
    - Hedging Language: {claim.hedging_language}

    OFFICIAL FINANCIAL DATA AND CONTEXT:
    {context}

    INSTRUCTIONS — Follow these steps exactly:

    STEP 1 - IDENTIFY: What exact financial metric is being claimed? Map it to the official data fields.
    STEP 2 - RETRIEVE: Find exact numbers from official data for all relevant periods. Quote them.
    STEP 3 - COMPUTE: Calculate the actual value. Show your math.
    STEP 4 - COMPARE: Compare claimed vs actual. State the difference.
    STEP 5 - TOLERANCE: Apply tolerance (hedging: ±2%/±5%; precise: ±0.5%/±1%).
    STEP 6 - MISLEADING CHECK: Evaluate framing:
      - Cherry-picking (positive highlighted, negative hidden)?
      - GAAP vs Non-GAAP divergence?
      - Period-shopping (YoY because QoQ looks bad)?
      - Base-effect (huge % off tiny base)?
      - Omission (acquisition growth as organic)?
    STEP 7 - VERDICT: VERIFIED | APPROXIMATELY_TRUE | FALSE | MISLEADING | UNVERIFIABLE
    STEP 8 - EVIDENCE: List exact strings/numbers from the context that support your verdict.

    Respond with ONLY valid JSON:
    {{
      "verdict": "...",
      "actual_value": 123.45,
      "claimed_value": 123.45,
      "difference": 0.0,
      "explanation": "Step-by-step reasoning...",
      "misleading_flags": [],
      "confidence": "high|medium|low",
      "data_sources_used": [],
      "evidence": ["exact quote 1", "exact quote 2"]
    }}
    """

    max_retries = 5
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.info(f"LLM Verification attempt {attempt + 1} for claim {claim.id} using {model_string}")
            
            response = litellm.completion(
                model=model_string,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                api_base="http://127.0.0.1:11434",
                timeout=300
            )

            content = response.choices[0].message.content
            # Clean up potential markdown blocks if LLM didn't strictly follow JSON-only instruction
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            verdict = Verdict(
                claim_id=claim.id,
                verdict=data.get("verdict", "UNVERIFIABLE"),
                actual_value=float(data.get("actual_value")) if data.get("actual_value") is not None else None,
                claimed_value=float(data.get("claimed_value", claim.value)),
                difference=float(data.get("difference")) if data.get("difference") is not None else None,
                explanation=data.get("explanation", ""),
                misleading_flags=data.get("misleading_flags", []),
                confidence=1.0 if data.get("confidence") == "high" else 0.5,
                data_sources=data.get("data_sources_used", []),
                evidence=data.get("evidence", [])
            )
            
            # Save to DB
            save_verdicts(db_session, [verdict])
            return verdict

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            if attempt == max_retries - 1:
                logger.error(f"Final retry failed for claim {claim.id}. Waiting 60s for full reset.")
                time.sleep(60)
                continue
                
            if "429" in error_str or "rate_limit" in error_str:
                wait_time = (2 ** attempt) + 2
                logger.warning(f"Rate limit hit. Fast retry {attempt+1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Unexpected error: {e}. Retrying in 5s...")
                time.sleep(5)

    # Final fallback if all retries fail
    logger.error(f"Failing LLM verification for claim {claim.id} after {max_retries} attempts: {last_error}")
    fallback_verdict = Verdict(
        claim_id=claim.id,
        verdict="UNVERIFIABLE",
        actual_value=None,
        claimed_value=claim.value,
        difference=None,
        explanation=f"LLM verification failed after {max_retries} retries. Error: {str(last_error)}",
        misleading_flags=[],
        confidence=0.0,
        data_sources=[]
    )
    save_verdicts(db_session, [fallback_verdict])
    return fallback_verdict
