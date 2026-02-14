import json
import logging
import re
import uuid
import litellm
from typing import List, Optional
from src.models import Claim
from src.config import MODEL_CONFIGS, OLLAMA_BASE_URL, OLLAMA_API_KEY, validate_ollama_config

logger = logging.getLogger(__name__)

def get_litellm_model_string(tier: str) -> str:
    """
    Maps configuration tiers to LiteLLM model strings.
    """
    mapping = {
        "groq_backup": "groq/llama-3.3-70b-versatile",
        "premium_claude": "anthropic/claude-3-5-sonnet-20240620",
        "premium_openai": "openai/gpt-4o",
        "local_qwq": "ollama/qwq:32b",
        "local_small": "ollama/deepseek-r1:7b"
    }
    # Check MODEL_CONFIGS first for dynamic values (like default), then fallback to mapping
    return MODEL_CONFIGS.get(tier, mapping.get(tier, MODEL_CONFIGS["default"]))

def _clean_json_response(response_text: str) -> list:
    """
    Cleans and parses the LLM response to extract a JSON array.
    """
    # Remove thoughts or other non-JSON content (e.g. from DeepSeek)
    if "<think>" in response_text:
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    
    # Try to find the JSON array in the text
    # Looking for a block that starts with [ and ends with ]
    match = re.search(r'(\[.*\])', response_text, re.DOTALL)
    if match:
        response_text = match.group(1).strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # More aggressive cleanup if needed
        logger.warning("JSON parsing failed, attempting aggressive cleanup")
        # Remove trailing commas before closing brackets and braces
        # Handle cases like [..., ] or {..., }
        response_text = re.sub(r',\s*\]', ']', response_text)
        response_text = re.sub(r',\s*\}', '}', response_text)
        
        # Also handle potential markdown code blocks if regex didn't strip them
        response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Aggressive JSON cleanup failed: {e}")
            # Try one last thing: find all {} blocks and wrap them in []
            try:
                objects = re.findall(r'(\{[^{}]*\})', response_text)
                if objects:
                    return [json.loads(obj) for obj in objects]
            except Exception:
                pass
            return []

def _batch_sentences(sentences: list[dict], max_tokens: int = 2000, overlap: int = 2) -> list[list[dict]]:
    """
    Batches sentences to stay within token limits. 
    Uses a rough estimation (4 characters per token).
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    for i, s in enumerate(sentences):
        # Rough estimation of tokens
        s_tokens = len(s["sentence"]) // 4 + 20 # 20 for metadata
        
        if current_tokens + s_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            # Overlap: keep the last 'overlap' sentences for context
            current_batch = current_batch[-overlap:] if len(current_batch) > overlap else []
            current_tokens = sum(len(x["sentence"]) // 4 + 20 for x in current_batch)
            
        current_batch.append(s)
        current_tokens += s_tokens
        
    if current_batch:
        batches.append(current_batch)
        
    return batches

def extract_claims_llm(
    sentences: list[dict], 
    ticker: str, 
    quarter: int, 
    year: int, 
    model_tier: str = "default"
) -> list[Claim]:
    """
    Extracts quantitative claims from filtered sentences using an LLM.
    """
    if not sentences:
        return []

    model_string = get_litellm_model_string(model_tier)
    
    # Fail fast if config is missing for Ollama
    if "ollama" in model_string:
         validate_ollama_config()

    batches = _batch_sentences(sentences, max_tokens=1800)  # Balanced for context and speed
    all_claims = []

    for batch_idx, batch in enumerate(batches):
        formatted_sentences = "\n".join([
            f"[{s['speaker']} ({s['role']})]: {s['sentence']}" 
            for s in batch
        ])
        
        # --- MANDATORY PAUSE ---
        # 10 seconds is ideal for Ollama Cloud to clear buffer
        import time
        if batch_idx > 0:
            logger.info(f"Clearing buffer... waiting 10s before batch {batch_idx}")
            time.sleep(10)
        # -------------------------------

        prompt = f"""
You are a financial analyst extracting quantitative claims from an earnings call.

TRANSCRIPT SENTENCES (Company: {ticker}, {quarter} {year}):
---
{formatted_sentences}
---

For each sentence, determine:
1. SELECTION: Does this sentence contain a verifiable quantitative claim? Skip opinions, forward-looking guidance, and qualitative statements.
2. DISAMBIGUATION: If the claim references something ambiguous, resolve it using context. If unresolvable, mark confidence as "low".
3. DECOMPOSITION: Break complex sentences into standalone verifiable claims.

For EACH quantitative claim found, return a JSON object:
{{
  "metric": "the financial metric (revenue, net_income, eps, operating_margin, etc.)",
  "claim_type": "absolute_value | percentage_growth | comparison | ratio | vague_quantitative",
  "stated_value": "the exact number or range for vague claims (e.g. '10-99' for double-digit)",
  "unit": "percent | dollars_millions | dollars_billions | count | ratio | basis_points",
  "period": "YoY | QoQ | TTM | annual | quarterly | unspecified",
  "is_gaap": true/false,
  "is_forward_looking": true/false,
  "hedging_language": true/false,
  "raw_text": "exact sentence from transcript",
  "speaker": "speaker name"
}}

Return ONLY a JSON array. If no quantitative claims exist, return [].
Do NOT include qualitative statements without numbers.
"""
        
        retries = 5  # Total attempts per batch
        for attempt in range(retries):
            try:
                # 1. Try the request
                kwargs = {
                    "model": model_string,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "timeout": 300
                }

                if "ollama" in model_string:
                    kwargs["api_base"] = OLLAMA_BASE_URL
                    kwargs["api_key"] = OLLAMA_API_KEY

                response = litellm.completion(**kwargs)
                
                content = response.choices[0].message.content
                raw_claims = _clean_json_response(content)
                
                for rc in raw_claims:
                    try:
                        # Map raw claim to Claim model
                        # Handle invalid stated values like '15.215.6' or empty strings
                        stated_value = rc.get("stated_value", "0")
                        
                        # Clean the value - first, handle possible thousands separators
                        value_str = str(stated_value).strip()
                        
                        # Handle cases like '53.893.12' - determine if it's thousands separator or decimal
                        if value_str.count('.') > 1:
                            # If last decimal is followed by exactly 2 digits, likely currency (e.g., 53.893.12 is 53893.12)
                            parts = value_str.split('.')
                            if len(parts[-1]) == 2:
                                # Combine all parts except last, replace remaining dots with nothing
                                integer_part = ''.join(parts[:-1]).replace('.', '').replace(',', '')
                                decimal_part = parts[-1]
                                cleaned_value = f"{integer_part}.{decimal_part}"
                            else:
                                # If multiple decimals but not standard currency format, remove all decimals
                                logger.warning(f"Unusual number format: '{stated_value}', removing all decimal points")
                                cleaned_value = re.sub(r'[^\d]', '', value_str)
                        else:
                            # Normal case with single decimal or integer
                            cleaned_value = re.sub(r'[^\d.]', '', value_str)
                        
                        # Handle empty string case
                        if not cleaned_value or cleaned_value == '.':
                            continue
                        
                        try:
                            numeric_value = float(cleaned_value)
                        except (ValueError, TypeError):
                            logger.warning(f"Failed to convert '{stated_value}' to float, skipping")
                            continue
                        
                        claim = Claim(
                            id=str(uuid.uuid4()),
                            ticker=ticker,
                            quarter=quarter,
                            year=year,
                            speaker=rc.get("speaker", "Unknown"),
                            metric=rc.get("metric", "Unknown"),
                            value=numeric_value,
                            unit=rc.get("unit", "unknown"),
                            period=rc.get("period", "unspecified"),
                            is_gaap=bool(rc.get("is_gaap", True)),
                            is_forward_looking=bool(rc.get("is_forward_looking", False)),
                            hedging_language=str(rc.get("hedging_language", "false")),
                            raw_text=rc.get("raw_text", ""),
                            extraction_method="llm",
                            confidence=0.9 if not rc.get("vague_quantitative") else 0.5,
                            context=formatted_sentences # Providing batch context as context
                        )
                        all_claims.append(claim)
                    except Exception as e:
                        logger.warning(f"Failed to parse individual claim: {e}")
                
                break  # Success, break retry loop
                
            except Exception as e:
                error_str = str(e).lower()
                
                # 2. Check if we hit the final attempt
                if attempt == retries - 1:
                    logger.error(f"Final retry failed for batch {batch_idx}. Waiting 60s for full reset.")
                    time.sleep(60)
                    continue
                
                # 3. Handle Rate Limits specifically
                if "429" in error_str or "rate_limit" in error_str:
                    wait_time = (2 ** attempt) + 2  # Exponential: 2, 4, 8, 16...
                    logger.warning(f"Rate limit hit. Fast retry {attempt+1}/{retries} in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # 4. Handle other errors (Network, etc.)
                    logger.error(f"Unexpected error: {e}. Retrying in 5s...")
                    time.sleep(5)

    return all_claims
