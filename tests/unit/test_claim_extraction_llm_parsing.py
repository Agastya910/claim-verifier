import sys
import os
import unittest
from unittest.mock import patch, MagicMock

"""
Unit Test: Claim Extraction LLM Parsing
This test verifies the low-level logic for parsing LLM JSON responses and batching sentences.
Requires:
- No external dependencies
When to use it:
- Run this if regex parsing or JSON cleaning seems broken.
"""

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.claim_extraction.llm_extractor import extract_claims_llm, _clean_json_response, _batch_sentences

class TestLLMExtractor(unittest.TestCase):

    def test_clean_json_response_with_think(self):
        raw_text = """
        <think>
        I should extract the revenue and EPS.
        </think>
        [
            {"metric": "revenue", "stated_value": "100M", "unit": "dollars_millions"}
        ]
        """
        cleaned = _clean_json_response(raw_text)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["metric"], "revenue")

    def test_clean_json_response_malformed(self):
        raw_text = """
        Sure, here are the claims:
        [
            {"metric": "eps", "stated_value": "2.10", "unit": "dollars"},
        ]
        """
        cleaned = _clean_json_response(raw_text)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["stated_value"], "2.10")

    def test_batch_sentences(self):
        sentences = [
            {"sentence": "S" * 400, "speaker": "A", "role": "CEO"}, # ~100 tokens
            {"sentence": "S" * 400, "speaker": "A", "role": "CEO"}, # ~100 tokens
            {"sentence": "S" * 400, "speaker": "A", "role": "CEO"}, # ~100 tokens
        ]
        # Max tokens 150 -> should split into multiple batches
        batches = _batch_sentences(sentences, max_tokens=250, overlap=1)
        self.assertGreater(len(batches), 1)
        # Check overlap
        self.assertEqual(batches[1][0], sentences[1]) 

    @patch('litellm.completion')
    def test_extract_claims_llm(self, mock_completion):
        # Mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '[{"metric": "revenue", "stated_value": "15%", "unit": "percent", "raw_text": "Revenue grew by 15%", "speaker": "Tim Cook"}]'
        mock_completion.return_value = mock_response

        sentences = [{"sentence": "Revenue grew by 15%", "speaker": "Tim Cook", "role": "CEO"}]
        claims = extract_claims_llm(sentences, "AAPL", 4, 2023)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].metric, "revenue")
        self.assertEqual(claims[0].value, 15.0)
        self.assertEqual(claims[0].unit, "percent")

if __name__ == "__main__":
    unittest.main()
