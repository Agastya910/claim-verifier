import sys
import os
from datetime import date
import unittest
from unittest.mock import patch, MagicMock

"""
Unit Test: Claim Extraction Pipeline (Mocked)
This test verifies the orchestration of the extraction pipeline (GLiNER -> LLM) using mocks.
Requires:
- No external dependencies
When to use it:
- Run this to verify the flow of data through the extraction steps.
"""

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Transcript, TranscriptSegment
from src.claim_extraction.pipeline import extract_all_claims

class TestFullPipeline(unittest.TestCase):

    @patch('src.claim_extraction.entity_filter.GLiNER')
    @patch('litellm.completion')
    def test_pipeline_flow(self, mock_litellm, mock_gliner_class):
        # 1. Mock GLiNER
        mock_gliner = MagicMock()
        mock_gliner.predict_entities.return_value = [
            {'start': 0, 'end': 7, 'text': 'Revenue', 'label': 'financial metric', 'score': 0.9}
        ]
        mock_gliner_class.from_pretrained.return_value = mock_gliner

        # 2. Mock LiteLLM
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '[{"metric": "revenue", "stated_value": "15%", "unit": "percent", "period": "year over year", "raw_text": "Revenue grew by 15% year over year.", "speaker": "Tim Cook"}]'
        mock_litellm.return_value = mock_response

        # 3. Create sample transcript
        segments = [
            TranscriptSegment(
                speaker="Tim Cook",
                role="CEO",
                text="Revenue grew by 15% year over year. This was a great result."
            )
        ]
        transcript = Transcript(
            ticker="AAPL",
            year=2023,
            quarter=4,
            date=date(2023, 10, 27),
            segments=segments
        )

        # 4. Run pipeline
        print("Running full pipeline...")
        claims = extract_all_claims(transcript)

        # 5. Assertions
        self.assertEqual(len(claims), 1)
        claim = claims[0]
        self.assertEqual(claim.metric, "revenue")
        self.assertEqual(claim.period, "YoY") # Normalized
        self.assertTrue(claim.is_gaap) # Default since no adjusted-word
        self.assertEqual(claim.ticker, "AAPL")
        self.assertIn("Tim Cook: Revenue grew by 15% year over year.", claim.context) # Enriched context
        
        print("\nPipeline verification successful!")

if __name__ == "__main__":
    unittest.main()
