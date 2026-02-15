"""
Unit tests for the smart retrieval engine.

Tests intent detection, query decomposition, scoring, and result sizing
without requiring a real database connection.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.rag.smart_retrieval import (
    _detect_verdict_intent,
    _detect_metrics,
    _detect_quarters,
    _detect_speaker,
    _is_comparison,
    _classify_intent,
    _extract_keywords,
    _keyword_score,
    _metric_match_score,
    _score_claim,
    _build_system_prompt,
)


# ─── Intent Detection Tests ────────────────────────────────────────────

class TestVerdictIntentDetection:
    def test_false_claim_query(self):
        assert _detect_verdict_intent("give me a false claim") == "FALSE"

    def test_lies_query(self):
        assert _detect_verdict_intent("were they lying about revenue?") == "FALSE"

    def test_verified_query(self):
        assert _detect_verdict_intent("which claims were verified?") == "VERIFIED"

    def test_misleading_query(self):
        assert _detect_verdict_intent("any misleading claims?") == "MISLEADING"

    def test_approximately_true_query(self):
        assert _detect_verdict_intent("which claims are approximately true?") == "APPROXIMATELY_TRUE"

    def test_unverifiable_query(self):
        assert _detect_verdict_intent("show unverifiable claims") == "UNVERIFIABLE"

    def test_no_verdict_intent(self):
        assert _detect_verdict_intent("what was the revenue?") is None

    def test_wrong_detection(self):
        assert _detect_verdict_intent("what went wrong with margins?") == "FALSE"


class TestMetricDetection:
    def test_revenue(self):
        assert "revenue" in _detect_metrics("what was the revenue?")

    def test_eps(self):
        assert "eps" in _detect_metrics("show me EPS in Q4")

    def test_earnings_per_share(self):
        assert "eps" in _detect_metrics("what was earnings per share?")

    def test_gross_margin(self):
        assert "gross_margin" in _detect_metrics("tell me about gross margin")

    def test_free_cash_flow(self):
        assert "free_cash_flow" in _detect_metrics("what about free cash flow?")

    def test_cloud_metric(self):
        assert "cloud" in _detect_metrics("how did Azure cloud do?")

    def test_ai_metric(self):
        assert "ai" in _detect_metrics("what about AI investment?")

    def test_no_metric(self):
        assert _detect_metrics("tell me about the company") == []

    def test_multiple_metrics(self):
        results = _detect_metrics("compare revenue and gross margin")
        assert "revenue" in results
        assert "gross_margin" in results


class TestQuarterDetection:
    def test_q4_2024(self):
        assert (2024, 4) in _detect_quarters("what happened in Q4 2024?")

    def test_2024_q3(self):
        assert (2024, 3) in _detect_quarters("show me 2024 Q3 results")

    def test_multiple_quarters(self):
        quarters = _detect_quarters("compare Q3 2024 vs Q4 2024")
        assert (2024, 3) in quarters
        assert (2024, 4) in quarters

    def test_no_quarter(self):
        assert _detect_quarters("what was the revenue?") == []


class TestSpeakerDetection:
    def test_ceo(self):
        assert _detect_speaker("what did the CEO say?") == "CEO"

    def test_cfo(self):
        assert _detect_speaker("CFO comments on margins") == "CFO"

    def test_no_speaker(self):
        assert _detect_speaker("what was the revenue?") is None


class TestComparisonDetection:
    def test_compare(self):
        assert _is_comparison("compare Q3 vs Q4 revenue") is True

    def test_trend(self):
        assert _is_comparison("show me the revenue trend") is True

    def test_change(self):
        assert _is_comparison("what changed in margins?") is True

    def test_not_comparison(self):
        assert _is_comparison("what was the revenue?") is False


class TestIntentClassification:
    def test_verdict_takes_priority(self):
        assert _classify_intent("FALSE", ["revenue"], [], None, False) == "VERDICT_FILTER"

    def test_comparison_with_quarters(self):
        assert _classify_intent(None, [], [(2024, 3), (2024, 4)], None, True) == "COMPARISON"

    def test_speaker_filter(self):
        assert _classify_intent(None, [], [], "CEO", False) == "SPEAKER_FILTER"

    def test_metric_lookup(self):
        assert _classify_intent(None, ["revenue"], [], None, False) == "METRIC_LOOKUP"

    def test_general_fallback(self):
        assert _classify_intent(None, [], [], None, False) == "GENERAL"


# ─── Keyword & Scoring Tests ───────────────────────────────────────────

class TestKeywordExtraction:
    def test_removes_stop_words(self):
        keywords = _extract_keywords("what is the revenue for this quarter?")
        assert "what" not in keywords
        assert "the" not in keywords
        assert "revenue" in keywords
        assert "quarter" in keywords

    def test_short_words_filtered(self):
        keywords = _extract_keywords("it is ok to be great")
        assert "ok" not in keywords  # length 2
        assert "great" in keywords

    def test_claim_related_stops(self):
        keywords = _extract_keywords("give me a false claim please")
        assert "give" not in keywords
        assert "claim" not in keywords
        assert "please" not in keywords
        assert "false" in keywords


class TestKeywordScore:
    def test_full_match(self):
        assert _keyword_score(["revenue", "growth"], "revenue growth was strong") == 1.0

    def test_partial_match(self):
        assert _keyword_score(["revenue", "growth"], "revenue was flat") == 0.5

    def test_no_match(self):
        assert _keyword_score(["revenue", "growth"], "margins improved") == 0.0

    def test_empty_keywords(self):
        assert _keyword_score([], "anything here") == 0.0


class TestMetricMatchScore:
    def test_exact_match(self):
        assert _metric_match_score("revenue", ["revenue"]) == 1.0

    def test_synonym_match(self):
        assert _metric_match_score("earnings per share (diluted)", ["eps"]) == 1.0

    def test_no_match(self):
        assert _metric_match_score("revenue", ["eps"]) == 0.0

    def test_empty_metrics(self):
        assert _metric_match_score("revenue", []) == 0.0


class TestScoreClaim:
    """Test the composite scoring function with mock claim/verdict objects."""

    @staticmethod
    def _make_claim(raw_text="Revenue was $100B", metric="revenue",
                    value=100.0, unit="B", year=2024, quarter=4, speaker="CEO"):
        claim = MagicMock()
        claim.raw_text = raw_text
        claim.metric = metric
        claim.value = value
        claim.unit = unit
        claim.year = year
        claim.quarter = quarter
        claim.speaker = speaker
        return claim

    @staticmethod
    def _make_verdict(verdict="VERIFIED", explanation="Matches 10-K filing",
                      evidence=["Revenue was $100B in the filing"]):
        v = MagicMock()
        v.verdict = verdict
        v.explanation = explanation
        v.evidence = evidence
        return v

    def test_perfect_match_scores_highest(self):
        claim = self._make_claim()
        verdict = self._make_verdict()
        score = _score_claim(
            claim, verdict,
            keywords=["revenue"],
            detected_metrics=["revenue"],
            target_verdict=None,
            target_quarters=[],
            max_year=2024, max_quarter=4,
        )
        # Should have high keyword + metric + recency + evidence scores
        assert score > 0.5

    def test_verdict_match_boosts_score(self):
        claim = self._make_claim()
        verdict_false = self._make_verdict(verdict="FALSE")

        score_with = _score_claim(
            claim, verdict_false,
            keywords=["revenue"],
            detected_metrics=[],
            target_verdict="FALSE",
            target_quarters=[],
            max_year=2024, max_quarter=4,
        )
        score_without = _score_claim(
            claim, verdict_false,
            keywords=["revenue"],
            detected_metrics=[],
            target_verdict=None,
            target_quarters=[],
            max_year=2024, max_quarter=4,
        )
        assert score_with > score_without

    def test_no_verdict_claim_scores_lower(self):
        claim = self._make_claim()
        verdict = self._make_verdict()

        score_with_verdict = _score_claim(
            claim, verdict,
            keywords=["revenue"],
            detected_metrics=[],
            target_verdict=None,
            target_quarters=[],
            max_year=2024, max_quarter=4,
        )
        score_no_verdict = _score_claim(
            claim, None,
            keywords=["revenue"],
            detected_metrics=[],
            target_verdict=None,
            target_quarters=[],
            max_year=2024, max_quarter=4,
        )
        assert score_with_verdict > score_no_verdict


# ─── Prompt Generation Tests ───────────────────────────────────────────

class TestSystemPrompt:
    def test_verdict_filter_prompt(self):
        prompt = _build_system_prompt("VERDICT_FILTER", {"verdict_type": "FALSE"})
        assert "FALSE" in prompt
        assert "fabricate" not in prompt or "Do not fabricate" in prompt

    def test_metric_lookup_prompt(self):
        prompt = _build_system_prompt("METRIC_LOOKUP", {"detected_metrics": ["revenue"]})
        assert "revenue" in prompt

    def test_comparison_prompt(self):
        prompt = _build_system_prompt("COMPARISON", {})
        assert "compare" in prompt.lower() or "quarter" in prompt.lower()

    def test_general_prompt(self):
        prompt = _build_system_prompt("GENERAL", {})
        assert "comprehensive" in prompt.lower() or "synthesiz" in prompt.lower()
