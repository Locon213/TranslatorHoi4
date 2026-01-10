"""Tests for cost_tracker.py - Token usage and cost tracking."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class TestTokenUsage:
    """Test TokenUsage dataclass."""
    
    def test_token_usage_creation(self):
        """Test creating TokenUsage instance."""
        from translatorhoi4.translator.cost import TokenUsage

        usage = TokenUsage(prompt_tokens=100, completion_tokens=200)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 200
        assert usage.total_tokens == 300

    def test_token_usage_addition(self):
        """Test adding TokenUsage instances."""
        from translatorhoi4.translator.cost import TokenUsage

        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=200)
        usage2 = TokenUsage(prompt_tokens=50, completion_tokens=100)

        combined = usage1 + usage2
        assert combined.prompt_tokens == 150
        assert combined.completion_tokens == 300
        assert combined.total_tokens == 450

    def test_token_usage_iadd(self):
        """Test in-place addition of TokenUsage."""
        from translatorhoi4.translator.cost import TokenUsage

        usage = TokenUsage(prompt_tokens=100, completion_tokens=200)
        usage += TokenUsage(prompt_tokens=50, completion_tokens=100)
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 300
        assert usage.total_tokens == 450

    def test_token_usage_to_dict(self):
        """Test converting TokenUsage to dictionary."""
        from translatorhoi4.translator.cost import TokenUsage

        usage = TokenUsage(prompt_tokens=100, completion_tokens=200)
        result = usage.to_dict()
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 200
        assert result["total_tokens"] == 300

    def test_token_usage_from_dict(self):
        """Test creating TokenUsage from dictionary."""
        from translatorhoi4.translator.cost import TokenUsage
        
        data = {"prompt_tokens": 150, "completion_tokens": 250}
        usage = TokenUsage.from_dict(data)
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 250
        assert usage.total_tokens == 400


class TestCostTracker:
    """Test CostTracker functionality."""
    
    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CostTracker instance for testing."""
        from translatorhoi4.translator.cost import CostTracker

        # Create a fresh instance bypassing singleton
        tracker = CostTracker.__new__(CostTracker)
        tracker.__init__()
        tracker._load()
        yield tracker

    def test_record_usage(self, tracker):
        """Test recording token usage."""
        from translatorhoi4.translator.cost import TokenUsage

        usage = TokenUsage(prompt_tokens=1000, completion_tokens=2000)
        entry = tracker.record_usage(
            provider="openai",
            model="gpt-4",
            usage=usage,
            source_lang="english",
            target_lang="russian",
        )

        assert entry.provider == "openai"
        assert entry.model == "gpt-4"
        assert entry.usage.total_tokens == 3000
        assert entry.total_cost > 0

    def test_get_summary(self, tracker):
        """Test getting cost summary."""
        from translatorhoi4.translator.cost import TokenUsage

        # Record some usage
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=2000)
        tracker.record_usage(provider="openai", model="gpt-4", usage=usage)

        summary = tracker.get_summary()
        assert summary["total_tokens"] == 3000
        assert summary["total_requests"] == 1
        assert "openai" in summary["providers"]

    def test_get_provider_cost(self, tracker):
        """Test getting cost for specific provider."""
        from translatorhoi4.translator.cost import TokenUsage

        usage = TokenUsage(prompt_tokens=1000, completion_tokens=2000)
        tracker.record_usage(provider="openai", model="gpt-4", usage=usage)

        cost = tracker.get_provider_cost("openai")
        assert cost["provider"] == "openai"
        assert cost["total_tokens"] == 3000
        assert cost["requests"] == 1

    def test_clear(self, tracker):
        """Test clearing all tracking data."""
        from translatorhoi4.translator.cost import TokenUsage

        usage = TokenUsage(prompt_tokens=1000, completion_tokens=2000)
        tracker.record_usage(provider="openai", model="gpt-4", usage=usage)

        assert len(tracker._entries) > 0

        tracker.clear()

        assert len(tracker._entries) == 0

    def test_set_cost_per_million(self, tracker):
        """Test setting custom cost per million tokens."""
        tracker.set_cost_per_million("custom", input_cost=1.0, output_cost=5.0)

        # Record usage with custom provider
        from translatorhoi4.translator.cost import TokenUsage
        usage = TokenUsage(prompt_tokens=1000000, completion_tokens=1000000)
        entry = tracker.record_usage(provider="custom", model="custom-model", usage=usage)

        # Cost should be 1.0 + 5.0 = 6.0
        assert entry.input_cost == 1.0
        assert entry.output_cost == 5.0
        assert entry.total_cost == 6.0

    def test_set_currency(self, tracker):
        """Test setting display currency."""
        tracker.set_currency("EUR")

        # Check that currency is set
        assert tracker._default_currency == "EUR"

    def test_session_tracking(self, tracker):
        """Test session start and end tracking."""
        tracker.start_session()

        from translatorhoi4.translator.cost import TokenUsage
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=2000)
        tracker.record_usage(provider="openai", model="gpt-4", usage=usage)

        summary = tracker.end_session()

        assert "start_time" in summary
        assert "end_time" in summary
        assert "duration_seconds" in summary
        assert summary["total_tokens"] == 3000


class TestTokenEstimator:
    """Test TokenEstimator functionality."""
    
    def test_estimate_tokens_english(self):
        """Test estimating tokens for English text."""
        from translatorhoi4.translator.cost import TokenEstimator

        text = "This is a test sentence with several words"
        tokens = TokenEstimator.estimate_tokens(text, "english")
        assert tokens > 0

    def test_estimate_tokens_empty(self):
        """Test estimating tokens for empty text."""
        from translatorhoi4.translator.cost import TokenEstimator

        tokens = TokenEstimator.estimate_tokens("", "english")
        assert tokens == 0

    def test_estimate_tokens_with_overhead(self):
        """Test that overhead is added when specified."""
        from translatorhoi4.translator.cost import TokenEstimator

        text = "Hello"
        tokens_no_overhead = TokenEstimator.estimate_tokens(text, "english", include_prompt_overhead=False)
        tokens_with_overhead = TokenEstimator.estimate_tokens(text, "english", include_prompt_overhead=True)

        assert tokens_with_overhead > tokens_no_overhead

    def test_estimate_batch_tokens(self):
        """Test estimating tokens for batch of texts."""
        from translatorhoi4.translator.cost import TokenEstimator

        texts = ["First text", "Second text", "Third text"]
        tokens = TokenEstimator.estimate_batch_tokens(texts, "english")
        assert tokens > 0

    def test_estimate_completion_tokens(self):
        """Test estimating completion tokens."""
        from translatorhoi4.translator.cost import TokenEstimator

        source = "This is the source text"
        tokens = TokenEstimator.estimate_completion_tokens(source, language="english")
        assert tokens > 0

    def test_estimate_completion_tokens_expansion(self):
        """Test that completion tokens are larger than source."""
        from translatorhoi4.translator.cost import TokenEstimator

        source = "A short source"
        source_tokens = TokenEstimator.estimate_tokens(source, "english", include_prompt_overhead=False)
        completion_tokens = TokenEstimator.estimate_completion_tokens(source, language="english")

        # Completion should generally be larger due to expansion factor
        assert completion_tokens >= source_tokens
