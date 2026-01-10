"""Token usage tracking and cost calculation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TokenUsage:
    """Track token usage for a single request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = field(init=False)
    
    def __post_init__(self):
        self.total_tokens = self.prompt_tokens + self.completion_tokens
    
    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )
    
    def __iadd__(self, other: "TokenUsage"):
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        return self
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "TokenUsage":
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
        )


@dataclass
class CostEntry:
    """Track cost for a single request."""
    timestamp: datetime
    provider: str
    model: str
    usage: TokenUsage
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    source_lang: str = ""
    target_lang: str = ""
    entry_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "currency": self.currency,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "entry_count": self.entry_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostEntry":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            provider=data["provider"],
            model=data["model"],
            usage=TokenUsage.from_dict(data["usage"]),
            input_cost=data["input_cost"],
            output_cost=data["output_cost"],
            total_cost=data["total_cost"],
            currency=data.get("currency", "USD"),
            source_lang=data.get("source_lang", ""),
            target_lang=data.get("target_lang", ""),
            entry_count=data.get("entry_count", 1),
        )


class CostTracker:
    """Track token usage and calculate costs across translation sessions."""
    
    _instance: Optional["CostTracker"] = None
    
    def __new__(cls) -> "CostTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._entries: List[CostEntry] = []
        self._session_start: Optional[datetime] = None
        self._session_usage: TokenUsage = TokenUsage()
        self._session_cost: float = 0.0
        
        # Cost configuration per provider (per million tokens)
        self._cost_config: Dict[str, Dict[str, float]] = {
            "g4f": {"input": 0.0, "output": 0.0},  # Free tier
            "openai": {"input": 2.50, "output": 10.00},  # gpt-4 prices
            "anthropic": {"input": 3.00, "output": 15.00},  # Claude prices
            "gemini": {"input": 0.125, "output": 0.375},  # Gemini 1.5 Flash
            "io": {"input": 0.59, "output": 0.79},  # IO Intelligence
            "yandex_translate": {"input": 0.0, "output": 0.0},  # Free tier
            "yandex_cloud": {"input": 0.0, "output": 0.0},  # To be configured
            "deepl": {"input": 0.0, "output": 0.0},  # To be configured
            "fireworks": {"input": 0.0, "output": 0.0},  # To be configured
            "groq": {"input": 0.0, "output": 0.0},  # To be configured
            "together": {"input": 0.0, "output": 0.0},  # To be configured
            "ollama": {"input": 0.0, "output": 0.0},  # Local, free
        }
        
        self._default_currency = "USD"
        
        # Load from disk if exists
        self._load()
    
    def start_session(self):
        """Start a new tracking session."""
        self._session_start = datetime.now()
        self._session_usage = TokenUsage()
        self._session_cost = 0.0
    
    def end_session(self) -> Dict[str, Any]:
        """End the current session and return summary."""
        if self._session_start is None:
            return {}
        
        session_duration = (datetime.now() - self._session_start).total_seconds()
        
        summary = {
            "start_time": self._session_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": session_duration,
            "total_prompt_tokens": self._session_usage.prompt_tokens,
            "total_completion_tokens": self._session_usage.completion_tokens,
            "total_tokens": self._session_usage.total_tokens,
            "total_cost": self._session_cost,
            "currency": self._default_currency,
        }
        
        self._session_start = None
        return summary
    
    def record_usage(
        self,
        provider: str,
        model: str,
        usage: TokenUsage,
        source_lang: str = "",
        target_lang: str = "",
        entry_count: int = 1,
    ) -> CostEntry:
        """Record token usage for a translation batch.
        
        Args:
            provider: LLM provider name (g4f, openai, anthropic, gemini, io)
            model: Model identifier
            usage: Token usage data
            source_lang: Source language code
            target_lang: Target language code
            entry_count: Number of translation entries processed
            
        Returns:
            CostEntry with cost breakdown
        """
        # Get cost configuration
        config = self._cost_config.get(provider.lower(), {"input": 0, "output": 0})
        input_cost_per_m = config.get("input", 0)
        output_cost_per_m = config.get("output", 0)
        
        # Calculate costs
        input_cost = (usage.prompt_tokens / 1_000_000) * input_cost_per_m
        output_cost = (usage.completion_tokens / 1_000_000) * output_cost_per_m
        total_cost = input_cost + output_cost
        
        # Create entry
        entry = CostEntry(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            usage=usage,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            currency=self._default_currency,
            source_lang=source_lang,
            target_lang=target_lang,
            entry_count=entry_count,
        )
        
        # Store entry
        self._entries.append(entry)
        
        # Update session stats
        self._session_usage += usage
        self._session_cost += total_cost
        
        # Auto-save periodically
        if len(self._entries) % 100 == 0:
            self._save()
        
        return entry
    
    def get_provider_cost(self, provider: str) -> Dict[str, Any]:
        """Get cost summary for a specific provider."""
        entries = [e for e in self._entries if e.provider.lower() == provider.lower()]
        
        if not entries:
            return {
                "provider": provider,
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_entries": 0,
                "requests": 0,
            }
        
        total_cost = sum(e.total_cost for e in entries)
        total_tokens = sum(e.usage.total_tokens for e in entries)
        total_entries = sum(e.entry_count for e in entries)
        
        return {
            "provider": provider,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_entries": total_entries,
            "requests": len(entries),
            "currency": self._default_currency,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall cost and usage summary."""
        if not self._entries:
            return {
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_entries": 0,
                "total_requests": 0,
                "currency": self._default_currency,
                "providers": {},
            }
        
        # Aggregate by provider
        providers = {}
        for entry in self._entries:
            p = entry.provider.lower()
            if p not in providers:
                providers[p] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_entries": 0,
                    "requests": 0,
                }
            providers[p]["total_cost"] += entry.total_cost
            providers[p]["total_tokens"] += entry.usage.total_tokens
            providers[p]["total_entries"] += entry.entry_count
            providers[p]["requests"] += 1
        
        total_cost = sum(e.total_cost for e in self._entries)
        total_tokens = sum(e.usage.total_tokens for e in self._entries)
        total_entries = sum(e.entry_count for e in self._entries)
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_entries": total_entries,
            "total_requests": len(self._entries),
            "currency": self._default_currency,
            "providers": providers,
        }
    
    def set_cost_per_million(
        self,
        provider: str,
        input_cost: float,
        output_cost: float,
    ):
        """Set custom cost per million tokens for a provider.
        
        Args:
            provider: Provider name
            input_cost: Cost per million input tokens
            output_cost: Cost per million output tokens
        """
        self._cost_config[provider.lower()] = {
            "input": input_cost,
            "output": output_cost,
        }
    
    def set_currency(self, currency: str):
        """Set the display currency.
        
        Args:
            currency: Currency code (USD, EUR, GBP, etc.)
        """
        self._default_currency = currency.upper()
    
    def get_recent_entries(self, limit: int = 50) -> List[CostEntry]:
        """Get recent cost entries."""
        return self._entries[-limit:]
    
    def export_csv(self, path: str):
        """Export cost data to CSV format.
        
        Args:
            path: Output file path
        """
        import csv
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "provider", "model", "prompt_tokens",
                "completion_tokens", "total_tokens", "input_cost",
                "output_cost", "total_cost", "currency",
                "source_lang", "target_lang", "entry_count",
            ])
            writer.writeheader()
            
            for entry in self._entries:
                row = entry.to_dict()
                row["prompt_tokens"] = entry.usage.prompt_tokens
                row["completion_tokens"] = entry.usage.completion_tokens
                row["total_tokens"] = entry.usage.total_tokens
                writer.writerow(row)
    
    def clear(self):
        """Clear all tracking data."""
        self._entries.clear()
        self._session_start = None
        self._session_usage = TokenUsage()
        self._session_cost = 0.0
        self._save()
    
    def _get_storage_path(self) -> Path:
        """Get path for storing cost data."""
        app_dir = Path(__file__).parent.parent
        storage_dir = app_dir / ".translatorhoi4_data"
        storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir / "cost_tracking.json"
    
    def _save(self):
        """Save cost data to disk."""
        path = self._get_storage_path()
        try:
            data = {
                "entries": [e.to_dict() for e in self._entries],
                "cost_config": self._cost_config,
                "currency": self._default_currency,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass
    
    def _load(self):
        """Load cost data from disk."""
        path = self._get_storage_path()
        if not path.exists():
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._entries = [CostEntry.from_dict(e) for e in data.get("entries", [])]
            self._cost_config = data.get("cost_config", self._cost_config)
            self._default_currency = data.get("currency", "USD")
        except Exception:
            pass


# Global cost tracker instance
cost_tracker = CostTracker()


# --- Token Estimator ---
class TokenEstimator:
    """Estimate token counts for text without making API calls."""
    
    # Average tokens per word for different languages
    TOKENS_PER_WORD = {
        "english": 0.75,
        "russian": 0.8,
        "german": 0.8,
        "french": 0.8,
        "spanish": 0.75,
        "italian": 0.75,
        "polish": 0.85,
        "portuguese": 0.75,
        "japanese": 1.5,
        "chinese": 1.8,
        "korean": 1.4,
        "arabic": 1.0,
        "default": 0.8,
    }
    
    # Characters per token (approximate)
    CHARS_PER_TOKEN = 4
    
    @classmethod
    def estimate_tokens(
        cls,
        text: str,
        language: str = "english",
        include_prompt_overhead: bool = True,
    ) -> int:
        """Estimate token count for a text.
        
        Args:
            text: Input text
            language: Language code for adjustment
            include_prompt_overhead: Include estimated prompt overhead
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Estimate by characters (TikToken approximation)
        char_count = len(text)
        token_count = char_count // cls.CHARS_PER_TOKEN
        
        # Adjust for language
        multiplier = cls.TOKENS_PER_WORD.get(language.lower(), cls.TOKENS_PER_WORD["default"])
        token_count = int(token_count * multiplier)
        
        if include_prompt_overhead:
            # Add overhead for system prompt and formatting
            overhead = 100  # System prompt
            overhead += 50  # Markup/markers
            token_count += overhead
        
        return token_count
    
    @classmethod
    def estimate_batch_tokens(
        cls,
        texts: List[str],
        language: str = "english",
        separator: str = "\n\n",
    ) -> int:
        """Estimate tokens for a batch of texts.
        
        Args:
            texts: List of text strings
            language: Language code
            separator: Separator between texts
            
        Returns:
            Total estimated token count
        """
        if not texts:
            return 0
        
        combined = separator.join(texts)
        return cls.estimate_tokens(combined, language)
    
    @classmethod
    def estimate_completion_tokens(
        cls,
        source_text: str,
        expansion_factor: float = 1.3,
        language: str = "english",
    ) -> int:
        """Estimate completion token count.
        
        Args:
            source_text: Source text being translated
            expansion_factor: Expected expansion factor (translations often expand)
            language: Target language
            
        Returns:
            Estimated completion token count
        """
        source_tokens = cls.estimate_tokens(source_text, language, include_prompt_overhead=False)
        
        # Adjust for language-specific expansion
        if language.lower() in ["german", "russian"]:
            expansion_factor *= 1.1
        
        return int(source_tokens * expansion_factor)
