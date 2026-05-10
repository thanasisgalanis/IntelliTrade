"""Abstract base classes that decouple the modules.

Every concrete module must implement exactly one of these interfaces. The
orchestrator depends only on the interfaces, never on the concrete classes,
which keeps the system swappable (e.g. NewsAPI -> Bloomberg, Claude -> GPT,
MT5 -> cTrader) without touching business logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterable


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class NewsItem:
    """Raw news article fetched by an INewsCollector."""
    article_id: str
    source: str
    title: str
    description: str
    url: str
    published_at: datetime
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    """Structured output of an INewsAnalyzer."""
    sentiment: Sentiment
    confidence: float
    pair: str
    rationale: str = ""


@dataclass(frozen=True)
class TradeSignal:
    """Risk-validated, ready-to-execute order spec."""
    pair: str
    side: OrderSide
    lot_size: float
    sl_price: float
    tp_price: float
    confidence: float
    comment: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of attempting to place a market order."""
    success: bool
    order_id: int | None
    fill_price: float | None
    error: str | None = None
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module interfaces
# ---------------------------------------------------------------------------

class INewsCollector(ABC):
    @abstractmethod
    def fetch(self, query: str, page_size: int = 20) -> list[NewsItem]:
        """Fetch fresh news; persistence is the implementation's responsibility."""

    @abstractmethod
    def load_recent(self, limit: int = 50) -> list[NewsItem]:
        """Read previously-stored news from local storage."""


class INewsAnalyzer(ABC):
    @abstractmethod
    def analyze(self, item: NewsItem) -> AnalysisResult | None:
        """Return a structured signal or None if analysis failed/was skipped."""

    def analyze_many(self, items: list[NewsItem]) -> list[AnalysisResult]:
        """Analyze a batch of items.

        Default: per-item iteration. Implementations may override to
        deduplicate or batch external API calls across items targeting the
        same instrument (see issue #3).
        """
        results: list[AnalysisResult] = []
        for item in items:
            r = self.analyze(item)
            if r is not None:
                results.append(r)
        return results


class IRiskManager(ABC):
    @abstractmethod
    def evaluate(
        self,
        analysis: AnalysisResult,
        sl_pips: float,
        tp_pips: float,
    ) -> TradeSignal | None:
        """Return a sized TradeSignal or None if rejected (low conf, bad data...)."""


class IExecutionEngine(ABC):
    @abstractmethod
    def execute(self, signal: TradeSignal) -> ExecutionResult:
        """Place the market order and return the broker's response."""

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanly disconnect from the broker."""
