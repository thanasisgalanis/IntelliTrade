"""Per-pair aggregation of per-article analysis results.

The batch analyzer emits one :class:`AnalysisResult` per article. The risk
manager and execution engine, however, act per-pair: at most one trade per
instrument per cycle. This module collapses N article-level signals for
the same pair into one consolidated signal using a configurable strategy.

Strategies (configured via the ``AGGREGATION_STRATEGY`` env var):

``max_confidence``
    Keep the single article with the highest confidence. Conservative —
    one strong story dominates many weak ones.

``average_confidence``
    Sentiment = majority vote across articles (ties → neutral).
    Confidence = arithmetic mean of all article confidences.

``majority_sentiment``
    Sentiment = majority vote across articles (ties → neutral).
    Confidence = mean confidence of the *winning* sentiment only.

``weighted_average``
    Map sentiments to scalars (bullish=+1, bearish=-1, neutral=0) and
    take a confidence-weighted mean. Sentiment is the sign of the
    resulting score, with |score| < ``WEIGHTED_NEUTRAL_BAND`` (default
    0.10) snapping to neutral. Confidence = ``|score|``.

The rationale field on the consolidated result is a short summary noting
the strategy and the article count.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from trading_system.core.interfaces import AnalysisResult, Sentiment
from trading_system.core.logger import get_logger

log = get_logger(__name__)

STRATEGIES = (
    "max_confidence",
    "average_confidence",
    "majority_sentiment",
    "weighted_average",
)

_DEFAULT_STRATEGY = "average_confidence"
_DEFAULT_NEUTRAL_BAND = 0.10


@dataclass(frozen=True)
class AggregationConfig:
    strategy: str = _DEFAULT_STRATEGY
    neutral_band: float = _DEFAULT_NEUTRAL_BAND  # used by weighted_average

    @classmethod
    def from_env(cls, strategy: str | None, neutral_band: str | None) -> "AggregationConfig":
        s = (strategy or _DEFAULT_STRATEGY).strip().lower()
        if s not in STRATEGIES:
            raise ValueError(
                f"Unknown AGGREGATION_STRATEGY={s!r}; "
                f"valid: {', '.join(STRATEGIES)}"
            )
        try:
            band = float(neutral_band) if neutral_band else _DEFAULT_NEUTRAL_BAND
        except ValueError:
            raise ValueError(
                f"WEIGHTED_NEUTRAL_BAND must be a float, got {neutral_band!r}"
            )
        if not 0.0 <= band <= 1.0:
            raise ValueError("WEIGHTED_NEUTRAL_BAND must be in [0.0, 1.0]")
        return cls(strategy=s, neutral_band=band)


def aggregate_by_pair(
    results: list[AnalysisResult],
    config: AggregationConfig,
) -> list[AnalysisResult]:
    """Group ``results`` by pair and apply ``config.strategy`` to each group.

    Returns one :class:`AnalysisResult` per distinct pair, in the order the
    pairs were first encountered.
    """
    if not results:
        return []

    groups: dict[str, list[AnalysisResult]] = {}
    for r in results:
        groups.setdefault(r.pair, []).append(r)

    consolidated: list[AnalysisResult] = []
    for pair, group in groups.items():
        agg = _apply_strategy(pair, group, config)
        log.info(
            "Aggregated %s: %d article(s) -> sentiment=%s confidence=%.2f (strategy=%s)",
            pair,
            len(group),
            agg.sentiment.value,
            agg.confidence,
            config.strategy,
        )
        consolidated.append(agg)

    return consolidated


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _apply_strategy(
    pair: str,
    group: list[AnalysisResult],
    config: AggregationConfig,
) -> AnalysisResult:
    if len(group) == 1:
        # Single article — nothing to aggregate. Keep its rationale intact.
        return group[0]

    if config.strategy == "max_confidence":
        winner = max(group, key=lambda r: r.confidence)
        return AnalysisResult(
            sentiment=winner.sentiment,
            confidence=winner.confidence,
            pair=pair,
            rationale=f"max_confidence over {len(group)} articles",
        )

    if config.strategy == "average_confidence":
        sentiment = _majority_sentiment(group)
        avg_conf = sum(r.confidence for r in group) / len(group)
        return AnalysisResult(
            sentiment=sentiment,
            confidence=avg_conf,
            pair=pair,
            rationale=f"average_confidence over {len(group)} articles",
        )

    if config.strategy == "majority_sentiment":
        sentiment = _majority_sentiment(group)
        winners = [r for r in group if r.sentiment is sentiment]
        # _majority_sentiment may return NEUTRAL on a tie even when no
        # article was neutral — fall back to the full group in that case.
        if not winners:
            winners = group
        conf = sum(r.confidence for r in winners) / len(winners)
        return AnalysisResult(
            sentiment=sentiment,
            confidence=conf,
            pair=pair,
            rationale=(
                f"majority_sentiment ({len(winners)}/{len(group)} articles)"
            ),
        )

    if config.strategy == "weighted_average":
        score = sum(_sentiment_sign(r.sentiment) * r.confidence for r in group) / len(group)
        if abs(score) < config.neutral_band:
            sentiment = Sentiment.NEUTRAL
        elif score > 0:
            sentiment = Sentiment.BULLISH
        else:
            sentiment = Sentiment.BEARISH
        return AnalysisResult(
            sentiment=sentiment,
            confidence=min(1.0, abs(score)),
            pair=pair,
            rationale=(
                f"weighted_average score={score:+.2f} over {len(group)} articles"
            ),
        )

    # Unreachable: AggregationConfig.from_env() validates the value.
    raise ValueError(f"Unknown strategy {config.strategy!r}")


def _majority_sentiment(group: list[AnalysisResult]) -> Sentiment:
    counts = Counter(r.sentiment for r in group)
    top = counts.most_common()
    # Tie at the top → neutral. Any unique top wins.
    if len(top) > 1 and top[0][1] == top[1][1]:
        return Sentiment.NEUTRAL
    return top[0][0]


def _sentiment_sign(s: Sentiment) -> int:
    if s is Sentiment.BULLISH:
        return 1
    if s is Sentiment.BEARISH:
        return -1
    return 0
