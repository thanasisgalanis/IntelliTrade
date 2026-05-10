"""Per-pair aggregation strategy tests.

Covers all four configured strategies plus the env-loading guard rails.
"""
from __future__ import annotations

import pytest

from trading_system.core.interfaces import AnalysisResult, Sentiment
from trading_system.modules.news_analyzer.aggregator import (
    AggregationConfig,
    aggregate_by_pair,
)


def r(pair: str, sentiment: Sentiment, confidence: float) -> AnalysisResult:
    return AnalysisResult(
        sentiment=sentiment, confidence=confidence, pair=pair, rationale=""
    )


# ---------------------------------------------------------------------------
# Config / env loading
# ---------------------------------------------------------------------------

def test_default_strategy_is_average_confidence():
    cfg = AggregationConfig.from_env(strategy=None, neutral_band=None)
    assert cfg.strategy == "average_confidence"
    assert cfg.neutral_band == pytest.approx(0.10)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="Unknown AGGREGATION_STRATEGY"):
        AggregationConfig.from_env(strategy="bogus", neutral_band=None)


def test_strategy_env_is_case_insensitive():
    cfg = AggregationConfig.from_env(strategy="MAX_CONFIDENCE", neutral_band=None)
    assert cfg.strategy == "max_confidence"


def test_neutral_band_must_be_in_range():
    with pytest.raises(ValueError, match="WEIGHTED_NEUTRAL_BAND"):
        AggregationConfig.from_env(strategy="weighted_average", neutral_band="1.5")


# ---------------------------------------------------------------------------
# Single-pair groups (no aggregation needed)
# ---------------------------------------------------------------------------

def test_single_article_per_pair_passthrough():
    results = [
        r("EURUSD", Sentiment.BULLISH, 0.8),
        r("GBPUSD", Sentiment.BEARISH, 0.7),
    ]
    cfg = AggregationConfig(strategy="average_confidence")

    out = aggregate_by_pair(results, cfg)

    assert {a.pair for a in out} == {"EURUSD", "GBPUSD"}
    assert next(a for a in out if a.pair == "EURUSD").confidence == pytest.approx(0.8)


def test_empty_input_returns_empty():
    assert aggregate_by_pair([], AggregationConfig()) == []


# ---------------------------------------------------------------------------
# max_confidence
# ---------------------------------------------------------------------------

def test_max_confidence_picks_highest():
    results = [
        r("EURUSD", Sentiment.BULLISH, 0.6),
        r("EURUSD", Sentiment.BEARISH, 0.9),
        r("EURUSD", Sentiment.BULLISH, 0.7),
    ]
    cfg = AggregationConfig(strategy="max_confidence")

    out = aggregate_by_pair(results, cfg)

    assert len(out) == 1
    assert out[0].sentiment is Sentiment.BEARISH
    assert out[0].confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# average_confidence
# ---------------------------------------------------------------------------

def test_average_confidence_majority_vote_and_mean():
    results = [
        r("EURUSD", Sentiment.BULLISH, 0.6),
        r("EURUSD", Sentiment.BULLISH, 0.8),
        r("EURUSD", Sentiment.BEARISH, 0.4),
    ]
    cfg = AggregationConfig(strategy="average_confidence")

    out = aggregate_by_pair(results, cfg)

    assert len(out) == 1
    assert out[0].sentiment is Sentiment.BULLISH  # 2 vs 1
    # mean over ALL articles, not just majority
    assert out[0].confidence == pytest.approx((0.6 + 0.8 + 0.4) / 3)


def test_average_confidence_tied_sentiment_falls_to_neutral():
    results = [
        r("EURUSD", Sentiment.BULLISH, 0.7),
        r("EURUSD", Sentiment.BEARISH, 0.7),
    ]
    cfg = AggregationConfig(strategy="average_confidence")

    out = aggregate_by_pair(results, cfg)

    assert out[0].sentiment is Sentiment.NEUTRAL


# ---------------------------------------------------------------------------
# majority_sentiment
# ---------------------------------------------------------------------------

def test_majority_sentiment_averages_only_winning_side():
    results = [
        r("EURUSD", Sentiment.BULLISH, 0.6),
        r("EURUSD", Sentiment.BULLISH, 0.8),
        r("EURUSD", Sentiment.BEARISH, 0.4),
    ]
    cfg = AggregationConfig(strategy="majority_sentiment")

    out = aggregate_by_pair(results, cfg)

    assert out[0].sentiment is Sentiment.BULLISH
    # Mean confidence of the winning (bullish) side only.
    assert out[0].confidence == pytest.approx((0.6 + 0.8) / 2)


# ---------------------------------------------------------------------------
# weighted_average
# ---------------------------------------------------------------------------

def test_weighted_average_strong_bullish():
    results = [
        r("EURUSD", Sentiment.BULLISH, 0.9),
        r("EURUSD", Sentiment.BULLISH, 0.8),
    ]
    cfg = AggregationConfig(strategy="weighted_average", neutral_band=0.10)

    out = aggregate_by_pair(results, cfg)

    # score = (1*0.9 + 1*0.8) / 2 = 0.85
    assert out[0].sentiment is Sentiment.BULLISH
    assert out[0].confidence == pytest.approx(0.85)


def test_weighted_average_cancellation_falls_in_neutral_band():
    results = [
        r("EURUSD", Sentiment.BULLISH, 0.5),
        r("EURUSD", Sentiment.BEARISH, 0.5),
    ]
    cfg = AggregationConfig(strategy="weighted_average", neutral_band=0.10)

    out = aggregate_by_pair(results, cfg)

    # score = 0 -> neutral
    assert out[0].sentiment is Sentiment.NEUTRAL
    assert out[0].confidence == pytest.approx(0.0)


def test_weighted_average_score_outside_band_signals_direction():
    results = [
        r("EURUSD", Sentiment.BULLISH, 0.7),
        r("EURUSD", Sentiment.BEARISH, 0.3),
    ]
    cfg = AggregationConfig(strategy="weighted_average", neutral_band=0.10)

    out = aggregate_by_pair(results, cfg)

    # score = (0.7 - 0.3) / 2 = 0.20 -> bullish, conf = 0.20
    assert out[0].sentiment is Sentiment.BULLISH
    assert out[0].confidence == pytest.approx(0.20)


def test_aggregator_handles_multiple_pairs_independently():
    results = [
        r("EURUSD", Sentiment.BULLISH, 0.8),
        r("EURUSD", Sentiment.BULLISH, 0.6),
        r("GBPUSD", Sentiment.BEARISH, 0.9),
    ]
    cfg = AggregationConfig(strategy="average_confidence")

    out = {a.pair: a for a in aggregate_by_pair(results, cfg)}

    assert out["EURUSD"].sentiment is Sentiment.BULLISH
    assert out["EURUSD"].confidence == pytest.approx(0.7)
    assert out["GBPUSD"].sentiment is Sentiment.BEARISH
    assert out["GBPUSD"].confidence == pytest.approx(0.9)
