"""Risk manager unit tests.

Covers the two non-negotiable guarantees:
  * lot size = exactly 1% of free margin per trade given SL distance
  * any signal with confidence < 0.70 is rejected
plus side concerns: neutral sentiment rejection, missing margin, lot
clamping/stepping, and SL/TP price math.
"""
from __future__ import annotations

import math

import pytest

from trading_system.core.interfaces import AnalysisResult, OrderSide, Sentiment
from trading_system.modules.risk_manager.risk_manager import (
    FixedPercentRiskManager,
    SymbolSpec,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeBroker:
    """In-memory BrokerInfoProvider — no MT5 required."""

    def __init__(self, free_margin: float, spec: SymbolSpec | None) -> None:
        self.free_margin = free_margin
        self.spec = spec

    def get_free_margin(self) -> float:
        return self.free_margin

    def get_symbol_spec(self, pair: str):  # noqa: ANN001
        return self.spec


def eurusd_spec(bid: float = 1.10000, ask: float = 1.10010) -> SymbolSpec:
    return SymbolSpec(
        pair="EURUSD",
        pip_size=0.0001,
        pip_value_per_lot=10.0,   # $10 per pip per 1.0 lot on USD account
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        bid=bid,
        ask=ask,
        digits=5,
    )


def bullish(confidence: float, pair: str = "EURUSD") -> AnalysisResult:
    return AnalysisResult(Sentiment.BULLISH, confidence, pair)


def bearish(confidence: float, pair: str = "EURUSD") -> AnalysisResult:
    return AnalysisResult(Sentiment.BEARISH, confidence, pair)


# ---------------------------------------------------------------------------
# 1% calculation
# ---------------------------------------------------------------------------

def test_one_percent_calculation_exact():
    """$10,000 free margin, 20 pip SL, $10/pip/lot -> exactly 0.50 lots."""
    broker = FakeBroker(free_margin=10_000.0, spec=eurusd_spec())
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    signal = rm.evaluate(bullish(0.80), sl_pips=20, tp_pips=40)

    assert signal is not None
    assert signal.lot_size == pytest.approx(0.50)
    risked = signal.lot_size * 20 * 10.0
    assert risked == pytest.approx(10_000.0 * 0.01)


@pytest.mark.parametrize(
    "balance, sl_pips, expected_lots",
    [
        (5_000.0,  10, 0.50),  # 50 / (10*10) = 0.50
        (10_000.0, 50, 0.20),  # 100 / (50*10) = 0.20
        (25_000.0, 25, 1.00),  # 250 / (25*10) = 1.00
        (1_000.0,  20, 0.05),  # 10 / (20*10) = 0.05
    ],
)
def test_one_percent_calculation_parametrised(balance, sl_pips, expected_lots):
    broker = FakeBroker(free_margin=balance, spec=eurusd_spec())
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    signal = rm.evaluate(bullish(0.90), sl_pips=sl_pips, tp_pips=sl_pips * 2)

    assert signal is not None
    assert signal.lot_size == pytest.approx(expected_lots)
    risked = signal.lot_size * sl_pips * 10.0
    assert risked == pytest.approx(balance * 0.01)


def test_lot_size_rounds_down_to_step():
    """Account size that produces a non-step lot size must round DOWN.

    $7,777 * 1% = $77.77 risk, /20 pips /$10 = 0.38885 -> 0.38 (step 0.01).
    """
    broker = FakeBroker(free_margin=7_777.0, spec=eurusd_spec())
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    signal = rm.evaluate(bullish(0.80), sl_pips=20, tp_pips=40)

    assert signal is not None
    assert signal.lot_size == pytest.approx(0.38)
    assert signal.lot_size * 20 * 10.0 <= 7_777.0 * 0.01 + 1e-9


def test_lot_size_clamped_to_max():
    spec = eurusd_spec()
    spec_capped = SymbolSpec(**{**spec.__dict__, "max_lot": 0.10})
    broker = FakeBroker(free_margin=1_000_000.0, spec=spec_capped)
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    signal = rm.evaluate(bullish(0.99), sl_pips=20, tp_pips=40)

    assert signal is not None
    assert signal.lot_size == pytest.approx(0.10)


def test_below_min_lot_rejected():
    broker = FakeBroker(free_margin=10.0, spec=eurusd_spec())
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    signal = rm.evaluate(bullish(0.80), sl_pips=20, tp_pips=40)

    assert signal is None


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confidence", [0.0, 0.10, 0.50, 0.69, 0.6999])
def test_low_confidence_blocks_trade(confidence):
    broker = FakeBroker(free_margin=10_000.0, spec=eurusd_spec())
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    assert rm.evaluate(bullish(confidence), sl_pips=20, tp_pips=40) is None


@pytest.mark.parametrize("confidence", [0.70, 0.71, 0.85, 1.0])
def test_high_confidence_allows_trade(confidence):
    broker = FakeBroker(free_margin=10_000.0, spec=eurusd_spec())
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    assert rm.evaluate(bullish(confidence), sl_pips=20, tp_pips=40) is not None


def test_threshold_is_configurable():
    broker = FakeBroker(free_margin=10_000.0, spec=eurusd_spec())
    strict = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.90)

    assert strict.evaluate(bullish(0.85), sl_pips=20, tp_pips=40) is None
    assert strict.evaluate(bullish(0.90), sl_pips=20, tp_pips=40) is not None


# ---------------------------------------------------------------------------
# Other guards
# ---------------------------------------------------------------------------

def test_neutral_sentiment_rejected():
    broker = FakeBroker(free_margin=10_000.0, spec=eurusd_spec())
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)
    neutral = AnalysisResult(Sentiment.NEUTRAL, 0.95, "EURUSD")

    assert rm.evaluate(neutral, sl_pips=20, tp_pips=40) is None


def test_zero_free_margin_rejected():
    broker = FakeBroker(free_margin=0.0, spec=eurusd_spec())
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    assert rm.evaluate(bullish(0.95), sl_pips=20, tp_pips=40) is None


def test_missing_symbol_spec_rejected():
    broker = FakeBroker(free_margin=10_000.0, spec=None)
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    assert rm.evaluate(bullish(0.95), sl_pips=20, tp_pips=40) is None


def test_non_positive_sl_rejected():
    broker = FakeBroker(free_margin=10_000.0, spec=eurusd_spec())
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    assert rm.evaluate(bullish(0.95), sl_pips=0, tp_pips=40) is None
    assert rm.evaluate(bullish(0.95), sl_pips=-5, tp_pips=40) is None


# ---------------------------------------------------------------------------
# Side + SL/TP math
# ---------------------------------------------------------------------------

def test_buy_signal_levels():
    broker = FakeBroker(free_margin=10_000.0, spec=eurusd_spec(bid=1.10000, ask=1.10010))
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    signal = rm.evaluate(bullish(0.90), sl_pips=20, tp_pips=40)

    assert signal is not None
    assert signal.side is OrderSide.BUY
    assert signal.sl_price == pytest.approx(1.10010 - 0.0020)
    assert signal.tp_price == pytest.approx(1.10010 + 0.0040)


def test_sell_signal_levels():
    broker = FakeBroker(free_margin=10_000.0, spec=eurusd_spec(bid=1.10000, ask=1.10010))
    rm = FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0.70)

    signal = rm.evaluate(bearish(0.90), sl_pips=20, tp_pips=40)

    assert signal is not None
    assert signal.side is OrderSide.SELL
    assert signal.sl_price == pytest.approx(1.10000 + 0.0020)
    assert signal.tp_price == pytest.approx(1.10000 - 0.0040)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_invalid_constructor_args():
    broker = FakeBroker(free_margin=10_000.0, spec=eurusd_spec())
    with pytest.raises(ValueError):
        FixedPercentRiskManager(broker, risk_percent=0)
    with pytest.raises(ValueError):
        FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=0)
    with pytest.raises(ValueError):
        FixedPercentRiskManager(broker, risk_percent=1.0, min_confidence=1.5)
