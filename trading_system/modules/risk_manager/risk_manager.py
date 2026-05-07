"""Position sizing and pre-trade risk gate.

Two hard rules — both MUST hold before a TradeSignal is emitted:

  1. AI confidence >= ``min_confidence`` (default 0.70).
  2. The notional risk of the trade equals exactly ``risk_percent`` % of
     the account's *free margin* given the supplied stop-loss distance.

The lot-size formula is:

    risk_amount   = free_margin * (risk_percent / 100)
    risk_per_lot  = sl_pips * pip_value_per_lot
    lot_size      = risk_amount / risk_per_lot
    lot_size      = clamp(lot_size, min_lot, max_lot)
    lot_size      = round_down_to(lot_size, lot_step)

``pip_value_per_lot`` and lot constraints come from MT5's ``symbol_info``
when an account is connected; for tests the manager accepts an injected
``BrokerInfoProvider`` so MT5 is never required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from trading_system.core.interfaces import (
    AnalysisResult,
    IRiskManager,
    OrderSide,
    Sentiment,
    TradeSignal,
)
from trading_system.core.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class SymbolSpec:
    """Subset of MT5 symbol info we actually need for sizing."""
    pair: str
    pip_size: float          # e.g. 0.0001 for EURUSD, 0.01 for USDJPY
    pip_value_per_lot: float # account-currency value of 1 pip per 1.0 lot
    min_lot: float
    max_lot: float
    lot_step: float
    bid: float
    ask: float
    digits: int


class BrokerInfoProvider(Protocol):
    """Anything that can answer 'how much margin do I have?' and
    'what are the trading specs for this pair?'. The concrete MT5-backed
    implementation lives next to the execution engine."""

    def get_free_margin(self) -> float: ...
    def get_symbol_spec(self, pair: str) -> SymbolSpec | None: ...


class FixedPercentRiskManager(IRiskManager):
    def __init__(
        self,
        broker: BrokerInfoProvider,
        risk_percent: float = 1.0,
        min_confidence: float = 0.70,
    ) -> None:
        if risk_percent <= 0:
            raise ValueError("risk_percent must be > 0")
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1]")
        self._broker = broker
        self._risk_percent = risk_percent
        self._min_confidence = min_confidence

    # ------------------------------------------------------------------
    def evaluate(
        self,
        analysis: AnalysisResult,
        sl_pips: float,
        tp_pips: float,
    ) -> TradeSignal | None:
        # Gate 1: confidence
        if analysis.confidence < self._min_confidence:
            log.info(
                "REJECT %s — confidence %.2f below threshold %.2f",
                analysis.pair, analysis.confidence, self._min_confidence,
            )
            return None

        # Gate 2: actionable sentiment
        if analysis.sentiment is Sentiment.NEUTRAL:
            log.info("REJECT %s — neutral sentiment, no trade", analysis.pair)
            return None

        if sl_pips <= 0 or tp_pips <= 0:
            log.warning("REJECT %s — non-positive SL/TP pips", analysis.pair)
            return None

        spec = self._broker.get_symbol_spec(analysis.pair)
        if spec is None:
            log.warning("REJECT %s — symbol spec unavailable", analysis.pair)
            return None

        free_margin = self._broker.get_free_margin()
        if free_margin <= 0:
            log.warning("REJECT %s — non-positive free margin %s", analysis.pair, free_margin)
            return None

        lot_size = self._compute_lot_size(free_margin, sl_pips, spec)
        if lot_size <= 0:
            log.warning("REJECT %s — computed lot size <= 0", analysis.pair)
            return None

        side = OrderSide.BUY if analysis.sentiment is Sentiment.BULLISH else OrderSide.SELL
        sl_price, tp_price = self._compute_levels(side, spec, sl_pips, tp_pips)

        signal = TradeSignal(
            pair=analysis.pair,
            side=side,
            lot_size=lot_size,
            sl_price=sl_price,
            tp_price=tp_price,
            confidence=analysis.confidence,
            comment=f"InteliTrade conf={analysis.confidence:.2f}",
        )
        log.info(
            "ACCEPT %s %s lots=%.2f SL=%.5f TP=%.5f conf=%.2f",
            signal.pair, signal.side.value, signal.lot_size,
            signal.sl_price, signal.tp_price, signal.confidence,
        )
        return signal

    # ------------------------------------------------------------------
    def _compute_lot_size(
        self,
        free_margin: float,
        sl_pips: float,
        spec: SymbolSpec,
    ) -> float:
        risk_amount = free_margin * (self._risk_percent / 100.0)
        risk_per_lot = sl_pips * spec.pip_value_per_lot
        if risk_per_lot <= 0:
            return 0.0
        raw_lots = risk_amount / risk_per_lot
        stepped = self._round_down_to_step(raw_lots, spec.lot_step)
        return max(spec.min_lot, min(stepped, spec.max_lot)) if stepped >= spec.min_lot else 0.0

    @staticmethod
    def _round_down_to_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        return math.floor(value / step) * step

    @staticmethod
    def _compute_levels(
        side: OrderSide,
        spec: SymbolSpec,
        sl_pips: float,
        tp_pips: float,
    ) -> tuple[float, float]:
        sl_distance = sl_pips * spec.pip_size
        tp_distance = tp_pips * spec.pip_size
        if side is OrderSide.BUY:
            entry = spec.ask
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            entry = spec.bid
            sl = entry + sl_distance
            tp = entry - tp_distance
        return round(sl, spec.digits), round(tp, spec.digits)
