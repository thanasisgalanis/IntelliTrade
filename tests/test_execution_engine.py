"""Execution-engine unit tests with a fully mocked MetaTrader5 module.

Focus: filling-mode selection (issue #5 — retcode=10030 "Unsupported
filling mode"). The engine must consult ``symbol_info.filling_mode``
(a bitmask) and pick a mode the symbol actually supports rather than
hard-coding IOC for every order.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trading_system.core.interfaces import OrderSide, TradeSignal
from trading_system.modules.execution_engine.engine import MT5ExecutionEngine


# ---------------------------------------------------------------------------
# MT5 module / session fakes
# ---------------------------------------------------------------------------

# Sentinel constants — the engine must pass *exactly* these objects through
# to ``order_send``. Using distinct sentinels lets each test assert on
# identity rather than reading equal-but-coincidental ints.
ORDER_FILLING_IOC = "ORDER_FILLING_IOC"
ORDER_FILLING_FOK = "ORDER_FILLING_FOK"
ORDER_FILLING_RETURN = "ORDER_FILLING_RETURN"
ORDER_TYPE_BUY = "ORDER_TYPE_BUY"
ORDER_TYPE_SELL = "ORDER_TYPE_SELL"
TRADE_ACTION_DEAL = "TRADE_ACTION_DEAL"
ORDER_TIME_GTC = "ORDER_TIME_GTC"
TRADE_RETCODE_DONE = 10009
RETCODE_UNSUPPORTED_FILLING = 10030


def make_mt5_mock(
    *,
    filling_mode_mask: int | None = 2,  # default: IOC supported
    bid: float = 1.2345,
    ask: float = 1.2347,
    order_send_retcode: int = TRADE_RETCODE_DONE,
    symbol_info_returns_none: bool = False,
) -> MagicMock:
    mt5 = MagicMock()
    mt5.ORDER_FILLING_IOC = ORDER_FILLING_IOC
    mt5.ORDER_FILLING_FOK = ORDER_FILLING_FOK
    mt5.ORDER_FILLING_RETURN = ORDER_FILLING_RETURN
    mt5.ORDER_TYPE_BUY = ORDER_TYPE_BUY
    mt5.ORDER_TYPE_SELL = ORDER_TYPE_SELL
    mt5.TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
    mt5.ORDER_TIME_GTC = ORDER_TIME_GTC
    mt5.TRADE_RETCODE_DONE = TRADE_RETCODE_DONE

    mt5.symbol_info_tick.return_value = SimpleNamespace(bid=bid, ask=ask)

    if symbol_info_returns_none:
        mt5.symbol_info.return_value = None
    else:
        mt5.symbol_info.return_value = SimpleNamespace(
            filling_mode=filling_mode_mask
        )

    mt5.order_send.return_value = SimpleNamespace(
        retcode=order_send_retcode,
        order=987654,
        price=ask,
        comment="",
        _asdict=lambda: {"retcode": order_send_retcode},
    )
    mt5.last_error.return_value = (0, "ok")
    return mt5


def make_engine(mt5_mock: MagicMock) -> MT5ExecutionEngine:
    """Construct the engine without going through MT5Session — that
    class's __init__ tries to import the real MetaTrader5 package."""
    fake_session = SimpleNamespace(mt5=mt5_mock, shutdown=lambda: None)
    return MT5ExecutionEngine(session=fake_session)  # type: ignore[arg-type]


def make_signal(pair: str = "EURUSD") -> TradeSignal:
    return TradeSignal(
        pair=pair,
        side=OrderSide.BUY,
        lot_size=0.10,
        sl_price=1.2300,
        tp_price=1.2400,
        confidence=0.82,
        comment="news",
    )


# ---------------------------------------------------------------------------
# Filling-mode selection (issue #5 regression)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mask,expected",
    [
        (0b10, ORDER_FILLING_IOC),       # only IOC supported
        (0b11, ORDER_FILLING_IOC),       # both supported → prefer IOC
        (0b01, ORDER_FILLING_FOK),       # only FOK supported (issue #5 case)
        (0b00, ORDER_FILLING_RETURN),    # neither → exchange-execution
    ],
)
def test_filling_mode_matches_symbol_capability(mask, expected):
    mt5 = make_mt5_mock(filling_mode_mask=mask)
    engine = make_engine(mt5)

    engine.execute(make_signal())

    request = mt5.order_send.call_args.args[0]
    assert request["type_filling"] == expected


def test_filling_mode_falls_back_to_return_when_symbol_info_missing():
    """If symbol_info() returns None we still need *some* filling mode —
    RETURN is the broadest and won't crash the request build."""
    mt5 = make_mt5_mock(symbol_info_returns_none=True)
    engine = make_engine(mt5)

    engine.execute(make_signal())

    request = mt5.order_send.call_args.args[0]
    assert request["type_filling"] == ORDER_FILLING_RETURN


def test_filling_mode_handles_none_filling_mode_attribute():
    """Some brokers expose the attribute but populate it with None.
    The engine must treat that as 'unknown' and fall through to RETURN."""
    mt5 = make_mt5_mock(filling_mode_mask=None)
    engine = make_engine(mt5)

    engine.execute(make_signal())

    request = mt5.order_send.call_args.args[0]
    assert request["type_filling"] == ORDER_FILLING_RETURN


def test_filling_mode_query_uses_signal_pair():
    """The capability lookup must target the same symbol as the order —
    looking up the wrong symbol re-introduces the original bug for
    multi-pair runs."""
    mt5 = make_mt5_mock(filling_mode_mask=0b01)
    engine = make_engine(mt5)

    engine.execute(make_signal(pair="GBPUSD"))

    mt5.symbol_info.assert_called_with("GBPUSD")


# ---------------------------------------------------------------------------
# Order request shape & result handling
# ---------------------------------------------------------------------------

def test_buy_order_uses_ask_price_and_buy_type():
    mt5 = make_mt5_mock(bid=1.10, ask=1.11)
    engine = make_engine(mt5)

    engine.execute(make_signal())

    request = mt5.order_send.call_args.args[0]
    assert request["type"] == ORDER_TYPE_BUY
    assert request["price"] == 1.11
    assert request["action"] == TRADE_ACTION_DEAL
    assert request["symbol"] == "EURUSD"
    assert request["volume"] == 0.10


def test_sell_order_uses_bid_price_and_sell_type():
    mt5 = make_mt5_mock(bid=1.10, ask=1.11)
    engine = make_engine(mt5)
    sell = TradeSignal(
        pair="EURUSD", side=OrderSide.SELL, lot_size=0.10,
        sl_price=1.20, tp_price=1.05, confidence=0.8, comment="",
    )

    engine.execute(sell)

    request = mt5.order_send.call_args.args[0]
    assert request["type"] == ORDER_TYPE_SELL
    assert request["price"] == 1.10


def test_successful_fill_returns_success_result():
    mt5 = make_mt5_mock()
    engine = make_engine(mt5)

    result = engine.execute(make_signal())

    assert result.success is True
    assert result.order_id == 987654
    assert result.fill_price == pytest.approx(1.2347)


def test_unsupported_filling_retcode_surfaces_in_result():
    """Sanity check: if the broker still rejects with retcode=10030 (e.g.
    a symbol the bitmask lied about), the engine returns a clean failure
    rather than crashing — same handling as any other rejected order."""
    mt5 = make_mt5_mock(order_send_retcode=RETCODE_UNSUPPORTED_FILLING)
    mt5.order_send.return_value = SimpleNamespace(
        retcode=RETCODE_UNSUPPORTED_FILLING,
        order=0,
        price=0.0,
        comment="Unsupported filling mode",
        _asdict=lambda: {"retcode": RETCODE_UNSUPPORTED_FILLING},
    )
    engine = make_engine(mt5)

    result = engine.execute(make_signal())

    assert result.success is False
    assert "10030" in (result.error or "")
    assert "Unsupported filling mode" in (result.error or "")


def test_no_tick_returns_failure_without_calling_order_send():
    mt5 = make_mt5_mock()
    mt5.symbol_info_tick.return_value = None
    engine = make_engine(mt5)

    result = engine.execute(make_signal())

    assert result.success is False
    assert result.error == "no tick data"
    mt5.order_send.assert_not_called()
