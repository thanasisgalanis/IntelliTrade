"""MetaTrader 5 execution engine.

Wraps the official ``MetaTrader5`` Python package. Also exposes a
``MT5BrokerInfo`` adapter that satisfies the risk manager's
``BrokerInfoProvider`` protocol so that both modules talk to the same MT5
session.

The MT5 package is Windows-only; this file therefore imports it lazily so
the rest of the codebase can be imported (and tested) on any platform.
"""
from __future__ import annotations

from typing import Any

from trading_system.core.interfaces import (
    ExecutionResult,
    IExecutionEngine,
    OrderSide,
    TradeSignal,
)
from trading_system.core.logger import get_logger
from trading_system.modules.risk_manager.risk_manager import (
    BrokerInfoProvider,
    SymbolSpec,
)

log = get_logger(__name__)


def _import_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MetaTrader5 package is unavailable on this platform. "
            "MT5 features only run on Windows."
        ) from exc
    return mt5


class MT5Session:
    """Single shared connection to the MT5 terminal."""

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str | None = None,
    ) -> None:
        self._mt5 = _import_mt5()
        kwargs = {"login": login, "password": password, "server": server}
        if path:
            kwargs["path"] = path
        if not self._mt5.initialize(**kwargs):
            err = self._mt5.last_error()
            raise RuntimeError(f"MT5 initialize() failed: {err}")
        log.info("MT5 connected to %s as %s", server, login)

    @property
    def mt5(self) -> Any:
        return self._mt5

    def shutdown(self) -> None:
        try:
            self._mt5.shutdown()
            log.info("MT5 session shut down")
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("MT5 shutdown error: %s", exc)


class MT5BrokerInfo(BrokerInfoProvider):
    """Adapter that lets the risk manager query the live MT5 account."""

    def __init__(self, session: MT5Session) -> None:
        self._mt5 = session.mt5

    def get_free_margin(self) -> float:
        info = self._mt5.account_info()
        if info is None:
            log.error("account_info() returned None: %s", self._mt5.last_error())
            return 0.0
        return float(info.margin_free)

    def get_symbol_spec(self, pair: str) -> SymbolSpec | None:
        if not self._mt5.symbol_select(pair, True):
            log.error("symbol_select(%s) failed: %s", pair, self._mt5.last_error())
            return None
        info = self._mt5.symbol_info(pair)
        tick = self._mt5.symbol_info_tick(pair)
        if info is None or tick is None:
            log.error("symbol_info/tick missing for %s", pair)
            return None
        pip_size = info.point * 10  # 1 pip = 10 points on 5-digit brokers
        # MT5 trade_tick_value is the value of one tick (point) per 1 lot
        # in account currency; multiply by 10 to convert tick -> pip.
        pip_value_per_lot = float(info.trade_tick_value) * 10
        return SymbolSpec(
            pair=pair,
            pip_size=pip_size,
            pip_value_per_lot=pip_value_per_lot,
            min_lot=float(info.volume_min),
            max_lot=float(info.volume_max),
            lot_step=float(info.volume_step),
            bid=float(tick.bid),
            ask=float(tick.ask),
            digits=int(info.digits),
        )


class MT5ExecutionEngine(IExecutionEngine):
    def __init__(
        self,
        session: MT5Session,
        magic_number: int = 20260504,
        max_slippage_points: int = 10,
    ) -> None:
        self._mt5 = session.mt5
        self._session = session
        self._magic = magic_number
        self._slippage = max_slippage_points

    # ------------------------------------------------------------------
    def execute(self, signal: TradeSignal) -> ExecutionResult:
        mt5 = self._mt5
        tick = mt5.symbol_info_tick(signal.pair)
        if tick is None:
            return ExecutionResult(False, None, None, error="no tick data")

        if signal.side is OrderSide.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.pair,
            "volume": signal.lot_size,
            "type": order_type,
            "price": price,
            "sl": signal.sl_price,
            "tp": signal.tp_price,
            "deviation": self._slippage,
            "magic": self._magic,
            "comment": signal.comment[:31],  # MT5 caps comment length
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            log.error("order_send returned None: %s", err)
            return ExecutionResult(False, None, None, error=str(err))

        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            log.error(
                "Order rejected for %s: retcode=%s comment=%s",
                signal.pair, result.retcode, getattr(result, "comment", ""),
            )
            return ExecutionResult(
                success=False,
                order_id=getattr(result, "order", None),
                fill_price=None,
                error=f"retcode={result.retcode} {getattr(result, 'comment', '')}",
                raw=result._asdict() if hasattr(result, "_asdict") else {},
            )

        log.info(
            "FILLED %s %s lots=%.2f price=%.5f order=%s",
            signal.pair, signal.side.value, signal.lot_size,
            float(result.price), result.order,
        )
        return ExecutionResult(
            success=True,
            order_id=result.order,
            fill_price=float(result.price),
            raw=result._asdict() if hasattr(result, "_asdict") else {},
        )

    def shutdown(self) -> None:
        self._session.shutdown()
