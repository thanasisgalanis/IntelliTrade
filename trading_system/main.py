"""End-to-end orchestrator for the news-trading MVP.

Pipeline:  collector -> analyzer -> risk manager -> execution engine.

Each step is wired through its abstract interface, so any module can be
swapped without touching this file.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from trading_system.core.interfaces import (
    AnalysisResult,
    IExecutionEngine,
    INewsAnalyzer,
    INewsCollector,
    IRiskManager,
)
from trading_system.core.logger import configure_logging, get_logger
from trading_system.modules.execution_engine import MT5ExecutionEngine
from trading_system.modules.execution_engine.engine import MT5BrokerInfo, MT5Session
from trading_system.modules.news_analyzer import ClaudeNewsAnalyzer
from trading_system.modules.news_collector import NewsApiCollector
from trading_system.modules.risk_manager import FixedPercentRiskManager


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val or ""


def build_pipeline() -> tuple[
    INewsCollector, INewsAnalyzer, IRiskManager, IExecutionEngine
]:
    session = MT5Session(
        login=int(_env("MT5_LOGIN", required=True)),
        password=_env("MT5_PASSWORD", required=True),
        server=_env("MT5_SERVER", required=True),
        path=_env("MT5_PATH") or None,
    )

    allowed_pairs = {
        p.strip().upper()
        for p in _env("DEFAULT_PAIRS", "EURUSD").split(",")
        if p.strip()
    }

    collector = NewsApiCollector(
        api_key=_env("NEWSAPI_KEY", required=True),
        sqlite_path=_env("SQLITE_PATH", "data/news.db"),
        language=_env("NEWS_LANGUAGE", "en"),
    )
    analyzer = ClaudeNewsAnalyzer(
        api_key=_env("ANTHROPIC_API_KEY", required=True),
        model=_env("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        allowed_pairs=allowed_pairs,
    )
    risk = FixedPercentRiskManager(
        broker=MT5BrokerInfo(session),
        risk_percent=float(_env("RISK_PERCENT", "1.0")),
        min_confidence=float(_env("MIN_CONFIDENCE", "0.70")),
    )
    engine = MT5ExecutionEngine(
        session=session,
        magic_number=int(_env("MAGIC_NUMBER", "20260504")),
        max_slippage_points=int(_env("MAX_SLIPPAGE_POINTS", "10")),
    )
    return collector, analyzer, risk, engine


def run_once() -> int:
    log = get_logger("intelitrade")
    collector, analyzer, risk, engine = build_pipeline()

    sl_pips = float(_env("DEFAULT_SL_PIPS", "20"))
    tp_pips = float(_env("DEFAULT_TP_PIPS", "40"))
    page_size = int(_env("NEWS_PAGE_SIZE", "20"))
    query = _env("NEWS_QUERY", "forex")

    placed = 0
    try:
        items = collector.fetch(query=query, page_size=page_size)
        log.info("Pipeline starting on %d articles", len(items))

        analyses: list[AnalysisResult] = analyzer.analyze_many(items)
        log.info("Analyzer produced %d consolidated signal(s)", len(analyses))

        for analysis in analyses:
            signal = risk.evaluate(analysis, sl_pips=sl_pips, tp_pips=tp_pips)
            if signal is None:
                continue

            result = engine.execute(signal)
            if result.success:
                placed += 1
            else:
                log.warning("Execution failed for %s: %s", signal.pair, result.error)
    finally:
        engine.shutdown()

    log.info("Pipeline complete: %d order(s) placed", placed)
    return placed


def main() -> int:
    load_dotenv()
    configure_logging(
        log_file=os.getenv("LOG_FILE", "logs/intelitrade.log"),
        level=os.getenv("LOG_LEVEL", "INFO"),
    )
    try:
        run_once()
        return 0
    except Exception as exc:
        get_logger("intelitrade").exception("Fatal: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
