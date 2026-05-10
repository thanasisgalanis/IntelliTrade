"""End-to-end orchestrator for the news-trading MVP.

Pipeline:

    1. Fetch fresh articles from NewsAPI and persist them.
    2. Load every article in the DB whose ``analyzed_at`` is null
       (i.e. the work queue carried across runs).
    3. Send those articles to Claude in chunks of at most BATCH_MAX_SIZE,
       receiving one verdict per article. Persist each verdict to its row.
    4. Aggregate per-article verdicts into one signal per pair using the
       configured AGGREGATION_STRATEGY.
    5. Run each per-pair signal through the risk manager and execution
       engine.

Each step is wired through its abstract interface, so any module can be
swapped without touching this file.

Logging note: section dividers prepend ``\\n`` to the next log message so
they produce a real empty line in both console and file outputs, which
makes the per-pair / per-stage groupings visually scannable.
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
    NewsItem,
)
from trading_system.core.logger import configure_logging, get_logger
from trading_system.modules.execution_engine import MT5ExecutionEngine
from trading_system.modules.execution_engine.engine import MT5BrokerInfo, MT5Session
from trading_system.modules.news_analyzer import (
    AggregationConfig,
    ClaudeNewsAnalyzer,
    aggregate_by_pair,
)
from trading_system.modules.news_collector import NewsApiCollector
from trading_system.modules.risk_manager import FixedPercentRiskManager


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val or ""


def build_pipeline() -> tuple[
    INewsCollector,
    INewsAnalyzer,
    IRiskManager,
    IExecutionEngine,
    AggregationConfig,
]:
    session = MT5Session(
        login=int(_env("MT5_LOGIN", required=True)),
        password=_env("MT5_PASSWORD", required=True),
        server=_env("MT5_SERVER", required=True),
        path=_env("MT5_PATH") or None,
    )

    allowed_pairs = {
        p.strip().upper()
        for p in _env("ALLOWED_PAIRS", "EURUSD").split(",")
        if p.strip()
    }

    batch_max_size = int(_env("BATCH_MAX_SIZE", "50"))

    collector = NewsApiCollector(
        api_key=_env("NEWSAPI_KEY", required=True),
        sqlite_path=_env("SQLITE_PATH", "data/news.db"),
        language=_env("NEWS_LANGUAGE", "en"),
    )
    analyzer = ClaudeNewsAnalyzer(
        api_key=_env("ANTHROPIC_API_KEY", required=True),
        model=_env("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        allowed_pairs=allowed_pairs,
        batch_max_size=batch_max_size,
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
    aggregation = AggregationConfig.from_env(
        strategy=os.getenv("AGGREGATION_STRATEGY"),
        neutral_band=os.getenv("WEIGHTED_NEUTRAL_BAND"),
    )
    return collector, analyzer, risk, engine, aggregation


def run_once() -> int:
    log = get_logger("intellitrade")
    collector, analyzer, risk, engine, aggregation = build_pipeline()

    sl_pips = float(_env("DEFAULT_SL_PIPS", "20"))
    tp_pips = float(_env("DEFAULT_TP_PIPS", "40"))
    page_size = int(_env("NEWS_PAGE_SIZE", "20"))
    query = _env("NEWS_QUERY", "forex")

    placed = 0
    try:
        # ----- 1. Fetch -------------------------------------------------
        log.info("\n=== Stage 1/4: Fetching news ===")
        fetched = collector.fetch(query=query, page_size=page_size)
        log.info(
            "Fetch complete: %d article(s) returned by NewsAPI",
            len(fetched),
        )

        # ----- 2. Load work queue --------------------------------------
        log.info("\n=== Stage 2/4: Loading unanalyzed queue ===")
        pending = collector.load_unanalyzed()
        log.info(
            "Work queue: %d article(s) pending analysis (analyzed_at IS NULL)",
            len(pending),
        )

        # ----- 3. Batch-analyze + persist results ----------------------
        log.info("\n=== Stage 3/4: Sending batch(es) to Claude ===")
        per_article = _analyze_and_persist(analyzer, collector, pending, log)

        # ----- 4. Aggregate, gate, execute -----------------------------
        log.info("\n=== Stage 4/4: Aggregating per pair and trading ===")
        analyses = aggregate_by_pair(list(per_article.values()), aggregation)
        log.info(
            "Aggregator produced %d consolidated signal(s) using strategy=%s",
            len(analyses),
            aggregation.strategy,
        )

        for analysis in analyses:
            log.info("\n--- Pair %s ---", analysis.pair)
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

    log.info("\n=== Pipeline complete: %d order(s) placed ===", placed)
    return placed


def _analyze_and_persist(
    analyzer: INewsAnalyzer,
    collector: INewsCollector,
    pending: list[NewsItem],
    log,
) -> dict[str, AnalysisResult]:
    """Run the batch analyzer and write each verdict back to the DB.

    Articles in ``pending`` that the analyzer dropped (pre-filter, timeout,
    bad JSON) are stamped via :meth:`mark_analysis_skipped` so they do not
    re-enter the queue on the next run.
    """
    if not pending:
        log.info("Nothing to analyze; skipping Claude call")
        return {}

    per_article = analyzer.analyze_batch(pending)
    log.info(
        "Claude returned %d/%d analysis result(s)",
        len(per_article),
        len(pending),
    )

    persisted = 0
    for article_id, result in per_article.items():
        collector.save_analysis(article_id, result)
        persisted += 1
    log.info("Persisted %d analysis result(s) to news.db", persisted)

    # Stamp the rest as processed-but-skipped so we don't re-send them.
    skipped_ids = [
        item.article_id for item in pending if item.article_id not in per_article
    ]
    for article_id in skipped_ids:
        collector.mark_analysis_skipped(article_id, "no usable Claude result")
    if skipped_ids:
        log.info(
            "Stamped %d article(s) as analyzed-but-skipped (no result from Claude)",
            len(skipped_ids),
        )

    return per_article


def main() -> int:
    load_dotenv()
    configure_logging(
        log_file=os.getenv("LOG_FILE", "logs/intellitrade.log"),
        level=os.getenv("LOG_LEVEL", "INFO"),
    )
    try:
        run_once()
        return 0
    except Exception as exc:
        get_logger("intellitrade").exception("Fatal: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
