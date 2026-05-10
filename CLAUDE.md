# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**IntelliTrade** is an automated trading system that uses artificial intelligence for decision-making. The project is implemented using AI under programmer supervision.

Phase 1 MVP scope: trade Forex news events by collecting articles from NewsAPI, analysing them with Anthropic Claude, gating signals through a strict risk manager, and executing market orders on MetaTrader 5.

Phase 1 has been merged to `main` (PR #6). Current active work: branch `feature/batch-news-analysis`, which restructures the analyzer to issue **one Claude call per chunk of N articles**, store per-article verdicts in SQLite, and then aggregate per pair via a configurable strategy (see "Pipeline" below).

Per the project's working agreement (`context/0-initial-prompt.md`): each new phase starts on its own git branch and is merged to `main` only when the phase is verified complete.

## Architecture

Modular monolith in Python. Modules are decoupled through Abstract Base Classes in `trading_system/core/interfaces.py`; the orchestrator depends only on those interfaces, so any implementation (news source, analyzer, broker) can be swapped without touching business logic.

```
trading_system/
├── core/
│   ├── interfaces.py    # INewsCollector (now also: load_unanalyzed,
│   │                    # save_analysis, mark_analysis_skipped),
│   │                    # INewsAnalyzer (analyze + analyze_batch),
│   │                    # IRiskManager, IExecutionEngine + domain
│   │                    # dataclasses (NewsItem, AnalysisResult,
│   │                    # TradeSignal, ExecutionResult).
│   └── logger.py        # Rotating file (5 MB × 5) + console; idempotent
├── modules/
│   ├── news_collector/  # NewsApiCollector — NewsAPI → SQLite. Schema is
│   │                    # auto-migrated to add analysis columns
│   │                    # (sentiment/confidence/pair/rationale/analyzed_at).
│   │                    # Articles with analyzed_at IS NULL form the work
│   │                    # queue across runs.
│   ├── news_analyzer/   # ClaudeNewsAnalyzer.analyze_batch() pre-filters
│   │                    # articles whose only pair tokens are non-allowed,
│   │                    # then sends the rest to Claude in chunks of
│   │                    # BATCH_MAX_SIZE (default 50) using
│   │                    # _BATCH_ARRAY_SYSTEM_PROMPT. Returns
│   │                    # dict[article_id -> AnalysisResult].
│   │                    # Ephemeral prompt caching on system prompts.
│   │                    # aggregator.py collapses per-article results into
│   │                    # one AnalysisResult per pair via the configured
│   │                    # strategy: max_confidence, average_confidence,
│   │                    # majority_sentiment, weighted_average.
│   ├── risk_manager/    # FixedPercentRiskManager — enforces 1%-of-free-margin
│   │                    # sizing and a min-confidence gate (default 0.70)
│   └── execution_engine/# MT5ExecutionEngine + MT5BrokerInfo adapter.
│                        # _select_filling_mode() reads symbol_info.filling_mode
│                        # bitmask and picks IOC → FOK → RETURN per symbol to
│                        # avoid retcode=10030 ("Unsupported filling mode").
└── main.py              # End-to-end orchestrator
tests/                   # pytest unit tests (mocked broker + mocked Anthropic SDK)
context/                 # Free-form planning notes / prior session transcripts
data/                    # SQLite news store (gitignored)
logs/                    # Rotating log files (gitignored)
```

## Pipeline

`run_once()` in `trading_system/main.py` runs four stages, each separated in the log by a blank line plus a `=== Stage N/4 ... ===` banner so a single run is scannable:

1. **Fetch** — `collector.fetch()` pulls fresh articles from NewsAPI and inserts them into SQLite (`INSERT OR IGNORE` on `article_id = sha1(url)`). The log distinguishes new vs. duplicate inserts via `cursor.rowcount`.
2. **Load work queue** — `collector.load_unanalyzed()` returns every row where `analyzed_at IS NULL`, including any rows left behind by a previous crashed run.
3. **Batch analyze + persist** — `analyzer.analyze_batch(pending)` makes one Claude call per chunk of `BATCH_MAX_SIZE` and returns `{article_id: AnalysisResult}`. Each result is written back to its row via `collector.save_analysis()`. Articles for which Claude produced no usable result are stamped via `mark_analysis_skipped()` so they don't re-enter the queue.
4. **Aggregate, gate, execute** — `aggregate_by_pair()` collapses N per-article results for the same pair into one `AnalysisResult` using `AGGREGATION_STRATEGY`. Each consolidated signal goes through the risk manager and, if accepted, the execution engine.

Key design rules carried over from the initial prompt:
- Every module must be self-contained enough that it could later be reimplemented in a different language behind the same API.
- The risk manager has two non-negotiable gates: confidence ≥ `min_confidence` **and** lot size = exactly `risk_percent` % of free margin given the SL distance.
- `MetaTrader5` is imported lazily because the package is Windows-only; the rest of the codebase must remain importable (and testable) on macOS/Linux.

## Commands

Setup (one-time):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # MetaTrader5 is Windows-only; on macOS/Linux
                                       # install the rest manually:
                                       # pip install anthropic requests python-dotenv pytest
cp .env.example .env                   # then fill in MT5 / NewsAPI / Anthropic creds
```

Run the test suite (no MT5 or live API keys required — both broker and Anthropic SDK are mocked):

```bash
pytest
```

Run the live pipeline (requires a populated `.env` and a running MT5 terminal):

```bash
python -m trading_system.main
```

## Configuration

Runtime config is loaded from `.env` via `python-dotenv`. See `.env.example` for the full list; notable keys:

- `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH` — broker credentials
- `NEWSAPI_KEY`, `NEWS_QUERY`, `NEWS_LANGUAGE`, `NEWS_PAGE_SIZE` — news source
- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` (default `claude-sonnet-4-6`)
- `BATCH_MAX_SIZE` (default `50`) — max articles per single Claude call
- `AGGREGATION_STRATEGY` (default `average_confidence`) — how to collapse N per-article results into one per-pair signal. Valid: `max_confidence`, `average_confidence`, `majority_sentiment`, `weighted_average`
- `WEIGHTED_NEUTRAL_BAND` (default `0.10`) — only used by `weighted_average`; `|score|` below the band falls to neutral
- `RISK_PERCENT` (default `1.0`), `MIN_CONFIDENCE` (default `0.70`)
- `ALLOWED_PAIRS`, `DEFAULT_SL_PIPS`, `DEFAULT_TP_PIPS`, `MAX_SLIPPAGE_POINTS`, `MAGIC_NUMBER`
- `SQLITE_PATH`, `LOG_FILE`, `LOG_LEVEL`

## GitHub Remote

`git@github.com:thanasisgalanis/IntelliTrade.git`

## Rules

- Always ask to update claude.md (this file), after any change in our code, so it reflects the current status of our codebase.