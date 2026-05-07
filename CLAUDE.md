# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**IntelliTrade** is an automated trading system that uses artificial intelligence for decision-making. The project is implemented using AI under programmer supervision.

Phase 1 MVP scope: trade Forex news events by collecting articles from NewsAPI, analysing them with Anthropic Claude, gating signals through a strict risk manager, and executing market orders on MetaTrader 5.

> Phase 1 source code currently lives on branch `feature/phase-1-news-trading-mvp` (commit `37bb170`) and has not yet been merged to `main`. On `main` the working tree only carries leftover `__pycache__/` directories from a prior checkout — switch branches to see the actual code.

Per the project's working agreement (`context/0-initial-prompt.md`): each new phase starts on its own git branch and is merged to `main` only when the phase is verified complete.

## Architecture

Modular monolith in Python. Modules are decoupled through Abstract Base Classes in `trading_system/core/interfaces.py`; the orchestrator depends only on those interfaces, so any implementation (news source, analyzer, broker) can be swapped without touching business logic.

```
trading_system/
├── core/
│   ├── interfaces.py    # INewsCollector, INewsAnalyzer, IRiskManager,
│   │                    # IExecutionEngine + domain dataclasses (NewsItem,
│   │                    # AnalysisResult, TradeSignal, ExecutionResult)
│   └── logger.py        # Rotating file (5 MB × 5) + console; idempotent
├── modules/
│   ├── news_collector/  # NewsApiCollector — NewsAPI → SQLite, dedup by URL hash
│   ├── news_analyzer/   # ClaudeNewsAnalyzer — Anthropic SDK, strict JSON-only
│   │                    # system prompt with ephemeral prompt caching
│   ├── risk_manager/    # FixedPercentRiskManager — enforces 1%-of-free-margin
│   │                    # sizing and a min-confidence gate (default 0.70)
│   └── execution_engine/# MT5ExecutionEngine + MT5BrokerInfo adapter
└── main.py              # End-to-end orchestrator
tests/                   # pytest unit tests (mocked broker + mocked Anthropic SDK)
context/                 # Free-form planning notes / prior session transcripts
data/                    # SQLite news store (gitignored)
logs/                    # Rotating log files (gitignored)
```

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
- `RISK_PERCENT` (default `1.0`), `MIN_CONFIDENCE` (default `0.70`)
- `DEFAULT_PAIRS`, `DEFAULT_SL_PIPS`, `DEFAULT_TP_PIPS`, `MAX_SLIPPAGE_POINTS`, `MAGIC_NUMBER`
- `SQLITE_PATH`, `LOG_FILE`, `LOG_LEVEL`

## GitHub Remote

`git@github.com:thanasisgalanis/IntelliTrade.git`

> Note: the GitHub repo is spelled **IntelliTrade** (double-l) while the local working directory is `InteliTrade` (single-l). Worth aligning eventually.
