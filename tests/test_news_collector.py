"""News collector persistence tests.

Covers:
  * fetch -> persist with rowcount-driven new/duplicate logging
  * load_unanalyzed returns rows with analyzed_at IS NULL only
  * save_analysis stamps the verdict and analyzed_at
  * mark_analysis_skipped stamps analyzed_at without sentiment
  * schema migration: ALTER TABLE on a pre-analysis-column DB
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trading_system.core.interfaces import AnalysisResult, Sentiment
from trading_system.modules.news_collector.collector import NewsApiCollector


def _fake_session(articles: list[dict]):
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = {"status": "ok", "articles": articles}
    response.raise_for_status.return_value = None
    session.get.return_value = response
    return session


def _article(url: str, title: str = "T", description: str = "D") -> dict:
    return {
        "url": url,
        "title": title,
        "description": description,
        "publishedAt": "2026-05-04T12:00:00Z",
        "source": {"name": "Reuters"},
    }


def make_collector(tmp_path: Path, articles: list[dict] | None = None) -> NewsApiCollector:
    return NewsApiCollector(
        api_key="unused",
        sqlite_path=tmp_path / "news.db",
        session=_fake_session(articles or []),
    )


def test_fetch_persists_articles_with_null_analysis(tmp_path):
    coll = make_collector(tmp_path, [_article("https://x/1"), _article("https://x/2")])

    items = coll.fetch(query="forex")

    assert len(items) == 2
    pending = coll.load_unanalyzed()
    assert len(pending) == 2
    assert {i.url for i in pending} == {"https://x/1", "https://x/2"}


def test_fetch_dedupes_by_url(tmp_path):
    coll = make_collector(tmp_path, [_article("https://x/1"), _article("https://x/1")])

    coll.fetch(query="forex")

    pending = coll.load_unanalyzed()
    assert len(pending) == 1


def test_save_analysis_removes_article_from_unanalyzed_queue(tmp_path):
    coll = make_collector(tmp_path, [_article("https://x/1")])
    items = coll.fetch(query="forex")
    item = items[0]

    coll.save_analysis(
        item.article_id,
        AnalysisResult(
            sentiment=Sentiment.BULLISH,
            confidence=0.82,
            pair="EURUSD",
            rationale="hawkish ECB",
        ),
    )

    assert coll.load_unanalyzed() == []

    # Verify the row was actually written with the verdict.
    with sqlite3.connect(tmp_path / "news.db") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT sentiment, confidence, pair, rationale, analyzed_at "
            "FROM news WHERE article_id = ?",
            (item.article_id,),
        ).fetchone()
    assert row["sentiment"] == "bullish"
    assert row["confidence"] == pytest.approx(0.82)
    assert row["pair"] == "EURUSD"
    assert row["rationale"] == "hawkish ECB"
    assert row["analyzed_at"] is not None


def test_mark_analysis_skipped_stamps_without_sentiment(tmp_path):
    coll = make_collector(tmp_path, [_article("https://x/1")])
    items = coll.fetch(query="forex")

    coll.mark_analysis_skipped(items[0].article_id, "no usable Claude result")

    assert coll.load_unanalyzed() == []
    with sqlite3.connect(tmp_path / "news.db") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT sentiment, analyzed_at, rationale FROM news WHERE article_id=?",
            (items[0].article_id,),
        ).fetchone()
    assert row["sentiment"] is None
    assert row["analyzed_at"] is not None
    assert row["rationale"].startswith("skipped:")


def test_load_unanalyzed_respects_limit(tmp_path):
    coll = make_collector(
        tmp_path,
        [_article(f"https://x/{i}") for i in range(5)],
    )
    coll.fetch(query="forex")

    assert len(coll.load_unanalyzed(limit=3)) == 3


def test_schema_migration_adds_columns_on_legacy_db(tmp_path):
    """An older DB created before the analysis columns existed must get
    ALTER TABLE'd, not crash."""
    db_path = tmp_path / "news.db"

    # Build a legacy schema by hand (no analysis columns).
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE news (
                article_id   TEXT PRIMARY KEY,
                source       TEXT,
                title        TEXT,
                description  TEXT,
                url          TEXT,
                published_at TEXT,
                raw_json     TEXT,
                fetched_at   TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "abc",
                "Reuters",
                "Old article",
                "desc",
                "https://x/old",
                datetime(2026, 5, 1, tzinfo=timezone.utc).isoformat(),
                json.dumps({}),
                datetime(2026, 5, 1, tzinfo=timezone.utc).isoformat(),
            ),
        )

    # Constructing the collector should migrate the schema, not raise.
    coll = NewsApiCollector(
        api_key="unused",
        sqlite_path=db_path,
        session=_fake_session([]),
    )

    # Legacy row is still there and queues for analysis.
    pending = coll.load_unanalyzed()
    assert len(pending) == 1
    assert pending[0].article_id == "abc"

    # And the new columns are addressable via save_analysis.
    coll.save_analysis(
        "abc",
        AnalysisResult(
            sentiment=Sentiment.NEUTRAL, confidence=0.4, pair="EURUSD"
        ),
    )
    assert coll.load_unanalyzed() == []
