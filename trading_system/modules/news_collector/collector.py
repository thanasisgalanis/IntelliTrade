"""NewsAPI -> SQLite collector.

The raw JSON of every article is persisted before any processing so that
nothing is lost if the analyzer or executor crashes downstream.

The ``news`` table also carries the result of the Claude analysis once it
runs (``sentiment``/``confidence``/``pair``/``rationale``/``analyzed_at``).
Articles with ``analyzed_at IS NULL`` are the work queue: they have been
fetched and stored but not yet sent to Claude.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

from trading_system.core.interfaces import (
    AnalysisResult,
    INewsCollector,
    NewsItem,
)
from trading_system.core.logger import get_logger

log = get_logger(__name__)

_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_DEFAULT_TIMEOUT = 10  # seconds

# Columns added after the original schema. Listed here so a fresh DB
# (handled by CREATE TABLE) and an upgraded one (handled by ALTER TABLE)
# converge on the same shape without duplicate definitions.
_ANALYSIS_COLUMNS: tuple[tuple[str, str], ...] = (
    ("sentiment", "TEXT"),
    ("confidence", "REAL"),
    ("pair", "TEXT"),
    ("rationale", "TEXT"),
    ("analyzed_at", "TEXT"),
)


class NewsApiCollector(INewsCollector):
    def __init__(
        self,
        api_key: str,
        sqlite_path: str | Path,
        language: str = "en",
        timeout: float = _DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("NewsAPI key is required")
        self._api_key = api_key
        self._language = language
        self._timeout = timeout
        self._session = session or requests.Session()
        self._db_path = Path(sqlite_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------ db
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS news (
                    article_id   TEXT PRIMARY KEY,
                    source       TEXT,
                    title        TEXT,
                    description  TEXT,
                    url          TEXT,
                    published_at TEXT,
                    raw_json     TEXT,
                    fetched_at   TEXT,
                    sentiment    TEXT,
                    confidence   REAL,
                    pair         TEXT,
                    rationale    TEXT,
                    analyzed_at  TEXT
                )
                """
            )
            existing = {
                row["name"] for row in conn.execute("PRAGMA table_info(news)")
            }
            for col, ddl in _ANALYSIS_COLUMNS:
                if col not in existing:
                    conn.execute(f"ALTER TABLE news ADD COLUMN {col} {ddl}")

    # -------------------------------------------------------------- public
    def fetch(self, query: str, page_size: int = 20) -> list[NewsItem]:
        params = {
            "q": query,
            "language": self._language,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "apiKey": self._api_key,
        }
        try:
            resp = self._session.get(_NEWSAPI_URL, params=params, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.error("NewsAPI request failed: %s", exc)
            return []

        payload = resp.json()
        if payload.get("status") != "ok":
            log.error("NewsAPI returned non-ok status: %s", payload)
            return []

        items: list[NewsItem] = []
        for article in payload.get("articles", []):
            item = self._article_to_item(article)
            if item is None:
                continue
            items.append(item)

        new_count = self._persist(items)
        log.info(
            "NewsAPI fetched %d article(s) for query=%r (%d new, %d duplicate)",
            len(items),
            query,
            new_count,
            len(items) - new_count,
        )
        return items

    def load_recent(self, limit: int = 50) -> list[NewsItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM news ORDER BY published_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def load_unanalyzed(self, limit: int | None = None) -> list[NewsItem]:
        """Return articles that have been stored but not yet analyzed.

        ``analyzed_at IS NULL`` is the single source of truth for the work
        queue, so a crash between fetch and analyze leaves the next run
        able to pick up where the previous one stopped.
        """
        sql = (
            "SELECT * FROM news WHERE analyzed_at IS NULL "
            "ORDER BY published_at ASC"
        )
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_item(r) for r in rows]

    def save_analysis(self, article_id: str, result: AnalysisResult) -> None:
        """Persist a per-article Claude result and stamp ``analyzed_at``."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE news
                   SET sentiment   = ?,
                       confidence  = ?,
                       pair        = ?,
                       rationale   = ?,
                       analyzed_at = ?
                 WHERE article_id  = ?
                """,
                (
                    result.sentiment.value,
                    float(result.confidence),
                    result.pair,
                    result.rationale,
                    now,
                    article_id,
                ),
            )

    def mark_analysis_skipped(self, article_id: str, reason: str) -> None:
        """Stamp ``analyzed_at`` for articles we deliberately did not send
        to Claude (e.g. only-non-allowed-pair pre-filter), so they don't
        re-enter the queue on the next run. The reason goes into
        ``rationale`` for traceability; sentiment/confidence/pair stay null.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE news
                   SET rationale   = ?,
                       analyzed_at = ?
                 WHERE article_id  = ?
                """,
                (f"skipped: {reason}", now, article_id),
            )

    # -------------------------------------------------------- helpers
    @staticmethod
    def _article_to_item(article: dict) -> NewsItem | None:
        url = article.get("url")
        title = article.get("title")
        if not url or not title:
            return None
        published_raw = article.get("publishedAt") or ""
        try:
            published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
        except ValueError:
            published = datetime.now(timezone.utc)
        article_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
        source = (article.get("source") or {}).get("name", "unknown")
        return NewsItem(
            article_id=article_id,
            source=source,
            title=title,
            description=article.get("description") or "",
            url=url,
            published_at=published,
            raw=article,
        )

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> NewsItem:
        try:
            raw = json.loads(row["raw_json"]) if row["raw_json"] else {}
        except json.JSONDecodeError:
            raw = {}
        return NewsItem(
            article_id=row["article_id"],
            source=row["source"] or "unknown",
            title=row["title"] or "",
            description=row["description"] or "",
            url=row["url"] or "",
            published_at=datetime.fromisoformat(row["published_at"]),
            raw=raw,
        )

    def _persist(self, items: list[NewsItem]) -> int:
        """Insert items, ignoring URL-hash collisions. Returns the count
        of *newly inserted* rows so the caller can log new vs. duplicate.
        """
        if not items:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO news
                (article_id, source, title, description, url, published_at, raw_json, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        i.article_id,
                        i.source,
                        i.title,
                        i.description,
                        i.url,
                        i.published_at.isoformat(),
                        json.dumps(i.raw),
                        now,
                    )
                    for i in items
                ],
            )
            return cursor.rowcount or 0
