"""NewsAPI -> SQLite collector.

The raw JSON of every article is persisted before any processing so that
nothing is lost if the analyzer or executor crashes downstream.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

from trading_system.core.interfaces import INewsCollector, NewsItem
from trading_system.core.logger import get_logger

log = get_logger(__name__)

_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_DEFAULT_TIMEOUT = 10  # seconds


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
                    fetched_at   TEXT
                )
                """
            )

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

        self._persist(items)
        log.info("Fetched %d articles for query=%r", len(items), query)
        return items

    def load_recent(self, limit: int = 50) -> list[NewsItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM news ORDER BY published_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_item(r) for r in rows]

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

    def _persist(self, items: list[NewsItem]) -> None:
        if not items:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.executemany(
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
