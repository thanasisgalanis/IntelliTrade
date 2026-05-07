"""Anthropic Claude-backed news analyzer.

The system prompt forces the model to return ONLY a JSON object with the
exact schema defined in ``_REQUIRED_KEYS``. Any deviation (extra prose,
malformed JSON, schema mismatch, network error, timeout) is caught and the
analyzer returns ``None`` instead of crashing the orchestrator.

The static system prompt is marked ``cache_control: ephemeral`` so that
Anthropic prompt-caching reuses it across calls.
"""
from __future__ import annotations

import json
import re
from typing import Any

import anthropic

from trading_system.core.interfaces import (
    AnalysisResult,
    INewsAnalyzer,
    NewsItem,
    Sentiment,
)
from trading_system.core.logger import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_TIMEOUT = 20.0  # seconds
_REQUIRED_KEYS = {"sentiment", "confidence", "pair"}
_VALID_SENTIMENTS = {s.value for s in Sentiment}

_SYSTEM_PROMPT = """You are a Forex news sentiment analyst.
Read the article supplied by the user and determine its likely short-term
impact on a single, specific currency pair.

You MUST respond with a single JSON object and NOTHING else — no prose,
no markdown fences, no preamble. The schema is:

{
  "sentiment":  "bullish" | "bearish" | "neutral",
  "confidence": <float between 0.0 and 1.0>,
  "pair":       "<6-letter ISO pair, e.g. EURUSD>"
}

Rules:
- "sentiment" is from the perspective of the BASE currency in "pair".
- "confidence" reflects how strongly the article supports the call; use
  values below 0.50 if the article is ambiguous or off-topic.
- "pair" must be exactly 6 uppercase letters. If multiple pairs are
  plausible, pick the single most affected one.
- If the article is irrelevant to FX, return neutral sentiment with low
  confidence and your best-guess pair.
- Output JSON only. No code fences. No commentary.
"""


class ClaudeNewsAnalyzer(INewsAnalyzer):
    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        timeout: float = _DEFAULT_TIMEOUT,
        max_tokens: int = 200,
        client: anthropic.Anthropic | None = None,
        allowed_pairs: set[str] | None = None,
    ) -> None:
        if client is None and not api_key:
            raise ValueError("Anthropic API key is required when no client is injected")
        self._client = client or anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens
        self._allowed_pairs = allowed_pairs

    # ------------------------------------------------------------------
    def analyze(self, item: NewsItem) -> AnalysisResult | None:
        user_msg = self._build_user_message(item)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
        except anthropic.APITimeoutError:
            log.warning("Claude timed out analyzing article %s", item.article_id)
            return None
        except anthropic.APIError as exc:
            log.error("Claude API error on %s: %s", item.article_id, exc)
            return None
        except Exception as exc:  # last-resort safety net
            log.exception("Unexpected error calling Claude: %s", exc)
            return None

        text = self._extract_text(response)
        parsed = self._parse_json(text)
        if parsed is None:
            log.warning("Claude returned non-JSON for %s: %r", item.article_id, text[:200])
            return None

        return self._validate(parsed, item.article_id)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_user_message(item: NewsItem) -> str:
        return (
            f"Source: {item.source}\n"
            f"Published: {item.published_at.isoformat()}\n"
            f"Title: {item.title}\n"
            f"Description: {item.description}\n"
            f"URL: {item.url}"
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            blocks = response.content or []
            return "".join(getattr(b, "text", "") for b in blocks).strip()
        except AttributeError:
            return ""

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Tolerate stray prose around the JSON object.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _validate(self, payload: dict, article_id: str) -> AnalysisResult | None:
        missing = _REQUIRED_KEYS - payload.keys()
        if missing:
            log.warning("Claude payload missing keys %s for %s", missing, article_id)
            return None

        sentiment_raw = str(payload["sentiment"]).lower()
        if sentiment_raw not in _VALID_SENTIMENTS:
            log.warning("Invalid sentiment %r for %s", sentiment_raw, article_id)
            return None

        try:
            confidence = float(payload["confidence"])
        except (TypeError, ValueError):
            log.warning("Non-numeric confidence for %s: %r", article_id, payload["confidence"])
            return None
        if not 0.0 <= confidence <= 1.0:
            log.warning("Confidence out of range for %s: %s", article_id, confidence)
            return None

        pair = str(payload["pair"]).upper().replace("/", "").replace("-", "")
        if not re.fullmatch(r"[A-Z]{6}", pair):
            log.warning("Invalid pair %r for %s", pair, article_id)
            return None
        if self._allowed_pairs and pair not in self._allowed_pairs:
            log.info("Pair %s not in allowlist; skipping %s", pair, article_id)
            return None

        return AnalysisResult(
            sentiment=Sentiment(sentiment_raw),
            confidence=confidence,
            pair=pair,
            rationale=str(payload.get("rationale", "")),
        )
