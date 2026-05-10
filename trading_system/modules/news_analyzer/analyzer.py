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

# Three orthogonal pair-mention forms. Each is matched independently so a
# match in one form does not consume characters that belong to another (the
# non-overlapping behaviour of ``finditer`` would otherwise cause e.g.
# "USD/INR AND EUR/USD" to swallow "AND EUR" between the two real pairs).
# Both halves are post-validated against ``_FX_CURRENCIES`` to suppress
# false positives from adjacent 3-letter acronyms (e.g. "THE ECB").
_PAIR_RE_CONTIGUOUS = re.compile(r"\b([A-Z]{6})\b")
_PAIR_RE_DELIMITED = re.compile(r"\b([A-Z]{3})[/\-]([A-Z]{3})\b")
_PAIR_RE_SPACED = re.compile(r"\b([A-Z]{3})\s+([A-Z]{3})\b")

# ISO 4217 codes for currencies that appear in mainstream FX coverage. The
# list is intentionally conservative — adding a code only widens the pre-
# filter (more articles get a Claude call), it never causes a wrongful skip.
_FX_CURRENCIES: frozenset[str] = frozenset({
    # Majors
    "USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF",
    # Asia
    "CNY", "CNH", "HKD", "SGD", "KRW", "TWD", "INR", "IDR",
    "MYR", "PHP", "THB", "VND", "PKR", "BDT", "LKR", "NPR",
    # EMEA
    "NOK", "SEK", "DKK", "ISK", "PLN", "HUF", "CZK", "RON",
    "BGN", "HRK", "RUB", "TRY", "ILS", "ZAR", "EGP", "NGN",
    "KES", "MAD", "TND",
    # Middle East
    "SAR", "AED", "QAR", "KWD", "BHD", "OMR", "JOD",
    # Americas
    "MXN", "BRL", "ARS", "CLP", "COP", "PEN", "UYU",
})

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

_BATCH_SYSTEM_PROMPT = """You are a Forex news sentiment analyst.
The user supplies a TARGET pair and a numbered list of news articles
about it. Synthesize ALL of them into a single short-term outlook for
that pair — do not analyse the articles separately.

You MUST respond with a single JSON object and NOTHING else — no prose,
no markdown fences, no preamble. The schema is:

{
  "sentiment":  "bullish" | "bearish" | "neutral",
  "confidence": <float between 0.0 and 1.0>,
  "pair":       "<the target pair, echoed verbatim>"
}

Rules:
- "sentiment" is from the perspective of the BASE currency in "pair".
- "confidence" reflects how strongly the *combined* articles support the
  call. If they contradict each other, lower the confidence.
- "pair" must echo the target pair supplied by the user.
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
        if not self._mentions_allowed_pair(item):
            log.info(
                "Article %s only mentions non-allowed pairs; skipping Claude call",
                item.article_id,
            )
            return None

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
    # Batched analysis (issue #3)
    # ------------------------------------------------------------------
    def analyze_many(self, items: list[NewsItem]) -> list[AnalysisResult]:
        """Analyze a batch of articles, consolidating multiple stories
        about the same pair into a single Claude call.

        Strategy:
          * Articles whose text explicitly names exactly one allowed pair
            are grouped by that pair; each group is sent in **one** Claude
            call that returns a single consolidated AnalysisResult.
          * Articles that don't pin to a single allowed pair (zero or many
            matches) fall back to per-item :meth:`analyze`, where Claude
            picks the pair on its own.
          * Results are deduplicated by pair (highest-confidence wins) so a
            grouped batch and an ungrouped fallback can never produce two
            competing signals for the same instrument.
        """
        groups: dict[str, list[NewsItem]] = {}
        ungrouped: list[NewsItem] = []
        for item in items:
            mentions = self._allowed_pair_mentions(item)
            if len(mentions) == 1:
                groups.setdefault(next(iter(mentions)), []).append(item)
            else:
                ungrouped.append(item)

        results: list[AnalysisResult] = []
        for pair, batch in groups.items():
            res = self._analyze_pair(pair, batch)
            if res is not None:
                results.append(res)
        for item in ungrouped:
            res = self.analyze(item)
            if res is not None:
                results.append(res)

        return self._dedupe_by_pair(results)

    def _analyze_pair(
        self, pair: str, items: list[NewsItem]
    ) -> AnalysisResult | None:
        """One Claude call consolidating *all* ``items`` into a signal for
        ``pair``. Same error-handling contract as :meth:`analyze` — any
        failure (timeout, API error, malformed JSON, schema violation)
        returns ``None`` rather than raising.
        """
        if not items:
            return None
        tag = f"batch:{pair}({len(items)})"
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": _BATCH_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": self._build_batch_message(pair, items),
                    }
                ],
            )
        except anthropic.APITimeoutError:
            log.warning("Claude timed out on %s", tag)
            return None
        except anthropic.APIError as exc:
            log.error("Claude API error on %s: %s", tag, exc)
            return None
        except Exception as exc:  # last-resort safety net
            log.exception("Unexpected error calling Claude on %s: %s", tag, exc)
            return None

        text = self._extract_text(response)
        parsed = self._parse_json(text)
        if parsed is None:
            log.warning("Claude returned non-JSON for %s: %r", tag, text[:200])
            return None

        # Enforce the target pair we supplied — Claude is told to echo it,
        # but we don't trust it to do so under prompt drift.
        parsed["pair"] = pair
        return self._validate(parsed, tag)

    def _allowed_pair_mentions(self, item: NewsItem) -> set[str]:
        """Return the subset of explicit pair tokens in the article that
        are also in the configured allowlist. Empty if no allowlist or
        no allowed pair appears in the text.
        """
        if not self._allowed_pairs:
            return set()
        return self._extract_pair_tokens(item) & self._allowed_pairs

    @staticmethod
    def _extract_pair_tokens(item: NewsItem) -> set[str]:
        """Extract validated FX pair tokens from the article text."""
        text = f"{item.title}\n{item.description}".upper()
        tokens: set[str] = set()
        for m in _PAIR_RE_CONTIGUOUS.finditer(text):
            word = m.group(1)
            if word[:3] in _FX_CURRENCIES and word[3:] in _FX_CURRENCIES:
                tokens.add(word)
        for pattern in (_PAIR_RE_DELIMITED, _PAIR_RE_SPACED):
            for m in pattern.finditer(text):
                base, quote = m.group(1), m.group(2)
                if base in _FX_CURRENCIES and quote in _FX_CURRENCIES:
                    tokens.add(base + quote)
        return tokens

    @staticmethod
    def _dedupe_by_pair(
        results: list[AnalysisResult],
    ) -> list[AnalysisResult]:
        best: dict[str, AnalysisResult] = {}
        for r in results:
            existing = best.get(r.pair)
            if existing is None or r.confidence > existing.confidence:
                best[r.pair] = r
        return list(best.values())

    # ------------------------------------------------------------------
    def _mentions_allowed_pair(self, item: NewsItem) -> bool:
        """Return False only when the article explicitly names FX pairs and
        none of them are in the allowlist — then the Claude call is wasted
        and we skip it. If no allowlist is configured, or the text contains
        no explicit pair tokens, we let the call proceed (Claude may still
        infer an allowed pair from context).
        """
        if not self._allowed_pairs:
            return True
        tokens = self._extract_pair_tokens(item)
        if not tokens:
            return True
        return any(pair in self._allowed_pairs for pair in tokens)

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
    def _build_batch_message(pair: str, items: list[NewsItem]) -> str:
        lines = [f"Target pair: {pair}", "", "Articles:"]
        for i, item in enumerate(items, start=1):
            lines.extend([
                f"\n[{i}] Source: {item.source}",
                f"    Published: {item.published_at.isoformat()}",
                f"    Title: {item.title}",
                f"    Description: {item.description}",
                f"    URL: {item.url}",
            ])
        return "\n".join(lines)

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
