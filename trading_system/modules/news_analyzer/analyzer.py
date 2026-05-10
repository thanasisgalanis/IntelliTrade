"""Anthropic Claude-backed news analyzer.

Two paths:

* :meth:`analyze` — single article, returns a single :class:`AnalysisResult`.
  Kept for completeness and the abstract-base contract; the live pipeline
  uses the batch path below.

* :meth:`analyze_batch` — many articles in **one** ``messages.create`` call,
  returning a per-article :class:`AnalysisResult` keyed by ``article_id``.
  The model is required to emit a JSON object of the form
  ``{"results": [{"article_id": "...", "sentiment": "...",
  "confidence": 0.x, "pair": "EURUSD", "rationale": "..."}, ...]}``.

A static system prompt is sent ``cache_control: ephemeral`` so Anthropic
prompt-caching reuses it across calls.

Any deviation (extra prose, malformed JSON, schema mismatch, network error,
timeout) is caught: the affected article(s) are dropped from the result
dict, never raised. The orchestrator therefore always gets a clean mapping.
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
_DEFAULT_BATCH_MAX_SIZE = 50
_REQUIRED_KEYS = {"sentiment", "confidence", "pair"}
_VALID_SENTIMENTS = {s.value for s in Sentiment}

# Scale per-article token budget with batch size — the model returns one
# JSON object per article, plus the wrapping array. 60 output tokens per
# article is comfortable for the schema we ask for.
_OUTPUT_TOKENS_PER_ARTICLE = 60
_OUTPUT_TOKENS_FLOOR = 200

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

_BATCH_ARRAY_SYSTEM_PROMPT = """You are a Forex news sentiment analyst.
The user supplies a numbered list of news articles, each tagged with an
``article_id``. Analyse EACH article independently — do not blend stories
across articles — and return one JSON object per article.

You MUST respond with a single JSON object and NOTHING else — no prose,
no markdown fences, no preamble. The schema is:

{
  "results": [
    {
      "article_id": "<echo verbatim from input>",
      "sentiment":  "bullish" | "bearish" | "neutral",
      "confidence": <float between 0.0 and 1.0>,
      "pair":       "<6-letter ISO pair, e.g. EURUSD>",
      "rationale":  "<one short sentence>"
    },
    ...
  ]
}

Rules:
- One entry per input article. Echo "article_id" exactly as supplied.
- "sentiment" is from the perspective of the BASE currency in "pair".
- "confidence" reflects how strongly that single article supports the
  call; use values below 0.50 if the article is ambiguous or off-topic.
- "pair" must be exactly 6 uppercase letters. If multiple pairs are
  plausible, pick the single most affected one.
- If an article is irrelevant to FX, still emit its entry with neutral
  sentiment, low confidence, and your best-guess pair.
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
        batch_max_size: int = _DEFAULT_BATCH_MAX_SIZE,
    ) -> None:
        if client is None and not api_key:
            raise ValueError("Anthropic API key is required when no client is injected")
        if batch_max_size <= 0:
            raise ValueError("batch_max_size must be > 0")
        self._client = client or anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens
        self._allowed_pairs = allowed_pairs
        self._batch_max_size = batch_max_size

    # ------------------------------------------------------------------
    def analyze(self, item: NewsItem) -> AnalysisResult | None:
        if not self._mentions_allowed_pair(item):
            log.info(
                "Article %s only mentions non-allowed pairs; skipping Claude call",
                item.article_id,
            )
            return None

        user_msg = self._build_user_message(item)
        candidates = sorted(self._extract_pair_tokens(item)) or ["auto"]
        log.info(
            "Calling Claude (model=%s, mode=single) article=%s articles=1 pair_candidates=%s",
            self._model,
            item.article_id,
            ",".join(candidates),
        )
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

        self._log_usage(f"single:{item.article_id}", response)
        text = self._extract_text(response)
        parsed = self._parse_json(text)
        if parsed is None:
            log.warning("Claude returned non-JSON for %s: %r", item.article_id, text[:200])
            return None

        result = self._validate(parsed, item.article_id)
        if result is not None:
            log.info(
                "Claude result single:%s pair=%s sentiment=%s confidence=%.2f",
                item.article_id,
                result.pair,
                result.sentiment.value,
                result.confidence,
            )
        return result

    # ------------------------------------------------------------------
    # Per-article batch analysis (one Claude call per chunk of <= batch_max_size)
    # ------------------------------------------------------------------
    def analyze_batch(
        self, items: list[NewsItem]
    ) -> dict[str, AnalysisResult]:
        """Analyze a batch of articles in as few Claude calls as possible.

        Articles whose explicit pair tokens are *all* outside the allowed
        list are dropped before the call (issue #2 pre-filter). The
        remainder is split into chunks of at most ``batch_max_size`` and
        each chunk is sent in **one** ``messages.create`` call. Per-article
        results are merged into a single ``article_id -> AnalysisResult``
        dict.
        """
        if not items:
            return {}

        eligible: list[NewsItem] = []
        skipped: list[str] = []
        for item in items:
            if self._mentions_allowed_pair(item):
                eligible.append(item)
            else:
                skipped.append(item.article_id)

        if skipped:
            log.info(
                "Pre-filter: %d article(s) only mention non-allowed pairs; skipping (ids=%s)",
                len(skipped),
                ",".join(skipped),
            )

        if not eligible:
            return {}

        results: dict[str, AnalysisResult] = {}
        chunks = list(_chunked(eligible, self._batch_max_size))
        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_results = self._call_batch(chunk, chunk_idx, len(chunks))
            results.update(chunk_results)

        return results

    def _call_batch(
        self,
        items: list[NewsItem],
        chunk_idx: int,
        chunk_total: int,
    ) -> dict[str, AnalysisResult]:
        tag = f"batch[{chunk_idx}/{chunk_total}]({len(items)})"
        log.info(
            "Calling Claude (model=%s, mode=batch) chunk=%d/%d articles=%d ids=%s",
            self._model,
            chunk_idx,
            chunk_total,
            len(items),
            ",".join(item.article_id for item in items),
        )
        max_tokens = max(
            _OUTPUT_TOKENS_FLOOR,
            len(items) * _OUTPUT_TOKENS_PER_ARTICLE,
        )
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": _BATCH_ARRAY_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": self._build_batch_array_message(items),
                    }
                ],
            )
        except anthropic.APITimeoutError:
            log.warning("Claude timed out on %s", tag)
            return {}
        except anthropic.APIError as exc:
            log.error("Claude API error on %s: %s", tag, exc)
            return {}
        except Exception as exc:  # last-resort safety net
            log.exception("Unexpected error calling Claude on %s: %s", tag, exc)
            return {}

        self._log_usage(tag, response)
        text = self._extract_text(response)
        parsed = self._parse_json(text)
        if parsed is None:
            log.warning("Claude returned non-JSON for %s: %r", tag, text[:200])
            return {}

        entries = parsed.get("results") if isinstance(parsed, dict) else None
        if not isinstance(entries, list):
            log.warning("Claude payload missing 'results' array for %s", tag)
            return {}

        valid_ids = {item.article_id for item in items}
        out: dict[str, AnalysisResult] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            article_id = str(entry.get("article_id", "")).strip()
            if not article_id or article_id not in valid_ids:
                log.warning(
                    "%s — entry references unknown article_id=%r; dropping",
                    tag,
                    article_id,
                )
                continue
            result = self._validate(entry, article_id)
            if result is None:
                continue
            out[article_id] = result
            log.info(
                "Claude result %s article=%s pair=%s sentiment=%s confidence=%.2f",
                tag,
                article_id,
                result.pair,
                result.sentiment.value,
                result.confidence,
            )

        missing = valid_ids - out.keys()
        if missing:
            log.warning(
                "%s — %d article(s) had no usable result (ids=%s)",
                tag,
                len(missing),
                ",".join(sorted(missing)),
            )
        return out

    # ------------------------------------------------------------------
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
    def _build_batch_array_message(items: list[NewsItem]) -> str:
        lines = [
            f"Articles to analyse ({len(items)}). Return one results entry per article.",
            "",
        ]
        for i, item in enumerate(items, start=1):
            lines.extend([
                f"[{i}] article_id: {item.article_id}",
                f"    Source: {item.source}",
                f"    Published: {item.published_at.isoformat()}",
                f"    Title: {item.title}",
                f"    Description: {item.description}",
                f"    URL: {item.url}",
                "",
            ])
        return "\n".join(lines)

    @staticmethod
    def _log_usage(tag: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        log.info(
            "Claude usage %s input=%s output=%s cache_read=%s cache_create=%s",
            tag,
            getattr(usage, "input_tokens", "?"),
            getattr(usage, "output_tokens", "?"),
            getattr(usage, "cache_read_input_tokens", "?"),
            getattr(usage, "cache_creation_input_tokens", "?"),
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


def _chunked(seq: list[NewsItem], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
