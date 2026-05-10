"""News analyzer unit tests with fully mocked Anthropic SDK.

Verifies the analyzer:
  * parses well-formed JSON responses,
  * returns None on malformed JSON without crashing,
  * returns None on Anthropic timeouts,
  * returns None on Anthropic API errors,
  * rejects schema violations (bad sentiment, out-of-range confidence,
    malformed pair, missing keys),
  * tolerates stray prose around a JSON object,
  * filters by allowed-pair set when provided.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pytest

from trading_system.core.interfaces import NewsItem, Sentiment
from trading_system.modules.news_analyzer.analyzer import ClaudeNewsAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_item(
    title: str = "ECB hints at faster rate cuts",
    description: str = "Sources suggest the ECB may accelerate cuts in Q3.",
    article_id: str = "abc123",
) -> NewsItem:
    return NewsItem(
        article_id=article_id,
        source="Reuters",
        title=title,
        description=description,
        url="https://example.com/article",
        published_at=datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc),
    )


def fake_response(text: str) -> SimpleNamespace:
    """Mimic anthropic.types.Message — only .content[].text is read."""
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def make_analyzer(
    response_text: str | None = None,
    side_effect: Exception | None = None,
    allowed_pairs: set[str] | None = None,
) -> ClaudeNewsAnalyzer:
    client = MagicMock(spec=anthropic.Anthropic)
    if side_effect is not None:
        client.messages.create.side_effect = side_effect
    else:
        client.messages.create.return_value = fake_response(response_text or "")
    return ClaudeNewsAnalyzer(
        api_key="unused",
        client=client,
        allowed_pairs=allowed_pairs,
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_valid_response_parsed():
    analyzer = make_analyzer(
        '{"sentiment": "bullish", "confidence": 0.82, "pair": "EURUSD"}'
    )

    result = analyzer.analyze(make_item())

    assert result is not None
    assert result.sentiment is Sentiment.BULLISH
    assert result.confidence == pytest.approx(0.82)
    assert result.pair == "EURUSD"


def test_pair_normalised_uppercase_and_stripped():
    analyzer = make_analyzer(
        '{"sentiment": "bearish", "confidence": 0.75, "pair": "eur/usd"}'
    )

    result = analyzer.analyze(make_item())

    assert result is not None
    assert result.pair == "EURUSD"
    assert result.sentiment is Sentiment.BEARISH


def test_tolerates_prose_around_json():
    analyzer = make_analyzer(
        'Sure! Here is the analysis:\n'
        '{"sentiment": "neutral", "confidence": 0.4, "pair": "GBPUSD"}\n'
        'Hope that helps.'
    )

    result = analyzer.analyze(make_item())

    assert result is not None
    assert result.sentiment is Sentiment.NEUTRAL
    assert result.pair == "GBPUSD"


# ---------------------------------------------------------------------------
# Failure modes — must NOT raise
# ---------------------------------------------------------------------------

def test_non_json_response_returns_none():
    analyzer = make_analyzer("I cannot help with that request.")

    assert analyzer.analyze(make_item()) is None


def test_empty_response_returns_none():
    analyzer = make_analyzer("")

    assert analyzer.analyze(make_item()) is None


def test_timeout_returns_none():
    timeout = anthropic.APITimeoutError(request=MagicMock())
    analyzer = make_analyzer(side_effect=timeout)

    assert analyzer.analyze(make_item()) is None


def test_api_error_returns_none():
    api_err = anthropic.APIError(
        message="boom", request=MagicMock(), body=None
    )
    analyzer = make_analyzer(side_effect=api_err)

    assert analyzer.analyze(make_item()) is None


def test_unexpected_exception_returns_none():
    analyzer = make_analyzer(side_effect=RuntimeError("network died"))

    assert analyzer.analyze(make_item()) is None


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_missing_keys_rejected():
    analyzer = make_analyzer('{"sentiment": "bullish", "confidence": 0.9}')

    assert analyzer.analyze(make_item()) is None


def test_invalid_sentiment_rejected():
    analyzer = make_analyzer(
        '{"sentiment": "very-bullish", "confidence": 0.9, "pair": "EURUSD"}'
    )

    assert analyzer.analyze(make_item()) is None


def test_non_numeric_confidence_rejected():
    analyzer = make_analyzer(
        '{"sentiment": "bullish", "confidence": "high", "pair": "EURUSD"}'
    )

    assert analyzer.analyze(make_item()) is None


@pytest.mark.parametrize("bad_conf", [-0.1, 1.5, 2.0])
def test_out_of_range_confidence_rejected(bad_conf):
    analyzer = make_analyzer(
        '{"sentiment": "bullish", "confidence": %s, "pair": "EURUSD"}' % bad_conf
    )

    assert analyzer.analyze(make_item()) is None


@pytest.mark.parametrize("bad_pair", ["EUR", "EURUSDX", "EUR1SD", "12USD"])
def test_invalid_pair_rejected(bad_pair):
    analyzer = make_analyzer(
        '{"sentiment": "bullish", "confidence": 0.9, "pair": "%s"}' % bad_pair
    )

    assert analyzer.analyze(make_item()) is None


def test_pair_not_in_allowlist_rejected():
    analyzer = make_analyzer(
        '{"sentiment": "bullish", "confidence": 0.9, "pair": "USDJPY"}',
        allowed_pairs={"EURUSD", "GBPUSD"},
    )

    assert analyzer.analyze(make_item()) is None


def test_pair_in_allowlist_accepted():
    analyzer = make_analyzer(
        '{"sentiment": "bullish", "confidence": 0.9, "pair": "EURUSD"}',
        allowed_pairs={"EURUSD", "GBPUSD"},
    )

    result = analyzer.analyze(make_item())
    assert result is not None
    assert result.pair == "EURUSD"


# ---------------------------------------------------------------------------
# Pre-filter: skip Claude call when only non-allowed pairs are mentioned
# (regression for issue #2 — wasted API quota / cost on USD/INR-style articles)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "title,description",
    [
        ("USD/INR rises 0.4% as crude eases", "Indian rupee weakens against dollar."),
        ("Asian FX wrap", "USDINR climbs while local stocks dip."),
        ("Emerging FX update", "Spot USD INR last seen at 83.2 on the day."),
        ("USD-INR session recap", "Pair settled near session highs."),
    ],
)
def test_skips_api_call_when_only_non_allowed_pair_mentioned(title, description):
    client = MagicMock(spec=anthropic.Anthropic)
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused",
        client=client,
        allowed_pairs={"EURUSD", "GBPUSD"},
    )

    result = analyzer.analyze(make_item(title=title, description=description))

    assert result is None
    client.messages.create.assert_not_called()


def test_calls_api_when_allowed_pair_mentioned():
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_response(
        '{"sentiment": "bullish", "confidence": 0.8, "pair": "EURUSD"}'
    )
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused",
        client=client,
        allowed_pairs={"EURUSD", "GBPUSD"},
    )

    result = analyzer.analyze(
        make_item(title="EUR/USD breaks higher", description="Euro firms vs dollar.")
    )

    assert result is not None
    assert result.pair == "EURUSD"
    client.messages.create.assert_called_once()


def test_calls_api_when_mixed_pairs_include_allowed_one():
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_response(
        '{"sentiment": "bullish", "confidence": 0.7, "pair": "EURUSD"}'
    )
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused",
        client=client,
        allowed_pairs={"EURUSD"},
    )

    result = analyzer.analyze(
        make_item(
            title="USD/INR and EUR/USD diverge",
            description="Rupee weakens while euro firms.",
        )
    )

    assert result is not None
    client.messages.create.assert_called_once()


def test_calls_api_when_no_explicit_pair_mentioned():
    """If the article has no pair token, defer to Claude — it may still
    infer an allowed pair from context (e.g. a story about ECB policy)."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_response(
        '{"sentiment": "bullish", "confidence": 0.8, "pair": "EURUSD"}'
    )
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused",
        client=client,
        allowed_pairs={"EURUSD"},
    )

    result = analyzer.analyze(make_item())  # default item has no pair tokens

    assert result is not None
    client.messages.create.assert_called_once()


def test_no_allowlist_disables_prefilter():
    """Without an allowlist there's nothing to filter against — every
    article must reach Claude as before."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_response(
        '{"sentiment": "neutral", "confidence": 0.4, "pair": "USDINR"}'
    )
    analyzer = ClaudeNewsAnalyzer(api_key="unused", client=client)

    result = analyzer.analyze(
        make_item(title="USD/INR drifts", description="Rupee flat on the day.")
    )

    assert result is not None
    client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# Per-article batch analysis (analyze_batch)
# Contract:
#   * one Claude call per chunk of <= batch_max_size
#   * pre-filter drops articles whose only pair tokens are non-allowed
#   * returns dict[article_id -> AnalysisResult] with only valid entries
# ---------------------------------------------------------------------------

def fake_batch_response(entries: list[dict]) -> SimpleNamespace:
    """Build a fake response wrapping the new {"results": [...]} schema."""
    import json as _json
    return fake_response(_json.dumps({"results": entries}))


def _entry(article_id, sentiment="bullish", confidence=0.8, pair="EURUSD",
           rationale="ok"):
    return {
        "article_id": article_id,
        "sentiment": sentiment,
        "confidence": confidence,
        "pair": pair,
        "rationale": rationale,
    }


def test_analyze_batch_single_call_for_many_articles():
    """N articles -> 1 Claude call -> N per-article results keyed by id."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_batch_response([
        _entry("a1", pair="EURUSD", confidence=0.7),
        _entry("a2", pair="GBPUSD", sentiment="bearish", confidence=0.6),
        _entry("a3", pair="EURUSD", sentiment="neutral", confidence=0.4),
    ])
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD", "GBPUSD"},
    )

    items = [
        make_item(article_id="a1", title="EUR/USD up", description="euro firms"),
        make_item(article_id="a2", title="GBP/USD weak", description="cable down"),
        make_item(article_id="a3", title="EUR/USD wobble", description="mixed signals"),
    ]

    out = analyzer.analyze_batch(items)

    assert client.messages.create.call_count == 1
    assert set(out.keys()) == {"a1", "a2", "a3"}
    assert out["a1"].pair == "EURUSD"
    assert out["a2"].pair == "GBPUSD"
    assert out["a2"].sentiment is Sentiment.BEARISH
    assert out["a3"].sentiment is Sentiment.NEUTRAL


def test_analyze_batch_chunks_at_batch_max_size():
    """Articles in excess of batch_max_size span multiple Claude calls."""
    client = MagicMock(spec=anthropic.Anthropic)
    # First chunk: a1, a2. Second chunk: a3.
    client.messages.create.side_effect = [
        fake_batch_response([_entry("a1"), _entry("a2")]),
        fake_batch_response([_entry("a3")]),
    ]
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
        batch_max_size=2,
    )

    items = [
        make_item(article_id=f"a{i}", title=f"EUR/USD {i}", description="euro")
        for i in (1, 2, 3)
    ]

    out = analyzer.analyze_batch(items)

    assert client.messages.create.call_count == 2
    assert set(out.keys()) == {"a1", "a2", "a3"}


def test_analyze_batch_pre_filters_non_allowed_pairs():
    """Pre-filter (issue #2) still applies: only non-allowed pair => no API call."""
    client = MagicMock(spec=anthropic.Anthropic)
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    items = [
        make_item(article_id="x1", title="USD/INR rises", description="rupee weak"),
        make_item(article_id="x2", title="USDINR climbs", description="dollar up"),
    ]

    assert analyzer.analyze_batch(items) == {}
    client.messages.create.assert_not_called()


def test_analyze_batch_pre_filter_keeps_eligible_articles():
    """Mixed list: pre-filtered ones drop, eligible ones reach Claude."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_batch_response([
        _entry("a1", pair="EURUSD"),
    ])
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    items = [
        make_item(article_id="x1", title="USD/INR drift", description="rupee"),
        make_item(article_id="a1", title="EUR/USD up", description="euro firms"),
    ]

    out = analyzer.analyze_batch(items)

    assert client.messages.create.call_count == 1
    assert set(out.keys()) == {"a1"}


def test_analyze_batch_empty_input_makes_no_api_call():
    client = MagicMock(spec=anthropic.Anthropic)
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    assert analyzer.analyze_batch([]) == {}
    client.messages.create.assert_not_called()


def test_analyze_batch_drops_entries_for_unknown_article_ids():
    """Claude must echo back valid article_ids; hallucinated ones are dropped."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_batch_response([
        _entry("a1", pair="EURUSD"),
        _entry("ghost", pair="EURUSD"),  # not in input
    ])
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    items = [make_item(article_id="a1", title="EUR/USD up", description="euro")]

    out = analyzer.analyze_batch(items)

    assert set(out.keys()) == {"a1"}


def test_analyze_batch_drops_entries_for_non_allowed_pair_in_response():
    """Even if Claude returns a non-allowed pair, the row is dropped (allowlist)."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_batch_response([
        _entry("a1", pair="USDJPY"),   # not allowed
        _entry("a2", pair="EURUSD"),   # allowed
    ])
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    items = [
        make_item(article_id="a1", title="some FX news", description="x"),
        make_item(article_id="a2", title="EUR/USD update", description="y"),
    ]

    out = analyzer.analyze_batch(items)

    assert set(out.keys()) == {"a2"}


def test_analyze_batch_handles_malformed_response():
    """Garbage response => empty dict, no crash."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_response("not json")
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    items = [make_item(article_id="a1", title="EUR/USD up", description="euro")]

    assert analyzer.analyze_batch(items) == {}


def test_analyze_batch_handles_timeout():
    timeout = anthropic.APITimeoutError(request=MagicMock())
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = timeout
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    items = [make_item(article_id="a1", title="EUR/USD up", description="euro")]

    assert analyzer.analyze_batch(items) == {}


def test_analyze_batch_message_contains_every_article():
    """The single user message must list every article verbatim — otherwise
    Claude analyses incomplete evidence."""
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return fake_batch_response([
            _entry("a1", pair="EURUSD"),
            _entry("a2", pair="EURUSD"),
        ])

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = capture
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    items = [
        make_item(article_id="a1", title="EUR/USD breaks 1.10",
                  description="Euro firms on hawkish ECB."),
        make_item(article_id="a2", title="ECB tone shifts EUR/USD",
                  description="Yields rise; euro extends gains."),
    ]

    analyzer.analyze_batch(items)

    body = captured["messages"][0]["content"]
    assert "article_id: a1" in body
    assert "article_id: a2" in body
    assert "breaks 1.10" in body
    assert "ECB tone shifts EUR/USD" in body
    assert "Euro firms on hawkish ECB." in body
    assert "Yields rise; euro extends gains." in body
