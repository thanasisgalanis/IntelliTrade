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
# Batched analysis — issue #3 regressions
# (multiple articles for the same pair must collapse into one Claude call,
#  and competing signals for one pair must dedupe to a single result.)
# ---------------------------------------------------------------------------

def test_analyze_many_groups_articles_about_same_pair_into_one_call():
    """Three articles all naming GBP/USD must reach Claude as a single
    batch request, producing one consolidated AnalysisResult."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_response(
        '{"sentiment": "bullish", "confidence": 0.82, "pair": "GBPUSD"}'
    )
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"GBPUSD"},
    )

    items = [
        make_item(article_id="a1", title="GBP/USD breaks higher",
                  description="Sterling firms vs dollar."),
        make_item(article_id="a2", title="Cable extends rally on GBP/USD",
                  description="Pound hits new high vs USD."),
        make_item(article_id="a3", title="GBPUSD outlook upgraded",
                  description="Analysts revise GBP/USD targets."),
    ]

    results = analyzer.analyze_many(items)

    assert client.messages.create.call_count == 1
    assert len(results) == 1
    assert results[0].pair == "GBPUSD"
    assert results[0].sentiment is Sentiment.BULLISH
    assert results[0].confidence == pytest.approx(0.82)


def test_analyze_many_uses_one_call_per_distinct_pair():
    """Articles about different allowed pairs each trigger their own
    batch call — never one shared call for unrelated pairs."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = [
        fake_response('{"sentiment": "bullish", "confidence": 0.8, "pair": "GBPUSD"}'),
        fake_response('{"sentiment": "bearish", "confidence": 0.7, "pair": "EURUSD"}'),
    ]
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"GBPUSD", "EURUSD"},
    )

    items = [
        make_item(article_id="g1", title="GBP/USD up", description="cable firms"),
        make_item(article_id="g2", title="GBPUSD higher", description="pound up"),
        make_item(article_id="e1", title="EUR/USD slips", description="euro down"),
    ]

    results = analyzer.analyze_many(items)

    assert client.messages.create.call_count == 2
    assert {r.pair for r in results} == {"GBPUSD", "EURUSD"}


def test_analyze_many_falls_back_to_single_when_no_pair_token():
    """An article with no explicit pair token can't be grouped — it must
    still reach Claude via the single-article path."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = fake_response(
        '{"sentiment": "neutral", "confidence": 0.4, "pair": "EURUSD"}'
    )
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    results = analyzer.analyze_many([make_item()])  # default has no pair tokens

    assert client.messages.create.call_count == 1
    assert len(results) == 1
    assert results[0].pair == "EURUSD"


def test_analyze_many_skips_articles_for_non_allowed_pairs():
    """Issue #2 pre-filter still applies inside the batch flow: USD/INR
    articles never reach the API."""
    client = MagicMock(spec=anthropic.Anthropic)
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    items = [
        make_item(article_id="x1", title="USD/INR rises", description="rupee weak"),
        make_item(article_id="x2", title="USDINR climbs", description="dollar up"),
    ]

    assert analyzer.analyze_many(items) == []
    client.messages.create.assert_not_called()


def test_analyze_many_combines_grouped_and_ungrouped_paths():
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = [
        # Batch call for the GBP/USD group
        fake_response('{"sentiment": "bullish", "confidence": 0.8, "pair": "GBPUSD"}'),
        # Single-article fallback (no pair token) — Claude infers EURUSD
        fake_response('{"sentiment": "bearish", "confidence": 0.6, "pair": "EURUSD"}'),
    ]
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"GBPUSD", "EURUSD"},
    )

    items = [
        make_item(article_id="g1", title="GBP/USD news", description="cable"),
        make_item(article_id="g2", title="GBPUSD outlook", description="sterling"),
        make_item(article_id="u1"),  # default — no pair token, ungrouped
    ]

    results = analyzer.analyze_many(items)

    assert client.messages.create.call_count == 2
    assert {r.pair for r in results} == {"GBPUSD", "EURUSD"}


def test_analyze_many_dedupes_by_pair_keeping_highest_confidence():
    """If batched and ungrouped paths both yield a signal for the same
    pair, the higher-confidence one wins — never two trades for one pair."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = [
        fake_response('{"sentiment": "bullish", "confidence": 0.55, "pair": "GBPUSD"}'),
        fake_response('{"sentiment": "bullish", "confidence": 0.85, "pair": "GBPUSD"}'),
    ]
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"GBPUSD"},
    )

    items = [
        make_item(article_id="g1", title="GBP/USD news", description="cable"),
        make_item(article_id="u1"),  # ungrouped → individual call
    ]

    results = analyzer.analyze_many(items)

    assert len(results) == 1
    assert results[0].pair == "GBPUSD"
    assert results[0].confidence == pytest.approx(0.85)


def test_analyze_many_handles_malformed_batch_response():
    """If a batch call returns garbage, that pair is dropped — other
    pairs in the same run are unaffected."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = [
        fake_response("not json"),  # GBPUSD batch fails
        fake_response('{"sentiment": "bullish", "confidence": 0.7, "pair": "EURUSD"}'),
    ]
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"GBPUSD", "EURUSD"},
    )

    items = [
        make_item(article_id="g1", title="GBP/USD up", description="cable"),
        make_item(article_id="e1", title="EUR/USD down", description="euro"),
    ]

    results = analyzer.analyze_many(items)

    assert client.messages.create.call_count == 2
    assert len(results) == 1
    assert results[0].pair == "EURUSD"


def test_analyze_many_empty_input_makes_no_api_calls():
    client = MagicMock(spec=anthropic.Anthropic)
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"EURUSD"},
    )

    assert analyzer.analyze_many([]) == []
    client.messages.create.assert_not_called()


def test_analyze_many_overrides_pair_when_claude_misformats_it():
    """The batch path passes the target pair to Claude and trusts our own
    detection — not whatever the model echoes back."""
    client = MagicMock(spec=anthropic.Anthropic)
    # Claude returns a *different* pair than the one we asked about.
    client.messages.create.return_value = fake_response(
        '{"sentiment": "bullish", "confidence": 0.8, "pair": "EURUSD"}'
    )
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"GBPUSD"},
    )

    items = [
        make_item(article_id="g1", title="GBP/USD up", description="cable"),
        make_item(article_id="g2", title="GBPUSD higher", description="pound"),
    ]

    results = analyzer.analyze_many(items)

    assert len(results) == 1
    assert results[0].pair == "GBPUSD"  # not EURUSD


def test_analyze_many_batch_message_contains_every_article():
    """All articles in a group must appear in the single user message —
    otherwise Claude is consolidating from incomplete evidence."""
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return fake_response(
            '{"sentiment": "bullish", "confidence": 0.8, "pair": "GBPUSD"}'
        )

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = capture
    analyzer = ClaudeNewsAnalyzer(
        api_key="unused", client=client, allowed_pairs={"GBPUSD"},
    )

    items = [
        make_item(article_id="a1", title="GBP/USD breaks 1.30",
                  description="Sterling spikes on jobs data."),
        make_item(article_id="a2", title="Cable extends GBP/USD gains",
                  description="Pound rallies on hawkish BoE tone."),
    ]

    analyzer.analyze_many(items)

    body = captured["messages"][0]["content"]
    assert "Target pair: GBPUSD" in body
    assert "breaks 1.30" in body
    assert "Cable extends GBP/USD gains" in body
    assert "Sterling spikes on jobs data." in body
    assert "hawkish BoE tone." in body
