# -*- coding: utf-8 -*-
"""Retry-every-minute behaviour when the LLM provider is overloaded."""

from unittest import mock

import pytest

from src import analyzer as analyzer_module
from src.analyzer import GeminiAnalyzer, is_provider_overload_error

OVERLOAD_503 = RuntimeError(
    'ServiceUnavailableError: GeminiException - {"error": {"code": 503, '
    '"message": "This model is currently experiencing high demand"}}'
)
PER_MINUTE_429 = RuntimeError(
    'RateLimitError: {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", '
    '"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}}'
)
PER_DAY_429 = RuntimeError(
    'RateLimitError: {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", '
    '"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}'
)


def test_classifier():
    assert is_provider_overload_error(OVERLOAD_503) is True
    assert is_provider_overload_error(PER_MINUTE_429) is True
    assert is_provider_overload_error(PER_DAY_429) is False
    assert is_provider_overload_error(ValueError("bad JSON")) is False


def test_classifier_follows_exception_chain():
    try:
        try:
            raise OVERLOAD_503
        except RuntimeError as inner:
            raise Exception("All LLM models failed") from inner
    except Exception as outer:
        assert is_provider_overload_error(outer) is True


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.setenv("LLM_OVERLOAD_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_OVERLOAD_RETRY_SECONDS", "60")
    sleeps = []
    monkeypatch.setattr(analyzer_module.time, "sleep", lambda s: sleeps.append(s))
    instance = GeminiAnalyzer.__new__(GeminiAnalyzer)
    instance._sleeps = sleeps
    return instance


def test_overload_retries_every_minute_then_succeeds(analyzer):
    ok = ("text", "gemini/x", {})
    with mock.patch.object(analyzer, "_call_litellm_once", side_effect=[OVERLOAD_503, PER_MINUTE_429, ok]) as call:
        assert analyzer._call_litellm("p", {}) == ok
    assert call.call_count == 3
    assert analyzer._sleeps == [60, 60]


def test_daily_quota_is_not_retried(analyzer):
    with mock.patch.object(analyzer, "_call_litellm_once", side_effect=PER_DAY_429) as call:
        with pytest.raises(RuntimeError):
            analyzer._call_litellm("p", {})
    assert call.call_count == 1
    assert analyzer._sleeps == []


def test_gives_up_after_max_retries(analyzer):
    with mock.patch.object(analyzer, "_call_litellm_once", side_effect=OVERLOAD_503) as call:
        with pytest.raises(RuntimeError):
            analyzer._call_litellm("p", {})
    assert call.call_count == 4  # first try + 3 retries
    assert analyzer._sleeps == [60, 60, 60]


def test_other_errors_fail_immediately(analyzer):
    with mock.patch.object(analyzer, "_call_litellm_once", side_effect=ValueError("bad JSON")) as call:
        with pytest.raises(ValueError):
            analyzer._call_litellm("p", {})
    assert call.call_count == 1
