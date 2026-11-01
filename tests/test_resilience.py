"""Offline tests for the optimized LLM layer. Run: python -m pytest tests -q   (no API key and no network needed)."""
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from groq import (APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, NotFoundError,
                  RateLimitError)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402
from backend import llm, runtime, usage         # noqa: E402

REQ = httpx.Request("POST", "http://test")


def _resp(status, headers=None):
    return httpx.Response(status, headers=headers or {}, request=REQ)


def rate_limit(retry_after="0"):
    return RateLimitError("429", response=_resp(429, {"retry-after": retry_after}), body=None)


def bad_request(msg="bad", body=None):
    return BadRequestError(msg, response=_resp(400), body=body)


def ok(text='{"a": 1}', finish="stop", prompt=100, completion=20):
    return NS(choices=[NS(message=NS(content=text), finish_reason=finish)],
              usage=NS(prompt_tokens=prompt, completion_tokens=completion))


class Script:
    """A fake client that plays a list of outcomes (exceptions are raised, anything else is returned)."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        o = self.outcomes.pop(0) if self.outcomes else ok()
        if isinstance(o, Exception):
            raise o
        return o


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    llm._ids[:] = []
    llm._dead.clear(), llm._cooldown.clear(), llm._no_extras.clear(), llm._cache.clear()
    monkeypatch.setattr(llm, "_ids", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    monkeypatch.setattr(time, "sleep", lambda s: None)          # never really wait in tests
    yield


def use(monkeypatch, outcomes):
    s = Script(outcomes)
    monkeypatch.setattr(llm, "get_client", lambda: s)
    return s


def test_roles_pick_different_models(monkeypatch):
    s = use(monkeypatch, [ok(), ok(), ok()])
    llm.ask("hi", role="chat"), llm.ask("hi", role="lesson"), llm.ask("hi", role="quiz")
    assert [c["model"] for c in s.calls] == ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert all("reasoning_effort" in c["extra_body"] for c in s.calls)
    assert all(c["timeout"] >= 5 for c in s.calls)                    # a timeout is always sent


def test_unknown_models_are_skipped(monkeypatch):
    monkeypatch.setattr(llm, "_ids", ["llama-3.3-70b-versatile"])
    s = use(monkeypatch, [ok()])
    llm.ask("hi", role="chat")
    assert s.calls[0]["model"] == "llama-3.3-70b-versatile" and "extra_body" not in s.calls[0]


def test_rate_limit_switches_model_instead_of_sleeping(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    s = use(monkeypatch, [rate_limit("30"), ok()])
    llm.ask("hi", role="chat")
    assert [c["model"] for c in s.calls] == ["openai/gpt-oss-20b", "llama-3.1-8b-instant"] and not slept


def test_short_rate_limit_uses_another_free_model_without_waiting(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    s = use(monkeypatch, [rate_limit("1"), ok()])
    llm.ask("hi", role="chat")
    assert not slept and s.calls[0]["model"] != s.calls[1]["model"]


def _fake_clock(monkeypatch):
    """time.sleep moves a fake clock forward, so waiting logic can be tested without really waiting."""
    clock, slept = [1000.0], []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))
    return slept


def test_every_model_rate_limited_briefly_waits_then_succeeds(monkeypatch):
    slept = _fake_clock(monkeypatch)
    s = use(monkeypatch, [rate_limit("2"), rate_limit("2"), rate_limit("2"), ok()])
    assert llm.ask("hi", role="chat") == '{"a": 1}'
    assert len(slept) == 1 and 2 <= slept[0] < 3 and len(s.calls) == 4


def test_long_rate_limit_gives_a_clear_message_with_the_time(monkeypatch):
    _fake_clock(monkeypatch)
    use(monkeypatch, [rate_limit("300")] * 6)
    with pytest.raises(llm.LLMError) as e:
        llm.ask("hi", role="chat")
    assert "5 minutes" in str(e.value) and "rate limited" in str(e.value)


def test_daily_limit_is_reported_as_daily(monkeypatch):
    _fake_clock(monkeypatch)
    err = RateLimitError("Rate limit reached for model: tokens per day (TPD): Limit 100000. Please try again in 12m30s.",
                         response=_resp(429), body=None)
    use(monkeypatch, [err] * 6)
    with pytest.raises(llm.LLMError) as e:
        llm.ask("hi", role="chat")
    assert "daily" in str(e.value) and "13 minutes" in str(e.value)


def test_parse_wait_reads_groq_messages():
    assert llm._parse_wait("please try again in 7m12.5s. visit") == pytest.approx(432.5)
    assert llm._parse_wait("try again in 850ms") == pytest.approx(0.85)
    assert llm._parse_wait("try again in 1h2m3s") == 3723
    assert llm._parse_wait("nothing here") == 0


def test_retries_are_bounded(monkeypatch):
    s = use(monkeypatch, [APITimeoutError(request=REQ)] * 20)
    with pytest.raises(llm.LLMError):
        llm.ask("hi", role="chat")
    assert len(s.calls) == config.MAX_ATTEMPTS                         # was up to 6 attempts with 15-90 s sleeps


def test_connection_and_server_errors_are_retried(monkeypatch):
    s = use(monkeypatch, [APIConnectionError(request=REQ), InternalServerError("500", response=_resp(500), body=None), ok("fine")])
    assert llm.ask("hi") == "fine" and len(s.calls) == 3


def test_not_found_marks_model_dead(monkeypatch):
    s = use(monkeypatch, [NotFoundError("gone", response=_resp(404), body=None), ok(), ok()])
    llm.ask("hi", role="chat")
    llm.ask("hi", role="chat")
    assert "openai/gpt-oss-20b" in llm._dead
    assert s.calls[2]["model"] != "openai/gpt-oss-20b"                 # not tried again


def test_unsupported_thinking_setting_is_dropped_once_and_remembered(monkeypatch):
    s = use(monkeypatch, [bad_request("unknown parameter reasoning_effort"), ok(), ok()])
    llm.ask("hi", role="chat")
    llm.ask("hi", role="chat")
    assert "extra_body" in s.calls[0] and "extra_body" not in s.calls[1] and "extra_body" not in s.calls[2]


def test_json_validate_failed_is_salvaged_without_a_new_call(monkeypatch):
    e = bad_request("json_validate_failed", body={"error": {"failed_generation": '{"x": 2}'}})
    s = use(monkeypatch, [e])
    r = llm.ask_full("hi", json_mode=True)
    assert json.loads(r.text) == {"x": 2} and len(s.calls) == 1


def test_cut_off_json_is_retried_once_with_more_room(monkeypatch):
    from pydantic import BaseModel

    class M(BaseModel):
        a: int
    s = use(monkeypatch, [ok('{"a": ', finish="length"), ok('{"a": 5}')])
    assert llm.ask_json("p", M, max_tokens=1000).a == 5
    assert s.calls[1]["max_tokens"] == 1600 > s.calls[0]["max_tokens"]


def test_bad_json_gets_bounded_tries(monkeypatch):
    from pydantic import BaseModel

    class M(BaseModel):
        a: int
    s = use(monkeypatch, [ok("not json")] * 10)
    with pytest.raises(llm.LLMError):
        llm.ask_json("p", M)
    assert len(s.calls) == 2                                           # was 3


def test_cache_reuses_identical_read_only_requests_and_never_keeps_failures(monkeypatch):
    s = use(monkeypatch, [ok("answer one"), ok("answer two")])
    assert llm.ask("q", cache=True) == "answer one"
    assert llm.ask("q", cache=True) == "answer one" and len(s.calls) == 1
    assert llm.ask("q") == "answer two"                                # not cached unless asked
    s2 = use(monkeypatch, [ok("", finish="length"), ok("good")])
    assert llm.ask("other", cache=True) == "" and llm.ask("other", cache=True) == "good"   # cut-off answer was not cached


def test_json_cache_only_keeps_validated_results(monkeypatch):
    from pydantic import BaseModel

    class M(BaseModel):
        a: int
    s = use(monkeypatch, [ok("oops"), ok('{"a": 1}'), ok('{"a": 99}')])
    assert llm.ask_json("p", M, cache=True).a == 1                     # first reply invalid -> second used
    assert llm.ask_json("p", M, cache=True).a == 1 and len(s.calls) == 2


def test_singleflight_shares_one_call_between_concurrent_requests():
    runs, gate = [], threading.Event()

    def work():
        runs.append(1)
        gate.wait(2)
        return "done"
    out = []
    th = [threading.Thread(target=lambda: out.append(runtime.singleflight("k", work))) for _ in range(4)]
    [t.start() for t in th]
    time.sleep = time.sleep  # (sleep is patched to a no-op, so wait on the registry instead)
    for _ in range(200):
        if runs:
            break
        threading.Event().wait(0.005)
    gate.set()
    [t.join() for t in th]
    assert out == ["done"] * 4 and len(runs) == 1


def test_singleflight_shares_failure_and_allows_retry_afterwards():
    def boom():
        raise llm.LLMError("nope")
    with pytest.raises(llm.LLMError):
        runtime.singleflight("k2", boom)
    assert runtime.singleflight("k2", lambda: "ok") == "ok"            # nothing stuck after a failure


def test_usage_records_actual_estimated_time_and_task(monkeypatch):
    use(monkeypatch, [ok(prompt=321, completion=45)])
    with usage.task("demo") as t:
        llm.ask("x" * 400, role="quiz", max_tokens=777)
    sm = usage.snapshot()
    assert (sm["input_tokens"], sm["output_tokens"], sm["calls"]) == (321, 45, 1)      # actual = what Groq said
    assert sm["est_input_tokens"] >= 100 and sm["output_budget"] == 777              # estimate made before the call
    assert sm["by_role"]["quiz"]["calls"] == 1 and sm["log"][-1]["seconds"] >= 0
    assert sm["tasks"][-1]["name"] == "demo" and sm["tasks"][-1]["calls"] == 1 and sm["tasks"][-1]["seconds"] >= 0


def test_task_counts_calls_made_in_helper_threads(monkeypatch):
    use(monkeypatch, [])
    with usage.task("par"):
        runtime.parallel([lambda: llm.ask("a"), lambda: llm.ask("b"), lambda: llm.ask("c")])
    assert usage.snapshot()["tasks"][-1]["calls"] == 3


def test_parallel_keeps_order_and_returns_errors():
    out = runtime.parallel([lambda: 1, lambda: (_ for _ in ()).throw(ValueError("x")), lambda: 3])
    assert out[0] == 1 and isinstance(out[1], ValueError) and out[2] == 3


def test_retrieval_select_sends_everything_when_it_fits_and_the_relevant_part_when_not():
    from backend import retrieval
    from backend.schemas import Span
    spans = [Span(id=f"S{i}", page=1, text=t) for i, t in enumerate(
        ["Chlorophyll absorbs red and blue light in the chloroplast. " * 6,
         "Stomata let carbon dioxide into the leaf from the air. " * 6,
         "Enzymes slow down when the temperature becomes too hot. " * 6], 1)]
    assert [s.id for s in retrieval.select(spans, "anything", 10_000)] == ["S1", "S2", "S3"]
    picked = retrieval.select(spans, "how does chlorophyll absorb light", 400)
    assert picked[0].id == "S1" and len(picked) < 3
