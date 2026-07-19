"""Unit tests for the shared ModelAdapter layer — no network."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gopro_garmin_pipeline.config import Settings
from gopro_garmin_pipeline.gemini_scan import _clip_cache_key, _model_fingerprint
from gopro_garmin_pipeline.models import (
    VISION_SOURCES,
    GeminiAdapter,
    OpenAIAdapter,
    cache_dir_name,
    get_model_adapter,
    provider_api_key,
)
from gopro_garmin_pipeline.models.base import ClipRubric, CoarseFrameBatch, FrameScore


# ─── Factory / config ─────────────────────────────────────────

def test_provider_api_key_selects_by_model_provider():
    g = Settings(_env_file=None, model_provider="gemini", gemini_api_key="gk", openai_api_key="ok")
    assert provider_api_key(g) == "gk"
    o = Settings(_env_file=None, model_provider="openai", gemini_api_key="gk", openai_api_key="ok")
    assert provider_api_key(o) == "ok"


def test_get_model_adapter_gemini():
    s = Settings(_env_file=None, model_provider="gemini", gemini_api_key="gk", gemini_model="gemini-3.5-flash")
    with patch("gopro_garmin_pipeline.models.gemini.genai", create=True):
        # GeminiAdapter imports google.genai inside __init__
        with patch("google.genai.Client") as Client:
            Client.return_value = MagicMock()
            adapter = get_model_adapter(s)
    assert adapter.name == "gemini"
    assert adapter.model_id == "gemini-3.5-flash"


def test_get_model_adapter_openai():
    s = Settings(
        _env_file=None,
        model_provider="openai",
        openai_api_key="sk-test",
        openai_model="gpt-4.1-mini",
    )
    with patch("openai.OpenAI") as OpenAI:
        OpenAI.return_value = MagicMock()
        adapter = get_model_adapter(s)
    assert adapter.name == "openai"
    assert adapter.model_id == "gpt-4.1-mini"


def test_get_model_adapter_unknown_raises():
    s = Settings(_env_file=None, model_provider="anthropic", gemini_api_key="x")
    with pytest.raises(ValueError, match="Unknown MODEL_PROVIDER"):
        get_model_adapter(s)


def test_cache_dir_name():
    assert cache_dir_name("gemini") == ".gemini_cache"
    assert cache_dir_name("openai") == ".openai_cache"


# ─── Cache keys ───────────────────────────────────────────────

def test_gemini_cache_key_legacy_shape():
    assert _clip_cache_key("GX010346.MP4", provider="gemini", model_id="gemini-3.5-flash") == (
        "GX010346_v10.json"
    )
    assert _clip_cache_key(
        "GX010346.MP4", [12.3, 45.6], provider="gemini", model_id="m",
    ).startswith("GX010346_v10_l")
    assert "_m" not in _clip_cache_key("GX010346.MP4", provider="gemini", model_id="gemini-3.5-flash")


def test_openai_cache_key_includes_model_fingerprint():
    fp = _model_fingerprint("gpt-4.1-mini")
    key = _clip_cache_key("GX010346.MP4", provider="openai", model_id="gpt-4.1-mini")
    assert key == f"GX010346_v10_m{fp}.json"

    other = _clip_cache_key("GX010346.MP4", provider="openai", model_id="gpt-5-mini")
    assert other != key
    assert f"m{_model_fingerprint('gpt-5-mini')}" in other


# ─── Transient errors ─────────────────────────────────────────

def test_gemini_is_transient_error():
    adapter = object.__new__(GeminiAdapter)
    assert adapter.is_transient_error(Exception("503 UNAVAILABLE"))
    assert adapter.is_transient_error(Exception("rate 429 limit"))
    assert not adapter.is_transient_error(Exception("INVALID_ARGUMENT bad prompt"))


def test_openai_is_transient_error():
    adapter = object.__new__(OpenAIAdapter)

    class FakeRateLimit(Exception):
        status_code = 429

    class FakeServer(Exception):
        status_code = 503

    class FakeBadRequest(Exception):
        status_code = 400

    # Typed openai exceptions (if importable) + status_code fallback
    try:
        from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
        # These may need constructor args — fall back to duck-typed checks
        _ = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    except ImportError:
        pass

    assert adapter.is_transient_error(FakeRateLimit("rate limited"))
    assert adapter.is_transient_error(FakeServer("backend error"))
    assert adapter.is_transient_error(Exception("connection Timeout"))
    assert not adapter.is_transient_error(FakeBadRequest("bad request"))


def test_call_with_retry_uses_adapter_transient(monkeypatch):
    from gopro_garmin_pipeline import gemini_scan as gs

    sleeps: list[float] = []
    monkeypatch.setattr(gs.time, "sleep", lambda s: sleeps.append(s))

    class Flaky:
        name = "test"
        model_id = "t"
        calls = 0

        def score_frames(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise Exception("503 UNAVAILABLE")
            return [{"frame_index": 0, "visual": 4, "action": 4}]

        def is_transient_error(self, exc):
            return "503" in str(exc)

    adapter = Flaky()
    # Avoid reading real files — stub _call_batch path via score_frames through _call_with_retry
    # by stubbing _read_images and prompt builders via calling _call_with_retry after patching _call_batch

    def fake_batch(adapter, frame_paths, telemetry_text, n_frames, interval, has_power=True):
        return adapter.score_frames(images=[], system="", user="")

    monkeypatch.setattr(gs, "_call_batch", fake_batch)
    out = gs._call_with_retry(adapter, [], "tel", 1, 10.0, label="t")
    assert out == [{"frame_index": 0, "visual": 4, "action": 4}]
    assert adapter.calls == 3
    assert len(sleeps) == 2


def test_call_with_retry_hard_fails_non_transient(monkeypatch):
    from gopro_garmin_pipeline import gemini_scan as gs

    class HardFail:
        name = "test"
        model_id = "t"

        def score_frames(self, **kwargs):
            raise Exception("INVALID_ARGUMENT")

        def is_transient_error(self, exc):
            return False

    monkeypatch.setattr(
        gs, "_call_batch",
        lambda *a, **k: HardFail().score_frames(),
    )
    # Use HardFail for is_transient_error
    adapter = HardFail()
    errors = gs._ScanErrors()
    out = gs._call_with_retry(adapter, [], "tel", 1, 10.0, label="t", errors=errors)
    assert out is None
    assert errors.count == 1


# ─── OpenAI structured unwrap ─────────────────────────────────

def test_openai_score_frames_unwraps_frames():
    adapter = object.__new__(OpenAIAdapter)
    adapter.model_id = "gpt-4.1-mini"
    client = MagicMock()
    parsed = CoarseFrameBatch(frames=[
        FrameScore(frame_index=0, visual=5, action=3, clip_type="landmark", crop_x=50, reason="bridge"),
    ])
    client.responses.parse.return_value = MagicMock(output_parsed=parsed)
    adapter._client = client

    out = adapter.score_frames(images=[b"\xff\xd8fake"], system="sys", user="user")
    assert out == [{
        "frame_index": 0, "visual": 5, "action": 3,
        "clip_type": "landmark", "crop_x": 50, "reason": "bridge",
    }]
    assert client.responses.parse.called


def test_openai_score_clip_rubric():
    adapter = object.__new__(OpenAIAdapter)
    adapter.model_id = "gpt-4.1-mini"
    client = MagicMock()
    parsed = ClipRubric(
        light=8, composition=7, motion=9, scenery=6, subject=9,
        peak_offset=0.5, clip_type="action", crop_x=50, reason="sprint",
    )
    client.responses.parse.return_value = MagicMock(output_parsed=parsed)
    adapter._client = client

    out = adapter.score_clip_rubric(images=[b"x"], system="sys", user="user")
    assert out["motion"] == 9
    assert out["clip_type"] == "action"


def test_openai_score_clip_rubric_none_raises():
    adapter = object.__new__(OpenAIAdapter)
    adapter.model_id = "gpt-4.1-mini"
    client = MagicMock()
    client.responses.parse.return_value = MagicMock(output_parsed=None)
    adapter._client = client

    with pytest.raises(ValueError, match="no structured output"):
        adapter.score_clip_rubric(images=[b"x"], system="sys", user="user")


def test_openai_complete_json_narrative_uses_structured_indices():
    from gopro_garmin_pipeline.models.base import NarrativeIndices

    adapter = object.__new__(OpenAIAdapter)
    adapter.model_id = "gpt-4.1-mini"
    client = MagicMock()
    client.responses.parse.return_value = MagicMock(
        output_parsed=NarrativeIndices(indices=[0, 5, 3]),
    )
    adapter._client = client

    out = adapter.complete_json(prompt="pick clips")
    assert out == [0, 5, 3]
    assert client.responses.parse.called
    assert client.responses.create.call_count == 0


def test_openai_complete_json_eval_uses_structured_schema():
    from gopro_garmin_pipeline.models.base import PromptEvalResult

    adapter = object.__new__(OpenAIAdapter)
    adapter.model_id = "gpt-4.1-mini"
    client = MagicMock()
    client.responses.parse.return_value = MagicMock(
        output_parsed=PromptEvalResult(analysis="over-scores tunnels"),
    )
    adapter._client = client

    out = adapter.complete_json(prompt="eval", system="you are an expert")
    assert out["analysis"] == "over-scores tunnels"
    assert "prompt_changes" in out


def test_gemini_score_frames_parses_list():
    adapter = object.__new__(GeminiAdapter)
    adapter.model_id = "gemini-3.5-flash"
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(
        text='[{"frame_index": 1, "visual": 4, "action": 2}]',
    )
    adapter._client = client

    out = adapter.score_frames(images=[b"x"], system="sys", user="user")
    assert out[0]["frame_index"] == 1


# ─── Downstream compat ────────────────────────────────────────

def test_learned_ranker_has_gemini_true_for_openai():
    from gopro_garmin_pipeline.learned_ranker import _extract_features

    feats = _extract_features({
        "score": 7.0,
        "sources": ["openai", "telemetry"],
        "notes": "",
        "rubric": {"composition": 8, "scenery": 7, "motion": 6, "subject": 5},
    })
    assert feats["has_gemini"] == 1.0


def test_learned_ranker_has_gemini_false_without_vision():
    from gopro_garmin_pipeline.learned_ranker import _extract_features

    feats = _extract_features({
        "score": 3.0,
        "sources": ["telemetry"],
        "notes": "",
    })
    assert feats["has_gemini"] == 0.0


def test_vision_sources_constant():
    assert VISION_SOURCES == frozenset({"gemini", "openai"})


def test_hits_to_segments_uses_provider_identity():
    from gopro_garmin_pipeline.gemini_scan import _hits_to_segments

    # Minimal fake synced clip / ride — skip if Segment construction needs more
    class FakeClip:
        path = Path("GX01.MP4")
        creation_time = None

    class FakeSC:
        clip = FakeClip()
        offset_secs = 0.0

    class FakeRide:
        start_time = None

    FakeClip.creation_time = __import__("datetime").datetime(2026, 1, 1)

    hits = [{
        "clip_name": "GX01.MP4",
        "video_start": 10.0,
        "video_end": 20.0,
        "anchor_video_secs": 15.0,
        "rubric": {"light": 5, "composition": 5, "motion": 5, "scenery": 5, "subject": 5},
        "raw_sum": 25,
        "reason": "test",
        "clip_type": "scenery",
        "crop_x": 50,
    }]
    segs = _hits_to_segments(hits, [FakeSC()], FakeRide(), None, provider="openai")
    assert len(segs) == 1
    assert segs[0].source == "openai"
    assert segs[0].label["type"] == "openai_vision"


def test_default_openai_model_is_not_gpt4o():
    s = Settings(_env_file=None)
    assert s.openai_model == "gpt-4.1-mini"
    assert s.model_provider == "gemini"


def test_openai_dependency_floor_supports_responses_parse():
    """pyproject floor must be high enough that client.responses.parse exists."""
    import importlib.metadata
    import re
    from pathlib import Path

    text = Path("pyproject.toml").read_text()
    m = re.search(r'"openai>=([^"]+)"', text)
    assert m, "openai dependency missing from pyproject.toml"
    floor = m.group(1)
    parts = [int(x) for x in floor.split(".")]
    assert parts >= [1, 66, 0], f"openai floor {floor} predates responses.parse"

    # Installed SDK must expose the API we call.
    ver = importlib.metadata.version("openai")
    assert tuple(int(x) for x in ver.split(".")[:3]) >= (1, 66, 0)
    from openai import OpenAI
    client = OpenAI(api_key="sk-test")
    assert hasattr(client, "responses")
    assert hasattr(client.responses, "parse")


def test_cache_files_filter_by_openai_model(tmp_path, monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.gemini_scan import _model_fingerprint
    from gopro_garmin_pipeline.prompt_eval import (
        _cache_files_for_active_model,
        _load_gemini_hits,
    )

    cache = tmp_path / ".openai_cache"
    cache.mkdir()
    fp_a = _model_fingerprint("gpt-4.1-mini")
    fp_b = _model_fingerprint("gpt-5-mini")
    (cache / f"GX01_v10_m{fp_a}.json").write_text('[{"clip_name": "a"}]')
    (cache / f"GX01_v10_m{fp_b}.json").write_text('[{"clip_name": "b"}]')
    (cache / f"GX02_v10_m{fp_a}_ldeadbeef.json").write_text('[{"clip_name": "a2"}]')

    matched = _cache_files_for_active_model(cache, "openai", "gpt-4.1-mini")
    assert {p.name for p in matched} == {
        f"GX01_v10_m{fp_a}.json",
        f"GX02_v10_m{fp_a}_ldeadbeef.json",
    }

    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg.get_settings.cache_clear()
    try:
        hits = _load_gemini_hits(tmp_path)
        names = {h["clip_name"] for h in hits}
        assert names == {"a", "a2"}
        assert "b" not in names
    finally:
        cfg.get_settings.cache_clear()


def test_load_hits_no_cross_provider_fallback(tmp_path, monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.prompt_eval import _load_gemini_hits

    gem = tmp_path / ".gemini_cache"
    gem.mkdir()
    (gem / "GX01_v10.json").write_text('[{"clip_name": "gemini-only"}]')

    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg.get_settings.cache_clear()
    try:
        assert _load_gemini_hits(tmp_path) == []
    finally:
        cfg.get_settings.cache_clear()
