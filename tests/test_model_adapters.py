"""Unit tests for the shared ModelAdapter layer — no network."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gopro_garmin_pipeline.config import Settings
from gopro_garmin_pipeline.gemini_scan import _clip_cache_key, _model_fingerprint
from gopro_garmin_pipeline.models import (
    VISION_SOURCES,
    GeminiAdapter,
    LocalOpenAIAdapter,
    OpenAIAdapter,
    cache_dir_name,
    get_model_adapter,
    provider_api_key,
    provider_model_id,
)
from gopro_garmin_pipeline.models.base import ClipRubric, CoarseFrameBatch, FrameScore
from gopro_garmin_pipeline.models.local_vlm import (
    _reset_local_request_semaphore_for_tests,
)


# ─── Factory / config ─────────────────────────────────────────

def test_provider_api_key_selects_by_model_provider():
    g = Settings(_env_file=None, model_provider="gemini", gemini_api_key="gk", openai_api_key="ok")
    assert provider_api_key(g) == "gk"
    o = Settings(_env_file=None, model_provider="openai", gemini_api_key="gk", openai_api_key="ok")
    assert provider_api_key(o) == "ok"
    ready = Settings(
        _env_file=None,
        model_provider="local",
        local_base_url="http://localhost:8080/v1",
        local_model="qwen2.5-vl",
        local_api_key="local",
    )
    assert provider_api_key(ready) == "local"
    incomplete = Settings(
        _env_file=None,
        model_provider="local",
        local_base_url="http://localhost:8080/v1",
        local_model="",
    )
    assert provider_api_key(incomplete) == ""


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


def test_get_model_adapter_local():
    _reset_local_request_semaphore_for_tests()
    s = Settings(
        _env_file=None,
        model_provider="local",
        local_base_url="http://localhost:8080/v1",
        local_model="mlx-community/Qwen2.5-VL-3B",
        local_api_key="local",
        local_max_concurrency=1,
        local_timeout_seconds=600,
    )
    with patch("openai.OpenAI") as OpenAI:
        OpenAI.return_value = MagicMock()
        adapter = get_model_adapter(s)
    assert adapter.name == "local"
    assert adapter.model_id == "mlx-community/Qwen2.5-VL-3B"
    OpenAI.assert_called_once()
    kwargs = OpenAI.call_args.kwargs
    assert kwargs["base_url"] == "http://localhost:8080/v1"
    assert kwargs["timeout"] == 600.0


def test_get_model_adapter_local_requires_model():
    s = Settings(
        _env_file=None,
        model_provider="local",
        local_base_url="http://localhost:8080/v1",
        local_model="",
    )
    with pytest.raises(ValueError, match="LOCAL_MODEL"):
        get_model_adapter(s)


def test_get_model_adapter_unknown_raises():
    s = Settings(_env_file=None, model_provider="anthropic", gemini_api_key="x")
    with pytest.raises(ValueError, match="Unknown MODEL_PROVIDER"):
        get_model_adapter(s)


def test_cache_dir_name():
    assert cache_dir_name("gemini") == ".gemini_cache"
    assert cache_dir_name("openai") == ".openai_cache"
    assert cache_dir_name("local") == ".local_cache"


def test_provider_model_id_selects_local_model():
    settings = Settings(
        _env_file=None,
        model_provider="local",
        gemini_model="gemini-3.5-flash",
        local_model="mlx-community/Qwen3-VL-8B-Instruct-3bit",
    )
    assert provider_model_id(settings) == "mlx-community/Qwen3-VL-8B-Instruct-3bit"


def test_cli_exposes_skip_vision_with_legacy_alias():
    from click.testing import CliRunner

    from gopro_garmin_pipeline.cli import main

    runner = CliRunner()
    for command in ("compose", "process", "compare-features"):
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0
        assert "--skip-vision, --skip-gemini" in result.output
        assert "configured vision-model scan" in result.output


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


def test_local_cache_key_includes_model_fingerprint():
    fp = _model_fingerprint("qwen2.5-vl")
    key = _clip_cache_key("GX010346.MP4", provider="local", model_id="qwen2.5-vl")
    assert key == f"GX010346_v10_m{fp}.json"


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


def test_local_is_transient_error_skips_connection_refusal():
    adapter = object.__new__(LocalOpenAIAdapter)

    class FakeTimeout(Exception):
        pass

    class FakeServer(Exception):
        status_code = 503

    class FakeRateLimit(Exception):
        status_code = 429

    assert not adapter.is_transient_error(
        ConnectionError("Local VLM unreachable at http://localhost:8080/v1")
    )
    assert not adapter.is_transient_error(Exception("Connection refused"))
    assert not adapter.is_transient_error(Exception("Failed to connect to localhost"))
    assert adapter.is_transient_error(FakeTimeout("Request Timeout"))
    assert adapter.is_transient_error(FakeServer("backend error"))
    assert adapter.is_transient_error(FakeRateLimit("rate limited"))


def test_local_is_transient_error_skips_permanent_model_configuration_errors():
    adapter = object.__new__(LocalOpenAIAdapter)

    class FakeLoadFailure(Exception):
        status_code = 500

    class FakeNotFound(Exception):
        status_code = 404

    assert not adapter.is_transient_error(
        FakeLoadFailure(
            "Failed to load model: Repo id must use alphanumeric chars: '<your-vlm>'"
        )
    )
    assert not adapter.is_transient_error(FakeNotFound("model not found"))
    assert not adapter.is_transient_error(Exception("repository not found"))


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


# ─── Local OpenAI-compatible adapter ──────────────────────────

def _local_adapter_with_client(client) -> LocalOpenAIAdapter:
    _reset_local_request_semaphore_for_tests()
    adapter = object.__new__(LocalOpenAIAdapter)
    adapter.model_id = "qwen2.5-vl"
    adapter.base_url = "http://localhost:8080/v1"
    adapter._timeout = 600.0
    adapter._semaphore = __import__(
        "gopro_garmin_pipeline.models.local_vlm", fromlist=["_get_local_request_semaphore"]
    )._get_local_request_semaphore(1)
    adapter._client = client
    return adapter


def _chat_response(content: str):
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))],
    )


def test_local_score_frames_uses_json_schema_response_format():
    client = MagicMock()
    payload = {
        "frames": [{
            "frame_index": 0, "visual": 5, "action": 3,
            "clip_type": "landmark", "crop_x": 50, "reason": "bridge",
        }],
    }
    client.chat.completions.create.return_value = _chat_response(json.dumps(payload))
    adapter = _local_adapter_with_client(client)

    out = adapter.score_frames(images=[b"\xff\xd8fake"], system="sys", user="user")
    assert out[0]["frame_index"] == 0
    assert out[0]["visual"] == 5

    kwargs = client.chat.completions.create.call_args.kwargs
    assert "response_format" in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "coarse_frame_batch"
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    assert any(p.get("type") == "image_url" for p in user_content)
    assert any(p.get("type") == "text" for p in user_content)


def test_local_score_clip_rubric_structured():
    client = MagicMock()
    payload = {
        "light": 8, "composition": 7, "motion": 9, "scenery": 6, "subject": 9,
        "peak_offset": 0.5, "clip_type": "action", "crop_x": 50, "reason": "sprint",
    }
    client.chat.completions.create.return_value = _chat_response(json.dumps(payload))
    adapter = _local_adapter_with_client(client)

    out = adapter.score_clip_rubric(images=[b"x"], system="sys", user="user")
    assert out["motion"] == 9
    assert out["clip_type"] == "action"


def test_local_complete_json_narrative_and_eval():
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(
        json.dumps({"indices": [0, 5, 3]}),
    )
    adapter = _local_adapter_with_client(client)
    assert adapter.complete_json(prompt="pick clips") == [0, 5, 3]

    client.chat.completions.create.return_value = _chat_response(
        json.dumps({"analysis": "over-scores tunnels", "prompt_changes": []}),
    )
    out = adapter.complete_json(prompt="eval", system="you are an expert")
    assert out["analysis"] == "over-scores tunnels"


def test_local_falls_back_when_structured_format_rejected():
    client = MagicMock()

    class UnsupportedFormat(Exception):
        status_code = 400

        def __str__(self):
            return "Unsupported response_format type: 'json_schema'"

    client.chat.completions.create.side_effect = [
        UnsupportedFormat(),
        _chat_response(json.dumps([
            {"frame_index": 2, "visual": 4, "action": 4, "clip_type": "x", "crop_x": 50, "reason": "y"},
        ])),
    ]
    adapter = _local_adapter_with_client(client)

    out = adapter.score_frames(images=[b"x"], system="sys", user="user")
    assert out[0]["frame_index"] == 2
    assert client.chat.completions.create.call_count == 2
    second_kwargs = client.chat.completions.create.call_args_list[1].kwargs
    assert "response_format" not in second_kwargs


def test_local_invalid_freeform_raises_not_empty():
    client = MagicMock()

    class UnsupportedFormat(Exception):
        status_code = 400

        def __str__(self):
            return "Unsupported response_format type: 'json_schema'"

    client.chat.completions.create.side_effect = [
        UnsupportedFormat(),
        _chat_response("sorry, I cannot help with that"),
    ]
    adapter = _local_adapter_with_client(client)

    with pytest.raises(ValueError, match="unparseable|expected"):
        adapter.score_frames(images=[b"x"], system="sys", user="user")


def test_local_wrong_shape_fine_pass_raises():
    client = MagicMock()

    class UnsupportedFormat(Exception):
        status_code = 400

        def __str__(self):
            return "Unsupported response_format type: 'json_schema'"

    client.chat.completions.create.side_effect = [
        UnsupportedFormat(),
        _chat_response("[]"),
    ]
    adapter = _local_adapter_with_client(client)

    with pytest.raises(ValueError, match="fine pass"):
        adapter.score_clip_rubric(images=[b"x"], system="sys", user="user")


def test_local_connection_refused_raises_clear_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("Connection refused")
    adapter = _local_adapter_with_client(client)

    with pytest.raises(ConnectionError, match="Local VLM unreachable"):
        adapter.score_frames(images=[b"x"], system="sys", user="user")


def test_local_shared_semaphore_across_adapter_instances():
    """Two adapters must share one process-wide semaphore."""
    _reset_local_request_semaphore_for_tests()
    from gopro_garmin_pipeline.models import local_vlm as lo

    with patch("openai.OpenAI") as OpenAI:
        OpenAI.return_value = MagicMock()
        a = LocalOpenAIAdapter(
            base_url="http://localhost:8080/v1",
            api_key="local",
            model="m1",
            max_concurrency=1,
        )
        b = LocalOpenAIAdapter(
            base_url="http://localhost:8080/v1",
            api_key="local",
            model="m2",
            max_concurrency=1,
        )
    assert a._semaphore is b._semaphore
    assert lo._LOCAL_SEMAPHORE_LIMIT == 1


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
    assert VISION_SOURCES == frozenset({"gemini", "openai", "local"})


def test_learned_ranker_has_gemini_true_for_local():
    from gopro_garmin_pipeline.learned_ranker import _extract_features

    feats = _extract_features({
        "score": 7.0,
        "sources": ["local", "telemetry"],
        "notes": "",
        "rubric": {"composition": 8, "scenery": 7, "motion": 6, "subject": 5},
    })
    assert feats["has_gemini"] == 1.0


def test_active_model_id_local():
    from gopro_garmin_pipeline.prompt_eval import _active_model_id

    s = Settings(
        _env_file=None,
        model_provider="local",
        local_model="mlx-community/Qwen2.5-VL-3B",
    )
    assert _active_model_id(s) == "mlx-community/Qwen2.5-VL-3B"


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
        _load_model_hits,
    )

    cache = tmp_path / ".openai_cache"
    cache.mkdir()
    fp_a = _model_fingerprint("gpt-4.1-mini")
    fp_b = _model_fingerprint("gpt-5-mini")
    (cache / f"GX01_v10_m{fp_a}.json").write_text(
        '[{"clip_name": "GX010001.MP4"}]',
    )
    (cache / f"GX01_v10_m{fp_b}.json").write_text(
        '[{"clip_name": "GX010099.MP4"}]',
    )
    (cache / f"GX02_v10_m{fp_a}_ldeadbeef.json").write_text(
        '[{"clip_name": "GX010002.MP4"}]',
    )

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
        hits = _load_model_hits(tmp_path)
        names = {h["clip_name"] for h in hits}
        assert names == {"GX010001.MP4", "GX010002.MP4"}
        assert "GX010099.MP4" not in names
    finally:
        cfg.get_settings.cache_clear()


def test_load_hits_no_cross_provider_fallback(tmp_path, monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.prompt_eval import _load_model_hits

    gem = tmp_path / ".gemini_cache"
    gem.mkdir()
    (gem / "GX01_v10.json").write_text('[{"clip_name": "GX010001.MP4"}]')

    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg.get_settings.cache_clear()
    try:
        assert _load_model_hits(tmp_path) == []
    finally:
        cfg.get_settings.cache_clear()


# ─── Compare naming / local chapters / report filenames ────────


def test_sanitize_model_id_strips_unsafe_chars():
    import hashlib

    from gopro_garmin_pipeline.prompt_eval import (
        _report_filename,
        _sanitize_model_id,
    )

    assert _sanitize_model_id("gpt-4.1-mini") == "gpt-4.1-mini"
    # Changed-by-sanitize ids get a stable hash suffix
    raw = "org/model:v1"
    fp = hashlib.sha1(raw.encode()).hexdigest()[:8]
    assert _sanitize_model_id(raw) == f"org-model-v1_{fp}"
    assert "/" not in _sanitize_model_id("x/y")
    assert "\\" not in _sanitize_model_id("x\\y")
    name = _report_filename("openai", "org/weird model:v2")
    assert name.startswith("prompt_eval_openai_org-weird-model-v2_")
    assert name.endswith(".json")
    assert "/" not in name
    assert " " not in name


def test_sanitize_model_id_avoids_collision_with_hash_suffix():
    from gopro_garmin_pipeline.prompt_eval import (
        _report_filename,
        _sanitize_model_id,
    )

    a = _sanitize_model_id("org/model:v1")
    b = _sanitize_model_id("org-model-v1")
    assert a != b
    assert b == "org-model-v1"  # already safe — no hash
    assert a.startswith("org-model-v1_")
    assert _report_filename("openai", "org/model:v1") != _report_filename(
        "openai", "org-model-v1",
    )


def test_load_hits_filters_to_local_chapters(tmp_path, monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.prompt_eval import _load_model_hits

    cache = tmp_path / ".gemini_cache"
    cache.mkdir()
    (cache / "GX01_v10.json").write_text(
        json.dumps([
            {"clip_name": "GX010001.MP4", "ride_time_secs": 10},
            {"clip_name": "GX010002.MP4", "ride_time_secs": 20},
        ]),
    )
    (tmp_path / "GX010001.MP4").write_bytes(b"fake")

    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg.get_settings.cache_clear()
    try:
        hits = _load_model_hits(tmp_path)
        assert [h["clip_name"] for h in hits] == ["GX010001.MP4"]
    finally:
        cfg.get_settings.cache_clear()


def test_load_hits_keeps_all_when_no_local_mp4(tmp_path, monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.prompt_eval import _load_model_hits

    cache = tmp_path / ".gemini_cache"
    cache.mkdir()
    (cache / "GX01_v10.json").write_text(
        json.dumps([
            {"clip_name": "GX010001.MP4"},
            {"clip_name": "GX010002.MP4"},
        ]),
    )

    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg.get_settings.cache_clear()
    try:
        hits = _load_model_hits(tmp_path)
        assert {h["clip_name"] for h in hits} == {
            "GX010001.MP4",
            "GX010002.MP4",
        }
    finally:
        cfg.get_settings.cache_clear()


def test_compare_without_mp4_uses_labeled_chapter_tier(tmp_path, monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.prompt_eval import compare_ride

    cache = tmp_path / ".gemini_cache"
    cache.mkdir()
    (cache / "GX01_v10.json").write_text(json.dumps([
        {
            "clip_name": "GX010001.MP4",
            "ride_time_secs": 10,
            "rubric": {
                "light": 5, "composition": 7, "motion": 4,
                "scenery": 7, "subject": 4,
            },
        },
        {
            "clip_name": "GX010099.MP4",
            "ride_time_secs": 20,
            "rubric": {
                "light": 5, "composition": 6, "motion": 4,
                "scenery": 6, "subject": 4,
            },
        },
    ]))
    (tmp_path / "ride_labels.json").write_text(json.dumps([{
        "clip_name": "GX010001.MP4",
        "ride_time_secs": 10,
        "visual": 7,
        "action": 4,
        "scale_version": 2,
    }]))

    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    cfg.get_settings.cache_clear()
    try:
        report = compare_ride(tmp_path)
        assert report.n_labels == 1
        assert report.n_model_hits == 1
        assert len(report.matched) == 1
    finally:
        cfg.get_settings.cache_clear()


def test_filter_labels_keeps_missing_clip_name():
    from gopro_garmin_pipeline.prompt_eval import _filter_labels_to_local_chapters

    local = {"GX010001.MP4"}
    labels = [
        {"notes": "legacy", "ride_time_secs": 1},
        {"clip_name": "", "notes": "empty", "ride_time_secs": 2},
        {"clip_name": "GX010001.MP4", "notes": "present", "ride_time_secs": 3},
        {"clip_name": "GX010099.MP4", "notes": "absent", "ride_time_secs": 4},
    ]
    kept = _filter_labels_to_local_chapters(labels, local)
    notes = [lab["notes"] for lab in kept]
    assert notes == ["legacy", "empty", "present"]


def test_align_moments_openai_model_only_source():
    from gopro_garmin_pipeline.prompt_eval import _align_moments

    labels = []
    hits = [{
        "clip_name": "GX010001.MP4",
        "ride_time_secs": 100,
        "visual": 8,
        "action": 3,
        "clip_type": "scenery",
        "reason": "view",
    }]
    matched, label_only, model_only = _align_moments(
        labels, hits, provider="openai",
    )
    assert matched == []
    assert label_only == []
    assert len(model_only) == 1
    assert model_only[0].source == "openai"
    assert model_only[0].model_visual == 8


def test_compare_report_emits_model_fields_and_scoped_filename(tmp_path, monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.prompt_eval import (
        CompareHit,
        _report_filename,
        compare_ride,
    )

    cache = tmp_path / ".openai_cache"
    cache.mkdir()
    from gopro_garmin_pipeline.gemini_scan import _model_fingerprint

    fp = _model_fingerprint("gpt-4.1-mini")
    (cache / f"GX01_v10_m{fp}.json").write_text(
        json.dumps([{
            "clip_name": "GX010001.MP4",
            "ride_time_secs": 50,
            "visual": 7,
            "action": 2,
            "clip_type": "scenery",
            "reason": "ridge",
        }]),
    )
    (tmp_path / "GX010001.MP4").write_bytes(b"fake")
    (tmp_path / "ride_labels.json").write_text("[]")

    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg.get_settings.cache_clear()
    try:
        report = compare_ride(tmp_path)
        assert report.provider == "openai"
        assert report.n_model_hits == 1
        d = report.to_dict()
        assert d["summary"]["n_model_hits"] == 1
        assert d["summary"]["provider"] == "openai"
        assert "model_only" in d
        assert "gemini_only" not in d
        assert d["model_only"][0]["source"] == "openai"
        assert d["model_only"][0]["model_visual"] == 7

        scoped = tmp_path / _report_filename("openai", "gpt-4.1-mini")
        assert scoped.exists()
        assert not (tmp_path / "prompt_eval.json").exists()
    finally:
        cfg.get_settings.cache_clear()

    hit = CompareHit(
        ride_time_secs=1, source="both",
        model_visual=5, model_action=4, model_clip_type="x", model_reason="y",
    )
    assert "model_visual" in hit.to_dict()
    assert "gemini_visual" not in hit.to_dict()


def test_scoped_report_takes_precedence_over_legacy(tmp_path, monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.prompt_eval import (
        _load_all_reports,
        _report_filename,
    )

    scoped = {
        "summary": {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "n_labels": 1,
            "n_model_hits": 1,
            "n_matched": 1,
            "n_label_only": 0,
            "n_model_only": 0,
        },
        "matched": [],
        "label_only": [],
        "model_only": [{"source": "scoped"}],
        "patterns": {},
    }
    legacy = {
        "summary": {
            "n_labels": 9,
            "n_gemini_hits": 9,
            "n_matched": 0,
            "n_label_only": 0,
            "n_gemini_only": 9,
        },
        "gemini_only": [{"source": "legacy"}],
        "label_only": [],
        "matched": [],
        "patterns": {},
    }
    (tmp_path / _report_filename("gemini", "gemini-3.5-flash")).write_text(
        json.dumps(scoped),
    )
    (tmp_path / "prompt_eval.json").write_text(json.dumps(legacy))

    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg.get_settings.cache_clear()
    try:
        reports = _load_all_reports([tmp_path])
        assert len(reports) == 1
        assert reports[0]["model_only"][0]["source"] == "scoped"
    finally:
        cfg.get_settings.cache_clear()


def test_openai_does_not_fallback_to_legacy_gemini_report(tmp_path, monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.prompt_eval import _load_all_reports

    (tmp_path / "prompt_eval.json").write_text(json.dumps({
        "summary": {
            "n_labels": 1,
            "n_gemini_hits": 1,
            "n_matched": 0,
            "n_label_only": 0,
            "n_gemini_only": 1,
        },
        "gemini_only": [{"gemini_visual": 8}],
        "label_only": [],
        "matched": [],
        "patterns": {},
    }))

    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg.get_settings.cache_clear()
    try:
        assert _load_all_reports([tmp_path]) == []
    finally:
        cfg.get_settings.cache_clear()


def test_gemini_falls_back_to_legacy_prompt_eval(tmp_path, monkeypatch):
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.prompt_eval import _load_all_reports

    (tmp_path / "prompt_eval.json").write_text(json.dumps({
        "summary": {
            "n_labels": 1,
            "n_gemini_hits": 2,
            "n_matched": 0,
            "n_label_only": 1,
            "n_gemini_only": 1,
        },
        "gemini_only": [{"gemini_visual": 8, "gemini_action": 2}],
        "label_only": [],
        "matched": [],
        "patterns": {"gemini_misses_by_type": {"scenery": 1}},
    }))

    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg.get_settings.cache_clear()
    try:
        reports = _load_all_reports([tmp_path])
        assert len(reports) == 1
        r = reports[0]
        assert r["summary"]["n_model_hits"] == 2
        assert r["summary"]["n_model_only"] == 1
        assert "n_gemini_hits" not in r["summary"]
        assert r["summary"]["provider"] == "gemini"
        assert "model_only" in r
        assert r["model_only"][0]["model_visual"] == 8
        assert "gemini_visual" not in r["model_only"][0]
        assert "model_misses_by_type" in r["patterns"]
    finally:
        cfg.get_settings.cache_clear()


def test_legacy_report_rejects_mismatching_model(tmp_path, monkeypatch):
    """Legacy prompt_eval.json with an explicit different model must be rejected."""
    from gopro_garmin_pipeline import config as cfg
    from gopro_garmin_pipeline.prompt_eval import _load_all_reports

    (tmp_path / "prompt_eval.json").write_text(json.dumps({
        "summary": {
            "provider": "gemini",
            "model": "gemini-2.0-flash",
            "n_labels": 1,
            "n_model_hits": 1,
            "n_matched": 0,
            "n_label_only": 0,
            "n_model_only": 1,
        },
        "model_only": [{"model_visual": 8}],
        "label_only": [],
        "matched": [],
        "patterns": {},
    }))

    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg.get_settings.cache_clear()
    try:
        assert _load_all_reports([tmp_path]) == []
    finally:
        cfg.get_settings.cache_clear()


def test_normalize_and_aggregate_mixed_old_new_schemas():
    from gopro_garmin_pipeline.prompt_eval import (
        _aggregate_reports,
        _normalize_report,
    )

    old = _normalize_report({
        "summary": {
            "n_labels": 2,
            "n_gemini_hits": 3,
            "n_matched": 1,
            "n_label_only": 1,
            "n_gemini_only": 2,
        },
        "label_only": [{"label_type": "scenery"}],
        "gemini_only": [
            {"gemini_visual": 9, "gemini_action": 1},
            {"gemini_visual": 4, "gemini_action": 1},
        ],
        "matched": [],
        "patterns": {"avg_gemini_only_scores": {"visual": 6.5, "action": 1}},
    })
    new = {
        "summary": {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "n_labels": 1,
            "n_model_hits": 1,
            "n_matched": 1,
            "n_label_only": 0,
            "n_model_only": 0,
        },
        "label_only": [],
        "model_only": [],
        "matched": [{"source": "both", "model_visual": 7}],
        "patterns": {},
    }
    agg = _aggregate_reports([old, new])
    assert agg["n_rides"] == 2
    assert agg["n_labels"] == 3
    assert agg["n_model_hits"] == 4
    assert agg["n_matched"] == 2
    assert agg["n_model_only"] == 2
    assert agg["all_model_only"][0]["model_visual"] == 9
    assert "gemini_only" not in old
