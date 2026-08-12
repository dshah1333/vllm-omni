from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.nemotron_voicechat.data_plane import (
    NemotronVoiceChatDataPlaneContext,
    NemotronVoiceChatDataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stage2_output(audio: np.ndarray) -> object:
    completion = SimpleNamespace(multimodal_output={"model_outputs": [audio], "sr": [22050]})
    return SimpleNamespace(
        stage_id=2,
        request_output=SimpleNamespace(request_id="req-0", outputs=[completion]),
    )


def test_codec_audio_projects_legacy_runtime_event() -> None:
    projector = NemotronVoiceChatDataPlaneSession(lambda *_: "encoded-pcm")
    context = NemotronVoiceChatDataPlaneContext(epoch=2, response_format="pcm16")

    events = list(projector.project_output(_stage2_output(np.ones(1764, dtype=np.float32)), context=context))

    assert len(events) == 1
    assert events[0]["audio_data"] == "encoded-pcm"
    assert events[0]["audio_format"] == "pcm16"
    assert events[0]["sample_rate_hz"] == 22050
    assert events[0]["audio_duration_ms"] == 80


def test_function_channel_projects_completed_call_without_ending_speech() -> None:
    projector = NemotronVoiceChatDataPlaneSession(lambda *_: None)
    projector._special_ids = {"bos": 0, "eos": 1, "pad": 12, "sotc": 20, "eotc": 21}
    projector._decode = lambda token_ids: (
        '[{"name":"weather","arguments":{"city":"Shanghai"}}]' if token_ids == [99] else ""
    )
    context = NemotronVoiceChatDataPlaneContext(epoch=0)

    events = []
    for function_token in (20, 99, 21):
        completion = SimpleNamespace(
            multimodal_output={"nvc_text_token_ids": [12], "nvc_function_token": [function_token]}
        )
        output = SimpleNamespace(
            stage_id=0,
            request_output=SimpleNamespace(request_id="req-fc", outputs=[completion]),
        )
        events.extend(projector.project_output(output, context=context))

    listen = [event for event in events if event.get("is_listen") is True]
    function = [event for event in events if event.get("function_call") is True]
    assert len(listen) == 3
    assert len(function) == 1
    assert function[0]["name"] == "weather"
    assert function[0]["arguments"] == '{"city":"Shanghai"}'
