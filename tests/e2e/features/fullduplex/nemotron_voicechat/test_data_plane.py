from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.engine.model_events import (
    DuplexFunctionCallDelta,
    DuplexFunctionCallEnd,
    DuplexFunctionCallStart,
    DuplexListen,
    DuplexSpeakChunk,
    DuplexSpeakStart,
)
from vllm_omni.experimental.fullduplex.nemotron_voicechat.data_plane import (
    NemotronVoiceChatDataPlaneContext,
    NemotronVoiceChatDataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stage2_output(audio: np.ndarray) -> object:
    completion = SimpleNamespace(
        multimodal_output={
            "model_outputs": [audio],
            "sr": [22050],
        }
    )
    inner = SimpleNamespace(
        request_id="req-0",
        outputs=[completion],
    )
    return SimpleNamespace(stage_id=2, request_output=inner)


def test_codec_audio_starts_output_when_bos_side_channel_is_late() -> None:
    encoded_inputs: list[tuple[object, int]] = []

    def encode_audio(audio, sample_rate, output_format, speed):
        del output_format, speed
        encoded_inputs.append((audio, sample_rate))
        return "encoded-pcm"

    projector = NemotronVoiceChatDataPlaneSession(encode_audio)
    fence = DuplexFence("session", incarnation=1, epoch=2)
    context = NemotronVoiceChatDataPlaneContext(
        fence=fence,
        source_input_seq=7,
        response_format="pcm16",
    )

    events = list(
        projector.project_output(
            _stage2_output(np.ones(1764, dtype=np.float32)),
            context=context,
        )
    )

    assert [type(event) for event in events] == [DuplexSpeakStart, DuplexSpeakChunk]
    assert events[0].output_id == events[1].output_id
    assert events[1].audio_data == "encoded-pcm"
    assert events[1].audio_format == "pcm16"
    assert events[1].sample_rate_hz == 22050
    assert events[1].audio_duration_ms == 80
    assert encoded_inputs[0][1] == 22050


def test_function_channel_projects_openai_typed_events_without_ending_speech() -> None:
    projector = NemotronVoiceChatDataPlaneSession(lambda *_: None)
    projector._special_ids = {"bos": 0, "eos": 1, "pad": 12, "sotc": 20, "eotc": 21}
    projector._decode = lambda token_ids: (
        '[{"name":"weather","arguments":{"city":"Shanghai"}}]' if token_ids == [99] else ""
    )
    fence = DuplexFence("session", incarnation=1, epoch=0)
    context = NemotronVoiceChatDataPlaneContext(fence=fence, source_input_seq=1)

    events = []
    for seq, function_token in enumerate((20, 99, 21), start=1):
        context = NemotronVoiceChatDataPlaneContext(fence=fence, source_input_seq=seq)
        completion = SimpleNamespace(
            multimodal_output={
                "nvc_text_token_ids": [12],
                "nvc_function_token": [function_token],
            }
        )
        output = SimpleNamespace(
            stage_id=0,
            request_output=SimpleNamespace(request_id="req-fc", outputs=[completion]),
        )
        events.extend(projector.project_output(output, context=context))

    function_events = [event for event in events if not isinstance(event, DuplexListen)]
    assert sum(isinstance(event, DuplexListen) for event in events) == 3
    assert [type(event) for event in function_events] == [
        DuplexFunctionCallStart,
        DuplexFunctionCallDelta,
        DuplexFunctionCallEnd,
    ]
    assert function_events[0].name == "weather"
    assert function_events[1].arguments_delta == '{"city":"Shanghai"}'
    assert function_events[2].call_id == function_events[0].call_id
