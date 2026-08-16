from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.engine.duplex_runtime import duplex_resource_request_id
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.nemotron_voicechat.data_plane import (
    NemotronVoiceChatDataPlaneContext,
    NemotronVoiceChatDataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_RUNTIME = {
    "nvc_text_bos_id": 0,
    "nvc_text_eos_id": 1,
    "nvc_text_pad_id": 12,
    "nvc_function_sotc_id": 20,
    "nvc_function_eotc_id": 21,
    "nvc_tokenizer_ref": "test-tokenizer",
}


def _projector(encode_audio=lambda *_: None) -> NemotronVoiceChatDataPlaneSession:
    projector = NemotronVoiceChatDataPlaneSession(encode_audio)
    projector.configure_runtime(_RUNTIME, tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: "text"))
    return projector


def _stage0_output(token_id: int) -> object:
    completion = SimpleNamespace(multimodal_output={"nvc_text_token_ids": [token_id]})
    return SimpleNamespace(
        stage_id=0,
        request_output=SimpleNamespace(request_id="req-0", outputs=[completion]),
    )


def _stage2_output(audio: np.ndarray, *, finished: bool = False) -> object:
    completion = SimpleNamespace(
        multimodal_output={
            "model_outputs": [audio],
            "sr": [22050],
        }
    )
    return SimpleNamespace(
        stage_id=2,
        request_output=SimpleNamespace(request_id="req-0", outputs=[completion], finished=finished),
    )


def test_codec_audio_projects_legacy_runtime_event() -> None:
    projector = _projector(lambda *_: "encoded-pcm")
    context = NemotronVoiceChatDataPlaneContext(epoch=2, response_format="pcm16")

    events = list(projector.project_output(_stage2_output(np.ones(1764, dtype=np.float32)), context=context))

    assert len(events) == 1
    assert events[0]["audio_data"] == "encoded-pcm"
    assert events[0]["audio_format"] == "pcm16"
    assert events[0]["sample_rate_hz"] == 22050
    assert events[0]["audio_duration_ms"] == 80


def test_function_channel_projects_completed_call_without_ending_speech() -> None:
    projector = _projector()
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


@pytest.mark.parametrize("stage2_first", [False, True])
def test_speech_end_joins_eos_and_stage2_completion_in_either_order(stage2_first: bool) -> None:
    projector = _projector(lambda *_: "audio")
    eos = _stage0_output(1)
    final_audio = _stage2_output(np.ones(1764, dtype=np.float32), finished=True)

    outputs = (final_audio, eos) if stage2_first else (eos, final_audio)
    events = [event for output in outputs for event in projector.project_output(output)]

    assert sum(event.get("end_of_turn") is True for event in events) == 1


def test_streaming_audio_chunk_advances_eos_join_without_finishing_request() -> None:
    projector = _projector(lambda *_: "audio")

    assert list(projector.project_output(_stage0_output(1))) == []
    events = list(
        projector.project_output(
            _stage2_output(np.ones(1764, dtype=np.float32), finished=False),
        )
    )

    assert sum(event.get("end_of_turn") is True for event in events) == 1


def test_empty_listen_segment_does_not_satisfy_later_speech_eos() -> None:
    projector = _projector(lambda *_: None)

    assert list(projector.project_output(_stage2_output(np.empty(0, dtype=np.float32), finished=True))) == []
    assert list(projector.project_output(_stage0_output(1))) == []

    speech = list(
        projector.project_output(
            _stage2_output(np.ones(1764, dtype=np.float32), finished=True),
        )
    )

    assert sum(event.get("end_of_turn") is True for event in speech) == 1


def test_previous_audio_frame_does_not_satisfy_later_text_eos() -> None:
    projector = _projector(lambda *_: "audio")
    projector._decode = lambda _token_ids: "hello"

    assert not [event for event in projector.project_output(_stage0_output(42)) if event.get("end_of_turn") is True]
    first_audio = list(
        projector.project_output(
            _stage2_output(np.ones(1764, dtype=np.float32), finished=True),
        )
    )
    assert not [event for event in first_audio if event.get("end_of_turn") is True]

    # The old boolean latch closed here because *some* earlier Stage-2 chunk
    # had finished.  EOS is text frame 2, so it must wait for audio frame 2.
    eos = list(projector.project_output(_stage0_output(1)))
    assert not [event for event in eos if event.get("end_of_turn") is True]

    second_audio = list(
        projector.project_output(
            _stage2_output(np.ones(1764, dtype=np.float32), finished=True),
        )
    )
    assert sum(event.get("end_of_turn") is True for event in second_audio) == 1


@pytest.mark.parametrize(
    ("modalities", "expected_text", "expected_audio"),
    [
        (("text",), True, False),
        (("audio",), False, True),
    ],
)
def test_modalities_filter_payloads_but_preserve_state(
    modalities: tuple[str, ...],
    expected_text: bool,
    expected_audio: bool,
) -> None:
    projector = _projector(lambda *_: "audio")
    projector._decode = lambda _token_ids: "hello"
    context = NemotronVoiceChatDataPlaneContext(modalities=modalities)

    text_events = list(projector.project_output(_stage0_output(42), context=context))
    audio_events = list(
        projector.project_output(
            _stage2_output(np.ones(1764, dtype=np.float32)),
            context=context,
        )
    )

    assert bool([event for event in text_events if event.get("text") == "hello"]) is expected_text
    assert bool([event for event in audio_events if event.get("audio_data") == "audio"]) is expected_audio


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        '{"arguments":{}}',
        '{"name":"weather","arguments":"not-json"}',
        '[{"name":"weather","arguments":[]}]',
    ],
)
def test_malformed_function_payload_is_never_reported_as_success(raw: str) -> None:
    projector = _projector()
    projector._decode = lambda _token_ids: raw
    context = NemotronVoiceChatDataPlaneContext()

    events = []
    for function_token in (20, 99, 21):
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

    assert not [event for event in events if event.get("function_call") is True]
    errors = [event for event in events if event.get("error_code") == "nemotron_function_call_parse_error"]
    assert len(errors) == 1


def test_function_channel_without_eotc_is_not_reported_as_success() -> None:
    projector = _projector()
    context = NemotronVoiceChatDataPlaneContext()

    events = []
    for text_token, function_token in ((12, 20), (1, 99)):
        completion = SimpleNamespace(
            multimodal_output={
                "nvc_text_token_ids": [text_token],
                "nvc_function_token": [function_token],
            }
        )
        output = SimpleNamespace(
            stage_id=0,
            request_output=SimpleNamespace(request_id="req-fc", outputs=[completion]),
        )
        events.extend(projector.project_output(output, context=context))

    assert not [event for event in events if event.get("function_call") is True]


def test_new_epoch_request_does_not_inherit_partial_function_or_frame_state() -> None:
    projector = _projector(lambda *_: "audio")
    old_request = duplex_resource_request_id(DuplexFence("sid", epoch=1), "stage0")
    new_request = duplex_resource_request_id(DuplexFence("sid", epoch=2), "stage0")

    def stage0(request_id: str, *, text_token: int, function_token: int | None = None) -> object:
        metadata: dict[str, object] = {"nvc_text_token_ids": [text_token]}
        if function_token is not None:
            metadata["nvc_function_token"] = [function_token]
        completion = SimpleNamespace(multimodal_output=metadata)
        return SimpleNamespace(
            stage_id=0,
            request_output=SimpleNamespace(request_id=request_id, outputs=[completion]),
        )

    projector.begin_request(old_request)
    list(projector.project_output(stage0(old_request, text_token=42, function_token=20)))
    projector.begin_request(new_request)

    # EOTC in the new epoch must not close the old epoch's partial function.
    new_events = list(projector.project_output(stage0(new_request, text_token=1, function_token=21)))
    assert not [event for event in new_events if event.get("function_call") is True]
    assert not [event for event in new_events if event.get("end_of_turn") is True]

    audio = SimpleNamespace(
        stage_id=2,
        request_output=SimpleNamespace(
            request_id=new_request,
            outputs=[
                SimpleNamespace(multimodal_output={"model_outputs": [np.ones(1764, dtype=np.float32)], "sr": [22050]})
            ],
        ),
    )
    audio_events = list(projector.project_output(audio))
    assert sum(event.get("end_of_turn") is True for event in audio_events) == 1
