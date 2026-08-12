from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from vllm.sampling_params import SamplingParams

from vllm_omni.experimental.fullduplex.engine.contracts import DuplexInputMode
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.nemotron_voicechat.runtime import (
    NemotronVoiceChatDuplexRuntimeExtension,
)
from vllm_omni.experimental.fullduplex.nemotron_voicechat.serving_adapter import (
    NemotronVoiceChatServingRuntimeAdapter,
    _normalized_tools,
    _render_tool_prompt,
)
from vllm_omni.model_executor.models.nemotron_voicechat.pipeline import (
    NEMOTRON_VOICECHAT_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _frame() -> dict[str, object]:
    raw = np.zeros(1280, dtype=np.float32).tobytes()
    return {
        "type": "audio",
        "audio": base64.b64encode(raw).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
    }


def _plan(extension, *, input_seq: int):
    return extension.plan_append(
        request_id="req",
        fence=DuplexFence("sid", incarnation=2, epoch=3),
        session_config={},
        runtime_config={
            "nvc_prompt_token_ids": [0, 42, 1],
            "nvc_text_pad_id": 12,
        },
        seq=input_seq,
        turn_seq=input_seq,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload=_frame(),
        final=False,
        sampling_params=SamplingParams(),
    )


def test_pipeline_enables_model_native_duplex_plugins() -> None:
    assert NEMOTRON_VOICECHAT_PIPELINE.duplex_control_enabled is True
    assert NEMOTRON_VOICECHAT_PIPELINE.duplex_runtime_extension == (
        "vllm_omni.experimental.fullduplex.nemotron_voicechat.runtime.NemotronVoiceChatDuplexRuntimeExtension"
    )
    assert NEMOTRON_VOICECHAT_PIPELINE.duplex_serving_adapter == (
        "vllm_omni.experimental.fullduplex.nemotron_voicechat.serving_adapter.NemotronVoiceChatServingRuntimeAdapter"
    )


def test_duplex_deploy_profile_is_one_frame_per_segment() -> None:
    with open("vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml") as stream:
        config = yaml.safe_load(stream)
    assert config["async_chunk"] is True
    assert config["duplex_session"]["max_sessions"] == 1
    assert config["connectors"]["connector_of_shared_memory"]["extra"]["codec_chunk_frames"] == 1
    assert config["stages"][0]["default_sampling_params"]["max_tokens"] == 1
    assert config["stages"][0]["default_sampling_params"]["ignore_eos"] is True


def test_first_append_prefills_prompt_then_each_append_consumes_one_frame() -> None:
    extension = NemotronVoiceChatDuplexRuntimeExtension()

    first = _plan(extension, input_seq=1)
    later = _plan(extension, input_seq=2)

    assert first.prompt["prompt_token_ids"] == [0, 42, 1, 12]
    assert later.prompt["prompt_token_ids"] == [12]
    assert first.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] == 1
    assert later.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] == 2


def test_runtime_rejects_non_frame_payloads() -> None:
    extension = NemotronVoiceChatDuplexRuntimeExtension()
    payload = _frame()
    payload["audio"] = base64.b64encode(np.zeros(1279, dtype=np.float32).tobytes()).decode("ascii")
    with pytest.raises(ValueError, match="1280"):
        extension.plan_append(
            request_id="req",
            fence=DuplexFence("sid"),
            session_config={},
            runtime_config={"nvc_prompt_token_ids": [0, 1], "nvc_text_pad_id": 12},
            seq=1,
            turn_seq=1,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload=payload,
            final=False,
            sampling_params=SamplingParams(),
        )


def test_runtime_forces_greedy_single_token_thinker() -> None:
    extension = NemotronVoiceChatDuplexRuntimeExtension()
    defaults = (SamplingParams(temperature=0.7, max_tokens=99), SamplingParams(), SamplingParams())
    configured = extension.configure_sampling_params(runtime_config={}, defaults=defaults)

    assert configured[0].temperature == 0.0
    assert configured[0].max_tokens == 1
    assert configured[0].ignore_eos is True


def test_stage0_token_is_a_direct_duplex_side_channel() -> None:
    decision = NemotronVoiceChatDuplexRuntimeExtension().decide_output(
        stage_id=0,
        final_stage_id=2,
        segment_finished=True,
        segment_token_ids=(42,),
        segment_output_metadata={},
        output=object(),
    )

    assert decision is not None
    assert decision.metadata["duplex_direct_response"] is True
    assert decision.metadata["nvc_text_token_ids"] == [42]


def test_tools_are_rendered_with_the_nvidia_function_call_contract() -> None:
    config = SimpleNamespace(
        extra_body={
            "realtime_tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "description": "Look up weather.",
                    "parameters": {"type": "object"},
                }
            ]
        }
    )

    tools, signature = _normalized_tools(config)
    prompt = _render_tool_prompt("system", tools)

    assert signature == ('[{"description":"Look up weather.","name":"weather","parameters":{"type":"object"}}]')
    assert "<AVAILABLE_TOOLS>" in prompt
    assert '"name": "weather"' in prompt
    assert "<TOOLCALL>" in prompt
    assert "<TOOL_RESPONSE>" in prompt


def test_more_than_five_tools_fail_before_engine_open() -> None:
    config = SimpleNamespace(extra_body={"realtime_tools": [{"name": f"tool_{index}"} for index in range(6)]})
    with pytest.raises(ValueError, match="at most 5"):
        _normalized_tools(config)


def test_serving_capabilities_report_native_80ms_append() -> None:
    capabilities = NemotronVoiceChatServingRuntimeAdapter.capabilities(max_sessions=1)
    assert capabilities.chunk_period_ms == 80
    assert capabilities.supports_model_native_turn_policy is True
    assert capabilities.supports_core_resumable_request is True
    assert capabilities.supports_core_kv_lease is False
    assert capabilities.supports_multi_session is False
