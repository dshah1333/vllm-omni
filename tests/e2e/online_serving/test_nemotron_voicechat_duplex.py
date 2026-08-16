"""Nightly model-level coverage for Nemotron VoiceChat native duplex serving."""

from __future__ import annotations

import asyncio
import hashlib
import os
import wave
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

from tests.e2e.online_serving.nemotron_voicechat_realtime_duplex import parse_args, run
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get(
    "VLLM_TEST_NEMOTRON_VOICECHAT_MODEL",
    "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
)
DEPLOY_CONFIG = get_deploy_config_path("nemotron_labs_voicechat_duplex.yaml")
TURN_TAKING_SHA256 = "9602d5f78799644964b631e632c5740c325d518847cbb1ffb8917dee7abd17c1"

_TOKENIZER_OVERRIDE = os.environ.get("VLLM_TEST_NEMOTRON_VOICECHAT_LLM_PATH")
_SERVER_ENV = {"NEMOTRON_VOICECHAT_LLM_PATH": _TOKENIZER_OVERRIDE} if _TOKENIZER_OVERRIDE else None
SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            env_dict=_SERVER_ENV,
        ),
        id="native-duplex",
    )
]

pytestmark = [pytest.mark.full_model, pytest.mark.omni]


def _turn_taking_fixture(model_prefix: str) -> Path:
    candidates = [Path(MODEL)]
    if model_prefix:
        candidates.insert(0, Path(model_prefix) / MODEL)
    model_root = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if model_root is None:
        model_root = Path(snapshot_download(MODEL, local_files_only=True))

    fixture = model_root / "turn_taking.wav"
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != TURN_TAKING_SHA256:
        raise AssertionError("Nemotron VoiceChat turn-taking fixture SHA256 mismatch")
    with wave.open(str(fixture), "rb") as wav_file:
        actual_format = (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getcomptype(),
        )
    if actual_format != (24_000, 2, 2, "NONE"):
        raise AssertionError(f"Nemotron VoiceChat turn-taking fixture must be 24 kHz stereo PCM16, got {actual_format}")
    return fixture


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_native_duplex_turn_taking_streams_model_audio(
    omni_server,
    model_prefix: str,
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--url",
            f"ws://{omni_server.host}:{omni_server.port}/v1/realtime",
            "--model",
            omni_server.model,
            "--input-wav",
            str(_turn_taking_fixture(model_prefix)),
            "--input-channel",
            "0",
            "--max-frames",
            "190",
            "--minimum-audio-chunks",
            "10",
            "--minimum-audio-rms",
            "0.00001",
            "--no-realtime",
            "--timeout-s",
            "300",
            "--output-dir",
            str(tmp_path / "native_duplex"),
        ]
    )

    result = asyncio.run(run(args))

    assert result["ok"] is True
    assert result["input_frames"] == 190
    assert result["audio_bytes"] > 0
    assert result["audio_rms"] >= 0.00001
    assert result["event_counts"]["response.listen"] > 0
    assert result["event_counts"]["response.speak"] > 0
    assert result["event_counts"]["response.audio.delta"] >= 10
    assert result["event_counts"]["response.audio_transcript.delta"] > 0
    assert result["event_counts"]["response.done"] > 0
    assert result["timing"]["request_metrics"]["ttfp_ms"] >= 0
    assert (tmp_path / "native_duplex" / "output.wav").is_file()
