from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.nemotron_voicechat.input import (
    NEMOTRON_VOICECHAT_FRAME_SAMPLES,
    NemotronVoiceChatPcmAppendBuffer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _payload(samples: np.ndarray, *, sample_rate_hz: int = 16000, fmt: str = "pcm_f32le") -> dict[str, object]:
    return {
        "type": "audio",
        "audio": base64.b64encode(np.asarray(samples, dtype=np.float32).tobytes()).decode("ascii"),
        "format": fmt,
        "sample_rate_hz": sample_rate_hz,
    }


def _append(buffer, samples, operation_id):
    return buffer.prepare_append(
        _payload(np.asarray(samples, dtype=np.float32)),
        operation_id=operation_id,
        chunk_period_ms=80,
        allow_emit=True,
    )


def test_irregular_browser_packets_emit_exact_ordered_80ms_frame() -> None:
    buffer = NemotronVoiceChatPcmAppendBuffer()
    assert _append(buffer, np.arange(1000), "packet-1") is None

    reservation = _append(buffer, np.arange(1000, 1280), "packet-2")

    assert reservation is not None
    assert reservation.operation_id == "packet-2"
    decoded = np.frombuffer(base64.b64decode(reservation.payload["audio"]), dtype=np.float32)
    np.testing.assert_array_equal(decoded, np.arange(1280, dtype=np.float32))
    assert buffer.pending_byte_count == 0


def test_rollback_restores_failed_frame() -> None:
    buffer = NemotronVoiceChatPcmAppendBuffer()
    reservation = _append(buffer, np.arange(1280), "packet")
    assert reservation is not None

    reservation.rollback()
    replay = _append(buffer, np.empty(0, dtype=np.float32), "retry")

    assert replay is not None
    decoded = np.frombuffer(base64.b64decode(replay.payload["audio"]), dtype=np.float32)
    np.testing.assert_array_equal(decoded, np.arange(1280, dtype=np.float32))


def test_commit_pads_only_the_terminal_tail() -> None:
    buffer = NemotronVoiceChatPcmAppendBuffer()
    _append(buffer, np.arange(440), "packet")

    reservation = buffer.prepare_commit(operation_id="commit", chunk_period_ms=80)

    decoded = np.frombuffer(base64.b64decode(reservation.payload["audio"]), dtype=np.float32)
    np.testing.assert_array_equal(decoded[:440], np.arange(440, dtype=np.float32))
    np.testing.assert_array_equal(decoded[440:], np.zeros(840, dtype=np.float32))
    assert reservation.payload["final"] is True


def test_exact_boundary_commit_has_no_synthetic_audio() -> None:
    buffer = NemotronVoiceChatPcmAppendBuffer()
    reservation = _append(buffer, np.zeros(NEMOTRON_VOICECHAT_FRAME_SAMPLES), "packet")
    assert reservation is not None
    reservation.commit()

    committed = buffer.prepare_commit(operation_id="commit", chunk_period_ms=80)

    assert committed.payload is None
    assert committed.byte_count == 0


@pytest.mark.parametrize("operation", ["commit", "flush"])
def test_deferred_multi_frame_tail_fails_explicitly(operation: str) -> None:
    buffer = NemotronVoiceChatPcmAppendBuffer()
    for index in range(3):
        reservation = buffer.prepare_append(
            _payload(np.zeros(640, dtype=np.float32)),
            operation_id=f"packet-{index}",
            chunk_period_ms=80,
            allow_emit=False,
        )
        assert reservation is None

    with pytest.raises(ValueError, match="auto_response"):
        if operation == "commit":
            buffer.prepare_commit(operation_id="commit", chunk_period_ms=80)
        else:
            buffer.flush(chunk_period_ms=80)
    assert buffer.pending_byte_count == 3 * 640 * np.dtype(np.float32).itemsize


def test_rejects_packets_larger_than_one_model_frame() -> None:
    buffer = NemotronVoiceChatPcmAppendBuffer()
    with pytest.raises(ValueError, match="at most 1280"):
        _append(buffer, np.zeros(1281), "bad")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload(np.zeros(1280), sample_rate_hz=24000), "16000"),
        (_payload(np.zeros(1280), fmt="pcm16"), "pcm_f32le"),
        (_payload(np.array([np.nan], dtype=np.float32)), "finite"),
    ],
)
def test_rejects_invalid_audio_contract(payload: dict[str, object], message: str) -> None:
    buffer = NemotronVoiceChatPcmAppendBuffer()
    with pytest.raises(ValueError, match=message):
        buffer.prepare_append(
            payload,
            operation_id="bad",
            chunk_period_ms=80,
            allow_emit=True,
        )
