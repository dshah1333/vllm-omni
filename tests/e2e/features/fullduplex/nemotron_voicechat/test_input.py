from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.nemotron_voicechat.input import (
    NEMOTRON_VOICECHAT_FRAME_SAMPLES,
    NemotronVoiceChatInputController,
)
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    DuplexInputAppendCommand,
    DuplexInputCommitCommand,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _payload(samples: np.ndarray, *, sample_rate_hz: int = 16000, fmt: str = "pcm_f32le") -> dict[str, object]:
    return {
        "type": "audio",
        "audio": base64.b64encode(np.asarray(samples, dtype=np.float32).tobytes()).decode("ascii"),
        "format": fmt,
        "sample_rate_hz": sample_rate_hz,
    }


def _append(controller, state, samples, operation_id):
    return controller.append(
        state,
        DuplexInputAppendCommand(
            payload=_payload(np.asarray(samples, dtype=np.float32)),
            operation_id=operation_id,
            chunk_period_ms=80,
            allow_emit=True,
        ),
    )


def test_irregular_browser_packets_emit_exact_ordered_80ms_frames() -> None:
    controller = NemotronVoiceChatInputController()
    state = controller.create_state()

    first = _append(controller, state, np.arange(1000), "packet-1")
    second = _append(controller, state, np.arange(1000, 3000), "packet-2")

    assert first.append_payloads == ()
    assert len(second.append_payloads) == 2
    assert [reservation.operation_id for reservation in second.reservations] == ["packet-2:0", "packet-2:1"]
    decoded = [
        np.frombuffer(base64.b64decode(payload["audio"]), dtype=np.float32) for payload in second.append_payloads
    ]
    assert all(frame.shape == (NEMOTRON_VOICECHAT_FRAME_SAMPLES,) for frame in decoded)
    np.testing.assert_array_equal(np.concatenate(decoded), np.arange(2560, dtype=np.float32))
    assert controller.snapshot(state).pending_byte_count == 440 * 4


def test_rollback_restores_failed_frame_and_later_reservations_in_order() -> None:
    controller = NemotronVoiceChatInputController()
    state = controller.create_state()
    effect = _append(controller, state, np.arange(3000), "packet")

    effect.reservations[0].rollback()
    replay = _append(controller, state, np.empty(0, dtype=np.float32), "retry")

    assert len(replay.append_payloads) == 2
    decoded = np.concatenate(
        [np.frombuffer(base64.b64decode(payload["audio"]), dtype=np.float32) for payload in replay.append_payloads]
    )
    np.testing.assert_array_equal(decoded, np.arange(2560, dtype=np.float32))


def test_commit_pads_only_the_terminal_tail() -> None:
    controller = NemotronVoiceChatInputController()
    state = controller.create_state()
    _append(controller, state, np.arange(440), "packet")

    effect = controller.commit(
        state,
        DuplexInputCommitCommand(operation_id="commit", chunk_period_ms=80),
    )

    assert len(effect.append_payloads) == 1
    decoded = np.frombuffer(base64.b64decode(effect.append_payloads[0]["audio"]), dtype=np.float32)
    np.testing.assert_array_equal(decoded[:440], np.arange(440, dtype=np.float32))
    np.testing.assert_array_equal(decoded[440:], np.zeros(840, dtype=np.float32))
    assert effect.append_payloads[0]["final"] is True


def test_exact_boundary_commit_has_no_synthetic_audio() -> None:
    controller = NemotronVoiceChatInputController()
    state = controller.create_state()
    effect = _append(controller, state, np.zeros(NEMOTRON_VOICECHAT_FRAME_SAMPLES), "packet")
    effect.reservations[0].commit()

    committed = controller.commit(
        state,
        DuplexInputCommitCommand(operation_id="commit", chunk_period_ms=80),
    )

    assert committed.append_payloads == ()
    assert len(committed.reservations) == 1
    assert committed.reservations[0].byte_count == 0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload(np.zeros(1280), sample_rate_hz=24000), "16000"),
        (_payload(np.zeros(1280), fmt="pcm16"), "pcm_f32le"),
        (_payload(np.array([np.nan], dtype=np.float32)), "finite"),
    ],
)
def test_rejects_invalid_audio_contract(payload: dict[str, object], message: str) -> None:
    controller = NemotronVoiceChatInputController()
    with pytest.raises(ValueError, match=message):
        controller.append(
            controller.create_state(),
            DuplexInputAppendCommand(
                payload=payload,
                operation_id="bad",
                chunk_period_ms=80,
                allow_emit=True,
            ),
        )
