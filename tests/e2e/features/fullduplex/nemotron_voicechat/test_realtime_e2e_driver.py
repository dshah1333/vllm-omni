from __future__ import annotations

import wave

import numpy as np
import pytest

from tests.e2e.online_serving.nemotron_voicechat_realtime_duplex import (
    _audio_packet_durations_ms,
    _close_and_capture_failures,
    _read_wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_stereo_wav(path, *, sample_rate: int = 24_000) -> None:
    # Opposite-polarity channels make accidental downmixing obvious: the old
    # driver averaged them to silence, whereas channel 0 is the user fixture.
    left = np.full(sample_rate // 10, 8192, dtype="<i2")
    right = np.full(sample_rate // 10, -8192, dtype="<i2")
    interleaved = np.column_stack((left, right)).reshape(-1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(interleaved.tobytes())


def test_read_wav_selects_user_channel_before_resampling(tmp_path) -> None:
    path = tmp_path / "combined-conversation.wav"
    _write_stereo_wav(path)

    pcm = _read_wav(path, input_channel=0)

    assert pcm.dtype == np.dtype("<f4")
    assert pcm.shape == (1600,)
    assert np.mean(pcm) == pytest.approx(0.25, abs=2e-3)


def test_read_wav_rejects_invalid_channel(tmp_path) -> None:
    path = tmp_path / "combined-conversation.wav"
    _write_stereo_wav(path)

    with pytest.raises(ValueError, match="outside WAV channel count"):
        _read_wav(path, input_channel=2)


def test_audio_packet_durations_measure_each_delta_not_cumulative_metadata() -> None:
    events = [
        {
            "type": "response.audio.delta",
            "delta": "AAAAAA==",  # two PCM16 samples
            "sample_rate_hz": 1000,
            "metadata": {"audio_duration_ms": 80},
        },
        {
            "type": "response.audio.delta",
            "delta": "AAAAAAAAAAA=",  # four PCM16 samples
            "sample_rate_hz": 1000,
            "metadata": {"audio_duration_ms": 160},
        },
    ]

    assert _audio_packet_durations_ms(events) == [2.0, 4.0]


@pytest.mark.asyncio
async def test_failed_probe_closes_session_and_saves_events(tmp_path) -> None:
    class Events:
        def __init__(self) -> None:
            self.events = [{"type": "session.created", "session": {"id": "s-0"}}]

        def count(self, event_type: str) -> int:
            return sum(event["type"] == event_type for event in self.events)

    class Client:
        def __init__(self) -> None:
            self.events = Events()
            self.close_calls = 0

        async def close_session(self, *, timeout_s: float) -> None:
            assert timeout_s == 30.0
            self.close_calls += 1
            self.events.events.append({"type": "session.closed"})

    client = Client()
    with pytest.raises(RuntimeError, match="validation failed"):
        async with _close_and_capture_failures(
            client,
            output_dir=tmp_path,
            timeout_s=300.0,
        ):
            raise RuntimeError("validation failed")

    assert client.close_calls == 1
    failed_events = (tmp_path / "events.failed.jsonl").read_text(encoding="utf-8")
    assert '"type": "session.created"' in failed_events
    assert '"type": "session.closed"' in failed_events
