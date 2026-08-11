# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""80 ms PCM packetization for Nemotron VoiceChat native duplex."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field

import numpy as np

from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    DuplexInputAppendCommand,
    DuplexInputClearCommand,
    DuplexInputCloseCommand,
    DuplexInputCommitCommand,
    DuplexInputEffect,
    DuplexInputFlushCommand,
    DuplexInputSnapshot,
)

NEMOTRON_VOICECHAT_SAMPLE_RATE = 16000
NEMOTRON_VOICECHAT_FRAME_SAMPLES = 1280
_SAMPLE_BYTES = 4
_FRAME_BYTES = NEMOTRON_VOICECHAT_FRAME_SAMPLES * _SAMPLE_BYTES


def decode_pcm_f32le(payload: object, *, exact_frame: bool = False) -> bytes:
    if not isinstance(payload, dict):
        raise ValueError("Nemotron VoiceChat duplex audio payload must be a mapping")
    if payload.get("format") != "pcm_f32le":
        raise ValueError("Nemotron VoiceChat duplex audio format must be pcm_f32le")
    if payload.get("sample_rate_hz") != NEMOTRON_VOICECHAT_SAMPLE_RATE:
        raise ValueError("Nemotron VoiceChat duplex audio sample_rate_hz must be 16000")
    encoded = payload.get("audio")
    if not isinstance(encoded, str):
        raise ValueError("Nemotron VoiceChat duplex audio must be base64 pcm_f32le")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Nemotron VoiceChat duplex audio is not valid base64") from exc
    if len(raw) % _SAMPLE_BYTES:
        raise ValueError("Nemotron VoiceChat pcm_f32le payload has a partial sample")
    if exact_frame and len(raw) != _FRAME_BYTES:
        raise ValueError(
            f"Nemotron VoiceChat native duplex append must contain exactly {NEMOTRON_VOICECHAT_FRAME_SAMPLES} samples"
        )
    values = np.frombuffer(raw, dtype="<f4")
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError("Nemotron VoiceChat duplex audio samples must be finite")
    return raw


@dataclass(slots=True)
class NemotronVoiceChatInputState:
    buffer: bytearray = field(default_factory=bytearray)
    reservations: list[NemotronVoiceChatInputReservation] = field(default_factory=list)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    flush_seq: int = 0


class NemotronVoiceChatInputReservation:
    def __init__(
        self,
        owner: NemotronVoiceChatInputState,
        *,
        operation_id: str,
        payload: dict[str, object] | None,
        raw: bytes,
    ) -> None:
        self._owner = owner
        self.operation_id = operation_id
        self.payload = payload
        self.raw = raw
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return len(self.raw)

    def commit(self) -> None:
        if not self._active:
            return
        self._active = False
        if self in self._owner.reservations:
            self._owner.reservations.remove(self)

    def rollback(self) -> None:
        if not self._active:
            return
        try:
            index = self._owner.reservations.index(self)
        except ValueError:
            self._active = False
            return
        restore = bytearray()
        for reservation in self._owner.reservations[index:]:
            if reservation._active:
                restore.extend(reservation.raw)
                reservation._active = False
        del self._owner.reservations[index:]
        self._owner.buffer[:0] = restore


def _frame_payload(raw: bytes, *, final: bool) -> dict[str, object]:
    return {
        "type": "audio",
        "audio": base64.b64encode(raw).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": NEMOTRON_VOICECHAT_SAMPLE_RATE,
        "final": final,
    }


class NemotronVoiceChatInputController:
    """Turn arbitrary browser packets into ordered, rollback-safe 80 ms frames."""

    @staticmethod
    def create_state() -> NemotronVoiceChatInputState:
        return NemotronVoiceChatInputState()

    @staticmethod
    def _state(state: object) -> NemotronVoiceChatInputState:
        if not isinstance(state, NemotronVoiceChatInputState):
            raise TypeError("invalid Nemotron VoiceChat serving input state")
        return state

    def snapshot(self, state: object) -> DuplexInputSnapshot:
        state = self._state(state)
        return DuplexInputSnapshot(
            pending_byte_count=len(state.buffer),
            has_pending=bool(state.buffer),
            has_reserved=any(reservation.active for reservation in state.reservations),
            input_since_commit=state.input_since_commit,
            speech_since_commit=state.speech_since_commit,
        )

    @staticmethod
    def _reserve_available(
        state: NemotronVoiceChatInputState,
        *,
        operation_id: str,
        final: bool,
    ) -> DuplexInputEffect:
        payloads: list[object] = []
        reservations: list[NemotronVoiceChatInputReservation] = []
        frame_index = 0
        while len(state.buffer) >= _FRAME_BYTES:
            raw = bytes(state.buffer[:_FRAME_BYTES])
            del state.buffer[:_FRAME_BYTES]
            payload = _frame_payload(raw, final=final and len(state.buffer) < _FRAME_BYTES)
            reservation = NemotronVoiceChatInputReservation(
                state,
                operation_id=f"{operation_id}:{frame_index}",
                payload=payload,
                raw=raw,
            )
            state.reservations.append(reservation)
            payloads.append(payload)
            reservations.append(reservation)
            frame_index += 1
        return DuplexInputEffect(
            append_payloads=tuple(payloads),
            reservations=tuple(reservations),
        )

    def append(
        self,
        state: object,
        command: DuplexInputAppendCommand,
    ) -> DuplexInputEffect:
        state = self._state(state)
        raw = decode_pcm_f32le(command.payload)
        state.buffer.extend(raw)
        state.input_since_commit = state.input_since_commit or bool(raw)
        if not command.allow_emit:
            return DuplexInputEffect()
        return self._reserve_available(
            state,
            operation_id=command.operation_id,
            final=False,
        )

    def commit(
        self,
        state: object,
        command: DuplexInputCommitCommand,
    ) -> DuplexInputEffect:
        state = self._state(state)
        if not state.buffer:
            reservation = NemotronVoiceChatInputReservation(
                state,
                operation_id=command.operation_id,
                payload=None,
                raw=b"",
            )
            state.reservations.append(reservation)
            state.input_since_commit = False
            state.speech_since_commit = False
            return DuplexInputEffect(reservations=(reservation,))
        raw = bytes(state.buffer)
        state.buffer.clear()
        padded = raw + bytes(_FRAME_BYTES - len(raw))
        payload = _frame_payload(padded, final=True)
        reservation = NemotronVoiceChatInputReservation(
            state,
            operation_id=command.operation_id,
            payload=payload,
            raw=raw,
        )
        state.reservations.append(reservation)
        state.input_since_commit = False
        state.speech_since_commit = False
        return DuplexInputEffect(
            append_payloads=(payload,),
            reservations=(reservation,),
        )

    def clear(
        self,
        state: object,
        command: DuplexInputClearCommand,
    ) -> DuplexInputEffect:
        state = self._state(state)
        del command
        released = len(state.buffer)
        state.buffer.clear()
        for reservation in state.reservations:
            if reservation.active:
                released += reservation.byte_count
                reservation._active = False
        state.reservations.clear()
        state.input_since_commit = False
        state.speech_since_commit = False
        return DuplexInputEffect(released_bytes=released)

    def flush(
        self,
        state: object,
        command: DuplexInputFlushCommand,
    ) -> DuplexInputEffect:
        state = self._state(state)
        del command
        if not state.buffer:
            return DuplexInputEffect()
        raw = bytes(state.buffer)
        state.buffer.clear()
        state.flush_seq += 1
        return DuplexInputEffect(
            append_payloads=(_frame_payload(raw + bytes(_FRAME_BYTES - len(raw)), final=True),),
        )

    def close(
        self,
        state: object,
        command: DuplexInputCloseCommand,
    ) -> DuplexInputEffect:
        del command
        return self.clear(
            state,
            DuplexInputClearCommand(reason="session_close"),
        )


__all__ = [
    "NEMOTRON_VOICECHAT_FRAME_SAMPLES",
    "NEMOTRON_VOICECHAT_SAMPLE_RATE",
    "NemotronVoiceChatInputController",
    "NemotronVoiceChatInputReservation",
    "NemotronVoiceChatInputState",
    "decode_pcm_f32le",
]
