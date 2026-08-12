# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Output projection for Nemotron VoiceChat's independent channels."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.engine.contracts import (
    duplex_resource_request_belongs_to_session,
)
from vllm_omni.experimental.fullduplex.output import (
    get_duplex_output_decision,
)

logger = init_logger(__name__)

EncodeAudio = Callable[[object, int, str, float | None], str | None]


@dataclass(frozen=True, slots=True)
class NemotronVoiceChatDataPlaneContext:
    epoch: int = 0
    turn_id: int = 0
    auto_responds: bool = True
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ("text", "audio")


@dataclass(slots=True)
class _RequestState:
    pending_speech_end: bool = False
    function_active: bool = False
    function_tokens: list[int] = field(default_factory=list)
    function_call_id: str | None = None
    terminal: bool = False


def _coerce_ints(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list | tuple):
        return []
    values: list[int] = []
    for item in value:
        if isinstance(item, torch.Tensor):
            if item.numel() != 1:
                continue
            item = item.item()
        elif isinstance(item, np.generic):
            item = item.item()
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return values


def _unwrap(output: object) -> tuple[object, object | None, int | None]:
    stage_id = getattr(output, "stage_id", None)
    inner = getattr(output, "request_output", None)
    if inner is not None and inner is not output:
        output = inner
    outputs = getattr(output, "outputs", None)
    completion = outputs[0] if isinstance(outputs, list) and outputs else None
    if stage_id is None:
        stage_id = getattr(output, "stage_id", None)
    return output, completion, int(stage_id) if isinstance(stage_id, int) else None


def _multimodal(output: object, completion: object | None) -> dict[str, object]:
    decision = get_duplex_output_decision(output)
    metadata = getattr(decision, "metadata", None)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    for candidate in (
        getattr(output, "multimodal_output", None),
        getattr(completion, "multimodal_output", None) if completion is not None else None,
    ):
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def _audio_value(metadata: Mapping[str, object]) -> object | None:
    value = next((metadata[key] for key in ("audio", "model_outputs", "latent") if key in metadata), None)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _sample_rate(metadata: Mapping[str, object]) -> int:
    value = metadata.get("sr", metadata.get("sample_rate_hz", 22050))
    if isinstance(value, list) and value:
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return int(value) if isinstance(value, int | float) else 22050


def _audio_samples(audio: object | None) -> int:
    if audio is None:
        return 0
    if isinstance(audio, torch.Tensor):
        return int(audio.numel())
    try:
        return int(np.asarray(audio, dtype=np.float32).size)
    except (TypeError, ValueError):
        return 0


class NemotronVoiceChatDataPlaneSession:
    """Join frame-locked text/function outputs with Stage-2 audio."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._requests: dict[str, _RequestState] = {}
        self._tokenizer = None
        self._special_ids: dict[str, int] | None = None

    def begin_request(self, request_id: str) -> None:
        self._requests.setdefault(request_id, _RequestState()).terminal = False

    def is_terminal(self, request_id: str | None) -> bool:
        return bool(request_id and self._requests.get(request_id) and self._requests[request_id].terminal)

    def mark_terminal(self, request_id: str) -> None:
        self._requests.setdefault(request_id, _RequestState()).terminal = True

    def close_stream(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id:
            self._requests.pop(active_request_id, None)
        for request_id in tuple(self._requests):
            if duplex_resource_request_belongs_to_session(request_id, session_id):
                self._requests.pop(request_id, None)

    def _load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            ref = os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH") or "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
            self._tokenizer = AutoTokenizer.from_pretrained(ref, trust_remote_code=False)
        return self._tokenizer

    def _ids(self) -> dict[str, int]:
        if self._special_ids is None:
            tok = self._load_tokenizer()
            self._special_ids = {
                "bos": int(tok.convert_tokens_to_ids("<s>")),
                "eos": int(tok.convert_tokens_to_ids("</s>")),
                "pad": int(tok.convert_tokens_to_ids("<SPECIAL_12>")),
                "sotc": int(tok.convert_tokens_to_ids("<SPECIAL_20>")),
                "eotc": int(tok.convert_tokens_to_ids("<SPECIAL_21>")),
            }
        return self._special_ids

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return str(self._load_tokenizer().decode(token_ids, skip_special_tokens=True))

    def project(
        self,
        result: object,
        *,
        context: NemotronVoiceChatDataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        if not isinstance(result, dict):
            return
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return
        for output in outputs:
            yield from self.project_output(output, context=context)

    def project_output(
        self,
        output: object,
        *,
        context: NemotronVoiceChatDataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        context = context or NemotronVoiceChatDataPlaneContext()
        outer = output
        output, completion, stage_id = _unwrap(output)
        request_id = getattr(output, "request_id", None) or getattr(outer, "request_id", None)
        request_id = str(request_id) if request_id is not None else ""
        state = self._requests.setdefault(request_id, _RequestState())
        metadata = _multimodal(outer, completion)
        ids = self._ids()

        if stage_id == 0 or "nvc_text_token_ids" in metadata:
            text_ids = _coerce_ints(metadata.get("nvc_text_token_ids"))
            if not text_ids and completion is not None:
                text_ids = _coerce_ints(
                    getattr(completion, "token_ids", None) or getattr(completion, "cumulative_token_ids", None)
                )
            for token_id in text_ids[-1:]:
                if token_id == ids["eos"]:
                    state.pending_speech_end = True
                elif token_id == ids["pad"]:
                    yield {
                        "stage_role": "llm",
                        "is_listen": True,
                        "model_listen": True,
                        "listen_source": "model_listen",
                        "data_plane_request_id": request_id,
                        "end_of_turn": False,
                    }
                elif token_id != ids["bos"]:
                    text = self._decode([token_id])
                    if text:
                        yield {
                            "stage_role": "llm",
                            "is_listen": False,
                            "data_plane_request_id": request_id,
                            "text": text,
                            "end_of_turn": False,
                        }

            for function_id in _coerce_ints(metadata.get("nvc_function_token"))[-1:]:
                yield from self._project_function_token(
                    function_id,
                    state=state,
                    request_id=request_id,
                    ids=ids,
                )
            return

        audio = _audio_value(metadata)
        sample_rate = _sample_rate(metadata)
        sample_count = _audio_samples(audio)
        encoded = self._encode_audio(audio, sample_rate, context.response_format, context.speed)
        segment_finished = bool(getattr(output, "finished", False) or getattr(outer, "finished", False))
        end_of_turn = state.pending_speech_end and segment_finished
        if encoded:
            yield {
                "stage_role": "tts",
                "is_listen": False,
                "data_plane_request_id": request_id,
                "text": "",
                "audio_data": encoded,
                "audio_format": context.response_format,
                "sample_rate_hz": sample_rate,
                "audio_duration_ms": round(sample_count * 1000 / max(1, sample_rate)),
                "audio_text_mark": True,
                "end_of_turn": end_of_turn,
            }
        elif end_of_turn:
            yield {
                "stage_role": "tts",
                "is_listen": False,
                "data_plane_request_id": request_id,
                "text": "",
                "end_of_turn": True,
            }
        if end_of_turn:
            state.pending_speech_end = False

    def _project_function_token(
        self,
        token_id: int,
        *,
        state: _RequestState,
        request_id: str,
        ids: Mapping[str, int],
    ) -> Iterator[dict[str, object]]:
        if token_id == ids["sotc"]:
            state.function_active = True
            state.function_tokens.clear()
            state.function_call_id = f"call_{uuid4().hex}"
            return
        if token_id == ids["eotc"] and state.function_active:
            state.function_active = False
            raw = self._decode(state.function_tokens)
            state.function_tokens.clear()
            for call in self._parse_calls(raw):
                yield {
                    "stage_role": "function",
                    "data_plane_request_id": request_id,
                    "function_call": True,
                    "call_id": state.function_call_id or f"call_{uuid4().hex}",
                    "name": str(call.get("name") or "unknown"),
                    "arguments": (
                        call.get("arguments")
                        if isinstance(call.get("arguments"), str)
                        else json.dumps(call.get("arguments", {}), separators=(",", ":"))
                    ),
                }
            state.function_call_id = None
            return
        if state.function_active and token_id != ids["pad"]:
            state.function_tokens.append(token_id)

    @staticmethod
    def _parse_calls(value: str) -> list[dict[str, object]]:
        text = value.strip()
        if "<TOOLCALL>" in text:
            text = text.split("<TOOLCALL>", 1)[1]
        if "</TOOLCALL>" in text:
            text = text.split("</TOOLCALL>", 1)[0]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [{"name": "unknown", "arguments": text}] if text else []
        if isinstance(parsed, dict):
            parsed = [parsed]
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


__all__ = [
    "NemotronVoiceChatDataPlaneContext",
    "NemotronVoiceChatDataPlaneSession",
]
