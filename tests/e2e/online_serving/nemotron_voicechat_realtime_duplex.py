# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end probe for Nemotron VoiceChat native duplex Realtime serving."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import time
import uuid
import wave
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

from vllm_omni.experimental.fullduplex.client import (
    RealtimeDuplexClient,
    wait_for,
    write_pcm16_wav,
)

INPUT_SAMPLE_RATE_HZ = 16_000
OUTPUT_SAMPLE_RATE_HZ = 22_050
FRAME_SAMPLES = 1_280
FRAME_PERIOD_S = FRAME_SAMPLES / INPUT_SAMPLE_RATE_HZ
DEFAULT_FUNCTION_TOOLS = [
    {
        "type": "function",
        "name": "generate_random_number",
        "description": "Generate a random integer between min and max (inclusive).",
        "parameters": {
            "type": "object",
            "properties": {
                "min": {"type": "integer", "description": "Minimum value (inclusive)"},
                "max": {"type": "integer", "description": "Maximum value (inclusive)"},
            },
            "required": ["min", "max"],
        },
    }
]


def _url(base_url: str, model: str, session_id: str) -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(duplex="1", model=model, autostart="0", session_id=session_id)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    dtypes = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}
    if width not in dtypes:
        raise ValueError(f"unsupported input sample width: {width}")
    pcm = np.frombuffer(raw, dtype=dtypes[width]).astype(np.float32)
    pcm = (pcm - 128.0) / 128.0 if width == 1 else pcm / float(1 << (width * 8 - 1))
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if source_rate != INPUT_SAMPLE_RATE_HZ:
        count = max(1, round(pcm.size * INPUT_SAMPLE_RATE_HZ / source_rate))
        pcm = np.interp(
            np.linspace(0, pcm.size - 1, count),
            np.arange(pcm.size),
            pcm,
        )
    return np.ascontiguousarray(pcm, dtype="<f4")


async def _stream(
    client: RealtimeDuplexClient,
    pcm: np.ndarray,
    *,
    max_frames: int | None,
    realtime: bool,
) -> int:
    count = math.ceil(pcm.size / FRAME_SAMPLES)
    if max_frames is not None:
        count = min(count, max_frames)
    for seq in range(count):
        frame = pcm[seq * FRAME_SAMPLES : (seq + 1) * FRAME_SAMPLES]
        frame = np.pad(frame, (0, FRAME_SAMPLES - frame.size)).astype("<f4")
        await client.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode("ascii"),
                "format": "pcm_f32le",
                "sample_rate_hz": INPUT_SAMPLE_RATE_HZ,
                "duration_ms": 80,
                "audio_end_ms": (seq + 1) * 80,
            }
        )
        if realtime:
            await asyncio.sleep(FRAME_PERIOD_S)
    return count


def _events(client: RealtimeDuplexClient, event_type: str) -> list[dict[str, object]]:
    return [event for event in client.events.events if event.get("type") == event_type]


async def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    instructions = (
        Path(args.instructions_file).read_text(encoding="utf-8") if args.instructions_file else args.instructions
    )
    tools = json.loads(Path(args.tools_file).read_text(encoding="utf-8")) if args.tools_file else None
    if tools is None and args.expect_function_call:
        tools = DEFAULT_FUNCTION_TOOLS
    session_id = f"nemotron-voicechat-{uuid.uuid4().hex}"
    client = RealtimeDuplexClient(_url(args.url, args.model, session_id))
    started_at = time.monotonic()
    async with client:
        session_payload: dict[str, object] = {
            "session_id": session_id,
            "model": args.model,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm_f32le",
            "output_audio_format": "pcm16",
            "instructions": instructions,
            "turn_detection": None,
            "extra_body": {"auto_response": True},
        }
        if tools is not None:
            session_payload["tools"] = tools
        await client.send(
            {
                "type": "session.update",
                "session": session_payload,
            }
        )
        await wait_for(
            lambda: client.events.count("session.created") > 0 or bool(client.events.errors()),
            timeout_s=args.timeout_s,
            label="session.created",
        )
        if client.events.errors():
            raise AssertionError(f"session setup failed: {client.events.errors()}")
        created = _events(client, "session.created")[-1]
        session = created.get("session")
        capabilities = session.get("capabilities") if isinstance(session, dict) else None
        expected = {
            "implementation_level": "model_native_duplex",
            "chunk_period_ms": 80,
            "supports_core_resumable_request": True,
            "supports_core_kv_lease": False,
            "supports_multi_session": False,
        }
        if not isinstance(capabilities, dict) or any(capabilities.get(key) != value for key, value in expected.items()):
            raise AssertionError(f"unexpected capabilities: {capabilities}")

        frame_count = await _stream(
            client,
            _read_wav(Path(args.input_wav)),
            max_frames=args.max_frames,
            realtime=not args.no_realtime,
        )
        await client.send({"type": "input_audio_buffer.commit", "final": True})
        try:
            await wait_for(
                lambda: (
                    bool(client.events.errors())
                    or (
                        client.events.count("response.function_call_arguments.done") > 0
                        if args.expect_function_call
                        else client.events.count("response.audio.delta") >= args.minimum_audio_chunks
                    )
                ),
                timeout_s=args.timeout_s,
                label="model output",
            )
        except BaseException:
            (output_dir / "events.failed.jsonl").write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in client.events.events),
                encoding="utf-8",
            )
            raise
        await asyncio.sleep(args.drain_s)
        if client.events.errors():
            raise AssertionError(f"Realtime session emitted errors: {client.events.errors()}")

        function_events = [
            event for event in client.events.events if str(event.get("type", "")).startswith("response.function_call")
        ]
        if args.expect_function_call and not any(
            event.get("type") == "response.function_call_arguments.done" for event in function_events
        ):
            raise AssertionError(f"no completed function call: {function_events}")

        audio = client.events.audio_bytes()
        rates = {
            int(event["sample_rate_hz"])
            for event in _events(client, "response.audio.delta")
            if isinstance(event.get("sample_rate_hz"), int)
        }
        if audio and rates != {OUTPUT_SAMPLE_RATE_HZ}:
            raise AssertionError(f"unexpected output sample rates: {rates}")
        if args.minimum_audio_chunks and not audio:
            raise AssertionError("model produced no audio")
        await client.send({"type": "session.close"})
        await wait_for(
            lambda: client.events.count("session.closed") > 0,
            timeout_s=args.timeout_s,
            label="session.closed",
        )

    (output_dir / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in client.events.events),
        encoding="utf-8",
    )
    if audio:
        write_pcm16_wav(output_dir / "output.wav", audio, sample_rate_hz=OUTPUT_SAMPLE_RATE_HZ)
    pcm = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
    result = {
        "ok": True,
        "session_id": session_id,
        "input_frames": frame_count,
        "elapsed_s": round(time.monotonic() - started_at, 3),
        "capabilities": capabilities,
        "event_counts": {
            event_type: client.events.count(event_type)
            for event_type in sorted({str(event.get("type")) for event in client.events.events})
        },
        "audio_bytes": len(audio),
        "audio_rms": float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0,
        "function_events": function_events,
        "output_dir": str(output_dir),
    }

    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8125/v1/realtime")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--output-dir", default="/tmp/nemotron-voicechat-duplex")
    parser.add_argument(
        "--instructions",
        default="You are NVIDIA Voice Chat. Answer briefly. Start by greeting the user.",
    )
    parser.add_argument("--instructions-file")
    parser.add_argument("--tools-file")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--minimum-audio-chunks", type=int, default=1)
    parser.add_argument("--expect-function-call", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--drain-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(asyncio.run(run(parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
