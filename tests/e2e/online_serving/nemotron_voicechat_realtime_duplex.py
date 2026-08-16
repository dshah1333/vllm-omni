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
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from scipy.signal import resample_poly

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
DEFAULT_INSTRUCTIONS = "You are NVIDIA Voice Chat. Answer briefly. Start by greeting the user."
DEFAULT_FUNCTION_INSTRUCTIONS = (
    "You are NVIDIA Voice Chat. If the user's request matches an available tool, "
    "you MUST call that tool instead of answering from your own knowledge. "
    "Use only argument values spoken by the user and never invent missing values."
)


def _url(base_url: str, model: str, session_id: str) -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(duplex="1", model=model, autostart="0", session_id=session_id)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _read_wav(path: Path, *, input_channel: int = 0) -> np.ndarray:
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
    if not 0 <= input_channel < channels:
        raise ValueError(f"input channel {input_channel} is outside WAV channel count {channels}")
    if channels > 1:
        # NVIDIA's bundled VoiceChat samples are combined conversations: user
        # audio is channel 0 and reference agent audio is channel 1. Averaging
        # them leaks the expected agent response back into the model input and
        # makes both turn-taking and function-call probes invalid.
        pcm = pcm.reshape(-1, channels)[:, input_channel]
    if source_rate != INPUT_SAMPLE_RATE_HZ:
        divisor = math.gcd(source_rate, INPUT_SAMPLE_RATE_HZ)
        pcm = resample_poly(
            pcm,
            up=INPUT_SAMPLE_RATE_HZ // divisor,
            down=source_rate // divisor,
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


def _audio_packet_durations_ms(events: Sequence[dict[str, object]]) -> list[float]:
    """Measure each PCM16 delta independently, without trusting metadata."""
    durations = []
    for event in events:
        delta = event.get("delta")
        sample_rate_hz = event.get("sample_rate_hz")
        if not isinstance(delta, str) or not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
            raise AssertionError(f"malformed response.audio.delta: {event}")
        try:
            payload = base64.b64decode(delta, validate=True)
        except ValueError as exc:
            raise AssertionError("response.audio.delta is not valid base64") from exc
        if len(payload) % 2:
            raise AssertionError(f"PCM16 audio packet has odd byte length: {len(payload)}")
        durations.append(len(payload) * 1000.0 / (2 * sample_rate_hz))
    return durations


def _write_events(path: Path, client: RealtimeDuplexClient) -> None:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in client.events.events),
        encoding="utf-8",
    )


@asynccontextmanager
async def _close_and_capture_failures(
    client: RealtimeDuplexClient,
    *,
    output_dir: Path,
    timeout_s: float,
):
    """Return the single-session lease even when probe validation fails."""
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        close_error: Exception | None = None
        if client.events.count("session.created") and not client.events.count("session.closed"):
            try:
                await client.close_session(timeout_s=min(timeout_s, 30.0))
            except Exception as exc:  # preserve the original probe failure
                close_error = exc
        if failed or close_error is not None:
            _write_events(output_dir / "events.failed.jsonl", client)
        if close_error is not None and not failed:
            raise close_error


async def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.instructions_file:
        instructions = Path(args.instructions_file).read_text(encoding="utf-8")
    elif args.expect_function_call and args.instructions == DEFAULT_INSTRUCTIONS:
        instructions = DEFAULT_FUNCTION_INSTRUCTIONS
    else:
        instructions = args.instructions
    tools = json.loads(Path(args.tools_file).read_text(encoding="utf-8")) if args.tools_file else None
    if tools is None and args.expect_function_call:
        tools = DEFAULT_FUNCTION_TOOLS
    session_id = f"nemotron-voicechat-{uuid.uuid4().hex}"
    client = RealtimeDuplexClient(_url(args.url, args.model, session_id))
    started_at = time.monotonic()
    async with (
        client,
        _close_and_capture_failures(
            client,
            output_dir=output_dir,
            timeout_s=args.timeout_s,
        ),
    ):
        session_payload: dict[str, object] = {
            "session_id": session_id,
            "model": args.model,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm_f32le",
            "output_audio_format": "pcm16",
            "instructions": instructions,
            # Keep the Realtime reader alive for at least as long as the probe
            # is willing to wait for model output. The default 300 s session
            # timeout is shorter than this eager fp32 pipeline needs for the
            # bundled turn-taking fixture, so it otherwise cancels an active
            # response before the probe's own timeout can decide the result.
            "idle_timeout_s": args.timeout_s,
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

        stream_started_at_s = time.monotonic()
        frame_count = await _stream(
            client,
            _read_wav(Path(args.input_wav), input_channel=args.input_channel),
            max_frames=args.max_frames,
            realtime=not args.no_realtime,
        )
        input_committed_at_s = time.monotonic()
        completed_responses_at_commit = client.events.count("response.done")
        await client.send({"type": "input_audio_buffer.commit", "final": True})
        await wait_for(
            lambda: (
                bool(client.events.errors())
                or (
                    client.events.count("response.function_call_arguments.done") > 0
                    if args.expect_function_call
                    else (
                        client.events.count("response.audio.delta") >= args.minimum_audio_chunks
                        and (
                            args.allow_incomplete_response
                            or client.events.count("response.done") > completed_responses_at_commit
                        )
                    )
                )
            ),
            timeout_s=args.timeout_s,
            label="model output",
        )
        await asyncio.sleep(args.drain_s)
        if client.events.errors():
            raise AssertionError(f"Realtime session emitted errors: {client.events.errors()}")
        done_events = _events(client, "response.done")
        if not args.expect_function_call and not args.allow_incomplete_response:
            response = done_events[-1].get("response") if done_events else None
            status = response.get("status") if isinstance(response, dict) else None
            if status != "completed":
                raise AssertionError(f"response did not complete successfully: {done_events[-1:]}")

        function_events = [
            event for event in client.events.events if str(event.get("type", "")).startswith("response.function_call")
        ]
        function_items = [
            event
            for event in _events(client, "response.output_item.done")
            if isinstance(event.get("item"), dict) and event["item"].get("type") == "function_call"
        ]
        if args.expect_function_call and not any(
            event.get("type") == "response.function_call_arguments.done" for event in function_events
        ):
            raise AssertionError(f"no completed function call: {function_events}")
        if args.expect_function_call:
            matching_items = [
                event["item"] for event in function_items if event["item"].get("name") == args.expected_function_name
            ]
            if not matching_items:
                raise AssertionError(f"expected function {args.expected_function_name!r}, got {function_items}")
            try:
                function_arguments = json.loads(str(matching_items[-1].get("arguments", "")))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"function arguments are not JSON: {matching_items[-1]}") from exc
            if not isinstance(function_arguments, dict):
                raise AssertionError(f"function arguments are not an object: {function_arguments!r}")

        audio = client.events.audio_bytes()
        audio_events = _events(client, "response.audio.delta")
        rates = {int(event["sample_rate_hz"]) for event in audio_events if isinstance(event.get("sample_rate_hz"), int)}
        if not args.expect_function_call and audio and rates != {OUTPUT_SAMPLE_RATE_HZ}:
            raise AssertionError(f"unexpected output sample rates: {rates}")
        if not args.expect_function_call and args.minimum_audio_chunks and not audio:
            raise AssertionError("model produced no audio")
        packet_durations_ms = _audio_packet_durations_ms(audio_events)
        expected_packet_ms = float(expected["chunk_period_ms"])
        if not args.expect_function_call and any(
            not math.isclose(duration_ms, expected_packet_ms, abs_tol=0.01) for duration_ms in packet_durations_ms
        ):
            raise AssertionError(
                f"audio deltas are not fixed {expected_packet_ms:g} ms codec increments: {packet_durations_ms}"
            )
        audio_pcm = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
        audio_rms = float(np.sqrt(np.mean(np.square(audio_pcm)))) if audio_pcm.size else 0.0
        if not args.expect_function_call and audio and audio_rms < args.minimum_audio_rms:
            raise AssertionError(
                f"model output RMS {audio_rms:.6f} is below {args.minimum_audio_rms:.6f}; "
                "received packets contain only silence"
            )
        response_id = next(
            (candidate for candidate in client.events.response_ids if client.events.audio_bytes(candidate)),
            None,
        )
        timing = client.events.timing_summary(
            after_s=started_at,
            # Native duplex can emit before commit, so the first input frame
            # (rather than the terminal commit) is the meaningful latency
            # origin for TTFT/TTFP/RTF.
            input_committed_at_s=stream_started_at_s,
            response_id=response_id,
            measurement_origin={
                "ttft": "first input frame send to first non-empty text delta",
                "ttfp": "first input frame send to first audio packet",
                "rtf": "stream-start-to-last-audio receive time divided by emitted audio duration",
            },
        )
        audio_timing = timing.get("audio_output")
        if isinstance(audio_timing, dict):
            audio_timing["stream_start_to_first_audio_ms"] = audio_timing.pop(
                "commit_to_first_audio_ms",
                None,
            )

    _write_events(output_dir / "events.jsonl", client)
    if audio:
        write_pcm16_wav(output_dir / "output.wav", audio, sample_rate_hz=OUTPUT_SAMPLE_RATE_HZ)
    result = {
        "ok": True,
        "session_id": session_id,
        "input_frames": frame_count,
        "input_channel": args.input_channel,
        "input_stream_s": round(input_committed_at_s - stream_started_at_s, 3),
        "elapsed_s": round(time.monotonic() - started_at, 3),
        "capabilities": capabilities,
        "event_counts": {
            event_type: client.events.count(event_type)
            for event_type in sorted({str(event.get("type")) for event in client.events.events})
        },
        "audio_bytes": len(audio),
        "audio_rms": audio_rms,
        "audio_packet_durations_ms": packet_durations_ms,
        "timing": timing,
        "function_events": function_events,
        "function_items": function_items,
        "output_dir": str(output_dir),
    }

    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8125/v1/realtime")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-wav", required=True)
    parser.add_argument(
        "--input-channel",
        type=int,
        default=0,
        help="Zero-based WAV channel to stream (bundled VoiceChat samples use user audio on channel 0).",
    )
    parser.add_argument("--output-dir", default="/tmp/nemotron-voicechat-duplex")
    parser.add_argument(
        "--instructions",
        default=DEFAULT_INSTRUCTIONS,
    )
    parser.add_argument("--instructions-file")
    parser.add_argument("--tools-file")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--minimum-audio-chunks", type=int, default=1)
    parser.add_argument("--minimum-audio-rms", type=float, default=1e-4)
    parser.add_argument("--expect-function-call", action="store_true")
    parser.add_argument("--expected-function-name", default="generate_random_number")
    parser.add_argument(
        "--allow-incomplete-response",
        action="store_true",
        help="Treat initial audio as success without waiting for response.done.",
    )
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--drain-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(asyncio.run(run(parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
