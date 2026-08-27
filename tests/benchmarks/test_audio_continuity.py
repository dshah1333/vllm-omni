# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the audio-streaming continuity helper.

The helper is intentionally vllm-free: it operates on a flat timeline of
``(arrival_time_s, num_bytes)`` chunks plus PCM format params.  These tests
exercise the underrun arithmetic against hand-crafted timelines so future
refactors of the streaming backend cannot silently regress the metric.
"""

from __future__ import annotations

import pytest

from vllm_omni.benchmarks.audio_continuity import (
    ContinuityStats,
    compute_continuity_stats,
    roll_up_continuity_stats,
)

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


# PCM s16le mono at 24 kHz -> 48 000 bytes/sec, the Qwen3-TTS / VoxCPM2 default.
_SR = 24_000
_BPS = _SR * 2  # 48 000


def test_empty_timeline_is_continuous() -> None:
    stats = compute_continuity_stats([], [], sample_rate=_SR)
    assert stats == ContinuityStats(
        max_underrun_s=0.0,
        underrun_event_count=0,
        is_continuous=True,
    )


def test_single_chunk_has_no_underrun() -> None:
    # One large chunk, no later arrivals to check against.
    stats = compute_continuity_stats(
        chunk_arrival_times_s=[0.05],
        chunk_bytes=[_BPS],  # one second of audio in a single chunk
        sample_rate=_SR,
    )
    assert stats.max_underrun_s == 0.0
    assert stats.underrun_event_count == 0
    assert stats.is_continuous is True


def test_chunks_faster_than_realtime_have_no_underrun() -> None:
    # 100ms of audio per chunk, but arrive every 50ms — buffer always builds.
    stats = compute_continuity_stats(
        chunk_arrival_times_s=[0.0, 0.05, 0.10, 0.15],
        chunk_bytes=[_BPS // 10] * 4,
        sample_rate=_SR,
    )
    assert stats.max_underrun_s == 0.0
    assert stats.underrun_event_count == 0
    assert stats.is_continuous is True


def test_single_gap_above_threshold_breaks_continuity() -> None:
    # First chunk carries 100ms of audio at t=0.  Next chunk arrives at t=0.5s,
    # so the player ran dry from t=0.1 to t=0.5 -> 400ms underrun.
    stats = compute_continuity_stats(
        chunk_arrival_times_s=[0.0, 0.5],
        chunk_bytes=[_BPS // 10, _BPS // 10],  # 100ms each
        sample_rate=_SR,
        threshold_s=0.1,
    )
    assert stats.max_underrun_s == pytest.approx(0.4, abs=1e-6)
    assert stats.underrun_event_count == 1
    assert stats.is_continuous is False


def test_underrun_is_max_across_gaps_not_sum() -> None:
    # Two gaps: a 200ms shortfall and a 50ms shortfall.  max_underrun_s
    # must surface the 200ms one, not the sum.
    stats = compute_continuity_stats(
        chunk_arrival_times_s=[0.0, 0.3, 0.45],
        chunk_bytes=[_BPS // 10, _BPS // 10, _BPS // 10],  # 100ms chunks
        sample_rate=_SR,
        threshold_s=0.1,
    )
    # At t=0.3 the player consumed 300ms but only 100ms received -> 200ms deficit
    # At t=0.45 player consumed 450ms, received 200ms -> 250ms deficit.
    assert stats.max_underrun_s == pytest.approx(0.25, abs=1e-6)
    assert stats.underrun_event_count == 2
    assert stats.is_continuous is False


def test_continuity_threshold_boundary_inclusive() -> None:
    # First chunk = 100ms audio at t=0.  Next arrives at t=0.2 -> 100ms deficit
    # exactly. With threshold=0.1, continuity_ok stays True (boundary).
    stats = compute_continuity_stats(
        chunk_arrival_times_s=[0.0, 0.2],
        chunk_bytes=[_BPS // 10, _BPS // 10],
        sample_rate=_SR,
        threshold_s=0.1,
    )
    assert stats.max_underrun_s == pytest.approx(0.1, abs=1e-6)
    assert stats.is_continuous is True


def test_mismatched_lengths_returns_zero_stats() -> None:
    # Defensive: never crash on malformed timeline.
    stats = compute_continuity_stats(
        chunk_arrival_times_s=[0.0, 0.1],
        chunk_bytes=[100],
        sample_rate=_SR,
    )
    assert stats == ContinuityStats(0.0, 0, True)


def test_threshold_is_configurable() -> None:
    # 200ms underrun: violates 100ms threshold, OK against 500ms threshold.
    arrivals = [0.0, 0.3]
    bytes_list = [_BPS // 10, _BPS // 10]
    strict = compute_continuity_stats(arrivals, bytes_list, sample_rate=_SR, threshold_s=0.1)
    lenient = compute_continuity_stats(arrivals, bytes_list, sample_rate=_SR, threshold_s=0.5)
    assert strict.is_continuous is False
    assert lenient.is_continuous is True
    assert strict.max_underrun_s == lenient.max_underrun_s


def test_roll_up_is_empty_when_no_response_carried_audio() -> None:
    assert roll_up_continuity_stats([]) is None


def test_roll_up_takes_the_worst_response_not_the_average() -> None:
    clean = ContinuityStats(max_underrun_s=0.0, underrun_event_count=0, is_continuous=True)
    stalled = ContinuityStats(max_underrun_s=3.0, underrun_event_count=2, is_continuous=False)

    rolled = roll_up_continuity_stats([clean] * 9 + [stalled])

    # A mean would report 0.3 s and pass the 100 ms budget while the listener
    # heard a 3 s hole.
    assert rolled == ContinuityStats(
        max_underrun_s=3.0,
        underrun_event_count=2,
        is_continuous=False,
    )


def test_roll_up_sums_events_across_responses() -> None:
    first = ContinuityStats(max_underrun_s=0.2, underrun_event_count=3, is_continuous=False)
    second = ContinuityStats(max_underrun_s=0.5, underrun_event_count=4, is_continuous=False)

    rolled = roll_up_continuity_stats([first, second])

    assert rolled is not None
    assert rolled.max_underrun_s == pytest.approx(0.5)
    assert rolled.underrun_event_count == 7
