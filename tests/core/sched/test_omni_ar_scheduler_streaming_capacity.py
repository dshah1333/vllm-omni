# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Capacity guard for orchestrator-fed streaming prompts.

A stage the orchestrator feeds directly never runs the connector receive path,
so the connector-side capacity guard cannot protect it. Without the scheduler
guard the streaming prompt grows past the stage context until the engine dies
inside ``InputBatch.add_request``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.distributed.omni_connectors.utils.config import streaming_stage_context_limit

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TALKER_RESERVE = 26


def _make_scheduler(*, max_model_len: int = 128, tts_max_position_embeddings: int | None = None) -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    hf_config = SimpleNamespace(
        tts_config=(
            None if tts_max_position_embeddings is None else {"max_position_embeddings": tts_max_position_embeddings}
        )
    )
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(stage_id=1, max_model_len=max_model_len, hf_config=hf_config)
    )
    sched._new_prompt_len_snapshot = {}
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.chunk_transfer_adapter = None
    sched.skipped_waiting = set()
    return sched


def _make_session(*, num_computed_tokens: int) -> Request:
    session = Request(
        request_id="req-capacity-test",
        prompt_token_ids=[0] * 8,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    session.num_computed_tokens = num_computed_tokens
    return session


def _make_update(*, prompt_len: int, reserve: int | None = TALKER_RESERVE) -> StreamingUpdate:
    update = StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[0] * prompt_len,
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=SamplingParams(max_tokens=16),
    )
    meta: dict[str, object] = {}
    if reserve is not None:
        meta["next_stage_generation_tokens"] = reserve
    update.model_intermediate_buffer = {"meta": meta}
    return update


def _infos(update: StreamingUpdate) -> tuple[object, ...]:
    return (getattr(update, "model_intermediate_buffer", None), getattr(update, "additional_information", None))


def test_rollover_needed_when_extension_would_overrun_context():
    sched = _make_scheduler(max_model_len=128)
    # 127 computed + 3 incoming + 26 reserve = 156 > 128.
    session = _make_session(num_computed_tokens=127)
    update = _make_update(prompt_len=3)

    assert sched._streaming_capacity_rollover_needed(session, update, _infos(update)) is True


def test_rollover_not_needed_when_extension_fits():
    sched = _make_scheduler(max_model_len=128)
    # 84 computed + 6 incoming + 26 reserve = 116 <= 128.
    session = _make_session(num_computed_tokens=84)
    update = _make_update(prompt_len=6)

    assert sched._streaming_capacity_rollover_needed(session, update, _infos(update)) is False


def test_payload_without_generation_reserve_is_not_capacity_managed():
    sched = _make_scheduler(max_model_len=128)
    session = _make_session(num_computed_tokens=1_000)
    update = _make_update(prompt_len=6, reserve=None)

    assert sched._streaming_capacity_rollover_needed(session, update, _infos(update)) is False


def test_boolean_generation_reserve_is_not_capacity_managed():
    sched = _make_scheduler(max_model_len=128)
    session = _make_session(num_computed_tokens=1_000)
    update = _make_update(prompt_len=6, reserve=None)
    update.model_intermediate_buffer = {"meta": {"next_stage_generation_tokens": True}}

    assert sched._streaming_capacity_rollover_needed(session, update, _infos(update)) is False


def test_undeclared_context_limit_disables_the_guard():
    sched = _make_scheduler(max_model_len=0)
    session = _make_session(num_computed_tokens=10_000)
    update = _make_update(prompt_len=6)

    assert sched._streaming_capacity_rollover_needed(session, update, _infos(update)) is False


def test_prompt_too_large_for_the_context_does_not_loop_rollovers():
    sched = _make_scheduler(max_model_len=32)
    session = _make_session(num_computed_tokens=0)
    # A fresh prompt that cannot fit even alone must not request a rollover:
    # replacing it would produce the same over-budget prompt forever.
    update = _make_update(prompt_len=16)

    assert sched._streaming_capacity_rollover_needed(session, update, _infos(update)) is False


def test_context_limit_takes_the_smaller_of_stage_and_tts_limits():
    assert streaming_stage_context_limit(
        SimpleNamespace(max_model_len=8192, hf_config=SimpleNamespace(tts_config={"max_position_embeddings": 4096}))
    ) == 4096
    assert streaming_stage_context_limit(
        SimpleNamespace(max_model_len=2048, hf_config=SimpleNamespace(tts_config={"max_position_embeddings": 4096}))
    ) == 2048
    assert streaming_stage_context_limit(SimpleNamespace(max_model_len=0, hf_config=None)) == 0


def _stub_inherited_update(monkeypatch) -> MagicMock:
    """Replace the inherited extension implementation, wherever it resolves."""
    extended = MagicMock()
    for klass in OmniARScheduler.__mro__[1:]:
        if "_update_request_as_session" in klass.__dict__:
            monkeypatch.setattr(
                klass,
                "_update_request_as_session",
                lambda self, session, update: extended(),
            )
            return extended
    raise AssertionError("no inherited _update_request_as_session to stub")


def test_update_request_as_session_rolls_over_instead_of_extending(monkeypatch):
    sched = _make_scheduler(max_model_len=128)
    session = _make_session(num_computed_tokens=127)
    update = _make_update(prompt_len=3)

    sched._release_replaced_streaming_prompt_cache = MagicMock()
    sched._replace_streaming_session = MagicMock()
    extended = _stub_inherited_update(monkeypatch)

    sched._update_request_as_session(session, update)

    sched._replace_streaming_session.assert_called_once_with(session, update)
    sched._release_replaced_streaming_prompt_cache.assert_called_once_with(session)
    extended.assert_not_called()


def test_update_request_as_session_extends_when_context_allows(monkeypatch):
    sched = _make_scheduler(max_model_len=4096)
    session = _make_session(num_computed_tokens=84)
    update = _make_update(prompt_len=6)

    sched._release_replaced_streaming_prompt_cache = MagicMock()
    sched._replace_streaming_session = MagicMock()
    extended = _stub_inherited_update(monkeypatch)

    sched._update_request_as_session(session, update)

    sched._replace_streaming_session.assert_not_called()
    extended.assert_called_once()
