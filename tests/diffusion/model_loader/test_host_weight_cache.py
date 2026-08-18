# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import multiprocessing as mp
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.model_loader.host_weight_cache import (
    build_host_weight_cache_plan,
    default_host_weight_cache_root,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

_SPAWN_TIMEOUT_SECONDS = 120


class _LoaderMarker:
    pass


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2)
        self.register_buffer("persistent", torch.arange(2, dtype=torch.float32))
        self.register_buffer("temporary", torch.ones(1), persistent=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
            self.proj.bias.copy_(torch.tensor([7.0, 8.0]))


class _Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _Transformer()


def test_default_root_is_not_split_by_replica_vllm_cache(monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/tmp/replica-specific-vllm-cache")

    assert default_host_weight_cache_root().endswith("/.cache/vllm-omni/dlo-host-weights")


def _build(pipeline: nn.Module, cache_root: Path, **overrides):
    kwargs = {
        "dit_modules": (("transformer", pipeline.transformer),),
        "loader_type": _LoaderMarker,
        "cache_root": cache_root,
        "lock_timeout_seconds": 2.0,
        "max_shard_bytes": 32,
        "model_identity": "test/model",
        "revision": "revision",
        "runtime_dtype": torch.float32,
        "load_format": "default",
        "loader_inputs": {"model_config": {"hidden_size": 3}},
        "tensor_parallel_size": 1,
        "tensor_parallel_rank": 0,
        "sequence_parallel_guard": {"sequence_parallel_size": 1, "backend": "none"},
        "use_hsdp": False,
        "enable_expert_parallel": False,
        "quantization_config": None,
        "cfg_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    kwargs.update(overrides)
    return build_host_weight_cache_plan(pipeline, **kwargs)


def _multiprocess_builder(cache_root: str, queue) -> None:
    result = _build(_Pipeline(), Path(cache_root))
    queue.put(
        (
            result.fallback_code,
            result.plan.runtime_layout_key if result.plan is not None else None,
            sorted(binding.file_path for binding in result.plan.bindings.values()) if result.plan is not None else [],
        )
    )


def test_equivalent_process_layouts_reuse_one_entry(tmp_path):
    first = _build(_Pipeline(), tmp_path)
    second = _build(_Pipeline(), tmp_path)

    assert first.plan is not None
    assert second.plan is not None
    assert first.plan.runtime_layout_key == second.plan.runtime_layout_key
    assert first.plan.post_load_complete
    assert {binding.file_path for binding in first.plan.bindings.values()} == {
        binding.file_path for binding in second.plan.bindings.values()
    }
    assert set(first.plan.bindings) == {
        "transformer.persistent",
        "transformer.proj.bias",
        "transformer.proj.weight",
    }


def test_equivalent_local_model_paths_reuse_one_entry(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_link = tmp_path / "model-link"
    model_link.symlink_to(model_dir, target_is_directory=True)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    identities = (str(model_dir), "~/model", "model", str(model_link))
    results = [_build(_Pipeline(), cache_root, model_identity=identity) for identity in identities]

    assert all(result.plan is not None for result in results)
    assert len({result.plan.runtime_layout_key for result in results if result.plan is not None}) == 1
    assert len(list((cache_root / "v1").iterdir())) == 1


def test_unstable_loader_identity_fails_closed(tmp_path):
    result = _build(_Pipeline(), tmp_path, loader_inputs={"opaque": object()})

    assert result.plan is None
    assert result.fallback_code == "unstable_identity"
    assert result.fallback_reason == "host weight cache identity does not support values of type builtins.object"
    assert list(tmp_path.rglob("*.safetensors")) == []


def test_tp_coordinate_and_sp_implementation_guard_split_entries(tmp_path):
    pipeline = _Pipeline()
    tp0 = _build(pipeline, tmp_path, tensor_parallel_size=2, tensor_parallel_rank=0)
    tp1 = _build(pipeline, tmp_path, tensor_parallel_size=2, tensor_parallel_rank=1)
    sp2 = _build(
        pipeline,
        tmp_path,
        sequence_parallel_guard={"sequence_parallel_size": 2, "backend": "ulysses"},
    )

    assert tp0.plan is not None and tp1.plan is not None and sp2.plan is not None
    assert len({tp0.plan.runtime_layout_key, tp1.plan.runtime_layout_key, sp2.plan.runtime_layout_key}) == 3


def test_final_runtime_content_is_authoritative(tmp_path):
    first_pipeline = _Pipeline()
    second_pipeline = _Pipeline()
    with torch.no_grad():
        second_pipeline.transformer.proj.weight[0, 0] += 1

    first = _build(first_pipeline, tmp_path)
    second = _build(second_pipeline, tmp_path)

    assert first.plan is not None and second.plan is not None
    assert first.plan.runtime_layout_key != second.plan.runtime_layout_key


def test_expert_parallel_ownership_fails_closed(tmp_path):
    result = _build(_Pipeline(), tmp_path, enable_expert_parallel=True)

    assert result.plan is None
    assert result.fallback_code == "unsupported_expert_parallel"
    assert list(tmp_path.rglob("*.safetensors")) == []


def test_pipeline_parallel_ownership_fails_closed(tmp_path):
    result = _build(_Pipeline(), tmp_path, pipeline_parallel_size=2)

    assert result.plan is None
    assert result.fallback_code == "unsupported_pp"
    assert list(tmp_path.rglob("*.safetensors")) == []


def test_corrupt_published_entry_is_rebuilt_under_the_key_lock(tmp_path):
    first = _build(_Pipeline(), tmp_path)
    assert first.plan is not None
    shard = Path(next(iter(first.plan.bindings.values())).file_path)
    old_inode = shard.stat().st_ino
    with shard.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 0xFF]))

    second = _build(_Pipeline(), tmp_path)

    assert second.plan is not None
    assert second.plan.runtime_layout_key == first.plan.runtime_layout_key
    assert Path(next(iter(second.plan.bindings.values())).file_path).stat().st_ino != old_inode


def test_lock_timeout_keeps_the_ordinary_tensors(tmp_path):
    initial = _build(_Pipeline(), tmp_path)
    assert initial.plan is not None
    cache_key = initial.plan.runtime_layout_key
    entry_dir = Path(next(iter(initial.plan.bindings.values())).file_path).parent
    shutil.rmtree(entry_dir)
    lock_path = tmp_path / ".locks" / f"{cache_key}.lock"

    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _build(_Pipeline(), tmp_path, lock_timeout_seconds=0.01)

    assert result.plan is None
    assert result.fallback_code == "lock_timeout"


def test_next_writer_removes_stale_temp_after_process_death(tmp_path):
    initial = _build(_Pipeline(), tmp_path)
    assert initial.plan is not None
    cache_key = initial.plan.runtime_layout_key
    entry_dir = Path(next(iter(initial.plan.bindings.values())).file_path).parent
    shutil.rmtree(entry_dir)
    stale_temp = tmp_path / ".tmp" / f"{cache_key}.stale-writer"
    stale_temp.mkdir(parents=True)
    (stale_temp / "partial").write_bytes(b"partial")

    result = _build(_Pipeline(), tmp_path)

    assert result.plan is not None
    assert not stale_temp.exists()


@pytest.mark.parametrize("case", ["alias", "external_alias", "noncontiguous"])
def test_unsupported_tensor_layouts_fail_closed(tmp_path, case):
    pipeline = _Pipeline()
    if case == "alias":
        pipeline.transformer.tied = pipeline.transformer.proj.weight
        expected_code = "unsupported_alias"
    elif case == "external_alias":
        pipeline.encoder = nn.Module()
        pipeline.encoder.shared = pipeline.transformer.proj.weight
        expected_code = "unsupported_alias"
    else:
        pipeline.transformer.proj.weight = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(3, 2).t())
        expected_code = "unsupported_tensor"

    result = _build(pipeline, tmp_path)

    assert result.plan is None
    assert result.fallback_code == expected_code
    assert list(tmp_path.rglob("*.safetensors")) == []


def test_concurrent_processes_publish_one_immutable_entry(tmp_path):
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_multiprocess_builder, args=(str(tmp_path), queue)) for _ in range(2)]
    for process in processes:
        process.start()
    # Spawn workers import the full vLLM stack before entering the target;
    # loaded CI hosts can spend more than 30 seconds in that startup path.
    results = [queue.get(timeout=_SPAWN_TIMEOUT_SECONDS) for _ in processes]
    for process in processes:
        process.join(timeout=_SPAWN_TIMEOUT_SECONDS)

    assert all(process.exitcode == 0 for process in processes)
    assert all(code is None for code, _, _ in results)
    assert len({key for _, key, _ in results}) == 1
    assert len({tuple(paths) for _, _, paths in results}) == 1
    assert len(list((tmp_path / "v1").iterdir())) == 1
