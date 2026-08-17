# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.offloader.cuda_host_registration as registration_module
from vllm_omni.diffusion.offloader.cuda_host_registration import (
    CudaHostRegistration,
    CudaHostRegistrationBudgetError,
    CudaHostRegistrationCleanupError,
    CudaHostRegistrationError,
    _coalesce_ranges,
)
from vllm_omni.diffusion.offloader.host_registration import (
    HostRegistrationError,
    register_host_mappings,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _FakeStorage:
    def __init__(self, pointer: int, size: int) -> None:
        self._pointer = pointer
        self._size = size

    def data_ptr(self) -> int:
        return self._pointer

    def nbytes(self) -> int:
        return self._size


class _FakeTensor:
    device = SimpleNamespace(type="cpu")

    def __init__(self, pointer: int, size: int, *, pinned: bool = True) -> None:
        self._storage = _FakeStorage(pointer, size)
        self._pinned = pinned

    def untyped_storage(self) -> _FakeStorage:
        return self._storage

    def numel(self) -> int:
        return self._storage.nbytes()

    def is_pinned(self) -> bool:
        return self._pinned


class _FakeRuntime:
    def __init__(
        self,
        register_results: list[int | Exception],
        unregister_results: list[int | Exception] | None = None,
    ) -> None:
        self._register_results = iter(register_results)
        self._unregister_results = iter(unregister_results or [])
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []

    def cudaHostRegister(self, pointer: int, size: int, flags: int) -> int:
        self.registered.append((pointer, size, flags))
        result = next(self._register_results)
        if isinstance(result, Exception):
            raise result
        return result

    def cudaHostUnregister(self, pointer: int) -> int:
        self.unregistered.append(pointer)
        result = next(self._unregister_results, 0)
        if isinstance(result, Exception):
            raise result
        return result

    @staticmethod
    def cudaGetErrorString(error: int) -> str:
        return f"error-{error}"


def test_platform_factory_reports_unsupported_registration() -> None:
    with pytest.raises(HostRegistrationError, match="not supported on cpu"):
        register_host_mappings({}, device=torch.device("cpu"), max_bytes=4096)


def test_platform_factory_dispatches_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    sources = {"weights": []}

    def create(actual_sources, *, max_bytes):
        assert actual_sources is sources
        assert max_bytes == 4096
        return sentinel

    monkeypatch.setattr(CudaHostRegistration, "create", staticmethod(create))

    assert register_host_mappings(sources, device=torch.device("cuda"), max_bytes=4096) is sentinel


def test_coalesce_ranges_aligns_and_merges_overlapping_pages() -> None:
    assert _coalesce_ranges(
        [(0x1003, 4096), (0x2800, 1024), (0x9001, 1)],
        page_size=4096,
    ) == (
        registration_module._AddressRange(0x1000, 0x3000),
        registration_module._AddressRange(0x9000, 0xA000),
    )


def test_registration_rejects_over_budget_before_calling_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registration_module.torch.cuda, "is_available", lambda: True)
    runtime = _FakeRuntime([0])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)

    with pytest.raises(CudaHostRegistrationBudgetError, match="exceeding"):
        CudaHostRegistration.create(
            {"weights": [_FakeTensor(0x1003, 4096)]},  # type: ignore[list-item]
            max_bytes=4096,
        )

    assert runtime.registered == []


def test_registration_rolls_back_partial_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registration_module.torch.cuda, "is_available", lambda: True)
    runtime = _FakeRuntime([0, 7])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)
    consumed_errors: list[int] = []
    monkeypatch.setattr(registration_module, "_consume_last_cuda_error", consumed_errors.append)

    with pytest.raises(CudaHostRegistrationError, match="error-7"):
        CudaHostRegistration.create(
            {
                "first": [_FakeTensor(0x1003, 1)],  # type: ignore[list-item]
                "second": [_FakeTensor(0x9003, 1)],  # type: ignore[list-item]
            },
            max_bytes=8192,
        )

    assert runtime.unregistered == [0x1000]
    assert consumed_errors == [7]


def test_registration_wraps_runtime_exception_and_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registration_module.torch.cuda, "is_available", lambda: True)
    runtime = _FakeRuntime([0, RuntimeError("driver rejected mapping")])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)

    with pytest.raises(CudaHostRegistrationError, match="driver rejected mapping"):
        CudaHostRegistration.create(
            {
                "first": [_FakeTensor(0x1003, 1)],  # type: ignore[list-item]
                "second": [_FakeTensor(0x9003, 1)],  # type: ignore[list-item]
            },
            max_bytes=8192,
        )

    assert runtime.unregistered == [0x1000]


def test_registration_fails_closed_when_rollback_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registration_module.torch.cuda, "is_available", lambda: True)
    runtime = _FakeRuntime([0, 7], unregister_results=[9])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)
    consumed_errors: list[int] = []
    monkeypatch.setattr(registration_module, "_consume_last_cuda_error", consumed_errors.append)

    with pytest.raises(CudaHostRegistrationCleanupError, match="rollback errors"):
        CudaHostRegistration.create(
            {
                "first": [_FakeTensor(0x1003, 1)],  # type: ignore[list-item]
                "second": [_FakeTensor(0x9003, 1)],  # type: ignore[list-item]
            },
            max_bytes=8192,
        )

    assert consumed_errors == [7, 9]


@pytest.mark.parametrize("pending_error", [0, 801])
def test_consume_last_cuda_error_accepts_cleared_or_matching_state(
    monkeypatch: pytest.MonkeyPatch,
    pending_error: int,
) -> None:
    class _GetLastError:
        def __call__(self) -> int:
            return pending_error

    runtime = SimpleNamespace(cudaGetLastError=_GetLastError())
    monkeypatch.setattr(registration_module.ctypes, "CDLL", lambda _name: runtime)

    registration_module._consume_last_cuda_error(801)


def test_consume_last_cuda_error_rejects_unrelated_pending_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _GetLastError:
        def __call__(self) -> int:
            return 700

    runtime = SimpleNamespace(cudaGetLastError=_GetLastError())
    monkeypatch.setattr(registration_module.ctypes, "CDLL", lambda _name: runtime)

    with pytest.raises(CudaHostRegistrationCleanupError, match="cudaGetLastError reported 700"):
        registration_module._consume_last_cuda_error(801)


def test_registration_closes_all_coalesced_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registration_module.torch.cuda, "is_available", lambda: True)
    runtime = _FakeRuntime([0, 0])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)
    registration = CudaHostRegistration.create(
        {
            "first": [_FakeTensor(0x1003, 4096), _FakeTensor(0x2800, 1024)],  # type: ignore[list-item]
            "second": [_FakeTensor(0x9003, 1)],  # type: ignore[list-item]
        },
        max_bytes=12288,
    )

    assert registration.total_bytes == 12288
    assert registration.region_count == 2
    assert registration.close() == []
    assert runtime.unregistered == [0x9000, 0x1000]
    assert registration.close() == []


def test_registration_keeps_adjacent_file_mappings_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registration_module.torch.cuda, "is_available", lambda: True)
    runtime = _FakeRuntime([0, 0])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)

    registration = CudaHostRegistration.create(
        {
            "first.safetensors": [_FakeTensor(0x1000, 4096)],  # type: ignore[list-item]
            "second.safetensors": [_FakeTensor(0x2000, 4096)],  # type: ignore[list-item]
        },
        max_bytes=8192,
    )

    assert registration.region_count == 2
    assert runtime.registered == [
        (0x1000, 4096, registration_module._CUDA_HOST_REGISTER_READ_ONLY),
        (0x2000, 4096, registration_module._CUDA_HOST_REGISTER_READ_ONLY),
    ]
    assert registration.close() == []


def test_registration_close_retries_failed_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registration_module.torch.cuda, "is_available", lambda: True)
    runtime = _FakeRuntime([0], unregister_results=[9, 0])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)
    consumed_errors: list[int] = []
    monkeypatch.setattr(registration_module, "_consume_last_cuda_error", consumed_errors.append)
    registration = CudaHostRegistration.create(
        {"weights": [_FakeTensor(0x1003, 1)]},  # type: ignore[list-item]
        max_bytes=4096,
    )

    assert registration.close() == ["cudaHostUnregister(0x1000) failed: error-9"]
    assert registration.close() == []
    assert runtime.unregistered == [0x1000, 0x1000]
    assert consumed_errors == [9]
