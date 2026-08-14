# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden shape test for the NemotronVoiceChat frame-locked timeline contract.

Pins the exact prefill/off-by-one arithmetic against values verified with the
NeMo reference run on sample_general.wav (fp32, greedy): perception produced
196 acoustic frames for the 249,734-sample 16 kHz fixture (the NeMo dump's
250,760 was the loader's padded batch width; perception consumed the true
length); the system prompt tokenized to 56 ids ([bos] + 54 + [eos]); NeMo's
timeline T = 252 = 56 + 196; and our fp32 pipeline matched the reference text
channel 196/196 tokens.
"""

import pytest

from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    compute_acoustic_frame_count,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_streaming_mel_slice_matches_nemo_cache_geometry() -> None:
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
        slice_perception_streaming_mel,
    )

    cfg = SimpleNamespace(
        chunk_size=[1, 8],
        shift_size=[1, 8],
        pre_encode_cache_size=[0, 9],
        drop_extra_pre_encoded=2,
    )
    first, first_drop = slice_perception_streaming_mel(
        torch.arange(9).reshape(1, 1, 9),
        0,
        cfg,
    )
    assert first.flatten().tolist() == [0]
    assert first_drop == 0

    second, second_drop = slice_perception_streaming_mel(
        torch.arange(17).reshape(1, 1, 17),
        1,
        cfg,
    )
    assert second.shape[-1] == 17
    assert second.flatten().tolist() == [0] * 9 + list(range(1, 9))
    assert second_drop == 2

    third, _ = slice_perception_streaming_mel(
        torch.arange(25).reshape(1, 1, 25),
        2,
        cfg,
    )
    assert third.flatten().tolist() == list(range(17))

# Perception config subset matching the checkpoint (mel 10 ms hop, n_fft 512,
# dw-striding 8x CAUSAL subsampling, conv kernel 3). causal_downsampling is
# what the real checkpoint sets; omitting it here is exactly what let the
# non-causal padding formula pass review round 0 (helper vs itself).
_STT_CFG = {
    "perception": {
        "preprocessor": {"sample_rate": 16000, "window_stride": 0.01, "window_size": 0.025, "n_fft": 512},
        "encoder": {
            "subsampling": "dw_striding",
            "subsampling_factor": 8,
            "subsampling_conv_kernel_size": 3,
            "causal_downsampling": True,
        },
    }
}

# (num_16k_samples, expected 12.5 Hz frames), all verified against the REAL
# vendored AudioPerceptionModule built from the checkpoint config. The first
# row is the sample_general.wav acceptance fixture (196 frames -> the NeMo
# T=252 timeline). 71271 is the reviewer's 4.45 s repro that the non-causal
# formula undersized by one; 250760 is the NeMo loader's padded width, which
# genuinely yields 197 (not the fixture's 196).
_GOLDEN = [
    (249734, 196),
    (250760, 197),
    (71271, 57),
    (16000, 14),
    (16000 * 2, 26),
    (16000 * 4, 51),
    (16000 * 8, 101),
]


@pytest.mark.parametrize(("num_samples", "expected_frames"), _GOLDEN)
def test_acoustic_frame_count_golden(num_samples: int, expected_frames: int) -> None:
    assert compute_acoustic_frame_count(_STT_CFG, num_samples) == expected_frames


def test_frame_count_matches_vendored_subsampling_geometry() -> None:
    """Pin the helper against the real ConvSubsampling, not against itself.

    Builds the vendored subsampling module with the checkpoint geometry (both
    causal and non-causal) and checks the helper's padding arithmetic against
    the module's actual ``_left_padding + _right_padding`` and its
    ``calc_length`` output over a spread of mel lengths.
    """
    import torch

    from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.asr.subsampling import (
        ConvSubsampling,
        calc_length,
    )

    for causal in (True, False):
        sub = ConvSubsampling(
            subsampling="dw_striding",
            subsampling_factor=8,
            feat_in=128,
            feat_out=8,
            conv_channels=8,
            is_causal=causal,
        )
        expected_paddings = sub._left_padding + sub._right_padding
        cfg = {
            "perception": {
                "preprocessor": _STT_CFG["perception"]["preprocessor"],
                "encoder": {
                    "subsampling": "dw_striding",
                    "subsampling_factor": 8,
                    "subsampling_conv_kernel_size": 3,
                    "causal_downsampling": causal,
                },
            }
        }
        for num_samples in (16000, 71271, 249734, 250760):
            # Recompute the helper's mel length, then ask the module's own
            # calc_length what the subsampled length should be.
            mel_len = (num_samples + 512 - 512) // 160 + 1
            module_frames = int(
                calc_length(
                    torch.tensor([mel_len]),
                    all_paddings=expected_paddings,
                    kernel_size=3,
                    stride=2,
                    ceil_mode=False,
                    repeat_num=3,
                )[0]
            )
            assert compute_acoustic_frame_count(cfg, num_samples) == module_frames, (
                f"helper diverged from vendored geometry (causal={causal}, samples={num_samples})"
            )


def test_timeline_max_model_len_validation() -> None:
    from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
        validate_timeline_fits_max_model_len,
    )

    # The acceptance fixture (56 + 1 + 196 = 253) fits the shipped 8192 limit.
    validate_timeline_fits_max_model_len(56, 196, 8192)
    # Exactly at the limit: the vLLM request occupies prompt + 1 (frame-0
    # placeholder) + frames tokens, so 57 + 8135 == 8192 is allowed...
    validate_timeline_fits_max_model_len(56, 8135, 8192)
    # ...and one more frame must be rejected, naming every offending length.
    with pytest.raises(ValueError) as excinfo:
        validate_timeline_fits_max_model_len(56, 8136, 8192)
    message = str(excinfo.value)
    for fragment in (
        "vllm_prefill_len=57",
        "acoustic_frame_count=8136",
        "vllm_total_len=8193",
        "max_model_len=8192",
        "logical_prompt_token_len=56",
        "logical_total_len=8192",
    ):
        assert fragment in message


def _bare_thinker():
    import torch
    from torch import nn

    from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
        NemotronVoiceChatThinkerForConditionalGeneration,
    )

    thinker = NemotronVoiceChatThinkerForConditionalGeneration.__new__(NemotronVoiceChatThinkerForConditionalGeneration)
    nn.Module.__init__(thinker)
    thinker._use_function_head = True
    thinker._dtype = torch.float32
    thinker.function_head = nn.Linear(8, 16, bias=False)
    return thinker


def test_postprocess_appends_function_token_to_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    thinker = _bare_thinker()
    hidden = torch.randn(3, 8)
    expected = int(thinker.function_head(hidden[-1:]).argmax(dim=-1))

    # Default: only the load-bearing prev-token; the cumulative debug timeline
    # (a per-step host sync + growing concat) stays off.
    update = thinker.postprocess(hidden)
    assert int(update["nvc_prev_function_token"]) == expected
    assert "nvc_function_tokens" not in update

    monkeypatch.setenv("NEMOTRON_VOICECHAT_DEBUG_FUNCTION_TIMELINE", "1")
    # No existing timeline: starts one.
    update = thinker.postprocess(hidden)
    assert update["nvc_function_tokens"].tolist() == [expected]
    # Existing timeline: appended, not replaced.
    existing = torch.tensor([7, 8, 9], dtype=torch.long)
    update = thinker.postprocess(hidden, nvc_function_tokens=existing)
    assert update["nvc_function_tokens"].tolist() == [7, 8, 9, expected]


def test_postprocess_skips_intermediate_prefill_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setenv("NEMOTRON_VOICECHAT_DEBUG_FUNCTION_TIMELINE", "1")
    thinker = _bare_thinker()
    hidden = torch.randn(4, 8)
    # Intermediate chunk: 0 computed + 4 rows < prompt_len 10 -> no token.
    assert thinker.postprocess(hidden, _omni_is_prefill=True, _omni_num_computed_tokens=0, _omni_prompt_len=10) == {}
    # Final chunk reaches the last prefill position (acoustic frame 0).
    update = thinker.postprocess(hidden, _omni_is_prefill=True, _omni_num_computed_tokens=6, _omni_prompt_len=10)
    assert "nvc_prev_function_token" in update
    assert update["nvc_function_tokens"].numel() == 1


def test_duplex_sampler_rejects_mixed_batches_explicitly() -> None:
    import torch

    thinker = _bare_thinker()
    thinker._duplex_sampling_rows = {0: "req-duplex"}
    thinker._duplex_previous_text_tokens = {}

    with pytest.raises(RuntimeError, match="mixed duplex/non-duplex batches are unsupported"):
        thinker.sample(torch.zeros((2, 16)), None)

    assert thinker._duplex_previous_text_tokens == {}


def test_prefill_contract_arithmetic() -> None:
    # vllm_prefill_len = logical_prompt_token_len + 1 (the +1 is acoustic
    # frame 0); decode steps == acoustic_frame_count; NeMo timeline
    # T == prompt + frames. Values from the verified sample_general.wav run.
    logical_prompt_len = 56
    frames = compute_acoustic_frame_count(_STT_CFG, 249734)
    vllm_prefill_len = logical_prompt_len + 1
    assert vllm_prefill_len == 57
    assert logical_prompt_len + frames == 252  # NeMo reference T
    # The talker consumes the full timeline from t=1 and the producer trims
    # the prompt region: rows P-1 onward of the (T-1)-row code stack.
    talker_rows = (logical_prompt_len + frames) - 1
    trimmed = talker_rows - (logical_prompt_len - 1)
    assert trimmed == frames


@pytest.mark.parametrize(
    ("num_iter", "exponent", "num_quantizers"),
    [
        (1, 3.0, 31),
        (4, 1.0, 8),
        (8, 2.0, 31),
        (8, 3.0, 31),  # shipped VoiceChat generation settings
        (16, 4.0, 64),
    ],
)
def test_cached_unmask_counts_match_original_gpu_schedule(
    num_iter: int,
    exponent: float,
    num_quantizers: int,
) -> None:
    import torch
    from torch.nn import functional as F

    from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.ear_tts_model import (
        get_masking_rate,
        get_unmask_counts,
    )

    # This is the original per-frame schedule construction, kept here as the
    # executable contract for the cached host-side implementation.
    rates = torch.linspace(0.0, 1.0, num_iter + 1)[:-1].unsqueeze(-1)
    num_maskings = torch.ceil(get_masking_rate(rates, exponent) * num_quantizers).long()
    expected = num_maskings - F.pad(num_maskings[1:], [0, 0, 0, 1])

    actual = get_unmask_counts(num_iter, exponent, num_quantizers)
    assert actual == tuple(expected.flatten().tolist())
    assert sum(actual) == num_quantizers
    assert get_unmask_counts(num_iter, exponent, num_quantizers) is actual


def test_incremental_depthsum_is_bitwise_equal_to_full_recompute() -> None:
    import torch
    from torch.nn import functional as F

    from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.ear_tts_model import (
        get_unmask_counts,
        update_depthsum_embedding,
    )

    generator = torch.Generator().manual_seed(6089)
    num_quantizers, codebook_size, latent_size = 31, 32, 16
    rvq_embs = torch.randn(
        num_quantizers,
        codebook_size,
        latent_size,
        generator=generator,
    )
    code = torch.full((2, 1, num_quantizers), codebook_size, dtype=torch.long)
    incremental = torch.zeros((2, 1, latent_size), dtype=torch.float32)
    padded = F.pad(rvq_embs, (0, 0, 0, 1))

    cnt = 0
    for k in get_unmask_counts(8, 3.0, num_quantizers):
        if k == 0:
            continue
        reference = torch.zeros_like(incremental)
        for quantizer_idx in range(num_quantizers):
            reference = reference + F.embedding(
                code[..., quantizer_idx],
                padded[quantizer_idx],
            )
        assert torch.equal(incremental, reference)

        code[..., cnt : cnt + k] = torch.randint(
            codebook_size,
            (2, 1, k),
            generator=generator,
        )
        if cnt + k < num_quantizers:
            incremental = update_depthsum_embedding(
                incremental,
                code,
                rvq_embs,
                cnt,
                k,
            )
        cnt += k

    assert cnt == num_quantizers


def test_precomputed_rvq_norm_preserves_residual_codes() -> None:
    import torch

    from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.ear_tts_model import (
        depthsum_encoding_step,
    )

    generator = torch.Generator().manual_seed(16080)
    embs = torch.randn(5, 16, 8, generator=generator)
    residual = torch.randn(2, 1, 8, generator=generator)
    code = torch.full((2, 1, 5), 16, dtype=torch.long)
    expected_norm = embs.pow(2).sum(-1)

    actual = depthsum_encoding_step(
        embs,
        expected_norm,
        residual.clone(),
        code.clone(),
        0,
        5,
    )

    reference = code.clone()
    running = residual.clone()
    for quantizer_idx in range(5):
        selected = (
            embs[quantizer_idx].pow(2).sum(-1)
            - 2
            * (
                running.unsqueeze(-2)
                @ embs[quantizer_idx].transpose(-1, -2)
            ).squeeze(-2)
        ).argmin(-1)
        running = running - torch.nn.functional.embedding(
            selected,
            embs[quantizer_idx],
        )
        reference[..., quantizer_idx] = selected

    assert torch.equal(actual, reference)


def test_precomputed_rvq_norm_is_not_a_checkpoint_weight() -> None:
    import torch

    from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.ear_tts_model import (
        RVQEARTTSModel,
    )

    module = torch.nn.Module()
    rvq_embs = torch.randn(3, 8, 4)
    RVQEARTTSModel.set_rvq_embs(module, rvq_embs)

    assert "rvq_embs" in module.state_dict()
    assert "rvq_embs_squared_norm" not in module.state_dict()
    assert torch.equal(module.rvq_embs_squared_norm, rvq_embs.pow(2).sum(-1))


def test_torchaudio_mel_filterbank_matches_librosa() -> None:
    """The perception frontend builds its mel filterbank with torchaudio.

    torchaudio's melscale_fbanks(norm="slaney", mel_scale="slaney") implements
    the same slaney-convention math as librosa.filters.mel (the NeMo reference
    dependency) up to float32 rounding. The 1e-6 tolerance is ~4x the observed
    2.6e-7 max diff; e2e this stays token- and WAV-identical to the reference
    on the parity fixture (per-feature normalization absorbs it).
    """
    librosa = pytest.importorskip("librosa")
    torchaudio = pytest.importorskip("torchaudio")

    # The checkpoint's perception preprocessor config (16 kHz, n_fft=512, 128 mels).
    ta = torchaudio.functional.melscale_fbanks(
        n_freqs=257, f_min=0.0, f_max=8000.0, n_mels=128, sample_rate=16000, norm="slaney", mel_scale="slaney"
    ).T.numpy()
    reference = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0, norm="slaney")
    assert ta.shape == reference.shape
    assert abs(ta - reference).max() < 1e-6
