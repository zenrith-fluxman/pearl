"""
Tests that the Pearl linear op refuses to mine inside a CUDA graph capture.

pearl_gemm_noisy does per-call host work (current mining job, key/target, pinned signal buffer,
ticket submission) that a CUDA graph replay would skip, so a mining-sized call during capture must
raise instead of silently mining with stale state. The quantize and GEMM kernels are mocked, so
these tests run on CPU.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm_miner import vllm_kernels
from vllm_miner.config import config as pearl_config


def _fake_quant(x, smooth_scale=None, block_size=None):
    return x.to(torch.int8), torch.ones((x.shape[0], 1)), None


def _call_apply_linear(m, n, k):
    x = torch.zeros((m, k), dtype=torch.bfloat16)
    w_q = torch.zeros((n, k), dtype=torch.int8)
    w_s = torch.ones((n, 1))
    return vllm_kernels._pearl_apply_linear(x, w_q, w_s, None, 0, 0, True)


@pytest.fixture
def mocked_kernels():
    noisy = MagicMock(name="pearl_gemm_noisy")
    vanilla = MagicMock(name="pearl_gemm_vanilla")
    settings = MagicMock(no_mining=False)
    async_manager = MagicMock()
    async_manager._conf.skip_block_submission = True

    with (
        patch.object(vllm_kernels, "quant_7bit", side_effect=_fake_quant),
        patch.object(vllm_kernels, "pearl_gemm_noisy", noisy),
        patch.object(vllm_kernels, "pearl_gemm_vanilla", vanilla),
        patch.object(vllm_kernels, "get_async_manager", return_value=async_manager),
        patch.object(pearl_config, "settings", settings),
    ):
        yield noisy, vanilla


@pytest.mark.parametrize("m, n, k", [(1024, 4096, 4096), (2048, 6144, 4096)])
def test_mining_call_during_graph_capture_raises(m, n, k, mocked_kernels):
    noisy, vanilla = mocked_kernels
    assert pearl_config.should_use_noisy_gemm(m, n, k)

    with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
        with pytest.raises(RuntimeError, match="captured CUDA graph"):
            _call_apply_linear(m, n, k)

    noisy.assert_not_called()
    vanilla.assert_not_called()


@pytest.mark.parametrize("m, n, k", [(1024, 4096, 4096)])
def test_mining_call_outside_graph_capture_mines(m, n, k, mocked_kernels):
    noisy, _ = mocked_kernels

    with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
        _call_apply_linear(m, n, k)

    noisy.assert_called_once()


@pytest.mark.parametrize("m, n, k", [(512, 4096, 4096)])
def test_small_call_during_graph_capture_uses_vanilla(m, n, k, mocked_kernels):
    noisy, vanilla = mocked_kernels

    with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
        _call_apply_linear(m, n, k)

    noisy.assert_not_called()
    vanilla.assert_called_once()
