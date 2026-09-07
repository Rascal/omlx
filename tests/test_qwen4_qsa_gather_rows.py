# SPDX-License-Identifier: Apache-2.0
"""The QSA row gather must read only the selected rows of the stored (B, H, N, D) cache."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _reference(kv, indices):
    """The previous token-major path: transpose, gather per batch (the old helper, inlined), transpose back."""
    rows = kv.transpose(0, 2, 1, 3)
    batch, tokens = rows.shape[:2]
    trailing = rows.shape[2:]
    offsets = mx.arange(batch, dtype=mx.int32).reshape((batch,) + (1,) * (indices.ndim - 1)) * tokens
    flat = (indices.astype(mx.int32) + offsets).reshape(-1)
    gathered = rows.reshape(batch * tokens, *trailing)[flat].reshape(*indices.shape, *trailing)
    return mx.contiguous(gathered.transpose(0, 2, 1, 3))


@pytest.mark.parametrize("batch", [1, 2])
def test_gather_kv_rows_matches_token_major_gather(batch):
    from mlx_vlm.models.qwen4_exp import qsa_fast

    mx.random.seed(3)
    kv = mx.random.normal((batch, 2, 300, 16)).astype(mx.bfloat16)
    indices = mx.sort(mx.random.randint(0, 300, (batch, 37)).astype(mx.int32), axis=-1)
    out = qsa_fast._gather_kv_rows(kv, indices)
    mx.eval(out)
    assert out.shape == (batch, 2, 37, 16)
    assert mx.array_equal(out.view(mx.uint16), _reference(kv, indices).view(mx.uint16)).item()


def test_gather_kv_rows_does_not_copy_the_whole_cache():
    """Cost must not scale with the cache length: gathering 64 rows from 8x the tokens takes ~the same time."""
    import time

    from mlx_vlm.models.qwen4_exp import qsa_fast

    def cost(tokens):
        kv = mx.random.normal((1, 2, tokens, 256)).astype(mx.bfloat16)
        idx = mx.sort(mx.random.randint(0, tokens, (1, 64)).astype(mx.int32), axis=-1)
        mx.eval(kv, idx)
        mx.eval(qsa_fast._gather_kv_rows(kv, idx))
        t0 = time.perf_counter()
        for _ in range(20):
            mx.eval(qsa_fast._gather_kv_rows(kv, idx))
        return (time.perf_counter() - t0) / 20

    small, large = cost(16_384), cost(131_072)
    assert large < 2.0 * small, (small, large)
