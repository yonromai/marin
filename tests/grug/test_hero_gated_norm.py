# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax.numpy as jnp
import ml_dtypes
import numpy as np

from experiments.grug.moe_hero_ep.model import GatedNorm


def test_gated_norm_bf16_sigmoid_matches_rounded_fp64() -> None:
    # The BF16 projection rounds these weights to a gate logit of -4.
    down = jnp.zeros((1, 128), dtype=jnp.bfloat16).at[0, 0].set(1)
    up = jnp.zeros((128, 1), dtype=jnp.bfloat16).at[0, 0].set(-5.46875)
    norm = GatedNorm(w_down=down, w_up=up)

    actual = np.asarray(norm(jnp.ones((1, 1), dtype=jnp.bfloat16)).astype(jnp.float32)).item()
    expected = np.asarray(1.0 / (1.0 + math.exp(4.0)), dtype=ml_dtypes.bfloat16).astype(np.float32).item()

    assert actual == expected
