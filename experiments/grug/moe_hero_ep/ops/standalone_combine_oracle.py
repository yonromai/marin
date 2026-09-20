# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit Hero expert-combine kernels on already captured, identical BF16 inputs.

Pass the ``arrays.npz`` from ``native_residual_moe_audit``. The expert GEMM is
not run here, so compiler recomputation of its output cannot confound the
comparison between reductions.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from levanter.grug._moe.sonic import sonic_gather_sum


def _bf16_as_float32(value: np.ndarray) -> np.ndarray:
    if value.dtype.kind != "V" or value.dtype.itemsize != 2:
        raise ValueError(f"Expected raw BF16 bytes, got {value.dtype}")
    return (value.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def audit(arrays_path: Path) -> dict[str, object]:
    if jax.default_backend() != "gpu":
        raise RuntimeError("The Sonic gather audit requires a GPU")

    with np.load(arrays_path, allow_pickle=False) as arrays:
        expert = _bf16_as_float32(arrays["expert_output"])
        weights = _bf16_as_float32(arrays["combine_weights"])

    if expert.ndim != 4 or expert.shape[1] != 1 or weights.shape != expert.shape[:3]:
        raise ValueError(f"Expected [rows, 1, K, H] experts and [rows, 1, K] weights, got {expert.shape}")
    rows, _, topk, hidden = expert.shape
    expert = expert[:, 0]
    weights = weights[:, 0]
    dispatch_output = jnp.asarray(expert.reshape(rows * topk, hidden), dtype=jnp.bfloat16)
    combine_weights = jnp.asarray(weights, dtype=jnp.bfloat16)
    dispatch_positions = jnp.arange(rows * topk, dtype=jnp.int32).reshape(rows, topk)

    @jax.jit
    def combine(values, positions, combine):
        routes = values.reshape(rows, topk, hidden)
        weighted = routes.astype(jnp.float32) * combine[:, :, None].astype(jnp.float32)
        gather = sonic_gather_sum(values, positions, combine)
        fixed_sum = jnp.sum(weighted, axis=1).astype(jnp.bfloat16)
        scatter = (
            jnp.zeros((rows, hidden), dtype=jnp.float32)
            .at[jnp.repeat(jnp.arange(rows), topk)]
            .add(weighted.reshape(rows * topk, hidden))
            .astype(jnp.bfloat16)
        )
        return gather, fixed_sum, scatter

    outputs = combine(dispatch_output, dispatch_positions, combine_weights)
    reference = np.sum(expert.astype(np.float64) * weights[:, :, None].astype(np.float64), axis=1)
    rounded_reference = np.asarray(reference, dtype=ml_dtypes.bfloat16).astype(np.float32)

    methods = {}
    for name, output in zip(("sonic_gather", "jax_fixed_sum", "fp32_scatter"), outputs, strict=True):
        value = np.asarray(output, dtype=np.float32)
        difference = value.astype(np.float64) - reference
        methods[name] = {
            "exact_bf16_oracle_coordinates": np.sum(value == rounded_reference, axis=1).tolist(),
            "rms_vs_fp64": np.sqrt(np.mean(difference * difference, axis=1)).tolist(),
            "max_abs_vs_fp64": np.max(np.abs(difference), axis=1).tolist(),
            "cross_row_different_coordinates": int(np.count_nonzero(value[0] != value[1])) if rows == 2 else None,
        }

    inputs_identical = None
    if rows == 2:
        inputs_identical = bool(np.array_equal(expert[0], expert[1]) and np.array_equal(weights[0], weights[1]))

    return {
        "arrays": str(arrays_path),
        "device": jax.devices()[0].device_kind,
        "rows": rows,
        "topk": topk,
        "hidden": hidden,
        "captured_inputs_identical": inputs_identical,
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", type=Path)
    args = parser.parse_args()
    print("STANDALONE_COMBINE_ORACLE=" + json.dumps(audit(args.arrays), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
