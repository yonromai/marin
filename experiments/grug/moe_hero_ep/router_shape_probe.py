# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Temporary B200 shape probe for the Hero router dot and its backward."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from experiments.grug.moe_hero_ep.router_precision_probe import _scores


def _benchmark(fn, x, weight, *, warmup=3, repeats=5):
    for _ in range(warmup):
        jax.block_until_ready(fn(x, weight))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(x, weight))
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)), [float(sample) for sample in samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--dump-hlo", action="store_true")
    args = parser.parse_args()
    if args.rows < 1:
        raise ValueError("rows must be positive")
    with np.load(args.fixture, allow_pickle=False) as fixture:
        valid = fixture["valid"].reshape(-1)
        saved = fixture["inputs"][valid].copy()
        saved_routes = fixture["routes"].reshape(-1, 8)[valid].copy()
        weight = fixture["weight"].copy()
        bias = fixture["bias"].copy()
    x = np.tile(saved, (int(np.ceil(args.rows / len(saved))), 1))[: args.rows]
    row_indices = np.arange(args.rows) % len(saved)
    reference_scores = saved.astype(np.float64) @ weight.astype(np.float64)
    reference_routes = np.argsort(-(reference_scores + bias), axis=1, kind="stable")[:, :8]
    expected_routes = saved_routes[row_indices]
    cotangent = np.sin(np.arange(args.rows * weight.shape[1], dtype=np.float32).reshape(args.rows, -1) * 0.01)
    cotangent = jnp.asarray(cotangent)
    output_dir = Path(os.environ.get("IRIS_OUTPUT_DIR", "router-shape-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_saved = np.unique(saved.view(np.uint16).reshape(len(saved), -1), axis=0)
    report = {
        "backend": jax.default_backend(),
        "command": [sys.executable, *sys.argv],
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "rows": args.rows,
        "saved_valid_entries": len(saved),
        "unique_saved_activation_vectors": len(unique_saved),
        "variants": {},
    }
    outputs = {}
    routes = {}
    grads = {}
    bias_device = jnp.asarray(bias)
    top_routes = jax.jit(lambda scores: jax.lax.top_k(scores + bias_device, 9)[1][:, :8])
    for variant in ("current", "preferred_fp32", "fp32_operands"):
        forward = jax.jit(lambda inputs, weights, _variant=variant: _scores(inputs, weights, _variant))

        def loss(inputs, weights, _variant=variant):
            return jnp.sum(_scores(inputs, weights, _variant) * cotangent)

        backward = jax.jit(jax.grad(loss, argnums=(0, 1)))
        if args.dump_hlo:
            for name, function in (("forward", forward), ("backward", backward)):
                lowered = function.lower(jnp.asarray(x), jnp.asarray(weight))
                (output_dir / f"{variant}-{name}-stablehlo.txt").write_text(
                    str(lowered.compiler_ir(dialect="stablehlo"))
                )
                (output_dir / f"{variant}-{name}-optimized-hlo.txt").write_text(lowered.compile().as_text())
        scores_device = forward(x, weight)
        outputs[variant] = np.asarray(scores_device)
        routes[variant] = np.asarray(top_routes(scores_device))
        grads[variant] = tuple(np.asarray(g) for g in backward(x, weight))
        forward_median, forward_samples = _benchmark(forward, x, weight)
        backward_median, backward_samples = _benchmark(backward, x, weight)
        report["variants"][variant] = {
            "forward_median_seconds": forward_median,
            "forward_samples_seconds": forward_samples,
            "backward_median_seconds": backward_median,
            "backward_samples_seconds": backward_samples,
            "output_dtype": str(outputs[variant].dtype),
            "bf16_exact_score_fraction": float(
                np.mean(outputs[variant] == outputs[variant].astype(ml_dtypes.bfloat16).astype(np.float32))
            ),
            "fp64_score_max_abs": float(np.max(np.abs(outputs[variant] - reference_scores[row_indices]))),
            "fp64_score_rms": float(np.sqrt(np.mean((outputs[variant] - reference_scores[row_indices]) ** 2))),
            "saved_route_order_matches": int(np.sum(np.all(routes[variant] == expected_routes, axis=1))),
            "saved_route_set_matches": int(
                np.sum(np.all(np.sort(routes[variant], axis=1) == np.sort(expected_routes, axis=1), axis=1))
            ),
            "reference_route_order_matches": int(
                np.sum(np.all(routes[variant] == reference_routes[row_indices], axis=1))
            ),
        }
    for candidate in ("preferred_fp32", "fp32_operands"):
        score_delta = outputs[candidate].astype(np.float64) - outputs["current"].astype(np.float64)
        x_delta = grads[candidate][0].astype(np.float64) - grads["current"][0].astype(np.float64)
        weight_delta = grads[candidate][1].astype(np.float64) - grads["current"][1].astype(np.float64)
        report[f"{candidate}_vs_current"] = {
            "score_max_abs": float(np.max(np.abs(score_delta))),
            "score_rms": float(np.sqrt(np.mean(score_delta**2))),
            "x_grad_max_abs": float(np.max(np.abs(x_delta))),
            "x_grad_rms": float(np.sqrt(np.mean(x_delta**2))),
            "weight_grad_max_abs": float(np.max(np.abs(weight_delta))),
            "weight_grad_rms": float(np.sqrt(np.mean(weight_delta**2))),
            "route_order_changes": int(np.sum(np.any(routes[candidate] != routes["current"], axis=1))),
            "route_set_changes": int(
                np.sum(np.any(np.sort(routes[candidate], axis=1) != np.sort(routes["current"], axis=1), axis=1))
            ),
        }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    (output_dir / "report.json").write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
