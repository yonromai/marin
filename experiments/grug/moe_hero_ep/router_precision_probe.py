# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Temporary saved-input probe for Hero router contraction arithmetic."""

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3

LAYER = 15
BUCKET = "marin-us-east-02a"
EXPORT_KEY = "marin/users/romain/hero-vllm-b200/" "hero-535b-step108000-bf16-split-v3/model-layer-015.safetensors"
SAVED_ARRAYS = Path(
    "/home/romain/data/sessions/devbox/codex/"
    "01a0b566-0b11-7e80-88a7-249b292ea7a0/artifacts/"
    "hero-fp32-qualification/scratch/native-router-input-v2-arrays.npz"
)


def _read_router_weight() -> np.ndarray:
    configure_coreweave_s3()
    with fsspec.open(f"s3://{BUCKET}/{EXPORT_KEY}", "rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        entry = header[f"model.layers.{LAYER}.mlp.router.weight"]
        if entry["dtype"] != "BF16" or entry["shape"] != [384, 6144]:
            raise ValueError(entry)
        start, stop = entry["data_offsets"]
        handle.seek(8 + header_size + start)
        raw = handle.read(stop - start)
    bits = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
    return bits.view("<f4").reshape(384, 6144).T.copy()


def prepare_fixture(path: Path, *, all_rows: bool = False) -> None:
    row_slice = slice(None) if all_rows else slice(6, 8)
    with np.load(SAVED_ARRAYS, allow_pickle=False) as saved:
        positions = saved["layer_probe_positions"]
        inputs = saved["layer_probe_router_input"][LAYER, row_slice].reshape(-1, 6144).copy()
        bias = saved["effective_router_bias"][LAYER].copy()
        routes = saved["route_expert_ids"][LAYER, row_slice][:, positions].copy()
        gaps = saved["route_cutoff_gaps"][LAYER, row_slice][:, positions].copy()
        valid = positions[None, :] < saved["valid_lengths"][row_slice, None]
    if not np.array_equal(inputs, inputs.astype(ml_dtypes.bfloat16).astype(np.float32)):
        raise ValueError("Captured router inputs are not exact BF16 values")
    weight = _read_router_weight()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, inputs=inputs, weight=weight, bias=bias, routes=routes, gaps=gaps, positions=positions, valid=valid
    )
    print(
        json.dumps({"fixture": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(inputs)})
    )


def _scores(x: jax.Array, weight: jax.Array, variant: str) -> jax.Array:
    x_bf16, weight_bf16 = x.astype(jnp.bfloat16), weight.astype(jnp.bfloat16)
    if variant == "current":
        return jnp.einsum("td,de->te", x_bf16, weight_bf16).astype(jnp.float32)
    if variant == "preferred_fp32":
        return jnp.einsum("td,de->te", x_bf16, weight_bf16, preferred_element_type=jnp.float32)
    if variant == "fp32_operands":
        return jnp.einsum(
            "td,de->te", x_bf16.astype(jnp.float32), weight_bf16.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
        )
    raise ValueError(variant)


def _error(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = actual.astype(np.float64) - reference
    return {"max_abs": float(np.max(np.abs(delta))), "rms": float(np.sqrt(np.mean(delta**2)))}


def _router_graph(x: jax.Array, weight: jax.Array, bias: jax.Array, variant: str):
    scores = _scores(x, weight, variant)
    biased_scores = scores + bias
    probabilities = jax.nn.softmax(scores, axis=-1)
    top_scores, top_indices = jax.lax.top_k(biased_scores, 9)
    selected = top_indices[:, :8]
    selected_scores = jnp.take_along_axis(scores, selected, axis=-1)
    weights = jax.nn.sigmoid(selected_scores)
    weights = weights * (2.5 / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-9))
    margins = scores - top_scores[:, -1:]
    return scores, selected, probabilities, weights, margins


def measure(path: Path, hlo_dir: Path) -> None:
    with np.load(path, allow_pickle=False) as fixture:
        x = fixture["inputs"].copy()
        weight = fixture["weight"].copy()
        bias = fixture["bias"].copy()
        saved_routes = fixture["routes"].copy().reshape(-1, 8)
        saved_gaps = fixture["gaps"].copy().reshape(-1)
        valid = fixture["valid"].copy().reshape(-1)
    reference = x.astype(np.float64) @ weight.astype(np.float64)
    cotangent = np.linspace(-0.5, 0.5, reference.size, dtype=np.float32).reshape(reference.shape)
    reference_x_grad = cotangent.astype(np.float64) @ weight.astype(np.float64).T
    reference_weight_grad = x.astype(np.float64).T @ cotangent.astype(np.float64)
    hlo_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference": "NumPy float64 dot on identical BF16 input and weight values",
        "valid_rows": int(np.sum(valid)),
        "variants": {},
    }
    all_scores = {}
    graph_scores = {}
    for variant in ("current", "preferred_fp32", "fp32_operands"):

        def forward(inputs, weights, _variant=variant):
            return _scores(inputs, weights, _variant)

        def loss(inputs, weights):
            return jnp.sum(forward(inputs, weights) * jnp.asarray(cotangent))

        forward_jit = jax.jit(forward)
        backward_jit = jax.jit(jax.grad(loss, argnums=(0, 1)))
        for name, function in (("forward", forward_jit), ("backward", backward_jit)):
            lowered = function.lower(jnp.asarray(x), jnp.asarray(weight))
            (hlo_dir / f"{variant}-{name}-stablehlo.txt").write_text(str(lowered.compiler_ir(dialect="stablehlo")))
            compiled = lowered.compile()
            (hlo_dir / f"{variant}-{name}-optimized-hlo.txt").write_text(compiled.as_text())
        scores = np.asarray(forward_jit(x, weight))
        all_scores[variant] = scores
        x_grad, weight_grad = backward_jit(x, weight)
        x_grad, weight_grad = np.asarray(x_grad), np.asarray(weight_grad)
        order = np.argsort(-(scores + bias), axis=1, kind="stable")
        predicted = order[:, :8]
        gaps = (scores + bias)[np.arange(len(order)), order[:, 7]] - (scores + bias)[np.arange(len(order)), order[:, 8]]
        report["variants"][variant] = {
            "output_dtype": str(scores.dtype),
            "score_error": _error(scores, reference),
            "x_gradient_error": _error(x_grad, reference_x_grad),
            "weight_gradient_error": _error(weight_grad, reference_weight_grad),
            "saved_route_set_matches": int(
                sum(set(a) == set(b) for a, b in zip(predicted[valid], saved_routes[valid], strict=True))
            ),
            "saved_route_order_matches": int(np.sum(np.all(predicted[valid] == saved_routes[valid], axis=1))),
            "saved_cutoff_gap_error": _error(gaps[valid], saved_gaps[valid]),
            "reference_route_set_matches": int(
                sum(
                    set(a) == set(b)
                    for a, b in zip(predicted[valid], np.argsort(-(reference + bias), axis=1)[valid, :8], strict=True)
                )
            ),
        }
        # Block on the device and exclude compilation from the arithmetic microbenchmark.
        for _ in range(3):
            forward_jit(x, weight).block_until_ready()
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            forward_jit(x, weight).block_until_ready()
            samples.append(time.perf_counter() - start)
        report["variants"][variant]["isolated_forward_median_seconds"] = float(np.median(samples))

        def graph(inputs, weights, _variant=variant):
            return _router_graph(inputs, weights, jnp.asarray(bias), _variant)

        def graph_loss(inputs, weights):
            graph_scores_value, _, probabilities, combine, margins = graph(inputs, weights)
            return (
                jnp.sum(graph_scores_value * jnp.asarray(cotangent))
                + jnp.sum(probabilities * 0.01)
                + jnp.sum(combine * 0.01)
                + jnp.sum(margins * 0.001)
            )

        graph_jit = jax.jit(graph)
        graph_backward_jit = jax.jit(jax.grad(graph_loss, argnums=(0, 1)))
        for name, function in (("router_graph_forward", graph_jit), ("router_graph_backward", graph_backward_jit)):
            lowered = function.lower(jnp.asarray(x), jnp.asarray(weight))
            (hlo_dir / f"{variant}-{name}-stablehlo.txt").write_text(str(lowered.compiler_ir(dialect="stablehlo")))
            compiled = lowered.compile()
            (hlo_dir / f"{variant}-{name}-optimized-hlo.txt").write_text(compiled.as_text())
        graph_score, graph_routes, _, _, _ = graph_jit(x, weight)
        graph_scores[variant] = np.asarray(graph_score)
        graph_routes = np.asarray(graph_routes)
        graph_x_grad, graph_weight_grad = graph_backward_jit(x, weight)
        report["variants"][variant]["router_graph"] = {
            "score_error": _error(graph_scores[variant], reference),
            "saved_route_order_matches": int(np.sum(np.all(graph_routes[valid] == saved_routes[valid], axis=1))),
            "x_grad_max_abs": float(np.max(np.abs(np.asarray(graph_x_grad)))),
            "weight_grad_max_abs": float(np.max(np.abs(np.asarray(graph_weight_grad)))),
        }
    report["score_pair_differences"] = {
        "current_vs_preferred_fp32": _error(all_scores["current"], all_scores["preferred_fp32"].astype(np.float64)),
        "current_vs_fp32_operands": _error(all_scores["current"], all_scores["fp32_operands"].astype(np.float64)),
    }
    report["router_graph_score_pair_differences"] = {
        "current_vs_preferred_fp32": _error(graph_scores["current"], graph_scores["preferred_fp32"].astype(np.float64)),
        "current_vs_fp32_operands": _error(graph_scores["current"], graph_scores["fp32_operands"].astype(np.float64)),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output_dir := os.environ.get("IRIS_OUTPUT_DIR"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "report.json").write_text(rendered + "\n")
    print(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "measure"))
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--hlo-dir", type=Path, default=Path("router-probe-hlo"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare_fixture(args.fixture, all_rows=args.all_rows)
    else:
        hlo_dir = Path(os.environ["IRIS_OUTPUT_DIR"]) / "hlo" if os.environ.get("IRIS_OUTPUT_DIR") else args.hlo_dir
        measure(args.fixture, hlo_dir)


if __name__ == "__main__":
    main()
