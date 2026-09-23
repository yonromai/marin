# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired production-shape B200 probe for Hero router arithmetic."""

import argparse
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3

VARIANTS = ("current", "preferred_fp32", "rounded_preferred", "preferred_current_vjp")


@jax.custom_vjp
def _preferred_current_vjp(inputs: jax.Array, weights: jax.Array) -> jax.Array:
    # Keep preferred forward values while substituting the current backward graph.
    return scores(inputs, weights, "preferred_fp32")


def _preferred_current_vjp_forward(inputs: jax.Array, weights: jax.Array):
    return scores(inputs, weights, "preferred_fp32"), (inputs, weights)


def _preferred_current_vjp_backward(residual, cotangent):
    inputs, weights = residual
    _, pullback = jax.vjp(lambda x, w: scores(x, w, "current"), inputs, weights)
    return pullback(cotangent)


_preferred_current_vjp.defvjp(_preferred_current_vjp_forward, _preferred_current_vjp_backward)


def scores(inputs: jax.Array, weights: jax.Array, variant: str) -> jax.Array:
    if variant == "preferred_current_vjp":
        return _preferred_current_vjp(inputs, weights)
    inputs, weights = inputs.astype(jnp.bfloat16), weights.astype(jnp.bfloat16)
    if variant == "current":
        return jnp.einsum("td,de->te", inputs, weights).astype(jnp.float32)
    result = jnp.einsum("td,de->te", inputs, weights, preferred_element_type=jnp.float32)
    if variant == "rounded_preferred":
        # Keep the BF16 intermediate observable to the optimizer.
        return jax.lax.optimization_barrier(result.astype(jnp.bfloat16)).astype(jnp.float32)
    if variant == "preferred_fp32":
        return result
    raise ValueError(variant)


def routing(inputs: jax.Array, weights: jax.Array, bias: jax.Array, variant: str):
    logits = scores(inputs, weights, variant)
    _, indices = jax.lax.top_k(logits + bias, 9)
    indices = indices[:, :8]
    selected = jnp.take_along_axis(logits, indices, axis=-1)
    gates = jax.nn.sigmoid(selected)
    gates = (gates * (2.5 / (jnp.sum(gates, axis=-1, keepdims=True) + 1e-9))).astype(inputs.dtype)
    return logits, indices, gates


def _ready(value):
    return jax.block_until_ready(value)


def _read_fixture(uri: str):
    if uri.startswith("s3://"):
        configure_coreweave_s3()
        with fsspec.open(uri, "rb") as handle:
            data = handle.read()
    else:
        data = Path(uri).read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as fixture:
        valid = fixture["valid"].reshape(-1)
        values = fixture["inputs"][valid].copy()
        weight = fixture["weight"].copy()
        bias = fixture["bias"].copy()
    return hashlib.sha256(data).hexdigest(), values, weight, bias


def _time_pair(functions, arguments, *, warmup: int, repeats: int, seed: int):
    for _ in range(warmup):
        for fn in functions.values():
            _ready(fn(*arguments))
    order = list(functions)
    rng = random.Random(seed)
    samples = {name: [] for name in order}
    for trial in range(repeats):
        rng.shuffle(order)
        for name in order:
            start = time.perf_counter()
            _ready(functions[name](*arguments))
            samples[name].append({"trial": trial, "order": order.index(name), "seconds": time.perf_counter() - start})
    return samples


def _compile(functions, arguments, output: Path, phase: str):
    compiled = {}
    for name, fn in functions.items():
        executable = fn.lower(*arguments).compile()
        (output / f"{phase}-{name}-optimized-hlo.txt").write_text(executable.as_text())
        compiled[name] = executable
    return compiled


def _compare_routing(inputs, weights, bias):
    values = {}
    for variant in VARIANTS:
        values[variant] = tuple(
            np.asarray(x) for x in jax.jit(routing, static_argnames="variant")(inputs, weights, bias, variant)
        )
    baseline = values["current"]
    preferred = values["preferred_fp32"]
    hybrid = values["preferred_current_vjp"]
    if any(not np.array_equal(a, b) for a, b in zip(preferred, hybrid, strict=True)):
        raise ValueError("Hybrid forward routing differs from preferred FP32 routing")
    result = {}
    for variant, (logits, indices, gates) in values.items():
        result[variant] = {
            "score_dtype": str(logits.dtype),
            "score_exact_fraction_vs_current": float(np.mean(logits == baseline[0])),
            "score_max_abs_vs_current": float(np.max(np.abs(logits - baseline[0]))),
            "route_order_changed_rows": int(np.sum(np.any(indices != baseline[1], axis=1))),
            "route_set_changed_rows": int(
                np.sum(np.any(np.sort(indices, axis=1) != np.sort(baseline[1], axis=1), axis=1))
            ),
            "gate_exact_fraction_vs_current": float(np.mean(gates == baseline[2])),
            "gate_max_abs_vs_current": float(np.max(np.abs(gates.astype(np.float32) - baseline[2].astype(np.float32)))),
            "route_shape": list(indices.shape),
            "gate_shape": list(gates.shape),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--master-weight", action="store_true")
    parser.add_argument("--output-prefix")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.rows, args.warmup, args.repeats) < 1:
        raise ValueError("rows, warmup and repeats must be positive")
    fixture_hash, saved, weight, bias = _read_fixture(args.fixture)
    if saved.shape[1] != 6144 or weight.shape != (6144, 384):
        raise ValueError(f"Unexpected production router dimensions: {saved.shape}, {weight.shape}")
    if not np.array_equal(saved, saved.astype(jnp.bfloat16).astype(np.float32)):
        raise ValueError("Saved activations are not BF16 values")
    indices = np.arange(args.rows) % len(saved)
    inputs = jnp.asarray(saved[indices], dtype=jnp.bfloat16)
    weights = jnp.asarray(weight, dtype=jnp.float32 if args.master_weight else jnp.bfloat16)
    router_bias = jnp.asarray(bias, dtype=jnp.float32)
    cotangent = jnp.asarray(np.sin(np.arange(args.rows * 384, dtype=np.float32).reshape(args.rows, 384) * 0.01))
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "router-causal-output"))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "command": [sys.executable, *sys.argv],
        "fixture_sha256": fixture_hash,
        "rows": args.rows,
        "saved_valid_entries": len(saved),
        "input_dtype": str(inputs.dtype),
        "weight_dtype": str(weights.dtype),
        "master_weight": args.master_weight,
        "input_shape": list(inputs.shape),
        "weight_shape": list(weights.shape),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "backend_version": jax.devices()[0].client.platform_version,
        "python_version": platform.python_version(),
        "source_commit": args.source_commit,
        "routing": _compare_routing(inputs, weights, router_bias),
    }
    forward = {variant: jax.jit(lambda x, w, v=variant: scores(x, w, v)) for variant in VARIANTS}
    backward = {
        variant: jax.jit(jax.grad(lambda x, w, dy, v=variant: jnp.sum(scores(x, w, v) * dy), argnums=(0, 1)))
        for variant in VARIANTS
    }
    combined = {
        variant: jax.jit(jax.value_and_grad(lambda x, w, dy, v=variant: jnp.sum(scores(x, w, v) * dy), argnums=(0, 1)))
        for variant in VARIANTS
    }
    for phase, functions, arguments in (
        ("forward", forward, (inputs, weights)),
        ("backward", backward, (inputs, weights, cotangent)),
        ("combined", combined, (inputs, weights, cotangent)),
    ):
        compiled = _compile(functions, arguments, output, phase)
        if phase == "forward":
            rounded = np.asarray(compiled["rounded_preferred"](*arguments))
            preferred = np.asarray(compiled["preferred_fp32"](*arguments))
            current = np.asarray(compiled["current"](*arguments))
            routed_rounded = np.asarray(
                jax.jit(routing, static_argnames="variant")(inputs, weights, router_bias, "rounded_preferred")[0]
            )
            report["compiled_rounding"] = {
                "exact_fraction_vs_current": float(np.mean(rounded == current)),
                "exact_fraction_vs_preferred": float(np.mean(rounded == preferred)),
                "exact_fraction_vs_routing_graph": float(np.mean(rounded == routed_rounded)),
            }
            if report["compiled_rounding"]["exact_fraction_vs_preferred"] == 1.0:
                (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
                raise ValueError("The rounded control was optimized into the unrounded FP32 result")
            if report["compiled_rounding"]["exact_fraction_vs_routing_graph"] != 1.0:
                raise ValueError("Standalone rounding differs from rounded routing graph")
        if phase == "backward":
            current_grads = compiled["current"](*arguments)
            hybrid_grads = compiled["preferred_current_vjp"](*arguments)
            report["hybrid_gradient_exact_vs_current"] = [
                bool(np.asarray(jnp.all(a == b))) for a, b in zip(current_grads, hybrid_grads, strict=True)
            ]
            if not all(report["hybrid_gradient_exact_vs_current"]):
                raise ValueError("Hybrid backward gradients differ from current gradients")
        report[f"{phase}_samples"] = _time_pair(
            compiled, arguments, warmup=args.warmup, repeats=args.repeats, seed=31337
        )
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"{phase} complete", flush=True)
        if args.profile:
            trace_dir = output / f"{phase}-trace"
            with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
                for variant in VARIANTS:
                    with jax.profiler.TraceAnnotation(f"router_{phase}_{variant}"):
                        _ready(compiled[variant](*arguments))
    if args.output_prefix:
        configure_coreweave_s3()
        for path in output.rglob("*"):
            if path.is_file():
                target = args.output_prefix.rstrip("/") + "/" + path.relative_to(output).as_posix()
                with fsspec.open(target, "wb") as handle:
                    handle.write(path.read_bytes())
        print(f"uploaded {args.output_prefix}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
