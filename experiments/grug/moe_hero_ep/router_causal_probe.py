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


VARIANTS = ("current", "preferred_fp32", "rounded_preferred")


def scores(inputs: jax.Array, weights: jax.Array, variant: str) -> jax.Array:
    if variant == "current":
        return jnp.einsum("td,de->te", inputs, weights).astype(jnp.float32)
    result = jnp.einsum("td,de->te", inputs, weights, preferred_element_type=jnp.float32)
    if variant == "rounded_preferred":
        return result.astype(jnp.bfloat16).astype(jnp.float32)
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
        values[variant] = tuple(np.asarray(x) for x in jax.jit(routing, static_argnames="variant")(
            inputs, weights, bias, variant
        ))
    baseline = values["current"]
    result = {}
    for variant, (logits, indices, gates) in values.items():
        result[variant] = {
            "score_dtype": str(logits.dtype),
            "score_exact_fraction_vs_current": float(np.mean(logits == baseline[0])),
            "score_max_abs_vs_current": float(np.max(np.abs(logits - baseline[0]))),
            "route_order_changed_rows": int(np.sum(np.any(indices != baseline[1], axis=1))),
            "route_set_changed_rows": int(np.sum(np.any(np.sort(indices, axis=1) != np.sort(baseline[1], axis=1), axis=1))),
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
    weights = jnp.asarray(weight, dtype=jnp.bfloat16)
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
    for phase, functions, arguments in (
        ("forward", forward, (inputs, weights)),
        ("backward", backward, (inputs, weights, cotangent)),
    ):
        compiled = _compile(functions, arguments, output, phase)
        report[f"{phase}_samples"] = _time_pair(compiled, arguments, warmup=args.warmup, repeats=args.repeats, seed=31337)
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
